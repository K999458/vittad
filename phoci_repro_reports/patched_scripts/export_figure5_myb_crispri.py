#!/usr/bin/env python
import argparse
import csv
import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoci_repro.apriori import mine_tss_centered_rules
from phoci_repro.config import PHOCI_FEATURE_ORDER
from phoci_repro.crispri import (
    build_myb_target_map,
    compute_synergy_rows,
    read_expression_workbook,
    summarize_expression_rows,
    write_tsv,
)
from phoci_repro.data import interaction_span
from phoci_repro.model import PHOCIModel
from phoci_repro.myb_rules import (
    DEFAULT_MYB_MODULES,
    DEFAULT_MYB_RELATED_BINS,
    interaction_to_transaction,
    map_related_bins_to_genome,
    module_support_rows,
    target_pair_rule_rows,
)
from phoci_repro.precomputed import load_window_shard
from phoci_repro.resources import apply_resource_limits, choose_device
from phoci_repro.sampling import build_adjacency, canonical_interaction


PREDICTION_FIELDS = [
    "sample_method",
    "model",
    "source_cell_line",
    "split",
    "chrom",
    "subgraph_index",
    "bin_start",
    "bin_end",
    "candidate_index",
    "nodes",
    "genomic_bins",
    "target_loci",
    "contains_tss",
    "contains_module",
    "start_bp",
    "end_bp",
    "bin_size",
    "order",
    "span_bins",
    "score",
]

RULE_FIELDS = [
    "antecedent",
    "consequent",
    "support",
    "confidence",
    "count",
    "transaction_total",
]

TSS_RULE_FIELDS = [
    "antecedent",
    "consequent",
    "signal_support",
    "background_support",
    "support_contrast",
    "confidence",
    "lift",
    "signal_count",
    "background_count",
    "signal_total",
    "background_total",
]

MODULE_FIELDS = ["module", "members", "count", "total", "support"]

EXPRESSION_FIELDS = [
    "target_gene",
    "sample_name",
    "loci",
    "order",
    "replicate_count",
    "mean_log2_fc",
    "sd_log2_fc",
    "min_log2_fc",
    "max_log2_fc",
]

SYNERGY_FIELDS = [
    "target_gene",
    "sample_name",
    "loci",
    "observed_mean_log2_fc",
    "additive_expected_log2_fc",
    "synergy_delta_log2_fc",
    "single_locus_terms",
    "interpretation",
]

MYB_TARGET_FIELDS = [
    "target_gene",
    "locus",
    "related_bin",
    "genomic_bin",
    "sgRNA_count",
    "single_mean_log2_fc",
]


def load_model(checkpoint_path: Path, device: torch.device) -> PHOCIModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model = PHOCIModel(
        in_channels=len(PHOCI_FEATURE_ORDER),
        hidden_channels=int(checkpoint_args.get("hidden_channels", 400)),
        implementation=str(checkpoint_args.get("model_implementation", "legacy")),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_shard(path: Path) -> Dict[str, object]:
    shard = load_window_shard(path)
    shard["metadata"]["bin_size"] = int(shard["metadata"].get("bin_size", 5000))
    shard["positive_count"] = int(shard["metadata"].get("positive_count", int(np.sum(shard["labels"] == 1))))
    return shard


def order_weights_from_positives(positives: Sequence[Sequence[int]], min_order: int, max_order: int) -> Dict[int, int]:
    counts = Counter(len(item) for item in positives if int(min_order) <= len(item) <= int(max_order))
    if not counts:
        raise RuntimeError("no positive interactions in requested order range")
    return dict(sorted(counts.items()))


def allocate_counts(total_count: int, weights: Mapping[int, int]) -> Dict[int, int]:
    keys = list(weights)
    total_weight = float(sum(int(weights[key]) for key in keys))
    raw = {key: float(total_count) * float(weights[key]) / total_weight for key in keys}
    out = {key: int(raw[key]) for key in keys}
    remainder = int(total_count) - sum(out.values())
    for key in sorted(keys, key=lambda item: raw[item] - out[item], reverse=True)[:remainder]:
        out[key] += 1
    return out


def random_walk_from_anchor(
    adjacency: Mapping[int, Set[int]],
    anchor: int,
    order_counts: Mapping[int, int],
    rng: random.Random,
    max_attempts_per_candidate: int,
    progress_every: int = 50000,
) -> List[Tuple[int, ...]]:
    out = []
    seen = set()
    neighbors_by_distance = nodes_by_distance(adjacency, anchor)
    for order, target in sorted(order_counts.items()):
        generated = 0
        attempts = 0
        pool = [node for node in neighbors_by_distance if node != int(anchor)]
        if len(pool) < int(order) - 1:
            raise RuntimeError(f"not enough MYB-neighborhood nodes for order {order}")
        combination_iterator = combinations(pool, int(order) - 1)
        while generated < int(target):
            try:
                combo = next(combination_iterator)
            except StopIteration:
                break
            attempts += 1
            candidate = canonical_interaction((int(anchor),) + tuple(combo))
            if len(candidate) == int(order) and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
                generated += 1
                if progress_every and generated % int(progress_every) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "candidate_generation_progress",
                                "order": int(order),
                                "generated_for_order": int(generated),
                                "target_for_order": int(target),
                                "attempts_for_order": int(attempts),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if generated < int(target):
            raise RuntimeError(f"generated {generated}/{target} MYB anchored candidates for order {order}")
    rng.shuffle(out)
    return out


def nodes_by_distance(adjacency: Mapping[int, Set[int]], anchor: int) -> List[int]:
    seen = {int(anchor)}
    frontier = [int(anchor)]
    ordered = [int(anchor)]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in sorted(adjacency.get(node, set())):
                neighbor = int(neighbor)
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                ordered.append(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
    return ordered


def score_candidates(
    model: PHOCIModel,
    embeddings: torch.Tensor,
    candidates: Sequence[Tuple[int, ...]],
    batch_size: int,
) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for start in range(0, len(candidates), int(batch_size)):
            batch = candidates[start : start + int(batch_size)]
            spans = [interaction_span(item) for item in batch]
            logits = model.classify_embeddings(embeddings, batch, spans=spans)
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def write_rows(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})
            count += 1
    return count


def stream_prediction_transactions(
    path: Path,
    min_score: float,
    max_score: float = None,
) -> List[Set[str]]:
    transactions = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            score = float(row["score"])
            if score < float(min_score):
                continue
            if max_score is not None and score > float(max_score):
                continue
            transactions.append(set(item for item in row["target_loci"].split(";") if item))
    return transactions


def read_prediction_sets_for_tss(
    path: Path,
    tss_item: str,
    signal_min_score: float,
    background_max_score: float,
) -> Tuple[List[Set[str]], List[Set[str]], int, int]:
    signal = []
    background = []
    total = 0
    tss_rows = 0
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            total += 1
            transaction = set(item for item in row["target_loci"].split(";") if item)
            if tss_item not in transaction:
                continue
            tss_rows += 1
            score = float(row["score"])
            if score >= float(signal_min_score):
                signal.append(transaction)
            if score <= float(background_max_score):
                background.append(transaction)
    return signal, background, total, tss_rows


def positive_transactions(shard: Dict[str, object], target_bins: Mapping[str, int]) -> List[Set[str]]:
    metadata = shard["metadata"]
    chrom = str(metadata["chrom"])
    bin_start = int(metadata["bin_start"])
    return [
        interaction_to_transaction(interaction, chrom=chrom, bin_start=bin_start, target_bins=target_bins)
        for interaction in shard["interactions"][: shard["positive_count"]]
    ]


def generate_prediction_rows(
    args,
    shard: Dict[str, object],
    candidates: Sequence[Tuple[int, ...]],
    scores: Sequence[float],
    target_bins: Mapping[str, int],
) -> Iterable[Dict[str, object]]:
    metadata = shard["metadata"]
    chrom = str(metadata["chrom"])
    bin_start = int(metadata["bin_start"])
    bin_end = int(metadata["bin_end"])
    bin_size = int(metadata.get("bin_size", 5000))
    module_sets = {name: set(members) for name, members in DEFAULT_MYB_MODULES.items()}
    for index, (candidate, score) in enumerate(zip(candidates, scores)):
        genomic_bins = [bin_start + int(node) for node in candidate]
        transaction = interaction_to_transaction(candidate, chrom=chrom, bin_start=bin_start, target_bins=target_bins)
        target_loci = sorted(transaction)
        contains_modules = [name for name, members in module_sets.items() if members.issubset(transaction)]
        yield {
            "sample_method": "myb_tss_random_walk",
            "model": args.model_name,
            "source_cell_line": args.cell_line,
            "split": metadata.get("split", args.split),
            "chrom": chrom,
            "subgraph_index": metadata.get("subgraph_index", args.subgraph_index),
            "bin_start": bin_start,
            "bin_end": bin_end,
            "candidate_index": index,
            "nodes": ",".join(str(node) for node in candidate),
            "genomic_bins": ",".join(str(node) for node in genomic_bins),
            "target_loci": ";".join(target_loci),
            "contains_tss": "TSS" in transaction,
            "contains_module": ";".join(contains_modules) if contains_modules else "",
            "start_bp": int(min(genomic_bins) * bin_size),
            "end_bp": int((max(genomic_bins) + 1) * bin_size),
            "bin_size": bin_size,
            "order": len(candidate),
            "span_bins": interaction_span(candidate),
            "score": float(score),
        }


def plot_rule_panels(rule_rows, module_rows, target_map_rows, out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_rules = sorted(rule_rows, key=lambda row: (float(row["support"]), float(row["confidence"])), reverse=True)[:12]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=180)
    labels = [f"{row['antecedent']}->{row['consequent']}" for row in top_rules]
    y = np.arange(len(labels))
    axes[0].barh(y, [float(row["support"]) for row in top_rules], color="#4f8f6f")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Support")
    axes[0].set_title("Figure 5a rules")
    module_labels = [row["module"] for row in module_rows]
    axes[1].bar(module_labels, [float(row["support"]) for row in module_rows], color="#d07c3f")
    axes[1].set_ylabel("Support")
    axes[1].set_title("MYB modules")
    axes[1].tick_params(axis="x", rotation=25)
    xs = [int(row["genomic_bin"]) for row in target_map_rows if str(row["genomic_bin"])]
    names = [row["locus"] for row in target_map_rows if str(row["genomic_bin"])]
    axes[2].scatter(xs, [0] * len(xs), s=70, color="#6b5fb5")
    for x, name in zip(xs, names):
        axes[2].text(x, 0.02, name, ha="center", va="bottom", fontsize=8)
    axes[2].set_yticks([])
    axes[2].set_xlabel("5 kb genomic bin")
    axes[2].set_title("Locus bins")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_crispri_panels(expression_rows, synergy_rows, out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    myb_expr = [row for row in expression_rows if row["target_gene"] == "MYB"]
    myb_syn = [row for row in synergy_rows if row["target_gene"] == "MYB"]
    pair_order = ["A;C", "B;C", "A;B"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), dpi=180)
    for ax, pair in zip(axes, pair_order):
        expr = [row for row in myb_expr if row["loci"] == pair]
        syn = [row for row in myb_syn if row["loci"] == pair]
        values = [float(row["mean_log2_fc"]) for row in expr]
        x = np.arange(len(values))
        colors = ["#c64f6a" if float(row["synergy_delta_log2_fc"]) < 0 else "#4f79b8" for row in syn]
        ax.bar(x, values, color=colors[: len(values)])
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_title(f"{pair} CRISPRi")
        ax.set_xlabel("sgRNA pair")
        ax.set_ylabel("MYB log2 fold change")
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = ["artifact", "path", "description"]
    write_rows(path, rows, fields)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Figure 5 MYB Apriori and CRISPRi panels.")
    parser.add_argument("--outdir", default="outputs/paper_figures/figure5_myb_crispri")
    parser.add_argument("--shard", default="outputs/precomputed_paper_windows/K562/train/chr6_0027.npz")
    parser.add_argument("--checkpoint", default="outputs/full_training/Comprehensive/sage_full_model.pt")
    parser.add_argument("--workbook", default="../papers/phoci_supplementary_data_crispri_gene_expression.xlsx")
    parser.add_argument("--cell-line", default="K562")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="Comprehensive")
    parser.add_argument("--subgraph-index", default="27")
    parser.add_argument("--tss-bin", type=int, default=27036)
    parser.add_argument("--candidate-count", type=int, default=1000000)
    parser.add_argument("--min-order", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--signal-min-score", type=float, default=0.3)
    parser.add_argument("--background-max-score", type=float, default=0.5)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--max-attempts-per-candidate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--max-gpu-gb", type=float, default=16.0)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    threads = apply_resource_limits(cpu_fraction=0.5, max_threads=args.max_threads)
    device = choose_device(prefer_gpu=args.prefer_gpu, max_gpu_gb=args.max_gpu_gb)

    target_bins = map_related_bins_to_genome(args.tss_bin, DEFAULT_MYB_RELATED_BINS)
    shard = load_shard(Path(args.shard))
    metadata = shard["metadata"]
    metadata.update({"split": args.split, "subgraph_index": args.subgraph_index})
    tss_node = int(args.tss_bin) - int(metadata["bin_start"])
    if tss_node < 0 or tss_node >= shard["features"].shape[0]:
        raise RuntimeError("MYB TSS bin is outside the selected shard")

    order_weights = order_weights_from_positives(shard["interactions"][: shard["positive_count"]], args.min_order, args.max_order)
    order_counts = allocate_counts(args.candidate_count, order_weights)
    adjacency = build_adjacency(zip(shard["edge_index"][0].tolist(), shard["edge_index"][1].tolist()))
    predictions_path = outdir / "myb_tss_random_walk_predictions.tsv"
    if args.reuse_predictions:
        if not predictions_path.exists():
            raise RuntimeError(f"--reuse-predictions requested but missing {predictions_path}")
        prediction_rows = None
        candidates = []
        print(json.dumps({"event": "reuse_predictions", "path": str(predictions_path)}, sort_keys=True), flush=True)
    else:
        print(
            json.dumps(
                {
                    "event": "candidate_generation_start",
                    "candidate_count": int(args.candidate_count),
                    "order_counts": order_counts,
                    "tss_node": int(tss_node),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        candidates = random_walk_from_anchor(
            adjacency,
            anchor=tss_node,
            order_counts=order_counts,
            rng=random.Random(args.seed),
            max_attempts_per_candidate=args.max_attempts_per_candidate,
        )
        print(json.dumps({"event": "candidate_generation_complete", "candidate_count": len(candidates)}, sort_keys=True), flush=True)
        model = load_model(Path(args.checkpoint), device)
        x = torch.as_tensor(shard["features"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(shard["edge_index"], dtype=torch.long, device=device)
        print(json.dumps({"event": "encoding_start", "device": str(device)}, sort_keys=True), flush=True)
        with torch.no_grad():
            embeddings = model.encode(x, edge_index)
        print(json.dumps({"event": "scoring_start", "candidate_count": len(candidates)}, sort_keys=True), flush=True)
        scores = score_candidates(model, embeddings, candidates, args.score_batch_size)
        print(json.dumps({"event": "scoring_complete", "candidate_count": len(scores)}, sort_keys=True), flush=True)
        prediction_rows = list(generate_prediction_rows(args, shard, candidates, scores, target_bins))
        write_rows(predictions_path, prediction_rows, PREDICTION_FIELDS)
        print(json.dumps({"event": "predictions_written", "path": str(predictions_path), "rows": len(prediction_rows)}, sort_keys=True), flush=True)

    signal, background, total_predictions, tss_predictions = read_prediction_sets_for_tss(
        predictions_path,
        tss_item="TSS",
        signal_min_score=args.signal_min_score,
        background_max_score=args.background_max_score,
    )
    tss_rules = mine_tss_centered_rules(
        signal,
        background,
        tss_item="TSS",
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )
    tss_rules_path = outdir / "figure5a_myb_tss_centered_rules.tsv"
    write_rows(tss_rules_path, tss_rules, TSS_RULE_FIELDS)

    high_transactions = stream_prediction_transactions(predictions_path, min_score=args.signal_min_score)
    pair_rules = target_pair_rule_rows(high_transactions)
    pair_rules_path = outdir / "figure5a_myb_target_pair_rules.tsv"
    write_rows(pair_rules_path, pair_rules, RULE_FIELDS)
    module_rows = module_support_rows(high_transactions)
    module_path = outdir / "figure5a_myb_module_support.tsv"
    write_rows(module_path, module_rows, MODULE_FIELDS)

    experimental_transactions = positive_transactions(shard, target_bins)
    experimental_rules = target_pair_rule_rows(experimental_transactions)
    experimental_rules_path = outdir / "supplementary_figure11_experimental_myb_rules.tsv"
    write_rows(experimental_rules_path, experimental_rules, RULE_FIELDS)
    experimental_module_rows = module_support_rows(experimental_transactions)
    experimental_module_path = outdir / "supplementary_figure11_experimental_myb_module_support.tsv"
    write_rows(experimental_module_path, experimental_module_rows, MODULE_FIELDS)

    expression_rows = summarize_expression_rows(read_expression_workbook(Path(args.workbook)))
    synergy_rows = compute_synergy_rows(expression_rows)
    expression_path = outdir / "crispri_expression_summary.tsv"
    synergy_path = outdir / "crispri_synergy_summary.tsv"
    write_tsv(expression_rows, expression_path, EXPRESSION_FIELDS)
    write_tsv(synergy_rows, synergy_path, SYNERGY_FIELDS)
    target_map_rows, target_map_summary = build_myb_target_map(
        expression_rows,
        synergy_rows,
        tss_genomic_bin=args.tss_bin,
        local_related_bins=DEFAULT_MYB_RELATED_BINS,
    )
    for row in target_map_rows:
        locus = str(row["locus"])
        row["genomic_bin"] = int(target_bins[locus])
    target_map_summary["genomic_bins"] = {locus: int(value) for locus, value in sorted(target_bins.items())}
    target_map_path = outdir / "myb_crispri_target_map.tsv"
    target_map_summary_path = outdir / "myb_crispri_target_map_summary.json"
    write_tsv(target_map_rows, target_map_path, MYB_TARGET_FIELDS)
    target_map_summary_path.write_text(json.dumps(target_map_summary, indent=2, sort_keys=True) + "\n")

    figure5_rules_png = outdir / "figure5a_b_myb_rule_panels.png"
    figure5_rules_pdf = outdir / "figure5a_b_myb_rule_panels.pdf"
    figure5_crispri_png = outdir / "figure5c_h_crispri_panels.png"
    figure5_crispri_pdf = outdir / "figure5c_h_crispri_panels.pdf"
    plot_rule_panels(pair_rules, module_rows, target_map_rows, figure5_rules_png, figure5_rules_pdf)
    plot_crispri_panels(expression_rows, synergy_rows, figure5_crispri_png, figure5_crispri_pdf)

    manifest_path = outdir / "figure5_supp11_manifest.tsv"
    write_manifest(
        manifest_path,
        [
            {"artifact": "myb_predictions", "path": str(predictions_path), "description": "Full MYB TSS-anchored random-walk predictions scored by the comprehensive PHOCI model."},
            {"artifact": "myb_tss_rules", "path": str(tss_rules_path), "description": "TSS-centered Apriori rules from high-score MYB predictions."},
            {"artifact": "myb_pair_rules", "path": str(pair_rules_path), "description": "Target-locus pair rules from high-score MYB predictions."},
            {"artifact": "myb_module_support", "path": str(module_path), "description": "Support for TSS-A-C, TSS-B-C, and TSS-A-B modules in high-score predictions."},
            {"artifact": "experimental_myb_rules", "path": str(experimental_rules_path), "description": "Apriori-style target-locus rules from experimental K562 Pore-C positives in the MYB shard."},
            {"artifact": "experimental_myb_modules", "path": str(experimental_module_path), "description": "Experimental Pore-C module support for MYB loci."},
            {"artifact": "crispri_expression", "path": str(expression_path), "description": "CRISPRi qRT-PCR expression summaries parsed from the supplementary workbook."},
            {"artifact": "crispri_synergy", "path": str(synergy_path), "description": "CRISPRi additive vs observed synergy summaries."},
            {"artifact": "figure5_rules_png", "path": str(figure5_rules_png), "description": "Figure 5a-b rule/module panel."},
            {"artifact": "figure5_crispri_png", "path": str(figure5_crispri_png), "description": "Figure 5c-h CRISPRi panel."},
        ],
    )

    summary = {
        "cell_line": args.cell_line,
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "shard": args.shard,
        "workbook": args.workbook,
        "device": str(device),
        "threads": int(threads),
        "candidate_count": int(args.candidate_count) if args.reuse_predictions else len(candidates),
        "prediction_rows": total_predictions,
        "order_weights": order_weights,
        "order_counts": order_counts,
        "target_bins": target_bins,
        "tss_node": tss_node,
        "signal_min_score": float(args.signal_min_score),
        "background_max_score": float(args.background_max_score),
        "total_predictions": total_predictions,
        "tss_predictions": tss_predictions,
        "signal_transactions": len(signal),
        "background_transactions": len(background),
        "high_transactions": len(high_transactions),
        "tss_rule_count": len(tss_rules),
        "pair_rule_count": len(pair_rules),
        "module_rows": module_rows,
        "experimental_transactions": len(experimental_transactions),
        "experimental_rule_count": len(experimental_rules),
        "expression_rows": len(expression_rows),
        "synergy_rows": len(synergy_rows),
        "outputs": {
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
            "tss_rules": str(tss_rules_path),
            "pair_rules": str(pair_rules_path),
            "module_support": str(module_path),
            "experimental_rules": str(experimental_rules_path),
            "experimental_module_support": str(experimental_module_path),
            "expression": str(expression_path),
            "synergy": str(synergy_path),
            "target_map": str(target_map_path),
            "target_map_summary": str(target_map_summary_path),
            "figure5_rules_png": str(figure5_rules_png),
            "figure5_crispri_png": str(figure5_crispri_png),
        },
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1024**3, 4)
        summary["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 4)
    summary_path = outdir / "figure5_myb_crispri_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "summary": str(summary_path), "candidate_count": summary["candidate_count"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
