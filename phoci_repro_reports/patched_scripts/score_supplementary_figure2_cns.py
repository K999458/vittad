#!/usr/bin/env python
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoci_repro.cns_perturbation import compute_feature_deltas, summarize_cns_perturbation_rows
from phoci_repro.config import PHOCI_FEATURE_ORDER
from phoci_repro.data import interaction_span
from phoci_repro.model import PHOCIModel
from phoci_repro.precomputed import read_manifest, unpack_interactions
from phoci_repro.resources import apply_resource_limits, choose_device


PAIR_FIELDS = [
    "cell_line",
    "model",
    "split",
    "chrom",
    "subgraph_index",
    "bin_start",
    "bin_end",
    "pair_index",
    "order",
    "positive_nodes",
    "cns_nodes",
    "positive_genomic_bins",
    "cns_genomic_bins",
    "positive_score",
    "cns_score",
    "score_delta",
    "positive_span_bins",
    "cns_span_bins",
    "span_delta",
    "changed_node_count",
    "common_node_count",
    "positive_only_node",
    "cns_only_node",
    "positive_only_genomic_bin",
    "cns_only_genomic_bin",
    "largest_abs_delta_feature",
    "largest_abs_delta",
] + [f"feature_delta_{feature}" for feature in PHOCI_FEATURE_ORDER]


EXAMPLE_FIELDS = PAIR_FIELDS + ["example_rank", "example_label"]


SCAN_FIELDS = [
    "example_rank",
    "example_label",
    "cell_line",
    "model",
    "chrom",
    "subgraph_index",
    "bin_start",
    "replacement_node",
    "replacement_genomic_bin",
    "replacement_start_bp",
    "replacement_label",
    "interaction_nodes",
    "interaction_genomic_bins",
    "score",
    "span_bins",
    "distance_from_positive_node_bins",
    "distance_from_cns_node_bins",
] + [f"feature_{feature}" for feature in PHOCI_FEATURE_ORDER]


def load_model(checkpoint_path: Path, device: torch.device) -> PHOCIModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    hidden = int(checkpoint_args.get("hidden_channels", 400))
    implementation = str(checkpoint_args.get("model_implementation", "legacy"))
    model = PHOCIModel(in_channels=len(PHOCI_FEATURE_ORDER), hidden_channels=hidden, implementation=implementation)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_positive_cns(path: Path) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        positives = unpack_interactions(data["positive_offsets"], data["positive_nodes"])
        cns = unpack_interactions(data["cns_offsets"], data["cns_nodes"])
        features = np.asarray(data["features"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    if len(positives) != len(cns):
        raise ValueError(f"positive/CNS count mismatch in {path}: {len(positives)} vs {len(cns)}")
    return positives, cns, features, edge_index


def select_rows(manifest: Path, chrom: str, split: str, region_start_bin: int, region_end_bin: int) -> List[Dict[str, object]]:
    rows = []
    for row in read_manifest(manifest):
        if row["chrom"] != chrom or row["split"] != split:
            continue
        if int(row["bin_end"]) <= int(region_start_bin) or int(row["bin_start"]) >= int(region_end_bin):
            continue
        rows.append(row)
    rows.sort(key=lambda item: int(item["subgraph_index"]))
    return rows


def in_region(interaction: Sequence[int], bin_start: int, region_start_bin: int, region_end_bin: int) -> bool:
    genomic = [int(bin_start) + int(node) for node in interaction]
    return min(genomic) >= int(region_start_bin) and max(genomic) < int(region_end_bin)


def score_interactions(
    model: PHOCIModel,
    embeddings: torch.Tensor,
    interactions: Sequence[Sequence[int]],
    batch_size: int,
) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for start in range(0, len(interactions), int(batch_size)):
            batch = [tuple(int(node) for node in item) for item in interactions[start : start + int(batch_size)]]
            spans = [interaction_span(item) for item in batch]
            logits = model.classify_embeddings(embeddings, batch, spans=spans)
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def build_pair_rows(
    args,
    model: PHOCIModel,
    device: torch.device,
    row: Dict[str, object],
    pair_counter_start: int,
) -> List[Dict[str, object]]:
    positives, cns_items, features, edge_index = load_positive_cns(Path(row["shard_path"]))
    selected_indexes = []
    for index, (positive, cns) in enumerate(zip(positives, cns_items)):
        order = len(positive)
        if order < int(args.min_order) or order > int(args.max_order):
            continue
        if not in_region(positive, int(row["bin_start"]), args.region_start_bin, args.region_end_bin):
            continue
        if not in_region(cns, int(row["bin_start"]), args.region_start_bin, args.region_end_bin):
            continue
        selected_indexes.append(index)
    if args.max_pairs_per_window is not None:
        selected_indexes = selected_indexes[: int(args.max_pairs_per_window)]
    if not selected_indexes:
        return []

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.encode(x, edge_index_tensor)

    selected_positives = [positives[index] for index in selected_indexes]
    selected_cns = [cns_items[index] for index in selected_indexes]
    positive_scores = score_interactions(model, embeddings, selected_positives, args.score_batch_size)
    cns_scores = score_interactions(model, embeddings, selected_cns, args.score_batch_size)

    out = []
    for offset, (positive_index, positive, cns, positive_score, cns_score) in enumerate(
        zip(selected_indexes, selected_positives, selected_cns, positive_scores, cns_scores)
    ):
        positive_span = interaction_span(positive)
        cns_span = interaction_span(cns)
        positive_set = set(int(node) for node in positive)
        cns_set = set(int(node) for node in cns)
        positive_only = sorted(positive_set - cns_set)
        cns_only = sorted(cns_set - positive_set)
        if positive_only and cns_only:
            feature_deltas = compute_feature_deltas(
                (positive_only[0],),
                (cns_only[0],),
                {node: features[node, :].tolist() for node in {positive_only[0], cns_only[0]}},
                PHOCI_FEATURE_ORDER,
            )
        else:
            feature_deltas = compute_feature_deltas(
                positive,
                cns,
                {node: features[node, :].tolist() for node in positive_set | cns_set},
                PHOCI_FEATURE_ORDER,
            )
        positive_genomic_bins = [int(row["bin_start"]) + int(node) for node in positive]
        cns_genomic_bins = [int(row["bin_start"]) + int(node) for node in cns]
        positive_only_node = positive_only[0] if len(positive_only) == 1 else ""
        cns_only_node = cns_only[0] if len(cns_only) == 1 else ""
        pair_row = {
            "cell_line": args.cell_line,
            "model": args.model_name,
            "split": row["split"],
            "chrom": row["chrom"],
            "subgraph_index": row["subgraph_index"],
            "bin_start": int(row["bin_start"]),
            "bin_end": int(row["bin_end"]),
            "pair_index": int(pair_counter_start + offset),
            "order": int(len(positive)),
            "positive_nodes": ",".join(str(node) for node in positive),
            "cns_nodes": ",".join(str(node) for node in cns),
            "positive_genomic_bins": ",".join(str(node) for node in positive_genomic_bins),
            "cns_genomic_bins": ",".join(str(node) for node in cns_genomic_bins),
            "positive_score": float(positive_score),
            "cns_score": float(cns_score),
            "score_delta": float(positive_score) - float(cns_score),
            "positive_span_bins": int(positive_span),
            "cns_span_bins": int(cns_span),
            "span_delta": int(cns_span) - int(positive_span),
            "changed_node_count": int(max(len(positive_only), len(cns_only))),
            "common_node_count": int(len(positive_set & cns_set)),
            "positive_only_node": positive_only_node,
            "cns_only_node": cns_only_node,
            "positive_only_genomic_bin": int(row["bin_start"]) + int(positive_only_node) if positive_only_node != "" else "",
            "cns_only_genomic_bin": int(row["bin_start"]) + int(cns_only_node) if cns_only_node != "" else "",
            **feature_deltas,
        }
        out.append(pair_row)
    return out


def select_override_examples(rows: Sequence[Dict[str, object]], max_examples: int) -> List[Dict[str, object]]:
    candidates = [
        row
        for row in rows
        if int(row["order"]) in {3, 4, 5}
        and float(row["score_delta"]) > 0
        and int(row["span_delta"]) < 0
    ]
    by_order: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_order[int(row["order"])].append(row)
    selected = []
    for order in [3, 4, 5]:
        order_rows = sorted(
            by_order.get(order, []),
            key=lambda row: (float(row["score_delta"]), float(row["positive_score"]), float(row["largest_abs_delta"])),
            reverse=True,
        )
        if order_rows:
            selected.append(dict(order_rows[0]))
    if len(selected) < int(max_examples):
        seen = {int(row["pair_index"]) for row in selected}
        for row in sorted(candidates, key=lambda item: float(item["score_delta"]), reverse=True):
            if int(row["pair_index"]) in seen:
                continue
            selected.append(dict(row))
            seen.add(int(row["pair_index"]))
            if len(selected) >= int(max_examples):
                break
    for index, row in enumerate(selected[: int(max_examples)], start=1):
        row["example_rank"] = int(index)
        row["example_label"] = f"order_{row['order']}_rank_{index}"
    return selected[: int(max_examples)]


def scan_examples(
    args,
    model: PHOCIModel,
    device: torch.device,
    examples: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows_by_key = {
        (row["chrom"], str(row["subgraph_index"])): row
        for row in select_rows(Path(args.manifest), args.chrom, args.split, args.region_start_bin, args.region_end_bin)
    }
    out = []
    for example in examples:
        manifest_row = rows_by_key[(example["chrom"], str(example["subgraph_index"]))]
        _, _, features, edge_index = load_positive_cns(Path(manifest_row["shard_path"]))
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        with torch.no_grad():
            embeddings = model.encode(x, edge_index_tensor)
        positive = tuple(int(item) for item in str(example["positive_nodes"]).split(",") if item != "")
        cns = tuple(int(item) for item in str(example["cns_nodes"]).split(",") if item != "")
        common = sorted(set(positive) & set(cns))
        positive_only_text = str(example.get("positive_only_node", ""))
        cns_only_text = str(example.get("cns_only_node", ""))
        if int(example.get("changed_node_count", 0)) == 1 and positive_only_text and cns_only_text:
            positive_only = int(positive_only_text)
            cns_only = int(cns_only_text)
            fixed_nodes = common
        else:
            positive_only_candidates = sorted(set(positive) - set(cns))
            cns_only_candidates = sorted(set(cns) - set(positive))
            if not positive_only_candidates or not cns_only_candidates:
                continue
            positive_only = positive_only_candidates[0]
            cns_only = cns_only_candidates[0]
            fixed_nodes = sorted(node for node in positive if node != positive_only)
        candidates = []
        labels = {}
        lower = max(0, min(positive_only, cns_only) - int(args.scan_radius_bins))
        upper = min(features.shape[0], max(positive_only, cns_only) + int(args.scan_radius_bins) + 1)
        for node in range(lower, upper):
            if node in fixed_nodes:
                continue
            genomic_bin = int(manifest_row["bin_start"]) + int(node)
            if genomic_bin < int(args.region_start_bin) or genomic_bin >= int(args.region_end_bin):
                continue
            interaction = tuple(sorted(fixed_nodes + [int(node)]))
            if len(interaction) != len(positive):
                continue
            if int(node) not in interaction or (set(interaction) - set(fixed_nodes)) != {int(node)}:
                continue
            labels[int(node)] = "positive" if int(node) == positive_only else ("cns" if int(node) == cns_only else "scan")
            candidates.append(interaction)
        scores = score_interactions(model, embeddings, candidates, args.score_batch_size)
        for interaction, score in zip(candidates, scores):
            replacement = sorted(set(interaction) - set(fixed_nodes))[0]
            feature_values = {f"feature_{feature}": float(features[int(replacement), index]) for index, feature in enumerate(PHOCI_FEATURE_ORDER)}
            genomic_bins = [int(manifest_row["bin_start"]) + int(node) for node in interaction]
            out.append(
                {
                    "example_rank": int(example["example_rank"]),
                    "example_label": example["example_label"],
                    "cell_line": args.cell_line,
                    "model": args.model_name,
                    "chrom": manifest_row["chrom"],
                    "subgraph_index": manifest_row["subgraph_index"],
                    "bin_start": int(manifest_row["bin_start"]),
                    "replacement_node": int(replacement),
                    "replacement_genomic_bin": int(manifest_row["bin_start"]) + int(replacement),
                    "replacement_start_bp": (int(manifest_row["bin_start"]) + int(replacement)) * int(args.bin_size),
                    "replacement_label": labels[int(replacement)],
                    "interaction_nodes": ",".join(str(node) for node in interaction),
                    "interaction_genomic_bins": ",".join(str(node) for node in genomic_bins),
                    "score": float(score),
                    "span_bins": int(interaction_span(interaction)),
                    "distance_from_positive_node_bins": int(replacement) - int(positive_only),
                    "distance_from_cns_node_bins": int(replacement) - int(cns_only),
                    **feature_values,
                }
            )
    return out


def write_rows(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def plot_score_panels(pair_rows: Sequence[Dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    orders = [3, 4, 5, 6]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), dpi=180)
    positive_data = [[float(row["positive_score"]) for row in pair_rows if int(row["order"]) == order] for order in orders]
    cns_data = [[float(row["cns_score"]) for row in pair_rows if int(row["order"]) == order] for order in orders]
    delta_data = [[float(row["score_delta"]) for row in pair_rows if int(row["order"]) == order] for order in orders]

    axes[0].boxplot(positive_data, positions=np.arange(len(orders)) - 0.18, widths=0.28, patch_artist=True, boxprops={"facecolor": "#d64f45", "alpha": 0.65})
    axes[0].boxplot(cns_data, positions=np.arange(len(orders)) + 0.18, widths=0.28, patch_artist=True, boxprops={"facecolor": "#2f6db3", "alpha": 0.65})
    axes[0].set_xticks(range(len(orders)))
    axes[0].set_xticklabels([str(order) for order in orders])
    axes[0].set_xlabel("Interaction order")
    axes[0].set_ylabel("Prediction score")
    axes[0].set_title("a  Positive vs CNS")
    axes[0].plot([], [], color="#d64f45", linewidth=6, label="Positive")
    axes[0].plot([], [], color="#2f6db3", linewidth=6, label="CNS")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].boxplot(delta_data, labels=[str(order) for order in orders], patch_artist=True, boxprops={"facecolor": "#4f8f6f", "alpha": 0.7})
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("Interaction order")
    axes[1].set_ylabel("Positive score - CNS score")
    axes[1].set_title("b  CNS perturbation shift")

    x = [int(row["span_delta"]) for row in pair_rows]
    y = [float(row["score_delta"]) for row in pair_rows]
    c = [int(row["order"]) for row in pair_rows]
    axes[2].scatter(x, y, c=c, s=3, cmap="viridis", alpha=0.25, linewidths=0)
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].axvline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_xlabel("CNS span - positive span (bins)")
    axes[2].set_ylabel("Positive score - CNS score")
    axes[2].set_title("c  Distance and score")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_scan_panels(scan_rows: Sequence[Dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = sorted({int(row["example_rank"]) for row in scan_rows})
    fig, axes = plt.subplots(len(examples), 1, figsize=(8.4, 2.6 * len(examples)), dpi=180, sharex=False)
    if len(examples) == 1:
        axes = [axes]
    panel_letters = ["d", "e", "f"]
    for axis, rank, letter in zip(axes, examples, panel_letters):
        rows = sorted([row for row in scan_rows if int(row["example_rank"]) == rank], key=lambda row: int(row["replacement_genomic_bin"]))
        x = [int(row["replacement_start_bp"]) / 1e6 for row in rows]
        y = [float(row["score"]) for row in rows]
        axis.plot(x, y, color="#222222", linewidth=1.0)
        for label, color in [("positive", "#d64f45"), ("cns", "#2f6db3")]:
            subset = [row for row in rows if row["replacement_label"] == label]
            if subset:
                axis.scatter([int(row["replacement_start_bp"]) / 1e6 for row in subset], [float(row["score"]) for row in subset], s=32, color=color, zorder=3, label=label)
        axis.set_title(f"{letter}  {rows[0]['example_label']} replacement scan")
        axis.set_xlabel(f"{rows[0]['chrom']} position (Mb)")
        axis.set_ylabel("Score")
        axis.legend(frameon=False, fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_feature_panels(examples: Sequence[Dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 1, figsize=(9.8, 2.8 * len(examples)), dpi=180, sharex=True)
    if len(examples) == 1:
        axes = [axes]
    panel_letters = ["h", "i", "j"]
    for axis, example, letter in zip(axes, examples, panel_letters):
        deltas = [float(example.get(f"feature_delta_{feature}", 0.0)) for feature in PHOCI_FEATURE_ORDER]
        colors = ["#d64f45" if value > 0 else "#2f6db3" for value in deltas]
        axis.bar(range(len(PHOCI_FEATURE_ORDER)), deltas, color=colors, width=0.75)
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(f"{letter}  {example['example_label']} CNS node minus positive node")
        axis.set_ylabel("Feature delta")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xticks(range(len(PHOCI_FEATURE_ORDER)))
    axes[-1].set_xticklabels(PHOCI_FEATURE_ORDER, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = ["artifact", "path", "description"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.8g}"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and plot PHOCI Supplementary Figure 2 CNS perturbation analysis.")
    parser.add_argument("--outdir", default="outputs/paper_figures/supplementary_figure2_cns")
    parser.add_argument("--manifest", default="outputs/precomputed_paper_windows/GM12878/manifest.tsv")
    parser.add_argument("--checkpoint", default="outputs/full_training/GM12878/sage_full_model.pt")
    parser.add_argument("--model-name", default="GM12878")
    parser.add_argument("--cell-line", default="GM12878")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chrom", default="chr14")
    parser.add_argument("--region-start-bp", type=int, default=76000000)
    parser.add_argument("--region-end-bp", type=int, default=81000000)
    parser.add_argument("--bin-size", type=int, default=5000)
    parser.add_argument("--min-order", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--max-pairs-per-window", type=int, default=None)
    parser.add_argument("--scan-radius-bins", type=int, default=220)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--max-gpu-gb", type=float, default=16.0)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    args.region_start_bin = int(args.region_start_bp) // int(args.bin_size)
    args.region_end_bin = int(math.ceil(int(args.region_end_bp) / float(args.bin_size)))

    threads = apply_resource_limits(cpu_fraction=0.5, max_threads=args.max_threads)
    device = choose_device(prefer_gpu=args.prefer_gpu, max_gpu_gb=args.max_gpu_gb)
    model = load_model(Path(args.checkpoint), device)
    rows = select_rows(Path(args.manifest), args.chrom, args.split, args.region_start_bin, args.region_end_bin)
    if not rows:
        raise RuntimeError("no manifest rows overlap requested Supplementary Figure 2 region")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    for row in rows:
        if args.progress:
            print(json.dumps({"event": "score_window", "chrom": row["chrom"], "subgraph_index": row["subgraph_index"], "current_pairs": len(pair_rows)}), flush=True)
        pair_rows.extend(build_pair_rows(args, model, device, row, pair_counter_start=len(pair_rows)))

    examples = select_override_examples(pair_rows, max_examples=args.max_examples)
    scan_rows = scan_examples(args, model, device, examples)
    summary = summarize_cns_perturbation_rows(pair_rows)
    summary.update(
        {
            "cell_line": args.cell_line,
            "model": args.model_name,
            "chrom": args.chrom,
            "split": args.split,
            "region_start_bp": int(args.region_start_bp),
            "region_end_bp": int(args.region_end_bp),
            "region_start_bin": int(args.region_start_bin),
            "region_end_bin": int(args.region_end_bin),
            "manifest_rows": len(rows),
            "selected_example_count": len(examples),
            "scan_row_count": len(scan_rows),
            "device": str(device),
            "threads": int(threads),
        }
    )
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1024**3, 4)
        summary["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 4)

    pair_path = outdir / "cns_perturbation_pairs.tsv"
    example_path = outdir / "selected_override_examples.tsv"
    scan_path = outdir / "cns_distance_scan.tsv"
    summary_path = outdir / "cns_perturbation_summary.json"
    score_png = outdir / "supplementary_figure2a_c_cns_scores.png"
    score_pdf = outdir / "supplementary_figure2a_c_cns_scores.pdf"
    scan_png = outdir / "supplementary_figure2d_f_distance_scan.png"
    scan_pdf = outdir / "supplementary_figure2d_f_distance_scan.pdf"
    feature_png = outdir / "supplementary_figure2h_j_feature_deltas.png"
    feature_pdf = outdir / "supplementary_figure2h_j_feature_deltas.pdf"
    manifest_path = outdir / "supplementary_figure2_manifest.tsv"

    write_rows(pair_path, pair_rows, PAIR_FIELDS)
    write_rows(example_path, examples, EXAMPLE_FIELDS)
    write_rows(scan_path, scan_rows, SCAN_FIELDS)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_score_panels(pair_rows, score_png, score_pdf)
    plot_scan_panels(scan_rows, scan_png, scan_pdf)
    plot_feature_panels(examples, feature_png, feature_pdf)
    manifest_rows = [
        {"artifact": "pairs", "path": str(pair_path), "description": "Positive/CNS perturbation scores for GM12878 chr14 76-81Mb, orders 3-6."},
        {"artifact": "examples", "path": str(example_path), "description": "Selected epigenetic override cases used for distance and feature panels."},
        {"artifact": "distance_scan", "path": str(scan_path), "description": "Replacement-node score scans around selected override cases."},
        {"artifact": "summary", "path": str(summary_path), "description": "CNS perturbation summary by order and override feature counts."},
        {"artifact": "supplementary_figure2a_c_png", "path": str(score_png), "description": "CNS score shift panels a-c."},
        {"artifact": "supplementary_figure2a_c_pdf", "path": str(score_pdf), "description": "CNS score shift panels a-c."},
        {"artifact": "supplementary_figure2d_f_png", "path": str(scan_png), "description": "Distance scan panels d-f."},
        {"artifact": "supplementary_figure2d_f_pdf", "path": str(scan_pdf), "description": "Distance scan panels d-f."},
        {"artifact": "supplementary_figure2h_j_png", "path": str(feature_png), "description": "Feature delta panels h-j."},
        {"artifact": "supplementary_figure2h_j_pdf", "path": str(feature_pdf), "description": "Feature delta panels h-j."},
    ]
    write_manifest(manifest_path, manifest_rows)
    print(json.dumps({"event": "complete", "pairs": len(pair_rows), "examples": len(examples), "scan_rows": len(scan_rows), "summary": str(summary_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
