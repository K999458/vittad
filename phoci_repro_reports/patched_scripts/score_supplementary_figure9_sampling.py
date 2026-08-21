#!/usr/bin/env python
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoci_repro.config import PHOCI_FEATURE_ORDER
from phoci_repro.data import interaction_span
from phoci_repro.model import PHOCIModel
from phoci_repro.precomputed import read_manifest
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
    "start_bp",
    "end_bp",
    "bin_size",
    "order",
    "span_bins",
    "score",
]


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


def load_shard_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int64)
        positive_offsets = np.asarray(data["positive_offsets"], dtype=np.int64)
    order_counts = Counter(int(length) for length in np.diff(positive_offsets) if int(length) > 0)
    return features, edge_index, dict(order_counts)


def select_rows(manifest: Path, cell_line: str, split: str, chrom: Optional[str], max_windows: Optional[int]) -> List[Dict[str, object]]:
    rows = []
    for row in read_manifest(manifest):
        if row["split"] != split:
            continue
        if chrom is not None and row["chrom"] != chrom:
            continue
        item = dict(row)
        item["source_cell_line"] = cell_line
        rows.append(item)
    rows.sort(key=lambda item: (item["chrom"], int(item["subgraph_index"])))
    if max_windows is not None:
        rows = rows[: int(max_windows)]
    return rows


def order_distribution(rows: Sequence[Dict[str, object]], min_order: int, max_order: int) -> Dict[int, int]:
    counts = Counter()
    for row in rows:
        _, _, shard_counts = load_shard_arrays(Path(row["shard_path"]))
        for order, count in shard_counts.items():
            if int(min_order) <= int(order) <= int(max_order):
                counts[int(order)] += int(count)
    if not counts:
        raise RuntimeError("no positive order distribution found for requested windows")
    return dict(sorted(counts.items()))


def allocate_counts(total_count: int, weights: Mapping[object, int]) -> Dict[object, int]:
    keys = list(weights)
    total_weight = float(sum(int(weights[key]) for key in keys))
    raw = {key: float(total_count) * float(weights[key]) / total_weight for key in keys}
    out = {key: int(raw[key]) for key in keys}
    remainder = int(total_count) - sum(out.values())
    for key in sorted(keys, key=lambda item: raw[item] - out[item], reverse=True)[:remainder]:
        out[key] += 1
    return out


def random_walk_candidates(
    adjacency: Mapping[int, set],
    order_counts: Mapping[int, int],
    rng: random.Random,
    max_attempts: int,
) -> List[Tuple[int, ...]]:
    walkable = [node for node, neighbors in adjacency.items() if neighbors]
    out = []
    for order, target in sorted(order_counts.items()):
        generated_for_order = 0
        attempts = 0
        while generated_for_order < int(target) and attempts < int(max_attempts) * max(1, int(target)):
            attempts += 1
            current = int(rng.choice(walkable))
            path = [current]
            visited = {current}
            while len(path) < int(order):
                choices = [node for node in adjacency.get(current, set()) if node not in visited]
                if not choices:
                    break
                current = int(rng.choice(choices))
                path.append(current)
                visited.add(current)
            if len(path) != int(order):
                continue
            candidate = canonical_interaction(path)
            if len(candidate) == int(order):
                out.append(candidate)
                generated_for_order += 1
        if generated_for_order < int(target):
            raise RuntimeError(f"random-walk generated {generated_for_order}/{target} candidates for order {order}")
    rng.shuffle(out)
    return out


def random_choice_candidates(
    node_count: int,
    order_counts: Mapping[int, int],
    rng: random.Random,
    max_attempts: int,
) -> List[Tuple[int, ...]]:
    nodes = list(range(int(node_count)))
    out = []
    for order, target in sorted(order_counts.items()):
        generated_for_order = 0
        attempts = 0
        while generated_for_order < int(target) and attempts < int(max_attempts) * max(1, int(target)):
            attempts += 1
            candidate = canonical_interaction(rng.sample(nodes, int(order)))
            if len(candidate) == int(order):
                out.append(candidate)
                generated_for_order += 1
        if generated_for_order < int(target):
            raise RuntimeError(f"random-choice generated {generated_for_order}/{target} candidates for order {order}")
    rng.shuffle(out)
    return out


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


def write_prediction_chunk(
    writer: csv.DictWriter,
    args,
    row: Dict[str, object],
    sample_method: str,
    candidates: Sequence[Tuple[int, ...]],
    scores: Sequence[float],
    start_index: int,
) -> int:
    for offset, (candidate, score) in enumerate(zip(candidates, scores)):
        genomic_bins = [int(row["bin_start"]) + int(node) for node in candidate]
        writer.writerow(
            {
                "sample_method": sample_method,
                "model": args.model_name,
                "source_cell_line": row["source_cell_line"],
                "split": row["split"],
                "chrom": row["chrom"],
                "subgraph_index": row["subgraph_index"],
                "bin_start": row["bin_start"],
                "bin_end": row["bin_end"],
                "candidate_index": int(start_index + offset),
                "nodes": ",".join(str(node) for node in candidate),
                "genomic_bins": ",".join(str(node) for node in genomic_bins),
                "start_bp": int(min(genomic_bins) * int(args.bin_size)),
                "end_bp": int((max(genomic_bins) + 1) * int(args.bin_size)),
                "bin_size": int(args.bin_size),
                "order": int(len(candidate)),
                "span_bins": int(interaction_span(candidate)),
                "score": float(score),
            }
        )
    return int(start_index + len(candidates))


def score_sampling(args, model: PHOCIModel, device: torch.device, rows: Sequence[Dict[str, object]], out_path: Path, order_weights: Dict[int, int]) -> Dict[str, object]:
    window_weights = {index: int(row["positive_count"]) for index, row in enumerate(rows)}
    per_method_window_counts = allocate_counts(int(args.count_per_method), window_weights)
    counts_by_method = {"random_walk": 0, "random_choice": 0}
    score_pass_counts = {method: Counter() for method in counts_by_method}
    thresholds = [float(item) for item in args.thresholds.split(",") if item]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, delimiter="\t")
        writer.writeheader()
        candidate_indexes = {"random_walk": 0, "random_choice": 0}
        for window_index, row in enumerate(rows):
            window_total = per_method_window_counts[window_index]
            window_order_counts = allocate_counts(window_total, order_weights)
            features, edge_index_np, _ = load_shard_arrays(Path(row["shard_path"]))
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
            with torch.no_grad():
                embeddings = model.encode(x, edge_index)
            edges = list(zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()))
            adjacency = build_adjacency(edges)
            for sample_method in ["random_walk", "random_choice"]:
                rng = random.Random(int(args.seed) + window_index * 1009 + (0 if sample_method == "random_walk" else 503))
                if sample_method == "random_walk":
                    candidates = random_walk_candidates(adjacency, window_order_counts, rng=rng, max_attempts=args.max_attempts_per_candidate)
                else:
                    candidates = random_choice_candidates(features.shape[0], window_order_counts, rng=rng, max_attempts=args.max_attempts_per_candidate)
                scores = score_candidates(model, embeddings, candidates, args.score_batch_size)
                candidate_indexes[sample_method] = write_prediction_chunk(
                    writer, args, row, sample_method, candidates, scores, candidate_indexes[sample_method]
                )
                counts_by_method[sample_method] += len(candidates)
                for threshold in thresholds:
                    score_pass_counts[sample_method][str(threshold)] += int(np.sum(scores >= threshold))
            if args.progress:
                print(
                    json.dumps(
                        {
                            "event": "window_scored",
                            "window_index": window_index,
                            "windows": len(rows),
                            "chrom": row["chrom"],
                            "subgraph_index": row["subgraph_index"],
                            "per_method_candidates": window_total,
                            "counts_by_method": counts_by_method,
                        }
                    ),
                    flush=True,
                )
            del x, edge_index, embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "counts_by_method": counts_by_method,
        "threshold_pass_counts": {method: dict(counter) for method, counter in score_pass_counts.items()},
        "thresholds": thresholds,
    }


def stream_prediction_rows(path: Path):
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row["score"] = float(row["score"])
            row["order"] = int(row["order"])
            row["span_bins"] = int(row["span_bins"])
            row["bin_size"] = int(row["bin_size"])
            yield row


def summarize_thresholds(path: Path, thresholds: Sequence[float]) -> List[Dict[str, object]]:
    totals = Counter()
    passed = {str(threshold): Counter() for threshold in thresholds}
    score_sums = Counter()
    score_max = defaultdict(lambda: float("-inf"))
    for row in stream_prediction_rows(path):
        method = row["sample_method"]
        totals[method] += 1
        score_sums[method] += float(row["score"])
        score_max[method] = max(score_max[method], float(row["score"]))
        for threshold in thresholds:
            if float(row["score"]) >= float(threshold):
                passed[str(threshold)][method] += 1
    rows = []
    for method in sorted(totals):
        for threshold in thresholds:
            count = passed[str(threshold)][method]
            total = totals[method]
            rows.append(
                {
                    "sample_method": method,
                    "threshold": float(threshold),
                    "pass_count": int(count),
                    "total_count": int(total),
                    "pass_fraction": float(count / total) if total else 0.0,
                    "mean_score": float(score_sums[method] / total) if total else 0.0,
                    "max_score": float(score_max[method]) if total else 0.0,
                }
            )
    return rows


def build_pairwise_counts(path: Path, max_rows_per_method: Optional[int] = None) -> Dict[str, Counter]:
    counters = {"random_walk": Counter(), "random_choice": Counter()}
    seen = Counter()
    for row in stream_prediction_rows(path):
        method = row["sample_method"]
        if max_rows_per_method is not None and seen[method] >= int(max_rows_per_method):
            continue
        bins = [int(item) for item in row["genomic_bins"].split(",") if item]
        for i, left in enumerate(bins):
            for right in bins[i + 1 :]:
                a, b = sorted((int(left), int(right)))
                counters[method][(row["chrom"], a, b)] += 1
        seen[method] += 1
    return counters


def write_threshold_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fields = ["sample_method", "threshold", "pass_count", "total_count", "pass_fraction", "mean_score", "max_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row[field]) for field in fields})


def write_pairwise_counts(counters: Mapping[str, Counter], path: Path, bin_size: int) -> None:
    fields = ["sample_method", "chrom", "bin1", "bin2", "start1", "start2", "count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for method, counter in sorted(counters.items()):
            for (chrom, bin1, bin2), count in counter.most_common():
                writer.writerow(
                    {
                        "sample_method": method,
                        "chrom": chrom,
                        "bin1": int(bin1),
                        "bin2": int(bin2),
                        "start1": int(bin1) * int(bin_size),
                        "start2": int(bin2) * int(bin_size),
                        "count": int(count),
                    }
                )


def read_pairwise_counts(path: Path) -> Dict[str, Counter]:
    counters = {"random_walk": Counter(), "random_choice": Counter()}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counters[row["sample_method"]][(row["chrom"], int(row["bin1"]), int(row["bin2"]))] = int(row["count"])
    return counters


def plot_thresholds(rows: Sequence[Dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    colors = {"random_walk": "#4f8f6f", "random_choice": "#8a8f98"}
    for method in sorted({row["sample_method"] for row in rows}):
        subset = sorted([row for row in rows if row["sample_method"] == method], key=lambda item: float(item["threshold"]))
        ax.plot([float(row["threshold"]) for row in subset], [float(row["pass_fraction"]) for row in subset], marker="o", linewidth=1.6, color=colors.get(method), label=method)
    ax.set_xlabel("Prediction score threshold")
    ax.set_ylabel("Fraction of candidates above threshold")
    ax.set_title("Supplementary Figure 9e")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_pairwise_maps(counters: Mapping[str, Counter], out_png: Path, out_pdf: Path, max_bins: int = 900) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180, sharex=True, sharey=True)
    for axis, method, title in zip(axes, ["random_walk", "random_choice"], ["Random walk", "Random choice"]):
        counter = counters.get(method, Counter())
        if counter:
            xs = np.asarray([key[1] for key in counter], dtype=np.float32)
            ys = np.asarray([key[2] for key in counter], dtype=np.float32)
            weights = np.asarray([counter[key] for key in counter], dtype=np.float32)
            bins = min(int(max_bins), max(50, int(np.sqrt(len(xs)))))
            heatmap, x_edges, y_edges = np.histogram2d(xs, ys, bins=bins, weights=weights)
            heatmap = np.log1p(heatmap.T)
            axis.imshow(
                heatmap,
                cmap="magma",
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
                rasterized=True,
            )
        axis.set_title(title)
        axis.set_xlabel("Genomic bin")
        axis.set_ylabel("Genomic bin")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Supplementary Figure 9c-d pairwise maps")
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
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score PHOCI random-walk/random-choice candidates for Supplementary Figure 9.")
    parser.add_argument("--outdir", default="outputs/paper_figures/supplementary_figure9_sampling")
    parser.add_argument("--manifest", default="outputs/precomputed_paper_windows/GM12878/manifest.tsv")
    parser.add_argument("--checkpoint", default="outputs/full_training/Comprehensive/sage_full_model.pt")
    parser.add_argument("--model-name", default="Comprehensive")
    parser.add_argument("--cell-line", default="GM12878")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--count-per-method", type=int, default=2000000)
    parser.add_argument("--min-order", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--bin-size", type=int, default=5000)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--max-attempts-per-candidate", type=int, default=100)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.88,0.9,0.95")
    parser.add_argument("--pairwise-max-rows-per-method", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--max-gpu-gb", type=float, default=16.0)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    threads = apply_resource_limits(cpu_fraction=0.5, max_threads=args.max_threads)
    device = choose_device(prefer_gpu=args.prefer_gpu, max_gpu_gb=args.max_gpu_gb)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = outdir / "sampling_predictions.tsv"
    rows = select_rows(Path(args.manifest), cell_line=args.cell_line, split=args.split, chrom=args.chrom, max_windows=args.max_windows)
    if not rows:
        raise RuntimeError("no manifest rows selected for sampling")
    order_weights = order_distribution(rows, min_order=args.min_order, max_order=args.max_order)
    if args.plots_only:
        if not predictions_path.exists():
            raise RuntimeError(f"--plots-only requested but missing predictions file: {predictions_path}")
        generation_summary = {
            "counts_by_method": dict(Counter(row["sample_method"] for row in stream_prediction_rows(predictions_path))),
            "thresholds": [float(item) for item in args.thresholds.split(",") if item],
        }
    else:
        model = load_model(Path(args.checkpoint), device)
        generation_summary = score_sampling(args, model, device, rows, predictions_path, order_weights)
    threshold_rows = summarize_thresholds(predictions_path, generation_summary["thresholds"])
    threshold_path = outdir / "sampling_threshold_summary.tsv"
    write_threshold_rows(threshold_rows, threshold_path)
    pairwise_path = outdir / "sampling_pairwise_counts.tsv"
    if args.plots_only and pairwise_path.exists():
        pairwise_counters = read_pairwise_counts(pairwise_path)
    else:
        pairwise_counters = build_pairwise_counts(predictions_path, max_rows_per_method=args.pairwise_max_rows_per_method)
        write_pairwise_counts(pairwise_counters, pairwise_path, bin_size=args.bin_size)
    threshold_png = outdir / "supplementary_figure9e_threshold_fraction.png"
    threshold_pdf = outdir / "supplementary_figure9e_threshold_fraction.pdf"
    pairwise_png = outdir / "supplementary_figure9c_d_pairwise_maps.png"
    pairwise_pdf = outdir / "supplementary_figure9c_d_pairwise_maps.pdf"
    plot_thresholds(threshold_rows, threshold_png, threshold_pdf)
    plot_pairwise_maps(pairwise_counters, pairwise_png, pairwise_pdf)
    summary = {
        **generation_summary,
        "model": args.model_name,
        "cell_line": args.cell_line,
        "split": args.split,
        "chrom": args.chrom or "all_test_chroms",
        "selected_windows": len(rows),
        "count_per_method": int(args.count_per_method),
        "order_weights": order_weights,
        "sampling_event_uniqueness": "duplicates_allowed; each sampled path still has unique nodes internally",
        "device": str(device),
        "scoring_device": "preserved_from_existing_predictions" if args.plots_only else str(device),
        "plotting_device": str(device) if args.plots_only else None,
        "threads": int(threads),
        "outputs": {
            "predictions": str(predictions_path),
            "threshold_summary": str(threshold_path),
            "pairwise_counts": str(pairwise_path),
            "threshold_plot_png": str(threshold_png),
            "threshold_plot_pdf": str(threshold_pdf),
            "pairwise_plot_png": str(pairwise_png),
            "pairwise_plot_pdf": str(pairwise_pdf),
        },
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1024**3, 4)
        summary["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 4)
    summary_path = outdir / "supplementary_figure9_sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest_path = outdir / "supplementary_figure9_manifest.tsv"
    write_manifest(
        manifest_path,
        [
            {"artifact": "predictions", "path": str(predictions_path), "description": "2M random-walk and 2M random-choice scored candidates."},
            {"artifact": "threshold_summary", "path": str(threshold_path), "description": "Fraction of generated candidates above score thresholds."},
            {"artifact": "pairwise_counts", "path": str(pairwise_path), "description": "Pairwise contact counts implied by generated multi-way candidates."},
            {"artifact": "summary", "path": str(summary_path), "description": "Sampling analysis summary."},
            {"artifact": "supplementary_figure9e_png", "path": str(threshold_png), "description": "Threshold-pass comparison panel."},
            {"artifact": "supplementary_figure9c_d_png", "path": str(pairwise_png), "description": "Random-walk/random-choice pairwise maps."},
        ],
    )
    print(json.dumps({"event": "complete", "summary": str(summary_path), "predictions": str(predictions_path), "counts_by_method": generation_summary["counts_by_method"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
