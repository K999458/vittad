#!/usr/bin/env python
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoci_repro.config import PHOCI_FEATURE_ORDER
from phoci_repro.embeddings import (
    assign_compartment_label,
    compute_embedding_analysis,
    plot_embedding_analysis,
    read_compartment_scores,
    write_embedding_summary,
    write_embedding_table,
)
from phoci_repro.model import PHOCIModel
from phoci_repro.precomputed import load_window_shard, read_manifest
from phoci_repro.resources import apply_resource_limits, choose_device


DEFAULT_CASES = [
    {
        "case_id": "gm12878_intra_chr14",
        "model": "GM12878",
        "cell_line": "GM12878",
        "title": "GM12878 intra-cell model on GM12878 chr14",
        "checkpoint": "outputs/full_training/GM12878/sage_full_model.pt",
        "manifest": "outputs/precomputed_paper_windows/GM12878/manifest.tsv",
        "compartment": "/storu/ysu/hiporec/deep-learning/Data/GM12878_compartments.bw",
        "porec_mcool": "/storu/ysu/hiporec/hiporec_data/mcool/GM12878_PoreC_1kb.mcool",
    },
    {
        "case_id": "k562_intra_chr14",
        "model": "K562",
        "cell_line": "K562",
        "title": "K562 intra-cell model on K562 chr14",
        "checkpoint": "outputs/full_training/K562/sage_full_model.pt",
        "manifest": "outputs/precomputed_paper_windows/K562/manifest.tsv",
        "compartment": "/storu/ysu/hiporec/deep-learning/Data/K562/K562_compartments.bw",
        "porec_mcool": "/storu/ysu/hiporec/hiporec_data/mcool/K562_PoreC_1kb.mcool",
    },
    {
        "case_id": "comprehensive_gm12878_chr14",
        "model": "Comprehensive",
        "cell_line": "GM12878",
        "title": "Comprehensive model on GM12878 chr14",
        "checkpoint": "outputs/full_training/Comprehensive/sage_full_model.pt",
        "manifest": "outputs/precomputed_paper_windows/GM12878/manifest.tsv",
        "compartment": "/storu/ysu/hiporec/deep-learning/Data/GM12878_compartments.bw",
        "porec_mcool": "/storu/ysu/hiporec/hiporec_data/mcool/GM12878_PoreC_1kb.mcool",
    },
    {
        "case_id": "comprehensive_k562_chr14",
        "model": "Comprehensive",
        "cell_line": "K562",
        "title": "Comprehensive model on K562 chr14",
        "checkpoint": "outputs/full_training/Comprehensive/sage_full_model.pt",
        "manifest": "outputs/precomputed_paper_windows/K562/manifest.tsv",
        "compartment": "/storu/ysu/hiporec/deep-learning/Data/K562/K562_compartments.bw",
        "porec_mcool": "/storu/ysu/hiporec/hiporec_data/mcool/K562_PoreC_1kb.mcool",
    },
]


def resolve_cases(training_root: Path, precompute_root: Path) -> List[Dict[str, str]]:
    cases = []
    for case in DEFAULT_CASES:
        item = dict(case)
        if item["model"] in {"GM12878", "K562", "Comprehensive"}:
            item["checkpoint"] = str(training_root / item["model"] / "sage_full_model.pt")
        if item["cell_line"] in {"GM12878", "K562"}:
            item["manifest"] = str(precompute_root / item["cell_line"] / "manifest.tsv")
        cases.append(item)
    return cases


NODE_FIELDS = [
    "case_id",
    "model",
    "cell_line",
    "chrom",
    "split",
    "node_id",
    "bin",
    "start",
    "end",
    "compartment_score",
    "compartment",
    "embedding_norm",
    "umap_1",
    "umap_2",
    "cluster",
]


CLUSTER_FIELDS = [
    "case_id",
    "model",
    "cell_line",
    "chrom",
    "split",
    "cluster",
    "node_count",
    "start_min",
    "end_max",
    "mean_compartment_score",
    "mean_embedding_norm",
    "compartment_A",
    "compartment_B",
    "compartment_neutral",
    "compartment_unknown",
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


def select_rows(manifest_path: Path, split: str, chrom: str) -> List[Dict[str, object]]:
    rows = [row for row in read_manifest(manifest_path) if row["split"] == split and row["chrom"] == chrom]
    rows.sort(key=lambda row: int(row["subgraph_index"]))
    return rows


def encode_case(
    case: Dict[str, str],
    model: PHOCIModel,
    split: str,
    chrom: str,
    bin_size: int,
    device: torch.device,
    progress: bool,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    rows = select_rows(Path(case["manifest"]), split=split, chrom=chrom)
    if not rows:
        raise RuntimeError(f"no rows found for {case['case_id']} split={split} chrom={chrom}")

    embedding_sums: Dict[int, np.ndarray] = {}
    embedding_counts: Dict[int, int] = {}
    processed_rows = 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            if progress:
                print(
                    json.dumps(
                        {
                            "event": "encode_window",
                            "case_id": case["case_id"],
                            "window_index": index,
                            "windows": len(rows),
                            "chrom": row["chrom"],
                            "subgraph_index": row["subgraph_index"],
                        }
                    ),
                    flush=True,
                )
            shard = load_window_shard(Path(row["shard_path"]), packed_only=True)
            x = torch.as_tensor(shard["features"], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(shard["edge_index"], dtype=torch.long, device=device)
            embeddings = model.encode(x, edge_index).detach().cpu().numpy().astype(np.float32)
            bin_start = int(row["bin_start"])
            for local_index, vector in enumerate(embeddings):
                absolute_bin = bin_start + int(local_index)
                if absolute_bin in embedding_sums:
                    embedding_sums[absolute_bin] += vector
                    embedding_counts[absolute_bin] += 1
                else:
                    embedding_sums[absolute_bin] = vector.copy()
                    embedding_counts[absolute_bin] = 1
            processed_rows += 1

    bins = np.array(sorted(embedding_sums), dtype=np.int64)
    matrix = np.vstack([embedding_sums[int(bin_id)] / float(embedding_counts[int(bin_id)]) for bin_id in bins]).astype(np.float32)
    min_bin = int(bins.min())
    max_bin = int(bins.max())
    full_scores = read_compartment_scores(Path(case["compartment"]), chrom=chrom, bin_start=min_bin, bin_count=max_bin - min_bin + 1, bin_size=bin_size)
    scores = [float(full_scores[int(bin_id) - min_bin]) for bin_id in bins]

    node_rows = []
    for node_id, (absolute_bin, score, vector) in enumerate(zip(bins.tolist(), scores, matrix)):
        start = int(absolute_bin) * int(bin_size)
        node_rows.append(
            {
                "case_id": case["case_id"],
                "model": case["model"],
                "cell_line": case["cell_line"],
                "chrom": chrom,
                "split": split,
                "node_id": int(node_id),
                "bin": int(absolute_bin),
                "start": int(start),
                "end": int(start + bin_size),
                "compartment_score": float(score),
                "compartment": assign_compartment_label(float(score)),
                "embedding_norm": float(np.linalg.norm(vector)),
            }
        )

    coverage = {
        "manifest_rows": int(len(rows)),
        "processed_rows": int(processed_rows),
        "node_count": int(matrix.shape[0]),
        "bin_start": int(min_bin),
        "bin_end": int(max_bin + 1),
        "start_bp": int(min_bin * bin_size),
        "end_bp": int((max_bin + 1) * bin_size),
    }
    return matrix, node_rows, coverage


def attach_case_fields(rows: Iterable[Dict[str, object]], case: Dict[str, str], chrom: str, split: str) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        item["case_id"] = case["case_id"]
        item["model"] = case["model"]
        item["cell_line"] = case["cell_line"]
        item["chrom"] = chrom
        item["split"] = split
        out.append(item)
    return out


def write_case_nodes(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in NODE_FIELDS})


def cluster_count_rows(case: Dict[str, str], rows: Sequence[Dict[str, object]], chrom: str, split: str) -> List[Dict[str, object]]:
    clusters = sorted({int(row["cluster"]) for row in rows})
    out = []
    for cluster in clusters:
        subset = [row for row in rows if int(row["cluster"]) == cluster]
        scores = [float(row["compartment_score"]) for row in subset if math.isfinite(float(row["compartment_score"]))]
        norms = [float(row["embedding_norm"]) for row in subset if math.isfinite(float(row["embedding_norm"]))]
        compartments = {"A": 0, "B": 0, "neutral": 0, "unknown": 0}
        for row in subset:
            compartments[str(row.get("compartment", "unknown"))] = compartments.get(str(row.get("compartment", "unknown")), 0) + 1
        out.append(
            {
                "case_id": case["case_id"],
                "model": case["model"],
                "cell_line": case["cell_line"],
                "chrom": chrom,
                "split": split,
                "cluster": int(cluster),
                "node_count": int(len(subset)),
                "start_min": int(min(int(row["start"]) for row in subset)),
                "end_max": int(max(int(row["end"]) for row in subset)),
                "mean_compartment_score": float(np.mean(scores)) if scores else float("nan"),
                "mean_embedding_norm": float(np.mean(norms)) if norms else float("nan"),
                "compartment_A": int(compartments.get("A", 0)),
                "compartment_B": int(compartments.get("B", 0)),
                "compartment_neutral": int(compartments.get("neutral", 0)),
                "compartment_unknown": int(compartments.get("unknown", 0)),
            }
        )
    return out


def write_cluster_counts(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLUSTER_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in CLUSTER_FIELDS})


def save_embedding_npz(
    path: Path,
    embeddings: np.ndarray,
    rows: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        bins=np.asarray([int(row["bin"]) for row in rows], dtype=np.int64),
        starts=np.asarray([int(row["start"]) for row in rows], dtype=np.int64),
        ends=np.asarray([int(row["end"]) for row in rows], dtype=np.int64),
        compartment_scores=np.asarray([float(row["compartment_score"]) for row in rows], dtype=np.float32),
        clusters=np.asarray([int(row["cluster"]) for row in rows], dtype=np.int32),
        umap=np.asarray([[float(row["umap_1"]), float(row["umap_2"])] for row in rows], dtype=np.float32),
    )


def choose_cooler_uri(mcool_path: Path, target_resolution: int, min_resolution: int) -> Optional[str]:
    import cooler

    uris = cooler.fileops.list_coolers(str(mcool_path))
    parsed = []
    for uri in uris:
        try:
            resolution = int(str(uri).rstrip("/").split("/")[-1])
        except ValueError:
            continue
        if resolution >= int(min_resolution):
            parsed.append((abs(resolution - int(target_resolution)), resolution, uri))
    if not parsed:
        return None
    parsed.sort()
    return parsed[0][2]


def load_contact_matrix(
    mcool_path: Path,
    chrom: str,
    start: int,
    end: int,
    target_resolution: int,
    min_resolution: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    import cooler

    if not mcool_path.exists():
        return None, {"status": "missing_mcool", "path": str(mcool_path)}
    uri = choose_cooler_uri(mcool_path, target_resolution=target_resolution, min_resolution=min_resolution)
    if uri is None:
        available = cooler.fileops.list_coolers(str(mcool_path))
        return None, {"status": "no_usable_resolution", "path": str(mcool_path), "available": available}
    cooler_uri = f"{mcool_path}::{uri}"
    cool = cooler.Cooler(cooler_uri)
    if chrom not in cool.chromsizes:
        return None, {"status": "missing_chrom", "path": str(mcool_path), "uri": uri, "chrom": chrom}
    chrom_size = int(cool.chromsizes[chrom])
    clipped_start = min(max(0, int(start)), chrom_size)
    clipped_end = min(max(clipped_start, int(end)), chrom_size)
    if clipped_end <= clipped_start:
        return None, {"status": "empty_region", "path": str(mcool_path), "uri": uri, "chrom": chrom, "start": int(start), "end": int(end), "chrom_size": chrom_size}
    region = f"{chrom}:{clipped_start}-{clipped_end}"
    matrix = cool.matrix(balance=False).fetch(region)
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix, {"status": "ok", "path": str(mcool_path), "uri": uri, "resolution": int(cool.binsize), "region": region}


def plot_contact_cluster_panel(
    rows: Sequence[Dict[str, object]],
    case: Dict[str, str],
    matrix: Optional[np.ndarray],
    contact_meta: Dict[str, object],
    out_png: Path,
    out_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    start = min(int(row["start"]) for row in rows)
    end = max(int(row["end"]) for row in rows)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), dpi=180, gridspec_kw={"height_ratios": [4, 1.15]}, sharex=True)
    ax_map, ax_track = axes
    if matrix is not None and matrix.size:
        finite = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.log1p(finite)
        extent = [start / 1e6, end / 1e6, end / 1e6, start / 1e6]
        ax_map.imshow(image, cmap="magma", extent=extent, aspect="auto", interpolation="nearest")
        resolution = contact_meta.get("resolution", "")
        ax_map.set_title(f"{case['title']} Pore-C contact map ({resolution} bp)")
        ax_map.set_ylabel(f"{rows[0]['chrom']} Mb")
    else:
        ax_map.text(0.5, 0.5, str(contact_meta.get("status", "contact map unavailable")), ha="center", va="center", transform=ax_map.transAxes)
        ax_map.set_title(f"{case['title']} Pore-C contact map unavailable")
        ax_map.set_ylabel(f"{rows[0]['chrom']} Mb")

    clusters = [int(row["cluster"]) for row in rows]
    starts = [(int(row["start"]) + int(row["end"])) / 2e6 for row in rows]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(cluster % 10) for cluster in clusters]
    ax_track.scatter(starts, clusters, c=colors, s=2, linewidths=0)
    ax_track.set_yticks(sorted(set(clusters)))
    ax_track.set_ylabel("Cluster")
    ax_track.set_xlabel(f"{rows[0]['chrom']} position (Mb)")
    ax_track.set_xlim(start / 1e6, end / 1e6)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_combined_figure(cases: Sequence[Dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 4, figsize=(15.5, 3.2 * len(cases)), dpi=180)
    if len(cases) == 1:
        axes = np.asarray([axes])
    comp_palette = {"A": "#d64f45", "B": "#2f6db3", "neutral": "#8a8f98", "unknown": "#c0c3c8"}
    panel_letters = list("abcdefghijklmnop")
    for row_index, case_payload in enumerate(cases):
        rows = case_payload["rows"]
        title = case_payload["case"]["title"]
        clusters = sorted({int(row["cluster"]) for row in rows})
        cmap = plt.get_cmap("tab10")

        ax = axes[row_index, 0]
        for label, color in comp_palette.items():
            subset = [row for row in rows if row.get("compartment") == label]
            if subset:
                ax.scatter([float(row["umap_1"]) for row in subset], [float(row["umap_2"]) for row in subset], s=2.5, alpha=0.85, linewidths=0, c=color, label=label)
        ax.set_title(f"{panel_letters[row_index * 4]}  A/B compartments")
        ax.set_ylabel(title)
        ax.legend(frameon=False, fontsize=6, markerscale=2, loc="best")

        ax = axes[row_index, 1]
        for cluster in clusters:
            subset = [row for row in rows if int(row["cluster"]) == cluster]
            ax.scatter([float(row["umap_1"]) for row in subset], [float(row["umap_2"]) for row in subset], s=2.5, alpha=0.85, linewidths=0, c=[cmap(cluster % 10)], label=str(cluster))
        ax.set_title(f"{panel_letters[row_index * 4 + 1]}  K-means clusters")

        ax = axes[row_index, 2]
        counts = [sum(1 for row in rows if int(row["cluster"]) == cluster) for cluster in clusters]
        ax.bar([str(cluster) for cluster in clusters], counts, color=[cmap(cluster % 10) for cluster in clusters], width=0.72)
        ax.set_title(f"{panel_letters[row_index * 4 + 2]}  Cluster node counts")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Nodes")

        ax = axes[row_index, 3]
        starts = [(int(row["start"]) + int(row["end"])) / 2e6 for row in rows]
        cluster_values = [int(row["cluster"]) for row in rows]
        ax.scatter(starts, cluster_values, s=2, linewidths=0, c=[cmap(cluster % 10) for cluster in cluster_values])
        ax.set_title(f"{panel_letters[row_index * 4 + 3]}  Genomic cluster positions")
        ax.set_xlabel(f"{rows[0]['chrom']} position (Mb)")
        ax.set_ylabel("Cluster")
        ax.set_yticks(clusters)

        for axis in axes[row_index]:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if axis in (axes[row_index, 0], axes[row_index, 1]):
                axis.set_xlabel("UMAP 1")
                axis.set_ylabel("UMAP 2" if axis is axes[row_index, 1] else title)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_manifest(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fields = ["case_id", "artifact", "path", "description"]
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
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PHOCI Figure 3 embedding, A/B compartment, k-means, and contact-map panels.")
    parser.add_argument("--outdir", default="outputs/paper_figures/figure3_embeddings")
    parser.add_argument("--training-root", default="outputs/full_training")
    parser.add_argument("--precompute-root", default="outputs/precomputed_paper_windows")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chrom", default="chr14")
    parser.add_argument("--bin-size", type=int, default=5000)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=20260628)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--max-gpu-gb", type=float, default=16.0)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--contact-resolution", type=int, default=100000)
    parser.add_argument("--min-contact-resolution", type=int, default=50000)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    threads = apply_resource_limits(cpu_fraction=0.5, max_threads=args.max_threads)
    device = choose_device(prefer_gpu=args.prefer_gpu, max_gpu_gb=args.max_gpu_gb)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cases = resolve_cases(Path(args.training_root), Path(args.precompute_root))

    model_cache: Dict[str, PHOCIModel] = {}
    case_payloads = []
    manifest_rows = []
    all_cluster_rows = []
    summary = {
        "chrom": args.chrom,
        "split": args.split,
        "bin_size": int(args.bin_size),
        "clusters": int(args.clusters),
        "device": str(device),
        "threads": int(threads),
        "training_root": args.training_root,
        "precompute_root": args.precompute_root,
        "cases": {},
    }

    for case in cases:
        if case["checkpoint"] not in model_cache:
            model_cache[case["checkpoint"]] = load_model(Path(case["checkpoint"]), device)
        print(json.dumps({"event": "case_start", "case_id": case["case_id"], "device": str(device)}), flush=True)
        embeddings, node_rows, coverage = encode_case(
            case=case,
            model=model_cache[case["checkpoint"]],
            split=args.split,
            chrom=args.chrom,
            bin_size=args.bin_size,
            device=device,
            progress=args.progress,
        )
        analysis_rows, analysis_summary = compute_embedding_analysis(
            embeddings=embeddings,
            node_rows=node_rows,
            n_clusters=args.clusters,
            random_state=args.random_state,
        )
        analysis_rows = attach_case_fields(analysis_rows, case=case, chrom=args.chrom, split=args.split)
        case_dir = outdir / case["case_id"]
        node_path = case_dir / f"{case['case_id']}_embedding_nodes.tsv"
        summary_path = case_dir / f"{case['case_id']}_embedding_summary.json"
        npz_path = case_dir / f"{case['case_id']}_embedding_arrays.npz"
        umap_png = case_dir / f"{case['case_id']}_embedding_umap.png"
        contact_png = case_dir / f"{case['case_id']}_contact_cluster.png"
        contact_pdf = case_dir / f"{case['case_id']}_contact_cluster.pdf"

        write_case_nodes(analysis_rows, node_path)
        save_embedding_npz(npz_path, embeddings, analysis_rows)
        case_summary = {
            **analysis_summary,
            "case_id": case["case_id"],
            "model": case["model"],
            "cell_line": case["cell_line"],
            "chrom": args.chrom,
            "split": args.split,
            "coverage": coverage,
            "checkpoint": case["checkpoint"],
            "manifest": case["manifest"],
            "compartment": case["compartment"],
            "porec_mcool": case["porec_mcool"],
        }
        write_embedding_summary(case_summary, summary_path)
        plot_embedding_analysis(analysis_rows, umap_png, case["title"])

        cluster_rows = cluster_count_rows(case, analysis_rows, chrom=args.chrom, split=args.split)
        all_cluster_rows.extend(cluster_rows)
        matrix, contact_meta = load_contact_matrix(
            Path(case["porec_mcool"]),
            chrom=args.chrom,
            start=coverage["start_bp"],
            end=coverage["end_bp"],
            target_resolution=args.contact_resolution,
            min_resolution=args.min_contact_resolution,
        )
        plot_contact_cluster_panel(analysis_rows, case, matrix, contact_meta or {}, contact_png, contact_pdf)

        summary["cases"][case["case_id"]] = {
            **case_summary,
            "outputs": {
                "nodes": str(node_path),
                "summary": str(summary_path),
                "arrays": str(npz_path),
                "umap": str(umap_png),
                "contact_cluster_png": str(contact_png),
                "contact_cluster_pdf": str(contact_pdf),
            },
            "contact_map": contact_meta,
        }
        case_payloads.append({"case": case, "rows": analysis_rows})
        for artifact, path, description in [
            ("nodes", node_path, "Node-level hidden embedding, UMAP, A/B compartment, and k-means cluster table."),
            ("summary", summary_path, "Case-level embedding analysis summary."),
            ("arrays", npz_path, "Compressed hidden embedding, UMAP, cluster, and coordinate arrays."),
            ("umap", umap_png, "Two-panel A/B and k-means UMAP plot."),
            ("contact_cluster_png", contact_png, "Pore-C contact map with k-means genomic cluster track."),
            ("contact_cluster_pdf", contact_pdf, "Pore-C contact map with k-means genomic cluster track."),
        ]:
            manifest_rows.append({"case_id": case["case_id"], "artifact": artifact, "path": str(path), "description": description})
        print(json.dumps({"event": "case_complete", "case_id": case["case_id"], "nodes": len(analysis_rows)}), flush=True)

    cluster_counts_path = outdir / "figure3_cluster_counts.tsv"
    write_cluster_counts(all_cluster_rows, cluster_counts_path)
    combined_png = outdir / "figure3_embedding_panels.png"
    combined_pdf = outdir / "figure3_embedding_panels.pdf"
    plot_combined_figure(case_payloads, combined_png, combined_pdf)
    manifest_rows.extend(
        [
            {"case_id": "all", "artifact": "cluster_counts", "path": str(cluster_counts_path), "description": "Cluster count and compartment composition by case."},
            {"case_id": "all", "artifact": "figure3_embedding_panels_png", "path": str(combined_png), "description": "Combined 16-panel Figure 3 embedding reproduction."},
            {"case_id": "all", "artifact": "figure3_embedding_panels_pdf", "path": str(combined_pdf), "description": "Combined 16-panel Figure 3 embedding reproduction."},
        ]
    )
    manifest_tsv = outdir / "figure3_manifest.tsv"
    manifest_json = outdir / "figure3_manifest.json"
    write_manifest(manifest_rows, manifest_tsv)
    manifest_json.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n")
    summary["outputs"] = {
        "cluster_counts": str(cluster_counts_path),
        "figure3_embedding_panels_png": str(combined_png),
        "figure3_embedding_panels_pdf": str(combined_pdf),
        "manifest_tsv": str(manifest_tsv),
        "manifest_json": str(manifest_json),
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1024**3, 4)
        summary["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 4)
    summary_path = outdir / "figure3_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "summary": str(summary_path), "cases": len(cases), "device": str(device)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
