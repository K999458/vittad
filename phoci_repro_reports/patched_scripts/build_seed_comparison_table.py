#!/usr/bin/env python3
"""Merge mns_maxspan seed1 vs seed2 metrics into a Table 6-style comparison TSV."""
import csv
from pathlib import Path

BASE = Path("/storu/ysu/hiporec/deep-learning/Multi-EPI/phoci_epimci_literature_review_20260627/phoci_paper_aligned_v2_20260630/outputs")
RUNS = {
    "GM12878": ("mns_maxspan_models_seed1/GM12878", "mns_maxspan_train_seed20260713/GM12878"),
    "K562": ("mns_maxspan_models_seed1/K562", "mns_maxspan_train_k562_seed20260713/K562"),
    "Comprehensive": ("mns_maxspan_models_seed1/Comprehensive", "mns_maxspan_train_comprehensive_seed20260713/Comprehensive"),
}
OUT = Path("/store/zkyang/phoci_repro_reports/paper_figures_mns_maxspan/tables/seed1_vs_seed2_metric_comparison.tsv")


def load(path: Path):
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(row["prefix"], row["negative_type"])] = row
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for cell, (d1, d2) in RUNS.items():
        seed1 = load(BASE / d1 / "sage_full_metrics_by_negative_type.tsv")
        seed2 = load(BASE / d2 / "sage_full_metrics_by_negative_type.tsv")
        for key in sorted(seed1):
            r1, r2 = seed1[key], seed2.get(key)
            if r2 is None:
                continue
            auc1, auc2 = float(r1["auc"]), float(r2["auc"])
            ap1, ap2 = float(r1["ap"]), float(r2["ap"])
            out_rows.append({
                "model": cell,
                "prefix": key[0],
                "negative_type": key[1],
                "seed1_auc": f"{auc1:.6f}",
                "seed2_auc": f"{auc2:.6f}",
                "delta_auc": f"{auc2 - auc1:+.6f}",
                "seed1_ap": f"{ap1:.6f}",
                "seed2_ap": f"{ap2:.6f}",
                "delta_ap": f"{ap2 - ap1:+.6f}",
                "n_positive": r1["n_positive"],
                "n_negative": r1["n_negative"],
            })
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    max_dauc = max(abs(float(r["delta_auc"])) for r in out_rows)
    max_dap = max(abs(float(r["delta_ap"])) for r in out_rows)
    print(f"rows={len(out_rows)} max|delta_auc|={max_dauc:.6f} max|delta_ap|={max_dap:.6f} -> {OUT}")


if __name__ == "__main__":
    main()
