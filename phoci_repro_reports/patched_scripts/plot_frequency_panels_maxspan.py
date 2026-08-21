#!/usr/bin/env python3
"""Render Figure 2j / Supp Fig 8 frequency panels for the maxspan export.

The upstream plot_figure2_supp_metrics.py skips these panels when the
frequency_bucket column is parsed as int (maxspan positives all have observed
frequency 1, so pandas infers a numeric dtype and isin(["0","1","2+"]) is
empty). This wrapper reads the bucket as string and uses point markers, which
also renders correctly with a single bucket.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FREQUENCY_BUCKETS = ["0", "1", "2+"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    frame = pd.read_csv(base_dir / "frequency_score_summary.tsv", sep="\t", dtype={"frequency_bucket": str})
    frame = frame[frame["frequency_bucket"].isin(FREQUENCY_BUCKETS)].copy()
    frame["frequency_bucket"] = pd.Categorical(frame["frequency_bucket"], categories=FREQUENCY_BUCKETS, ordered=True)
    made = []
    for prefixes, stem, title in [
        (["test_intra"], "figure2j_frequency_score_intra", "Figure 2j observed frequency vs prediction score, intra-cell"),
        (["test_intra", "test_inter", "test"], "supplementary_figure8_frequency_score_intra_inter",
         "Supplementary Figure 8 observed frequency vs prediction score"),
    ]:
        subset = frame[frame["prefix"].isin(prefixes)].copy()
        if subset.empty:
            continue
        subset["series"] = subset["model"] + " / " + subset["prefix"]
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        sns.pointplot(data=subset, x="frequency_bucket", y="mean_score", hue="series", dodge=0.25, ax=ax)
        ax.set_title(title + "\n(maxspan shards: all positives have observed frequency 1)")
        ax.set_ylabel("Mean prediction score")
        ax.set_xlabel("Observed frequency bucket")
        ax.legend(title="Model / split", frameon=False, loc="best", fontsize=8)
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            fig.savefig(plot_dir / f"{stem}{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(stem)
    print(json.dumps({"event": "complete", "plots": made, "plot_dir": str(plot_dir)}))


if __name__ == "__main__":
    main()
