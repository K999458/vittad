"""用已保存的 CSV 重画泛癌火山图和汇总条形图（修正中文字体）。不重新取数。"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/store/zkyang/tgm2_gdsc/pancancer"
LFC, FDRC = 1.0, 0.05

S = pd.read_csv(f"{OUT}/TGM2_泛癌_肿瘤vs正常_汇总.csv")
S = S.sort_values("log2FC", ascending=False).reset_index(drop=True)


def draw(ax, D, title, sub):
    x = D.log2FC.to_numpy()
    y = -np.log10(np.clip(D.fdr.to_numpy(), 1e-300, None))
    c = np.full(len(D), "#d0d0d0", dtype=object)
    c[(D.fdr < FDRC) & (D.log2FC >= LFC)] = "#e04b4b"
    c[(D.fdr < FDRC) & (D.log2FC <= -LFC)] = "#3b76c4"
    ax.scatter(x, y, s=2.5, c=list(c), linewidths=0, alpha=0.5, rasterized=True)
    ax.axhline(-np.log10(FDRC), ls="--", lw=0.7, c="#999")
    ax.axvline(LFC, ls="--", lw=0.7, c="#999")
    ax.axvline(-LFC, ls="--", lw=0.7, c="#999")
    t = D[D.gene == "TGM2"]
    if len(t):
        tx = float(t.log2FC.iloc[0])
        ty = -np.log10(max(float(t.fdr.iloc[0]), 1e-300))
        ax.scatter([tx], [ty], s=95, facecolor="#ffd400", edgecolor="k",
                   linewidths=1.4, zorder=6)
        ax.annotate("TGM2", (tx, ty), textcoords="offset points",
                    xytext=(9, 6), fontsize=9, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#fff6c2",
                              ec="k", lw=0.7))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=20)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, color="#555")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------- 大网格：18 个癌种 ----------
ncol = 5
nrow = int(np.ceil(len(S) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(S):]:
    ax.axis("off")
for ax, (_, r) in zip(axes, S.iterrows()):
    D = pd.read_csv(f"{OUT}/volcano_{r.cancer}.csv")
    draw(ax, D, f"{r.cancer} {r['name']}",
         f"肿瘤 {r.n_tumor} vs 癌旁 {r.n_normal}   "
         f"TGM2 log2FC={r.log2FC:+.2f}, FDR={r.fdr:.1e}")
for i, ax in enumerate(axes[:len(S)]):
    if i // ncol == nrow - 1:
        ax.set_xlabel("log2FC（肿瘤/正常）", fontsize=9)
    if i % ncol == 0:
        ax.set_ylabel("-log10 FDR", fontsize=9)
fig.legend(handles=[Patch(fc="#e04b4b", label=f"显著上调 (FDR<0.05, log2FC≥{LFC})"),
                    Patch(fc="#3b76c4", label=f"显著下调 (FDR<0.05, log2FC≤-{LFC})"),
                    Patch(fc="#ffd400", ec="k", label="TGM2")],
           loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=11,
           frameon=True)
fig.suptitle("泛癌 TGM2 火山图 —— TCGA 肿瘤 vs 癌旁正常（HiSeqV2, Mann-Whitney + BH）\n"
             "按 TGM2 log2FC 从高到低排列",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/火山图_泛癌_TGM2.png", dpi=150)
fig.savefig(f"{OUT}/火山图_泛癌_TGM2.pdf")
plt.close(fig)
print("[1] 泛癌大网格已重画")

# ---------- 重点癌种 ----------
KEY = ["KIRC", "ESCA", "PAAD", "THCA", "READ", "KIRP"]
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
axes = axes.ravel()
for ax, ca in zip(axes, KEY):
    r = S[S.cancer == ca].iloc[0]
    D = pd.read_csv(f"{OUT}/volcano_{ca}.csv")
    draw(ax, D, f"{ca} {r['name']}",
         f"肿瘤 {r.n_tumor} vs 癌旁 {r.n_normal}   "
         f"TGM2 log2FC={r.log2FC:+.2f}, FDR={r.fdr:.1e}")
    ax.set_xlabel("log2FC（肿瘤/正常）", fontsize=9)
    ax.set_ylabel("-log10 FDR", fontsize=9)
fig.suptitle("重点癌种 TGM2 火山图（TCGA 癌旁正常做对照）\n"
             "注意 PAAD 癌旁只有 4 例、ESCA 只有 11 例，检验功效不足",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/火山图_重点癌种_TGM2.png", dpi=160)
plt.close(fig)
print("[2] 重点癌种已重画")

# ---------- 汇总条形图 ----------
fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(S))[::-1]
col = ["#c62828" if (r.fdr < 0.05 and r.log2FC >= 1) else
       "#ef8a62" if (r.fdr < 0.05 and r.log2FC > 0) else
       "#1565c0" if (r.fdr < 0.05 and r.log2FC <= -1) else
       "#67a9cf" if (r.fdr < 0.05 and r.log2FC < 0) else "#bdbdbd"
       for _, r in S.iterrows()]
ax.barh(y, S.log2FC, color=col, height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.cancer}  {r['name']}  (T={r.n_tumor}/N={r.n_normal})"
                    for _, r in S.iterrows()], fontsize=10)
ax.axvline(0, c="k", lw=0.9)
ax.axvline(1, ls="--", c="#888", lw=0.8)
ax.axvline(-1, ls="--", c="#888", lw=0.8)
for yy, (_, r) in zip(y, S.iterrows()):
    star = ("***" if r.fdr < 1e-3 else "**" if r.fdr < 1e-2
            else "*" if r.fdr < 0.05 else "ns")
    off = 0.12 if r.log2FC >= 0 else -0.12
    ax.text(r.log2FC + off, yy, f"{r.log2FC:+.2f} {star}",
            va="center", ha="left" if r.log2FC >= 0 else "right", fontsize=9)
ax.set_xlabel("log2 fold change（肿瘤 / 癌旁正常）", fontsize=12)
ax.set_xlim(-3.9, 2.6)
ax.set_title("泛癌 TGM2 表达差异汇总（TCGA 癌旁正常对照）\n"
             "* FDR<0.05  ** FDR<0.01  *** FDR<0.001",
             fontsize=14, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc="#c62828", label="显著高表达 (log2FC≥1)"),
                   Patch(fc="#ef8a62", label="高表达 (0<log2FC<1)"),
                   Patch(fc="#bdbdbd", label="无显著差异"),
                   Patch(fc="#67a9cf", label="低表达"),
                   Patch(fc="#1565c0", label="显著低表达 (log2FC≤-1)")],
          loc="upper left", fontsize=9.5, framealpha=0.95)
fig.tight_layout()
fig.savefig(f"{OUT}/泛癌汇总_TGM2_log2FC.png", dpi=160)
plt.close(fig)
print("[3] 汇总条形图已重画")
