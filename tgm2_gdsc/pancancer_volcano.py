"""泛癌 TGM2 表达：肿瘤 vs 正常 差异分析 + 火山图

数据：UCSC Xena TCGA HiSeqV2，log2(RSEM normalized_count + 1)
分组：肿瘤 = barcode 后缀 -01；正常 = -11（癌旁正常）
统计：Mann-Whitney U + log2FC（组均值差）+ BH FDR
输出：每个癌种一张火山图（TGM2 标红）+ 泛癌汇总图 + 汇总表
"""
import os, gzip, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = "/store/zkyang/tgm2_gdsc/tcga"
OUT = "/store/zkyang/tgm2_gdsc/pancancer"
os.makedirs(OUT, exist_ok=True)
LFC, FDRC = 1.0, 0.05
MIN_NORMAL = 8
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r)); q[i] = prev
    return q


CANCERS = ["BRCA", "KIRC", "LUAD", "THCA", "PRAD", "LUSC", "LIHC", "HNSC",
           "COAD", "STAD", "KIRP", "KICH", "UCEC", "BLCA", "ESCA", "READ",
           "CHOL", "PAAD"]
FULLNAME = {
    "BRCA": "乳腺癌", "KIRC": "肾透明细胞癌", "LUAD": "肺腺癌", "THCA": "甲状腺癌",
    "PRAD": "前列腺癌", "LUSC": "肺鳞癌", "LIHC": "肝癌", "HNSC": "头颈鳞癌",
    "COAD": "结肠癌", "STAD": "胃癌", "KIRP": "肾乳头状癌", "KICH": "肾嫌色细胞癌",
    "UCEC": "子宫内膜癌", "BLCA": "膀胱癌", "ESCA": "食管癌", "READ": "直肠癌",
    "CHOL": "胆管癌", "PAAD": "胰腺癌",
}

summary = []
volcano_data = {}

for CA in CANCERS:
    f = f"{BASE}/{CA}_HiSeqV2.gz"
    if not os.path.exists(f):
        continue
    with gzip.open(f, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    tum = [c for c in df.columns if c.split("-")[-1].startswith("01")]
    nor = [c for c in df.columns if c.split("-")[-1].startswith("11")]
    if len(nor) < 4:
        log(f"{CA}: 正常样本仅 {len(nor)} 例，跳过火山图")
        continue

    A = df[tum].to_numpy(float)
    B = df[nor].to_numpy(float)
    keep = (np.mean(A > 0, axis=1) > 0.2) | (np.mean(B > 0, axis=1) > 0.2)
    genes = df.index.to_numpy()[keep]
    A, B = A[keep], B[keep]
    lfc = A.mean(axis=1) - B.mean(axis=1)
    _, p = stats.mannwhitneyu(A, B, axis=1, alternative="two-sided")
    q = bh(p)

    d = pd.DataFrame({"gene": genes, "log2FC": lfc, "p": p, "fdr": q})
    d.to_csv(f"{OUT}/volcano_{CA}.csv", index=False)
    volcano_data[CA] = d

    tg = d[d.gene == "TGM2"]
    if len(tg):
        r = tg.iloc[0]
        rank_up = int((d.log2FC > r.log2FC).sum()) + 1
        summary.append({
            "cancer": CA, "name": FULLNAME.get(CA, CA),
            "n_tumor": len(tum), "n_normal": len(nor),
            "log2FC": float(r.log2FC), "FC": float(2 ** r.log2FC),
            "p": float(r.p), "fdr": float(r.fdr),
            "tumor_mean": float(A[genes == "TGM2"].mean()),
            "normal_mean": float(B[genes == "TGM2"].mean()),
            "rank_by_lfc": rank_up, "n_genes": len(d),
            "sig_up": int(((d.fdr < FDRC) & (d.log2FC >= LFC)).sum()),
            "sig_dn": int(((d.fdr < FDRC) & (d.log2FC <= -LFC)).sum()),
        })

S = pd.DataFrame(summary).sort_values("log2FC", ascending=False)
S.to_csv(f"{OUT}/TGM2_泛癌_肿瘤vs正常_汇总.csv", index=False)

log("=" * 108)
log("泛癌 TGM2 表达：肿瘤 vs 癌旁正常（TCGA HiSeqV2，Mann-Whitney + BH）")
log("log2FC > 0 = 肿瘤中高表达")
log("=" * 108)
log("")
log(f"{'癌种':<7}{'中文名':<14}{'肿瘤n':>6}{'正常n':>6}{'log2FC':>9}{'FC':>7}"
    f"{'FDR':>11}{'肿瘤均值':>9}{'正常均值':>9}{'FC排名':>9}  判定")
log("-" * 108)
for _, r in S.iterrows():
    if r.fdr < 0.05 and r.log2FC >= 1:
        v = "★ 显著高表达"
    elif r.fdr < 0.05 and r.log2FC > 0:
        v = "✓ 高表达"
    elif r.fdr < 0.05 and r.log2FC <= -1:
        v = "▼ 显著低表达"
    elif r.fdr < 0.05:
        v = "▽ 低表达"
    else:
        v = "— 无差异"
    log(f"{r.cancer:<7}{r['name']:<14}{r.n_tumor:>6}{r.n_normal:>6}{r.log2FC:>9.2f}"
        f"{r.FC:>7.2f}{r.fdr:>11.2e}{r.tumor_mean:>9.2f}{r.normal_mean:>9.2f}"
        f"{r.rank_by_lfc:>5}/{r.n_genes:<4}  {v}")
log("-" * 108)
n_up = int(((S.fdr < 0.05) & (S.log2FC > 0)).sum())
n_dn = int(((S.fdr < 0.05) & (S.log2FC < 0)).sum())
log(f"汇总：{len(S)} 个癌种中，TGM2 显著高表达 {n_up} 个，显著低表达 {n_dn} 个，"
    f"无差异 {len(S)-n_up-n_dn} 个")

# ---------------- 火山图 ----------------
def draw(d, CA, ax):
    x = d.log2FC.to_numpy()
    y = -np.log10(np.clip(d.fdr.to_numpy(), 1e-300, None))
    col = np.full(len(d), "#c8c8c8", dtype=object)
    col[(d.fdr < FDRC) & (d.log2FC >= LFC)] = "#e04b4b"
    col[(d.fdr < FDRC) & (d.log2FC <= -LFC)] = "#3b76c4"
    ax.scatter(x, y, s=3, c=list(col), linewidths=0, alpha=0.55, rasterized=True)
    ax.axhline(-np.log10(FDRC), ls="--", lw=0.7, c="#888")
    ax.axvline(LFC, ls="--", lw=0.7, c="#888")
    ax.axvline(-LFC, ls="--", lw=0.7, c="#888")
    t = d[d.gene == "TGM2"]
    if len(t):
        tx = float(t.log2FC.iloc[0])
        ty = -np.log10(max(float(t.fdr.iloc[0]), 1e-300))
        ax.scatter([tx], [ty], s=95, facecolor="#ffd400", edgecolor="k",
                   linewidths=1.3, zorder=6, marker="o")
        ax.annotate("TGM2", (tx, ty), textcoords="offset points", xytext=(9, 6),
                    fontsize=9, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#fff6c2",
                              ec="k", lw=0.7, alpha=0.95))
    n_t = int(S.loc[S.cancer == CA, "n_tumor"].iloc[0]) if (S.cancer == CA).any() else 0
    n_n = int(S.loc[S.cancer == CA, "n_normal"].iloc[0]) if (S.cancer == CA).any() else 0
    ax.set_title(f"{CA}  (T={n_t}, N={n_n})", fontsize=10)
    ax.set_xlabel("log2 fold change (Tumor / Normal)", fontsize=8)
    ax.set_ylabel("-log10 FDR", fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


order = S.cancer.tolist()
ncol = 4
nrow = int(np.ceil(len(order) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow))
axes = np.atleast_1d(axes).ravel()
for i, CA in enumerate(order):
    draw(volcano_data[CA], CA, axes[i])
for j in range(len(order), len(axes)):
    axes[j].axis("off")
fig.suptitle("TGM2 in TCGA pan-cancer volcano plots (Tumor vs adjacent Normal)",
             fontsize=15, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(f"{OUT}/火山图_泛癌_TGM2.png", dpi=155)
fig.savefig(f"{OUT}/火山图_泛癌_TGM2.pdf")
plt.close(fig)

# 单独大图：三个目标癌种 + 最显著的
focus = [c for c in ["KIRC", "ESCA", "PAAD"] if c in volcano_data]
focus += [c for c in order if c not in focus][:3]
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4))
for i, CA in enumerate(focus[:6]):
    draw(volcano_data[CA], CA, axes.ravel()[i])
fig.suptitle("TGM2 volcano — 目标癌种 (KIRC/ESCA/PAAD) 与 TGM2 上调最强的癌种",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/火山图_重点癌种_TGM2.png", dpi=165)
plt.close(fig)

# ---------------- 泛癌汇总条形图 ----------------
fig, ax = plt.subplots(figsize=(11, 6.2))
Sp = S.sort_values("log2FC")
cols = ["#e04b4b" if (r.fdr < 0.05 and r.log2FC > 0)
        else "#3b76c4" if (r.fdr < 0.05 and r.log2FC < 0) else "#bbbbbb"
        for _, r in Sp.iterrows()]
ax.barh(range(len(Sp)), Sp.log2FC, color=cols, edgecolor="k", linewidth=0.4)
ax.set_yticks(range(len(Sp)))
ax.set_yticklabels(Sp.cancer, fontsize=9)
ax.axvline(0, c="k", lw=0.8)
ax.axvline(1, ls="--", c="#888", lw=0.7)
ax.axvline(-1, ls="--", c="#888", lw=0.7)
ax.set_xlabel("TGM2 log2 fold change (Tumor / Normal)", fontsize=11)
ax.set_title("TGM2 差异表达 泛癌汇总（红=显著高表达，蓝=显著低表达，灰=无差异）",
             fontsize=12)
for i, (_, r) in enumerate(Sp.iterrows()):
    star = "***" if r.fdr < 1e-10 else "**" if r.fdr < 1e-4 else "*" if r.fdr < 0.05 else "ns"
    off = 0.06 if r.log2FC >= 0 else -0.06
    ax.text(r.log2FC + off, i, star, va="center",
            ha="left" if r.log2FC >= 0 else "right", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/泛癌汇总_TGM2_log2FC.png", dpi=165)
plt.close(fig)

json.dump(summary, open(f"{OUT}/TGM2_泛癌汇总.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/泛癌_TGM2_报告.txt", "w").write("\n".join(lines))
print("\n>>> 图与表已写入", OUT)
