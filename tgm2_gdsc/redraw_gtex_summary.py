"""重画 TCGA+GTEx 版本的泛癌 TGM2 汇总图 + 箱线图（修正中文字体）。用缓存数据，不重新取数。"""
import sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/store/zkyang/tgm2_gdsc/pancancer"

CN = {
    "Pancreatic Adenocarcinoma": "胰腺癌 PAAD",
    "Skin Cutaneous Melanoma": "皮肤黑色素瘤 SKCM",
    "Kidney Clear Cell Carcinoma": "肾透明细胞癌 KIRC",
    "Stomach Adenocarcinoma": "胃癌 STAD",
    "Cholangiocarcinoma": "胆管癌 CHOL",
    "Testicular Germ Cell Tumor": "睾丸生殖细胞瘤 TGCT",
    "Rectum Adenocarcinoma": "直肠癌 READ",
    "Kidney Papillary Cell Carcinoma": "肾乳头状癌 KIRP",
    "Glioblastoma Multiforme": "胶质母细胞瘤 GBM",
    "Colon Adenocarcinoma": "结肠癌 COAD",
    "Sarcoma": "软组织肉瘤 SARC",
    "Brain Lower Grade Glioma": "低级别胶质瘤 LGG",
    "Breast Invasive Carcinoma": "乳腺癌 BRCA",
    "Acute Myeloid Leukemia": "急性髓系白血病 LAML",
    "Ovarian Serous Cystadenocarcinoma": "卵巢癌 OV",
    "Prostate Adenocarcinoma": "前列腺癌 PRAD",
    "Thyroid Carcinoma": "甲状腺癌 THCA",
    "Liver Hepatocellular Carcinoma": "肝癌 LIHC",
    "Esophageal Carcinoma": "食管癌 ESCA",
    "Adrenocortical Cancer": "肾上腺皮质癌 ACC",
    "Bladder Urothelial Carcinoma": "膀胱癌 BLCA",
    "Lung Adenocarcinoma": "肺腺癌 LUAD",
    "Kidney Chromophobe": "肾嫌色细胞癌 KICH",
    "Head & Neck Squamous Cell Carcinoma": "头颈鳞癌 HNSC",
    "Uterine Corpus Endometrioid Carcinoma": "子宫内膜癌 UCEC",
    "Lung Squamous Cell Carcinoma": "肺鳞癌 LUSC",
}

S = pd.DataFrame(json.load(open(f"{OUT}/TGM2_TCGA_GTEx.json")))
S["cn"] = S.disease.map(lambda d: CN.get(d, d))
S = S.sort_values("log2FC", ascending=False).reset_index(drop=True)

# ---------- 汇总条形图 ----------
fig, ax = plt.subplots(figsize=(12, 9))
y = np.arange(len(S))[::-1]
col = ["#c62828" if (r.fdr < 0.05 and r.log2FC >= 1) else
       "#ef8a62" if (r.fdr < 0.05 and r.log2FC > 0) else
       "#1565c0" if (r.fdr < 0.05 and r.log2FC <= -1) else
       "#67a9cf" if (r.fdr < 0.05 and r.log2FC < 0) else "#bdbdbd"
       for _, r in S.iterrows()]
ax.barh(y, S.log2FC, color=col, height=0.72)
ax.set_yticks(y)
ax.set_yticklabels(
    [f"{r.cn}  (T={r.n_tumor} / N={r.n_gtex + r.n_tcga_normal})"
     for _, r in S.iterrows()], fontsize=10)
ax.axvline(0, c="k", lw=0.9)
for v in (1, -1):
    ax.axvline(v, ls="--", c="#888", lw=0.8)
for yy, (_, r) in zip(y, S.iterrows()):
    star = ("***" if r.fdr < 1e-3 else "**" if r.fdr < 1e-2
            else "*" if r.fdr < 0.05 else "ns")
    off = 0.1 if r.log2FC >= 0 else -0.1
    ax.text(r.log2FC + off, yy, f"{r.log2FC:+.2f} {star}", va="center",
            ha="left" if r.log2FC >= 0 else "right", fontsize=9)
ax.set_xlim(-4.6, 5.6)
ax.set_xlabel("log2 fold change（TCGA 肿瘤 / GTEx正常+TCGA癌旁）", fontsize=12)
ax.set_title("泛癌 TGM2 表达差异 —— TCGA 肿瘤 vs GTEx 正常组织\n"
             "Toil 统一流程 RSEM log2(TPM+0.001)，Mann-Whitney + BH\n"
             "* FDR<0.05  ** FDR<0.01  *** FDR<0.001",
             fontsize=14, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc="#c62828", label="显著高表达 (log2FC≥1)"),
                   Patch(fc="#ef8a62", label="高表达 (0<log2FC<1)"),
                   Patch(fc="#bdbdbd", label="无显著差异"),
                   Patch(fc="#67a9cf", label="低表达"),
                   Patch(fc="#1565c0", label="显著低表达 (log2FC≤-1)")],
          loc="lower right", fontsize=9.5, framealpha=0.95)
fig.tight_layout()
fig.savefig(f"{OUT}/泛癌汇总_TGM2_TCGA_vs_GTEx.png", dpi=160)
fig.savefig(f"{OUT}/泛癌汇总_TGM2_TCGA_vs_GTEx.pdf")
plt.close(fig)
print("[1] GTEx 汇总条形图已重画")

# ---------- 箱线图 ----------
raw = json.load(open(f"{OUT}/toil_tgm2.json"))
tg = dict(zip(raw["samples"], raw["tgm2"]))
ph = raw["pheno"]
pkeys = list(ph.keys())
psamples = raw["psamples"]
site = ph.get("_primary_site")
styp = ph.get("_sample_type")
study = ph.get("_study")
dis = ph.get("detailed_category")
meta = {s: (site[i], styp[i], study[i], dis[i]) for i, s in enumerate(psamples)}

SITE = {r.disease: r.site for _, r in S.iterrows()}
order = list(S.disease)

data, labels, cols, ns = [], [], [], []
for d in order:
    gs = SITE.get(d)
    t = [tg[s] for s in tg if meta.get(s) and meta[s][2] == "TCGA"
         and meta[s][3] == d and "Normal" not in str(meta[s][1])
         and tg[s] is not None]
    n = [tg[s] for s in tg if meta.get(s) and meta[s][2] == "GTEX"
         and meta[s][0] == gs and tg[s] is not None]
    n += [tg[s] for s in tg if meta.get(s) and meta[s][2] == "TCGA"
          and meta[s][3] == d and "Normal" in str(meta[s][1])
          and tg[s] is not None]
    if len(t) < 20 or len(n) < 5:
        continue
    data += [n, t]
    labels.append(CN.get(d, d))
    ns.append((len(t), len(n)))

fig, ax = plt.subplots(figsize=(max(14, 0.66 * len(labels) * 2), 7.5))
pos = []
for i in range(len(labels)):
    pos += [i * 2.6, i * 2.6 + 0.95]
bp = ax.boxplot(data, positions=pos, widths=0.82, patch_artist=True,
                showfliers=False, medianprops=dict(color="k", lw=1.3))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor("#8ec7e8" if i % 2 == 0 else "#e88b8b")
    b.set_edgecolor("#333")
    b.set_linewidth(0.8)
ax.set_xticks([i * 2.6 + 0.48 for i in range(len(labels))])
ax.set_xticklabels([f"{l}\nT={n[0]}/N={n[1]}" for l, n in zip(labels, ns)],
                   rotation=55, ha="right", fontsize=9)
ax.set_ylabel("TGM2 表达量  log2(TPM+0.001)", fontsize=12)
ax.set_title("泛癌 TGM2 表达：正常组织（蓝）vs 肿瘤（红）\n"
             "按 log2FC 从高到低排列，左侧为 TGM2 在肿瘤中升高最明显的癌种",
             fontsize=14, fontweight="bold")
for yy, (_, r) in enumerate(S.iterrows()):
    pass
# 显著性标记
k = 0
top = max(max(d) for d in data if len(d))
for i, d in enumerate(order):
    if CN.get(d, d) not in labels:
        continue
    r = S[S.disease == d].iloc[0]
    star = ("***" if r.fdr < 1e-3 else "**" if r.fdr < 1e-2
            else "*" if r.fdr < 0.05 else "ns")
    ax.text(k * 2.6 + 0.48, top * 0.99, star, ha="center", fontsize=10,
            fontweight="bold",
            color="#c62828" if r.log2FC > 0 else "#1565c0")
    k += 1
ax.legend(handles=[Patch(fc="#8ec7e8", ec="#333", label="正常组织 (GTEx + TCGA癌旁)"),
                   Patch(fc="#e88b8b", ec="#333", label="TCGA 肿瘤")],
          loc="upper right", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/泛癌箱线图_TGM2_TCGA_vs_GTEx.png", dpi=150)
plt.close(fig)
print("[2] 泛癌箱线图已画")

# ---------- 三个目标癌种放大箱线图 ----------
TARGET = ["Pancreatic Adenocarcinoma", "Kidney Clear Cell Carcinoma",
          "Esophageal Carcinoma"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
for ax, d in zip(axes, TARGET):
    gs = SITE.get(d)
    t = [tg[s] for s in tg if meta.get(s) and meta[s][2] == "TCGA"
         and meta[s][3] == d and "Normal" not in str(meta[s][1])
         and tg[s] is not None]
    n = [tg[s] for s in tg if meta.get(s) and meta[s][2] == "GTEX"
         and meta[s][0] == gs and tg[s] is not None]
    na = [tg[s] for s in tg if meta.get(s) and meta[s][2] == "TCGA"
          and meta[s][3] == d and "Normal" in str(meta[s][1])
          and tg[s] is not None]
    groups = [n, na, t]
    gl = [f"GTEx 正常\nn={len(n)}", f"TCGA 癌旁\nn={len(na)}",
          f"TCGA 肿瘤\nn={len(t)}"]
    keep = [i for i, g in enumerate(groups) if len(g) >= 3]
    bp = ax.boxplot([groups[i] for i in keep], positions=range(len(keep)),
                    widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.4))
    fc = ["#8ec7e8", "#b6dfa8", "#e88b8b"]
    for j, i in enumerate(keep):
        bp["boxes"][j].set_facecolor(fc[i])
        bp["boxes"][j].set_edgecolor("#333")
    rng = np.random.default_rng(0)
    for j, i in enumerate(keep):
        g = groups[i]
        ax.scatter(j + rng.normal(0, 0.075, len(g)), g, s=7, c="#333",
                   alpha=0.32, linewidths=0, zorder=3)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([gl[i] for i in keep], fontsize=9.5)
    r = S[S.disease == d].iloc[0]
    ax.set_title(f"{CN.get(d,d)}\nlog2FC={r.log2FC:+.2f}   FDR={r.fdr:.2e}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("TGM2  log2(TPM+0.001)", fontsize=10)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("三个目标癌种 TGM2 表达（GTEx 正常 / TCGA 癌旁 / TCGA 肿瘤）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/箱线图_目标癌种_TGM2.png", dpi=165)
plt.close(fig)
print("[3] 目标癌种箱线图已画")
