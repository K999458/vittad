"""校正 GTEx-vs-TCGA 比较里的组成性偏倚（compositional bias）。

问题：
  PAAD 的火山图里 10638 个基因"显著上调"、只有 613 个下调，整张图向右平移。
  下调 Top10 全是胰腺消化酶（CPA1 / CELA3A / AMY2A / PNLIP / CTRB1 ...）。
  GTEx 正常胰腺是腺泡组织，这几十个消化酶基因就占掉转录组一半以上的 reads，
  TPM 是"占比"，分母被这些基因吃掉，导致其余所有基因的 TPM 被系统性压低。
  所以 TGM2 那个 log2FC=+3.86 里有很大一部分是归一化假象，不是真实上调。

校正：
  1) 全局中位数中心化（median-centering / 类似 upper-quartile 归一化）：
     每个样本减掉自己在表达基因上的中位数，消除库组成差异。
  2) 剔除腺泡/消化酶等超高表达基因后重新算 TPM 占比，再比较。
  3) 报告 TGM2 校正前后的 log2FC、FDR 和排名分位，以及全基因组 log2FC 的中位数
     （中位数≈0 说明偏倚已消除）。
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt

BASE = "/store/zkyang/tgm2_gdsc"
OUT = f"{BASE}/pancancer"
LFC, FDRC = 1.0, 0.05
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r))
        q[i] = prev
    return q


z = np.load(f"{OUT}/gtex_expr_cache.npz", allow_pickle=True)
mat = z["mat"]
ids = list(z["ids"])
union = list(z["samples"])
uidx = {s: i for i, s in enumerate(union)}
log(f"[0] 缓存矩阵 {mat.shape}")

pm = {}
for i, l in enumerate(open(f"{BASE}/gencode_v23_probemap.tsv")):
    if i == 0:
        continue
    a = l.rstrip("\n").split("\t")
    pm.setdefault(a[1], []).append(a[0])
id2sym = {}
for s, gl in pm.items():
    for g in gl:
        id2sym[g] = s

# symbol 层面去重：同 symbol 取全局均值最高的 ENSG
mean_all = np.nanmean(mat, axis=1)
best = {}
for k, g in enumerate(ids):
    s = id2sym.get(g, g)
    if s not in best or mean_all[k] > mean_all[best[s]]:
        best[s] = k
keep_idx = np.array(sorted(best.values()))
syms_all = np.array([id2sym.get(ids[k], ids[k]) for k in keep_idx])
M = mat[keep_idx]
log(f"[0] symbol 去重后 {M.shape}")

# 分组：复用 gtex_volcano.py 的分组逻辑，这里直接从 toil_tgm2.json 的表型重建
raw = json.load(open(f"{OUT}/toil_tgm2.json"))
ph, psamples = raw["pheno"], raw["psamples"]
meta = {s: (ph["_primary_site"][i], ph["_sample_type"][i],
            ph["_study"][i], ph["detailed_category"][i])
        for i, s in enumerate(psamples)}

TASKS = {
    "PAAD": ("Pancreatic Adenocarcinoma", ["Pancreas"], "胰腺癌"),
    "KIRC": ("Kidney Clear Cell Carcinoma", ["Kidney"], "肾透明细胞癌"),
    "ESCA": ("Esophageal Carcinoma", ["Esophagus", "Stomach"], "食管癌"),
}

results = {}
for CA, (dname, gsites, cn) in TASKS.items():
    tum, nor = [], []
    for s in union:
        m = meta.get(s)
        if not m:
            continue
        site, styp, study, dis = m
        norm = "Normal" in str(styp)
        if study == "TCGA" and dis == dname:
            (nor if norm else tum).append(s)
        elif study == "GTEX" and site in gsites:
            nor.append(s)
    ti = np.array([uidx[s] for s in tum])
    ni = np.array([uidx[s] for s in nor])

    sub = M[:, np.concatenate([ti, ni])]
    ok = (~np.isnan(sub)).sum(1) > 0.8 * sub.shape[1]
    ok &= np.nanmean(sub, 1) > -5
    E = M[ok]
    S = syms_all[ok]
    A0, N0 = E[:, ti], E[:, ni]

    log("")
    log("=" * 100)
    log(f"{CA} {cn}   肿瘤 {len(tum)}  vs  正常 {len(nor)}   基因 {len(S)}")
    log("=" * 100)

    variants = {}

    # --- 原始 ---
    variants["原始 TPM"] = (A0, N0)

    # --- 中位数中心化 ---
    off = np.nanmedian(np.concatenate([A0, N0], axis=1), axis=0)
    A1 = A0 - off[:len(ti)]
    N1 = N0 - off[len(ti):]
    variants["中位数中心化"] = (A1, N1)

    # --- 剔除超高表达基因后重算占比 ---
    lin = np.power(2.0, np.concatenate([A0, N0], axis=1)) - 0.001
    lin = np.clip(lin, 0, None)
    frac = lin / np.nansum(lin, axis=0, keepdims=True)
    top = np.argsort(-np.nanmax(frac, axis=1))[:200]
    mask = np.ones(len(S), bool)
    mask[top] = False
    lin2 = lin[mask]
    re = lin2 / np.nansum(lin2, axis=0, keepdims=True) * 1e6
    L = np.log2(re + 0.001).astype(np.float32)
    variants["剔除Top200高丰度基因后重算TPM"] = (L[:, :len(ti)], L[:, len(ti):])
    S2 = S[mask]
    log(f"  剔除的高丰度基因（占比最高的前 10 个）: "
        f"{', '.join(S[top[:10]])}")
    fr_nor = np.nansum(frac[top][:, len(ti):], axis=0)
    fr_tum = np.nansum(frac[top][:, :len(ti)], axis=0)
    log(f"  这 200 个基因占转录组比例：正常 {100*np.nanmedian(fr_nor):.1f}%  "
        f"肿瘤 {100*np.nanmedian(fr_tum):.1f}%")

    for name, (A, N) in variants.items():
        ss = S2 if name.startswith("剔除") else S
        lfc = np.nanmean(A, 1) - np.nanmean(N, 1)
        _, p = stats.mannwhitneyu(A, N, axis=1, alternative="two-sided",
                                  nan_policy="omit")
        p = np.nan_to_num(np.asarray(p, float), nan=1.0)
        q = bh(p)
        up = int(((q < FDRC) & (lfc >= LFC)).sum())
        dn = int(((q < FDRC) & (lfc <= -LFC)).sum())
        med = float(np.nanmedian(lfc))
        j = np.where(ss == "TGM2")[0]
        line = (f"  [{name:<28s}] 全基因组 log2FC 中位数 {med:+.2f}   "
                f"上调 {up:5d} / 下调 {dn:5d}  (比例 {up/max(dn,1):.1f}:1)")
        log(line)
        if len(j):
            j = j[0]
            rk = int((lfc > lfc[j]).sum()) + 1
            pct = 100 * rk / len(lfc)
            verdict = ("★ 显著高表达" if q[j] < FDRC and lfc[j] >= LFC else
                       "✓ 高表达" if q[j] < FDRC and lfc[j] > 0 else
                       "▼ 显著低表达" if q[j] < FDRC and lfc[j] <= -LFC else
                       "▽ 低表达" if q[j] < FDRC and lfc[j] < 0 else "— 无差异")
            log(f"       -> TGM2 log2FC={lfc[j]:+.2f}  FDR={q[j]:.2e}  "
                f"排名 {rk}/{len(lfc)} (前 {pct:.1f}%)  {verdict}")
            results.setdefault(CA, {})[name] = dict(
                log2FC=float(lfc[j]), fdr=float(q[j]), rank=rk,
                pct=float(pct), n_genes=int(len(lfc)), median_lfc=med,
                up=up, dn=dn, verdict=verdict)
            if name == "中位数中心化":
                results[CA]["_plot"] = (lfc.copy(), q.copy(), ss.copy(),
                                        len(tum), len(nor), cn)

# ---------- 校正后的火山图 ----------
plot = {k: v["_plot"] for k, v in results.items() if "_plot" in v}
fig, axes = plt.subplots(1, len(plot), figsize=(6.4 * len(plot), 6.2))
axes = np.atleast_1d(axes)
for ax, (CA, (lfc, q, ss, nt, nn, cn)) in zip(axes, plot.items()):
    y = -np.log10(np.clip(q, 1e-300, None))
    c = np.full(len(lfc), "#cccccc", dtype=object)
    c[(q < FDRC) & (lfc >= LFC)] = "#e04b4b"
    c[(q < FDRC) & (lfc <= -LFC)] = "#3b76c4"
    ax.scatter(lfc, y, s=4, c=list(c), linewidths=0, alpha=0.55,
               rasterized=True)
    ax.axhline(-np.log10(FDRC), ls="--", lw=0.8, c="#888")
    for v in (LFC, -LFC):
        ax.axvline(v, ls="--", lw=0.8, c="#888")
    ax.axvline(0, lw=0.6, c="#444")
    j = np.where(ss == "TGM2")[0]
    if len(j):
        j = j[0]
        tx, ty = lfc[j], -np.log10(max(q[j], 1e-300))
        ax.scatter([tx], [ty], s=175, facecolor="#ffd400", edgecolor="k",
                   linewidths=1.8, zorder=6)
        ax.annotate(f"TGM2\nlog2FC={tx:+.2f}\nFDR={q[j]:.1e}", (tx, ty),
                    textcoords="offset points", xytext=(14, 10), fontsize=10,
                    fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff6c2",
                              ec="k", lw=0.9))
    up = int(((q < FDRC) & (lfc >= LFC)).sum())
    dn = int(((q < FDRC) & (lfc <= -LFC)).sum())
    ax.set_title(f"{CA} {cn}   肿瘤 n={nt} vs 正常 n={nn}\n"
                 f"上调 {up} / 下调 {dn}   全基因组中位数 "
                 f"{np.nanmedian(lfc):+.2f}", fontsize=11)
    ax.set_xlabel("log2 fold change（肿瘤 / 正常，中位数中心化后）", fontsize=10)
    ax.set_ylabel("-log10 FDR", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("TGM2 火山图 —— 组成性偏倚校正后（TCGA 肿瘤 vs GTEx 正常）",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/火山图_GTEx校正后_目标癌种.png", dpi=170)
plt.close(fig)
log("")
log(f"[图] {OUT}/火山图_GTEx校正后_目标癌种.png")

clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in results.items()}
json.dump(clean, open(f"{OUT}/TGM2_GTEx归一化校正.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/GTEx归一化校正_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
