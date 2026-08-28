"""TGM2 的"肿瘤高表达"是真实的，还是间质成分变化带来的？

上一步发现 TGM2 在 GTEx 里是平滑肌/间质高表达基因（食管肌层 7.1 vs 黏膜 4.4）。
PAAD 是典型的富间质（desmoplastic）肿瘤，KIRC 血管丰富，
所以"肿瘤 vs 正常"里 TGM2 升高有可能只是间质占比升高的副产物。

做三件事：
 1) 把 TGM2 的 log2FC 放到间质/平滑肌 panel 和上皮 panel 里对比，看它跟谁走。
 2) 在肿瘤样本内部算 TGM2 与各 panel 的相关性。
 3) 用间质 signature 做偏相关 / 分层校正，看 TGM2 的升高能剩下多少。
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
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


PANELS = {
    "平滑肌/肌成纤维": ["ACTA2", "TAGLN", "MYH11", "CNN1", "DES", "MYL9"],
    "成纤维/胶原 ECM": ["COL1A1", "COL1A2", "COL3A1", "FN1", "DCN", "LUM",
                   "POSTN", "SPARC"],
    "血管内皮/血管": ["PECAM1", "VWF", "CDH5", "PDGFRB", "ENG"],
    "免疫": ["PTPRC", "CD68", "CD3E", "CD14", "LYZ"],
    "上皮": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "CLDN4"],
    "增殖": ["MKI67", "PCNA", "TOP2A", "CCNB1"],
    "管家": ["GAPDH", "ACTB", "TBP", "RPLP0", "PPIA"],
}

z = np.load(f"{OUT}/gtex_expr_cache.npz", allow_pickle=True)
mat = z["mat"]
ids = list(z["ids"])
union = list(z["samples"])
uidx = {s: i for i, s in enumerate(union)}

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
mean_all = np.nanmean(mat, axis=1)
best = {}
for k, g in enumerate(ids):
    s = id2sym.get(g, g)
    if s not in best or mean_all[k] > mean_all[best[s]]:
        best[s] = k
keep = np.array(sorted(best.values()))
syms = np.array([id2sym.get(ids[k], ids[k]) for k in keep])
M = mat[keep]
row = {s: i for i, s in enumerate(syms)}

raw = json.load(open(f"{OUT}/toil_tgm2.json"))
ph, psamples = raw["pheno"], raw["psamples"]
meta = {s: (ph["_primary_site"][i], ph["_sample_type"][i],
            ph["_study"][i], ph["detailed_category"][i])
        for i, s in enumerate(psamples)}

TASKS = {
    "PAAD": ("Pancreatic Adenocarcinoma", ["Pancreas"], "胰腺癌"),
    "KIRC": ("Kidney Clear Cell Carcinoma", ["Kidney"], "肾透明细胞癌"),
}

summary = {}
for CA, (dname, gsites, cn) in TASKS.items():
    tum, nor = [], []
    for s in union:
        m = meta.get(s)
        if not m:
            continue
        site, styp, study, dis = m
        if study == "TCGA" and dis == dname:
            (nor if "Normal" in str(styp) else tum).append(s)
        elif study == "GTEX" and site in gsites:
            nor.append(s)
    ti = [uidx[s] for s in tum]
    ni = [uidx[s] for s in nor]
    A, N = M[:, ti], M[:, ni]
    off = np.nanmedian(np.concatenate([A, N], axis=1), axis=0)
    A = A - off[:len(ti)]
    N = N - off[len(ti):]
    lfc_all = np.nanmean(A, 1) - np.nanmean(N, 1)

    log("")
    log("=" * 92)
    log(f"{CA} {cn}   肿瘤 {len(tum)} vs 正常 {len(nor)}"
        f"   （全部数值已做中位数中心化校正）")
    log("=" * 92)
    tg = row["TGM2"]
    log(f"  TGM2 自身 log2FC = {lfc_all[tg]:+.2f}")
    log("")
    log(f"  {'panel':<16s} {'基因':<52s} {'log2FC中位数':>12s}")
    log("  " + "-" * 84)
    panel_lfc = {}
    panel_score = {}
    for pname, gl in PANELS.items():
        gi = [row[g] for g in gl if g in row]
        if not gi:
            continue
        v = lfc_all[gi]
        panel_lfc[pname] = float(np.nanmedian(v))
        det = ", ".join(f"{g}{lfc_all[row[g]]:+.1f}" for g in gl if g in row)
        log(f"  {pname:<16s} {det[:52]:<52s} {np.nanmedian(v):>+12.2f}")
        if len(det) > 52:
            log(f"  {'':<16s} {det[52:]:<52s}")
        # panel score = 各基因 z-score 均值（在全部样本上）
        sub = M[gi][:, ti + ni]
        zz = (sub - np.nanmean(sub, 1, keepdims=True)) / \
             (np.nanstd(sub, 1, keepdims=True) + 1e-9)
        panel_score[pname] = np.nanmean(zz, 0)

    log("")
    log("  肿瘤内部：TGM2 与各 panel score 的 Spearman 相关（只用肿瘤样本）")
    tgv_t = M[tg, ti]
    for pname, sc in panel_score.items():
        s_t = sc[:len(ti)]
        ok = ~(np.isnan(tgv_t) | np.isnan(s_t))
        if ok.sum() < 20:
            continue
        r, p = stats.spearmanr(tgv_t[ok], s_t[ok])
        log(f"    {pname:<16s} rho={r:+.3f}  P={p:.2e}")

    # 成分校正。斜率必须在组内估计：
    # 如果直接在合并样本上回归，斜率会被"肿瘤组间质多"这件事本身带偏，
    # 等于把想检验的组间效应先塞进协变量里，属于过度校正。
    # 正确做法是先把 TGM2 和 panel score 各自在组内去均值，用组内变异估斜率，
    # 再用 校正后差值 = 原始差值 - beta × panel的组间差值。
    log("")
    log("  成分校正（斜率在组内估计，避免过度校正）")
    tgv = M[tg, ti + ni]
    grp = np.array([1] * len(ti) + [0] * len(ni))
    base = lfc_all[tg]

    def adjust(names):
        S = np.c_[[panel_score[k] for k in names]].T
        ok = ~(np.isnan(tgv) | np.isnan(S).any(1))
        y, X, g = tgv[ok], S[ok], grp[ok]
        yc = y.copy().astype(float)
        Xc = X.copy().astype(float)
        for gv in (0, 1):
            m = g == gv
            yc[m] -= yc[m].mean()
            Xc[m] -= Xc[m].mean(0)
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        dy = y[g == 1].mean() - y[g == 0].mean()
        dX = X[g == 1].mean(0) - X[g == 0].mean(0)
        adj = float(dy - beta @ dX)
        resid = y - X @ beta
        _, pp = stats.mannwhitneyu(resid[g == 1], resid[g == 0])
        return adj, float(pp), beta

    for pname in ["平滑肌/肌成纤维", "成纤维/胶原 ECM", "血管内皮/血管",
                  "免疫", "增殖"]:
        if pname not in panel_score:
            continue
        adj, pp, b = adjust([pname])
        log(f"    校正 {pname:<16s} 校正后 log2FC={adj:+.2f}  "
            f"(保留 {100*adj/base:.0f}%)  斜率={b[0]:+.2f}  P={pp:.2e}")

    keys = [k for k in ["平滑肌/肌成纤维", "成纤维/胶原 ECM", "血管内皮/血管",
                        "免疫", "增殖"] if k in panel_score]
    d, pp, _ = adjust(keys)
    log(f"    校正 全部成分 panel            校正后 log2FC={d:+.2f}  "
        f"(保留 {100*d/base:.0f}%)  P={pp:.2e}")
    summary[CA] = dict(cn=cn, tgm2_lfc=float(base), panel_lfc=panel_lfc,
                       adj_all=d, adj_all_p=float(pp),
                       retained_pct=float(100 * d / base))

# ---------- 图 ----------
fig, axes = plt.subplots(1, len(summary), figsize=(7.0 * len(summary), 5.6))
axes = np.atleast_1d(axes)
for ax, (CA, s) in zip(axes, summary.items()):
    names = list(s["panel_lfc"].keys())
    vals = [s["panel_lfc"][k] for k in names]
    names = ["TGM2"] + names
    vals = [s["tgm2_lfc"]] + vals
    cols = ["#ffd400"] + ["#e88b8b" if v > 0 else "#8ec7e8" for v in vals[1:]]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, edgecolor="#333", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, c="k", lw=0.9)
    for yy, v in zip(y, vals):
        ax.text(v + (0.06 if v >= 0 else -0.06), yy, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("log2FC（肿瘤 / 正常，中位数中心化后）", fontsize=10)
    ax.set_title(f"{CA} {s['cn']}\nTGM2 校正全部成分 panel 后残差 "
                 f"{s['adj_all']:+.2f}（保留 {s['retained_pct']:.0f}%）",
                 fontsize=11.5, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
fig.suptitle("TGM2 的肿瘤高表达 vs 间质/上皮/增殖成分对照",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/间质成分对照_TGM2.png", dpi=165)
plt.close(fig)
log("")
log(f"[图] {OUT}/间质成分对照_TGM2.png")

json.dump(summary, open(f"{OUT}/间质成分对照.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/间质成分对照_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
