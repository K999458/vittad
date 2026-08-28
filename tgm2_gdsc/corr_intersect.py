"""TGM2 与双轴交集差异基因的相关性分析（原始任务最后一项）

交集基因来源：axis2/交集_{癌种}_核心_方向一致.csv
  = 轴1（TGM2 高/低表达差异）∩ 轴2（耐药/非耐药差异）且方向一致

三层相关性：
  1) 原始 Spearman：TGM2 vs 基因，TCGA 原发肿瘤样本内
  2) 纯度偏相关：把 ABSOLUTE 肿瘤纯度作为协变量剔除
     —— 排除"两个基因都只是间质含量的读数"这种假相关
  3) 成纤维评分偏相关：把成纤维/ECM 评分作为协变量剔除
     —— 更直接地控制 CAF 含量

偏相关用秩变换后的三变量偏相关公式（等价于 partial Spearman）：
  r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
P 值用 t = r*sqrt((n-3)/(1-r^2))，df = n-3
"""
import sys, os, gzip, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt

BASE = "/store/zkyang/tgm2_gdsc"
OUT = f"{BASE}/corr"
os.makedirs(OUT, exist_ok=True)
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


def partial_r(x, y, z):
    """秩空间偏相关：控制 z 后 x 与 y 的相关"""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    n = int(ok.sum())
    if n < 20:
        return np.nan, np.nan, n
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    r = (rxy - rxz * ryz) / den
    r = float(np.clip(r, -0.999999, 0.999999))
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return r, float(p), n


FIB = ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "DCN", "LUM", "FBN1",
       "FN1", "POSTN", "THY1", "PDGFRB", "FAP", "SPARC", "VIM"]

# ---------- 纯度 ----------
PUR = pd.read_csv(f"{BASE}/tcga_absolute_purity.txt", sep="\t")
PUR = PUR[["array", "purity"]].dropna()
PUR["pid"] = PUR["array"].str[:15]
pur_map = dict(zip(PUR.pid, PUR.purity.astype(float)))
log(f"[0] ABSOLUTE 纯度覆盖 {len(pur_map)} 个 TCGA 样本")

summary = {}
for CA, cn in [("PAAD", "胰腺癌"), ("KIRC", "肾透明细胞癌"), ("ESCA", "食管癌")]:
    gf = f"{BASE}/axis2/交集_{CA}_核心_方向一致.csv"
    ef = f"{BASE}/tcga/{CA}_HiSeqV2.gz"
    if not (os.path.exists(gf) and os.path.exists(ef)):
        log(f"[{CA}] 缺文件，跳过")
        continue
    gs = pd.read_csv(gf)
    genes = gs.gene.astype(str).tolist()

    log("")
    log("=" * 100)
    log(f"### {CA} {cn}   交集基因 {len(genes)} 个")
    log("=" * 100)

    need = set(genes) | {"TGM2"} | set(FIB)
    got = {}
    with gzip.open(ef, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")[1:]
        for line in fh:
            g, rest = line.split("\t", 1)
            if g in need:
                got[g] = np.array([np.nan if x in ("", "NA") else float(x)
                                   for x in rest.rstrip("\n").split("\t")])
    if "TGM2" not in got:
        log("! 无 TGM2，跳过")
        continue

    # 原发肿瘤
    idx = np.array([i for i, s in enumerate(hdr) if s[13:15] == "01"])
    sids = [hdr[i] for i in idx]
    tg = got["TGM2"][idx]
    purv = np.array([pur_map.get(s[:15], np.nan) for s in sids])
    purv[(purv <= 0.05) | (purv >= 1.0)] = np.nan
    fibm = np.vstack([got[g][idx] for g in FIB if g in got])
    fibz = ((fibm - fibm.mean(axis=1, keepdims=True)) /
            (fibm.std(axis=1, keepdims=True) + 1e-9)).mean(axis=0)

    log(f"原发肿瘤 {len(idx)} 例；其中有纯度数据 {int(np.isfinite(purv).sum())} 例")
    log(f"交集基因在表达矩阵中命中 {sum(1 for g in genes if g in got)}/{len(genes)}")
    r_tp, p_tp = stats.spearmanr(tg[np.isfinite(purv)], purv[np.isfinite(purv)])
    r_tf, _ = stats.spearmanr(tg, fibz)
    log(f"TGM2 与纯度 rho={r_tp:+.3f} (P={p_tp:.1e})；"
        f"TGM2 与成纤维评分 rho={r_tf:+.3f}")

    rows = []
    for g in genes:
        v = got.get(g)
        if v is None:
            continue
        y = v[idx]
        ok = ~np.isnan(y)
        if ok.sum() < 30:
            continue
        r, p = stats.spearmanr(tg[ok], y[ok])
        rp, pp, npur = partial_r(tg, y, purv)
        rf, pf, _ = partial_r(tg, y, fibz)
        d = gs[gs.gene == g].iloc[0]
        rows.append(dict(
            gene=g, rho=float(r), p=float(p), n=int(ok.sum()),
            rho_adj_purity=rp, p_adj_purity=pp, n_purity=npur,
            rho_adj_fibro=rf, p_adj_fibro=pf,
            log2FC_axis1=float(d.log2FC_axis1),
            log2FC_axis2=float(d.log2FC_axis2),
            direction_axis1=d.direction_axis1))
    R = pd.DataFrame(rows)
    R["fdr"] = bh(R.p)
    R["fdr_adj_purity"] = bh(R.p_adj_purity.fillna(1))
    R["fdr_adj_fibro"] = bh(R.p_adj_fibro.fillna(1))
    R = R.sort_values("rho", ascending=False)
    R.to_csv(f"{OUT}/相关性_{CA}_交集基因.csv", index=False)

    sig = R[(R.fdr < 0.05) & (R.rho.abs() >= 0.3)]
    sig_p = R[(R.fdr_adj_purity < 0.05) & (R.rho_adj_purity.abs() >= 0.3)]
    sig_f = R[(R.fdr_adj_fibro < 0.05) & (R.rho_adj_fibro.abs() >= 0.3)]
    log("")
    log(f"|rho|>=0.3 且 FDR<0.05：")
    log(f"  原始            {len(sig):>3}/{len(R)} "
        f"（正相关 {int((sig.rho>0).sum())}，负相关 {int((sig.rho<0).sum())}）")
    log(f"  纯度校正后      {len(sig_p):>3}/{len(R)}   "
        f"保留率 {len(sig_p)/max(1,len(sig)):.0%}")
    log(f"  成纤维评分校正后 {len(sig_f):>3}/{len(R)}   "
        f"保留率 {len(sig_f)/max(1,len(sig)):.0%}")
    log(f"  rho 中位数 {R.rho.median():+.3f} -> 纯度校正 "
        f"{R.rho_adj_purity.median():+.3f} -> 成纤维校正 "
        f"{R.rho_adj_fibro.median():+.3f}")

    log("")
    log("  Top20 正相关（按原始 rho）")
    log(f"  {'基因':<12}{'rho':>8}{'FDR':>10}{'纯度校正rho':>12}"
        f"{'成纤维校正rho':>14}{'轴1 log2FC':>11}{'轴2 log2FC':>11}")
    for _, r in R.head(20).iterrows():
        log(f"  {r.gene[:11]:<12}{r.rho:>+8.3f}{r.fdr:>10.1e}"
            f"{r.rho_adj_purity:>+12.3f}{r.rho_adj_fibro:>+14.3f}"
            f"{r.log2FC_axis1:>+11.2f}{r.log2FC_axis2:>+11.2f}")
    neg = R.tail(10).iloc[::-1]
    if (neg.rho < 0).any():
        log("")
        log("  Top10 负相关")
        for _, r in neg.iterrows():
            log(f"  {r.gene[:11]:<12}{r.rho:>+8.3f}{r.fdr:>10.1e}"
                f"{r.rho_adj_purity:>+12.3f}{r.rho_adj_fibro:>+14.3f}"
                f"{r.log2FC_axis1:>+11.2f}{r.log2FC_axis2:>+11.2f}")

    # 纯度校正后仍然稳的核心基因
    core = R[(R.fdr_adj_purity < 0.05) & (R.rho_adj_purity >= 0.3) &
             (R.fdr_adj_fibro < 0.05) & (R.rho_adj_fibro >= 0.3)]
    core = core.sort_values("rho_adj_purity", ascending=False)
    core.to_csv(f"{OUT}/核心共表达_{CA}_双校正后.csv", index=False)
    log("")
    log(f"  ★ 双校正（纯度+成纤维）后仍 rho>=0.3 且 FDR<0.05 的基因："
        f"{len(core)} 个")
    if len(core):
        log("    " + ", ".join(core.gene.head(40)))

    summary[CA] = dict(cn=cn, n_genes=int(len(R)), n_tumor=int(len(idx)),
                       n_purity=int(np.isfinite(purv).sum()),
                       n_sig=int(len(sig)), n_sig_purity=int(len(sig_p)),
                       n_sig_fibro=int(len(sig_f)), n_core=int(len(core)),
                       tgm2_purity_rho=float(r_tp),
                       tgm2_fibro_rho=float(r_tf),
                       rho_median=float(R.rho.median()),
                       rho_median_purity=float(R.rho_adj_purity.median()),
                       rho_median_fibro=float(R.rho_adj_fibro.median()),
                       top=core.gene.head(15).tolist(),
                       R=R)

# ---------------- 图 1：Top 基因三种 rho 对比 ----------------
plotca = [c for c in ("PAAD", "KIRC") if c in summary]
if plotca:
    fig, axes = plt.subplots(1, len(plotca), figsize=(8.2 * len(plotca), 9))
    axes = np.atleast_1d(axes)
    for ax, CA in zip(axes, plotca):
        d = summary[CA]
        R = d["R"].sort_values("rho", ascending=False).head(22)
        R = R.iloc[::-1]
        y = np.arange(len(R))
        h = 0.26
        ax.barh(y + h, R.rho, height=h, color="#4c78a8",
                edgecolor="#333", lw=0.5, label="原始 rho")
        ax.barh(y, R.rho_adj_purity, height=h, color="#f58518",
                edgecolor="#333", lw=0.5, label="纯度校正")
        ax.barh(y - h, R.rho_adj_fibro, height=h, color="#54a24b",
                edgecolor="#333", lw=0.5, label="成纤维评分校正")
        ax.set_yticks(y)
        ax.set_yticklabels(R.gene, fontsize=9)
        ax.axvline(0, c="k", lw=0.9)
        ax.axvline(0.3, c="#999", ls="--", lw=0.8)
        ax.set_xlabel("与 TGM2 的 Spearman rho（TCGA 原发肿瘤）", fontsize=10.5)
        ax.set_title(f"{CA} {d['cn']}   交集基因 {d['n_genes']} 个   "
                     f"肿瘤 n={d['n_tumor']}（纯度 n={d['n_purity']}）\n"
                     f"Top22 正相关基因，虚线 = rho 0.3",
                     fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("TGM2 与双轴交集差异基因的相关性 —— 原始 vs 纯度/间质校正",
                 fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{OUT}/相关性_交集基因_Top.png", dpi=160)
    plt.close(fig)
    log("")
    log(f"[图] {OUT}/相关性_交集基因_Top.png")

# ---------------- 图 2：散点 ----------------
for CA in plotca:
    d = summary[CA]
    R = d["R"].sort_values("rho_adj_purity", ascending=False).head(6)
    if not len(R):
        continue
    ef = f"{BASE}/tcga/{CA}_HiSeqV2.gz"
    need = set(R.gene) | {"TGM2"}
    got = {}
    with gzip.open(ef, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")[1:]
        for line in fh:
            g, rest = line.split("\t", 1)
            if g in need:
                got[g] = np.array([np.nan if x in ("", "NA") else float(x)
                                   for x in rest.rstrip("\n").split("\t")])
    idx = np.array([i for i, s in enumerate(hdr) if s[13:15] == "01"])
    tg = got["TGM2"][idx]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    for ax, (_, r) in zip(axes.ravel(), R.iterrows()):
        y = got[r.gene][idx]
        ok = ~(np.isnan(tg) | np.isnan(y))
        ax.scatter(tg[ok], y[ok], s=15, alpha=0.55, c="#4c78a8",
                   edgecolors="none")
        b1, b0 = np.polyfit(tg[ok], y[ok], 1)
        xs = np.linspace(tg[ok].min(), tg[ok].max(), 20)
        ax.plot(xs, b0 + b1 * xs, c="#d62728", lw=2)
        ax.set_xlabel("TGM2  log2(norm_count+1)", fontsize=9.5)
        ax.set_ylabel(f"{r.gene}  log2(norm_count+1)", fontsize=9.5)
        ax.set_title(f"{r.gene}   rho={r.rho:+.2f}  FDR={r.fdr:.1e}\n"
                     f"纯度校正 {r.rho_adj_purity:+.2f}   "
                     f"成纤维校正 {r.rho_adj_fibro:+.2f}",
                     fontsize=10.5, fontweight="bold")
        ax.grid(ls=":", lw=0.6, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"{CA} {d['cn']}：TGM2 与交集基因共表达散点"
                 f"（纯度校正后 rho 最高的 6 个）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/散点_{CA}_TGM2共表达.png", dpi=155)
    plt.close(fig)
    log(f"[图] {OUT}/散点_{CA}_TGM2共表达.png")

# ---------------- 汇总 ----------------
log("")
log("=" * 100)
log("汇总")
log("=" * 100)
log(f"  {'癌种':<8}{'交集基因':>9}{'原始显著':>9}{'纯度校正':>9}"
    f"{'成纤维校正':>11}{'双校正核心':>11}{'rho中位数':>10}")
for CA, d in summary.items():
    log(f"  {CA:<8}{d['n_genes']:>9}{d['n_sig']:>9}{d['n_sig_purity']:>9}"
        f"{d['n_sig_fibro']:>11}{d['n_core']:>11}{d['rho_median']:>+10.3f}")

js = {k: {kk: vv for kk, vv in v.items() if kk != "R"}
      for k, v in summary.items()}
json.dump(js, open(f"{OUT}/相关性_汇总.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/相关性_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成", OUT)
