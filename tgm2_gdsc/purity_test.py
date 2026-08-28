"""决定性检验：TGM2 到底是肿瘤细胞表达的，还是间质表达的？

思路：
  TCGA 的 bulk 样本 = 肿瘤细胞 + 间质 + 免疫细胞的混合。
  ABSOLUTE 给出每个样本的肿瘤细胞占比（purity）。
  一个基因如果由肿瘤细胞表达，它的表达量应该随 purity 上升；
  如果由间质表达，应该随 purity 下降。这个方向是不受归一化影响的。

  再进一步，把表达量对 purity 做线性回归，外推到：
    purity = 1  ->  纯肿瘤细胞的表达量
    purity = 0  ->  纯间质的表达量
  两者一比就知道 TGM2 主要来自哪一侧。

参照基因：
  EPCAM / KRT19 / KRT8   —— 上皮/肿瘤细胞，应该是正相关
  COL1A1 / DCN / LUM     —— 成纤维，应该是负相关
  PTPRC / CD68           —— 免疫，应该是负相关
  PECAM1 / VWF           —— 内皮，应该是负相关
"""
import sys, os, gzip, json
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


PUR = pd.read_csv(f"{BASE}/tcga_absolute_purity.txt", sep="\t")
PUR = PUR[["array", "purity"]].dropna()
PUR["pid"] = PUR["array"].str[:15]
pur = dict(zip(PUR.pid, PUR.purity.astype(float)))
log(f"[0] ABSOLUTE 纯度覆盖 {len(pur)} 个样本")

REF = {
    "TGM2": "★ 待检验",
    "EPCAM": "上皮/肿瘤", "KRT19": "上皮/肿瘤", "KRT8": "上皮/肿瘤",
    "COL1A1": "成纤维", "DCN": "成纤维", "LUM": "成纤维",
    "ACTA2": "平滑肌/肌成纤维", "TAGLN": "平滑肌/肌成纤维",
    "PTPRC": "免疫", "CD68": "免疫",
    "PECAM1": "内皮", "VWF": "内皮",
    "GAPDH": "管家", "ACTB": "管家",
}

results = {}
for CA, cn in [("PAAD", "胰腺癌"), ("KIRC", "肾透明细胞癌"),
               ("ESCA", "食管癌")]:
    f = f"{BASE}/tcga/{CA}_HiSeqV2.gz"
    if not os.path.exists(f):
        log(f"[{CA}] 缺少表达文件，跳过")
        continue
    with gzip.open(f, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")[1:]
        want = {g: None for g in REF}
        for line in fh:
            g, rest = line.split("\t", 1)
            if g in want:
                want[g] = np.array([np.nan if x in ("", "NA") else float(x)
                                    for x in rest.rstrip("\n").split("\t")])
    # 只保留原发肿瘤且有纯度的样本
    idx, pv = [], []
    for i, s in enumerate(hdr):
        code = s[13:15]
        if code != "01":
            continue
        p = pur.get(s[:15])
        if p is None or not (0.05 < p < 1.0):
            continue
        idx.append(i)
        pv.append(p)
    idx = np.array(idx)
    pv = np.array(pv)
    log("")
    log("=" * 96)
    log(f"{CA} {cn}   有纯度数据的原发肿瘤 {len(idx)} 例   "
        f"纯度中位数 {np.median(pv):.2f}（{pv.min():.2f}~{pv.max():.2f}）")
    log("=" * 96)
    log(f"  {'基因':<9s} {'类别':<16s} {'与纯度 rho':>11s} {'P':>10s} "
        f"{'纯肿瘤外推':>10s} {'纯间质外推':>10s} {'肿瘤/间质':>9s}")
    log("  " + "-" * 90)
    rows = []
    for g, tag in REF.items():
        v = want.get(g)
        if v is None:
            continue
        y = v[idx]
        ok = ~np.isnan(y)
        if ok.sum() < 30:
            continue
        r, p = stats.spearmanr(pv[ok], y[ok])
        b1, b0 = np.polyfit(pv[ok], y[ok], 1)
        at1, at0 = b0 + b1, b0
        rows.append(dict(gene=g, tag=tag, rho=float(r), p=float(p),
                         at_pure_tumor=float(at1), at_pure_stroma=float(at0),
                         diff=float(at1 - at0)))
        log(f"  {g:<9s} {tag:<16s} {r:>+11.3f} {p:>10.1e} "
            f"{at1:>10.2f} {at0:>10.2f} {at1-at0:>+9.2f}")
    results[CA] = dict(cn=cn, n=len(idx), rows=rows,
                       purity_median=float(np.median(pv)))

    tg = [r for r in rows if r["gene"] == "TGM2"]
    if tg:
        t = tg[0]
        epi = np.mean([r["rho"] for r in rows if r["tag"] == "上皮/肿瘤"])
        fib = np.mean([r["rho"] for r in rows if r["tag"] == "成纤维"])
        log("")
        if t["rho"] > 0.1:
            v = "→ TGM2 随纯度升高，说明主要由肿瘤细胞表达"
        elif t["rho"] < -0.1:
            v = "→ TGM2 随纯度下降，说明主要由间质表达"
        else:
            v = "→ TGM2 与纯度基本无关，肿瘤细胞和间质都有贡献"
        log(f"  TGM2 rho={t['rho']:+.3f}  "
            f"（上皮参照均值 {epi:+.3f}，成纤维参照均值 {fib:+.3f}）")
        log(f"  {v}")

# ---------- 图 ----------
if results:
    fig, axes = plt.subplots(1, len(results), figsize=(6.6 * len(results), 6.4))
    axes = np.atleast_1d(axes)
    CMAP = {"★ 待检验": "#ffd400", "上皮/肿瘤": "#e88b8b", "成纤维": "#f0b27a",
            "平滑肌/肌成纤维": "#f5cba7", "免疫": "#a9d18e",
            "内皮": "#c39bd3", "管家": "#cccccc"}
    for ax, (CA, d) in zip(axes, results.items()):
        rows = sorted(d["rows"], key=lambda r: r["rho"])
        y = np.arange(len(rows))
        ax.barh(y, [r["rho"] for r in rows],
                color=[CMAP.get(r["tag"], "#ccc") for r in rows],
                edgecolor="#333", height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['gene']}  {r['tag']}" for r in rows],
                           fontsize=9.5)
        for yy, r in zip(y, rows):
            if r["gene"] == "TGM2":
                ax.get_yticklabels()[yy].set_fontweight("bold")
            off = 0.012 if r["rho"] >= 0 else -0.012
            ax.text(r["rho"] + off, yy, f"{r['rho']:+.2f}", va="center",
                    ha="left" if r["rho"] >= 0 else "right", fontsize=8.5)
        ax.axvline(0, c="k", lw=0.9)
        ax.set_xlabel("与 ABSOLUTE 肿瘤纯度的 Spearman rho", fontsize=10)
        ax.set_title(f"{CA} {d['cn']}   n={d['n']}\n"
                     f"rho>0 = 肿瘤细胞表达    rho<0 = 间质表达",
                     fontsize=11.5, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
    fig.suptitle("TGM2 是肿瘤细胞表达还是间质表达？—— 肿瘤纯度相关性检验",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/纯度检验_TGM2.png", dpi=165)
    plt.close(fig)
    log("")
    log(f"[图] {OUT}/纯度检验_TGM2.png")

json.dump(results, open(f"{OUT}/纯度检验_TGM2.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/纯度检验_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
