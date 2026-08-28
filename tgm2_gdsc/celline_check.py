"""正交验证：纯肿瘤细胞里 TGM2 到底高不高？

bulk RNA 有个绕不开的问题 —— TGM2 本身就是 ECM 交联酶，
校正 ECM signature 等于把要检验的东西塞进协变量，结论会循环。
细胞系没有间质、没有免疫细胞、没有血管，是纯肿瘤细胞，
拿它看 TGM2 在胰腺癌/肾癌谱系里的位置，可以绕开成分混杂。
"""
import sys, json
import numpy as np
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


tg = json.load(open(f"{BASE}/ccle_TGM2_rpkm.json"))
CN = {
    "PANCREAS": "胰腺", "KIDNEY": "肾", "OESOPHAGUS": "食管",
    "LUNG": "肺", "LARGE_INTESTINE": "结直肠", "SKIN": "皮肤",
    "BREAST": "乳腺", "OVARY": "卵巢", "STOMACH": "胃", "LIVER": "肝",
    "CENTRAL_NERVOUS_SYSTEM": "中枢神经", "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "血液淋巴",
    "URINARY_TRACT": "泌尿道", "PROSTATE": "前列腺", "ENDOMETRIUM": "子宫内膜",
    "SOFT_TISSUE": "软组织", "BONE": "骨", "THYROID": "甲状腺",
    "UPPER_AERODIGESTIVE_TRACT": "头颈", "AUTONOMIC_GANGLIA": "神经节",
    "BILIARY_TRACT": "胆道", "PLEURA": "胸膜", "CERVIX": "宫颈",
    "SALIVARY_GLAND": "唾液腺", "SMALL_INTESTINE": "小肠",
}

by = {}
for name, v in tg.items():
    if v is None or (isinstance(v, float) and np.isnan(v)):
        continue
    parts = name.split("_", 1)
    if len(parts) < 2:
        continue
    lin = parts[1]
    by.setdefault(lin, []).append(float(v))

by = {k: v for k, v in by.items() if len(v) >= 8}
allv = np.array([x for v in by.values() for x in v])
log(f"[0] CCLE 细胞系 {len(allv)} 株，谱系 {len(by)} 个")
log(f"[0] TGM2 全体中位数 {np.median(allv):.2f} RPKM")
log("")
log("=" * 82)
log("TGM2 在各谱系肿瘤细胞系中的表达（纯肿瘤细胞，无间质）")
log("=" * 82)
log(f"  {'谱系':<14s} {'n':>4s} {'中位数':>8s} {'vs其余 P':>11s} {'排名':>8s}")
log("  " + "-" * 76)

rows = []
for lin, v in by.items():
    v = np.array(v)
    other = np.array([x for k, vv in by.items() if k != lin for x in vv])
    _, p = stats.mannwhitneyu(v, other, alternative="two-sided")
    rows.append((lin, len(v), float(np.median(v)), float(p)))
rows.sort(key=lambda r: -r[2])
for i, (lin, n, m, p) in enumerate(rows, 1):
    mark = "  <<<" if lin in ("PANCREAS", "KIDNEY", "OESOPHAGUS") else ""
    log(f"  {CN.get(lin,lin):<14s} {n:>4d} {m:>8.2f} {p:>11.2e} "
        f"{i:>4d}/{len(rows)}{mark}")

log("")
for lin in ("PANCREAS", "KIDNEY", "OESOPHAGUS"):
    if lin not in by:
        continue
    v = np.array(by[lin])
    rk = [r[0] for r in rows].index(lin) + 1
    pct = stats.percentileofscore(allv, np.median(v))
    log(f"  {CN[lin]}癌细胞系: n={len(v)} 中位数 {np.median(v):.2f}，"
        f"在 {len(rows)} 个谱系里排第 {rk}，"
        f"高于全体细胞系的 {pct:.0f}%")

# ---------- 图 ----------
order = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(13, 6.4))
data = [[max(x, 0.05) for x in by[l]] for l in order]
bp = ax.boxplot(data, positions=range(len(order)), widths=0.66,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="k", lw=1.3))
for b, l in zip(bp["boxes"], order):
    b.set_facecolor("#ffd400" if l in ("PANCREAS", "KIDNEY", "OESOPHAGUS")
                    else "#b8cfe0")
    b.set_edgecolor("#333")
rng = np.random.default_rng(2)
for i, d in enumerate(data):
    ax.scatter(i + rng.normal(0, 0.08, len(d)), d, s=7, c="#333",
               alpha=0.35, linewidths=0, zorder=3)
ax.axhline(np.median(allv), ls="--", c="#c62828", lw=1.1,
           label=f"全体细胞系中位数 {np.median(allv):.1f} RPKM")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{CN.get(l,l)}\nn={len(by[l])}" for l in order],
                   rotation=55, ha="right", fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("TGM2 表达量  RPKM（对数刻度）", fontsize=11)
ax.set_title("TGM2 在 CCLE 肿瘤细胞系中的谱系分布（纯肿瘤细胞、无间质成分）\n"
             "黄色为三个目标癌种对应的谱系；FIBROBLAST 行是正常成纤维细胞系，可作间质参照",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/细胞系_TGM2_谱系分布.png", dpi=160)
plt.close(fig)
log("")
log(f"[图] {OUT}/细胞系_TGM2_谱系分布.png")

open(f"{OUT}/细胞系验证_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
