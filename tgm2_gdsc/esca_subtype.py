"""ESCA 结果与假设相反，拆组织学亚型核实。

TCGA-ESCA 是个混合队列：食管鳞癌(ESCC) + 食管腺癌/胃食管连接部腺癌(EAC)。
两者的正常对照组织完全不同：
  ESCC 对应 GTEx Esophagus-Mucosa（复层鳞状上皮）
  EAC  对应 GTEx Esophagus-Gastroesophageal Junction / Stomach（柱状上皮）
TGM2 在正常鳞状上皮里本身就是分化相关基因、表达很高，
拿它当分母，鳞癌一定显得"低表达"。必须分开算。
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt
import xenaPython as xena

os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")

BASE = "/store/zkyang/tgm2_gdsc"
OUT = f"{BASE}/pancancer"
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


# ---------- 取 ESCA 组织学亚型 ----------
HUB = "https://tcga.xenahubs.net"
CLIN = "TCGA.ESCA.sampleMap/ESCA_clinicalMatrix"


def decode(hub, ds, samples, field):
    c = xena.field_codes(hub, ds, [field])[0].get("code")
    v = xena.dataset_fetch(hub, ds, samples, [field])[0]
    if not c:
        return v
    cl = c.split("\t")
    o = []
    for x in v:
        try:
            i = int(float(x))
            o.append(cl[i] if 0 <= i < len(cl) else None)
        except (TypeError, ValueError):
            o.append(None)
    return o


csamp = xena.dataset_samples(HUB, CLIN, None)
fields = xena.dataset_field(HUB, CLIN)
cand = [f for f in fields if "histolog" in f.lower()]
log(f"[0] ESCA 临床表型里的组织学字段: {cand}")
HF = "histological_type" if "histological_type" in cand else cand[0]
hist = decode(HUB, CLIN, csamp, HF)
h_by_pt = {}
for s, h in zip(csamp, hist):
    if h:
        h_by_pt[s[:12]] = h
from collections import Counter
log(f"[0] 亚型分布: {dict(Counter(h_by_pt.values()))}")

# ---------- GTEx 食管亚区 ----------
GHUB = "https://toil.xenahubs.net"
GP = "GTEX_phenotype"
gs = xena.dataset_samples(GHUB, GP, None)
gfields = xena.dataset_field(GHUB, GP)
log(f"[0] GTEX_phenotype 字段: {gfields}")
tf = next((f for f in gfields if "specimen" in f.lower()
           or "detailed" in f.lower() or "body_site" in f.lower()), None)
sub = decode(GHUB, GP, gs, tf) if tf else [None] * len(gs)
log(f"[0] 用字段 {tf}；食管相关亚区: "
    f"{sorted({x for x in sub if x and 'sophag' in x})}")
gsub = {s: x for s, x in zip(gs, sub)}

# ---------- 载入缓存矩阵 ----------
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
syms_all = np.array([id2sym.get(ids[k], ids[k]) for k in keep])
M = mat[keep]
TG = int(np.where(syms_all == "TGM2")[0][0])

raw = json.load(open(f"{OUT}/toil_tgm2.json"))
ph, psamples = raw["pheno"], raw["psamples"]
meta = {s: (ph["_primary_site"][i], ph["_sample_type"][i],
            ph["_study"][i], ph["detailed_category"][i])
        for i, s in enumerate(psamples)}

esca_t = [s for s in union if meta.get(s) and meta[s][2] == "TCGA"
          and meta[s][3] == "Esophageal Carcinoma"
          and "Normal" not in str(meta[s][1])]
esca_n = [s for s in union if meta.get(s) and meta[s][2] == "TCGA"
          and meta[s][3] == "Esophageal Carcinoma"
          and "Normal" in str(meta[s][1])]
gtex_eso = [s for s in union if meta.get(s) and meta[s][2] == "GTEX"
            and meta[s][0] == "Esophagus"]
gtex_sto = [s for s in union if meta.get(s) and meta[s][2] == "GTEX"
            and meta[s][0] == "Stomach"]

scc = [s for s in esca_t if "Squamous" in str(h_by_pt.get(s[:12], ""))]
adc = [s for s in esca_t if "Adeno" in str(h_by_pt.get(s[:12], ""))]
log("")
log(f"[1] ESCA 肿瘤 {len(esca_t)}：鳞癌 {len(scc)}  腺癌 {len(adc)}  "
    f"未分类 {len(esca_t)-len(scc)-len(adc)}")

muc = [s for s in gtex_eso if "Mucosa" in str(gsub.get(s, ""))]
mus = [s for s in gtex_eso if "Muscularis" in str(gsub.get(s, ""))]
gej = [s for s in gtex_eso if "Junction" in str(gsub.get(s, ""))]
log(f"[1] GTEx 食管 {len(gtex_eso)}：黏膜 {len(muc)}  肌层 {len(mus)}  "
    f"连接部 {len(gej)}；GTEx 胃 {len(gtex_sto)}")


def med(names):
    if not names:
        return np.nan, 0
    v = M[TG, [uidx[s] for s in names]]
    v = v[~np.isnan(v)]
    return (float(np.median(v)), len(v)) if len(v) else (np.nan, 0)


log("")
log("=" * 78)
log("TGM2 表达中位数  log2(TPM+0.001)")
log("=" * 78)
for lab, g in [("GTEx 食管黏膜（鳞状上皮）", muc),
               ("GTEx 食管肌层", mus),
               ("GTEx 食管胃连接部", gej),
               ("GTEx 胃", gtex_sto),
               ("TCGA ESCA 癌旁", esca_n),
               ("TCGA ESCA 鳞癌 ESCC", scc),
               ("TCGA ESCA 腺癌 EAC", adc)]:
    m, n = med(g)
    log(f"  {lab:<26s} n={n:<5d} 中位数 {m:.2f}")

log("")
log("=" * 78)
log("亚型 vs 对应正常组织的 TGM2 差异（含全基因组中位数中心化校正）")
log("=" * 78)

CMP = [
    ("ESCC 鳞癌", scc, "GTEx 食管黏膜", muc),
    ("ESCC 鳞癌", scc, "GTEx 食管黏膜+癌旁", muc + esca_n),
    ("EAC 腺癌", adc, "GTEx 连接部+胃", gej + gtex_sto),
    ("EAC 腺癌", adc, "GTEx 胃", gtex_sto),
]
out = []
for tn, tg_, nn, ng in CMP:
    if len(tg_) < 15 or len(ng) < 10:
        log(f"  {tn} vs {nn}: 样本不足 (T={len(tg_)}, N={len(ng)})，跳过")
        continue
    ti = [uidx[s] for s in tg_]
    ni = [uidx[s] for s in ng]
    A, N = M[:, ti], M[:, ni]
    ok = (~np.isnan(A)).sum(1) > 0.8 * len(ti)
    ok &= (~np.isnan(N)).sum(1) > 0.8 * len(ni)
    ok &= (np.nanmean(A, 1) > -5) | (np.nanmean(N, 1) > -5)
    A, N, S = A[ok], N[ok], syms_all[ok]
    off = np.nanmedian(np.concatenate([A, N], axis=1), axis=0)
    A = A - off[:len(ti)]
    N = N - off[len(ti):]
    lfc = np.nanmean(A, 1) - np.nanmean(N, 1)
    _, p = stats.mannwhitneyu(A, N, axis=1, alternative="two-sided",
                              nan_policy="omit")
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    q = bh(p)
    j = int(np.where(S == "TGM2")[0][0])
    rk = int((lfc > lfc[j]).sum()) + 1
    verdict = ("★ 显著高表达" if q[j] < 0.05 and lfc[j] >= 1 else
               "✓ 高表达" if q[j] < 0.05 and lfc[j] > 0 else
               "▼ 显著低表达" if q[j] < 0.05 and lfc[j] <= -1 else
               "▽ 低表达" if q[j] < 0.05 and lfc[j] < 0 else "— 无差异")
    log(f"  {tn} (n={len(tg_)}) vs {nn} (n={len(ng)}):")
    log(f"      TGM2 log2FC={lfc[j]:+.2f}  FDR={q[j]:.2e}  "
        f"排名 {rk}/{len(lfc)} (前 {100*rk/len(lfc):.1f}%)  {verdict}")
    out.append(dict(tumor=tn, n_tumor=len(tg_), normal=nn, n_normal=len(ng),
                    log2FC=float(lfc[j]), fdr=float(q[j]), rank=rk,
                    n_genes=int(len(lfc)), verdict=verdict))

# ---------- 箱线图 ----------
BOX = [("GTEx 食管黏膜\n(鳞状上皮)", muc, "#8ec7e8"),
       ("GTEx 食管肌层\n(平滑肌)", mus, "#f0b27a"),
       ("GTEx 连接部\n(含平滑肌)", gej, "#f0b27a"),
       ("GTEx 胃\n(柱状上皮)", gtex_sto, "#8ec7e8"),
       ("TCGA 癌旁", esca_n, "#b6dfa8"),
       ("ESCC 鳞癌", scc, "#e88b8b"), ("EAC 腺癌", adc, "#c98be8")]
BOX = [(l, g, c) for l, g, c in BOX if len(g) >= 5]
fig, ax = plt.subplots(figsize=(11, 6))
data = [M[TG, [uidx[s] for s in g]] for _, g, _ in BOX]
data = [d[~np.isnan(d)] for d in data]
bp = ax.boxplot(data, positions=range(len(BOX)), widths=0.66,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="k", lw=1.4))
for b, (_, _, c) in zip(bp["boxes"], BOX):
    b.set_facecolor(c)
    b.set_edgecolor("#333")
rng = np.random.default_rng(1)
for i, d in enumerate(data):
    ax.scatter(i + rng.normal(0, 0.08, len(d)), d, s=6, c="#333",
               alpha=0.3, linewidths=0, zorder=3)
ax.set_xticks(range(len(BOX)))
ax.set_xticklabels([f"{l}\nn={len(d)}" for (l, _, _), d in zip(BOX, data)],
                   fontsize=9.5)
ax.set_ylabel("TGM2  log2(TPM+0.001)", fontsize=11)
ax.set_title("TGM2 在食管相关组织与 ESCA 亚型中的表达\n"
             "TGM2 是平滑肌/间质基因：GTEx 食管肌层与连接部很高(≈7.1)、"
             "黏膜上皮低(4.4)\n"
             "把整个 GTEx 食管(58% 是肌层+连接部)当对照，"
             "上皮来源的肿瘤就会被误判成\"低表达\"",
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
ax.set_ylim(0.5, 10.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#8ec7e8", ec="#333", label="GTEx 正常上皮"),
                   Patch(fc="#f0b27a", ec="#333", label="GTEx 正常平滑肌"),
                   Patch(fc="#b6dfa8", ec="#333", label="TCGA 癌旁"),
                   Patch(fc="#e88b8b", ec="#333", label="ESCC 鳞癌"),
                   Patch(fc="#c98be8", ec="#333", label="EAC 腺癌")],
          loc="lower left", fontsize=9, ncol=2, framealpha=0.95)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/箱线图_ESCA亚型_TGM2.png", dpi=165)
plt.close(fig)
log("")
log(f"[图] {OUT}/箱线图_ESCA亚型_TGM2.png")

json.dump(out, open(f"{OUT}/ESCA亚型_TGM2.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/ESCA亚型_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
