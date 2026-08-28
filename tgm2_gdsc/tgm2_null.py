"""经验零分布：随机抽 400 个基因跑同一套 GDSC 药谱相关，
给 TGM2 的「83% 药物呈耐药方向 / 中位 rho 0.212」一个真实的经验 P 值。

这是审稿人一定会问的问题：单基因 vs 全药谱的广谱相关，有多少是细胞系药理数据
本身的全局混杂（生长速率、谱系）？只有和随机基因的零分布比较才能回答。
"""
import os, json, re
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
import pandas as pd
from scipy import stats
import xenaPython as xena

OUT = "/store/zkyang/tgm2_gdsc"
HUB = "https://ucscpublic.xenahubs.net"
EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
N_RANDOM = 400
MIN_N = 50
rng = np.random.default_rng(20260828)
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def ck(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).split("_", 1)[0].upper())


# ---------- GDSC ----------
g = pd.read_parquet(f"{OUT}/gdsc_merged.parquet")
if "CL" not in g.columns:
    g["CL"] = g["CELL_LINE_NAME"].map(ck)
g = g.dropna(subset=["LN_IC50"])
g["KEY"] = g["DRUG_NAME"].astype(str) + "@" + g["SRC"]
drug_tbl = {}
for k, sub in g.groupby("KEY"):
    sub = sub.drop_duplicates("CL")
    if len(sub) >= MIN_N:
        drug_tbl[k] = (sub["CL"].to_numpy(), sub["LN_IC50"].to_numpy(float))
log(f"[0] 可评估药物组合 {len(drug_tbl)}")

# ---------- 随机基因表达 ----------
cache = f"{OUT}/ccle_random_genes.npz"
samples = None
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    Xm, gene_names, samples = z["X"], list(z["genes"]), list(z["samples"])
else:
    samples = xena.dataset_samples(HUB, EXPR, None)
    fields = list(xena.dataset_field(HUB, EXPR))
    TGM2 = [f for f in fields if f.split(".")[0] == "ENSG00000198959"][0]
    pool = [f for f in fields if f != TGM2]
    pick = [TGM2] + list(rng.choice(pool, N_RANDOM, replace=False))
    rows = []
    B = 50
    for i in range(0, len(pick), B):
        rows.extend(xena.dataset_fetch(HUB, EXPR, samples, pick[i:i + B]))
        log(f"    取表达 {min(i+B,len(pick))}/{len(pick)}")
    Xm = np.array([[np.nan if v is None else float(v) for v in r] for r in rows])
    gene_names = pick
    np.savez_compressed(cache, X=Xm, genes=np.array(gene_names, object),
                        samples=np.array(samples, object))
log(f"[0] 表达矩阵 {Xm.shape}（基因 x 细胞系），首个= TGM2")

cl_index = {}
for j, s in enumerate(samples):
    cl_index.setdefault(ck(s), j)

# ---------- 向量化相关 ----------
n_gene = Xm.shape[0]
frac_pos = np.zeros(n_gene)
med_rho = np.zeros(n_gene)
n_eval = np.zeros(n_gene, int)
rho_store = np.full((n_gene, len(drug_tbl)), np.nan)

for di, (k, (cls, ic50)) in enumerate(drug_tbl.items()):
    cols = np.array([cl_index.get(c, -1) for c in cls])
    ok = cols >= 0
    if ok.sum() < MIN_N:
        continue
    sub = Xm[:, cols[ok]]
    y = stats.rankdata(ic50[ok])
    y = (y - y.mean()) / (y.std() + 1e-12)
    valid = ~np.isnan(sub)
    allv = valid.all(axis=1)
    if allv.sum() == 0:
        continue
    S = sub[allv]
    # 常数行（表达全 0）会产生 nan，先剔除
    R = np.apply_along_axis(stats.rankdata, 1, S)
    Rm = R - R.mean(axis=1, keepdims=True)
    Rs = R.std(axis=1)
    good = Rs > 1e-9
    Z = np.zeros_like(Rm)
    Z[good] = Rm[good] / Rs[good][:, None]
    r = Z @ y / len(y)
    idx = np.where(allv)[0][good]
    rho_store[idx, di] = r[good]

for i in range(n_gene):
    v = rho_store[i]
    v = v[~np.isnan(v)]
    if len(v) < 100:
        continue
    n_eval[i] = len(v)
    frac_pos[i] = (v > 0).mean()
    med_rho[i] = np.median(v)

valid = n_eval >= 100
log(f"[1] 有效基因 {valid.sum()} / {n_gene}")

tg_fp, tg_mr = frac_pos[0], med_rho[0]
null_fp = frac_pos[1:][valid[1:]]
null_mr = med_rho[1:][valid[1:]]

log("")
log("=" * 96)
log("经验零分布：随机基因 vs TGM2（GDSC 全药谱，CCLE 表达）")
log("=" * 96)
log("")
log(f"TGM2：耐药方向药物占比 = {100*tg_fp:.1f}%   中位 rho = {tg_mr:+.3f}   "
    f"可评估药物 {n_eval[0]}")
log("")
log(f"随机 {len(null_fp)} 个基因的零分布：")
for lab, arr, tv in [("耐药方向占比", null_fp, tg_fp), ("中位 rho", null_mr, tg_mr)]:
    q = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    unit = "%" if "占比" in lab else ""
    sc = 100 if "占比" in lab else 1
    log(f"  {lab:<12} 均值 {sc*arr.mean():.3f}{unit}  标准差 {sc*arr.std():.3f}  "
        f"| P1 {sc*q[0]:.2f} P5 {sc*q[1]:.2f} P25 {sc*q[2]:.2f} 中位 {sc*q[3]:.2f} "
        f"P75 {sc*q[4]:.2f} P95 {sc*q[5]:.2f} P99 {sc*q[6]:.2f}")
    n_ge = int((arr >= tv).sum())
    pemp = (n_ge + 1) / (len(arr) + 1)
    z = (tv - arr.mean()) / (arr.std() + 1e-12)
    log(f"  {'':12} -> TGM2 超过 {100*(1-n_ge/len(arr)):.1f}% 的随机基因；"
        f"经验单侧 P = {pemp:.4f}；Z = {z:+.2f}")
log("")
log("解读要点：")
log("  · 若零分布本身就集中在 50% 附近且方差小，则 TGM2 的 83% 是真信号。")
log("  · 若零分布很宽（很多随机基因也能到 80%+），说明这是细胞系药理数据的全局混杂，")
log("    单基因-全药谱相关不能作为独立证据，必须做谱系分层与增殖校正。")

pd.DataFrame({"gene": [gene_names[i] for i in range(n_gene) if valid[i]],
              "n_drug": n_eval[valid], "frac_pos": frac_pos[valid],
              "med_rho": med_rho[valid]}).to_csv(f"{OUT}/null_distribution.csv", index=False)

open(f"{OUT}/tgm2_null_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/tgm2_null_report.txt")
