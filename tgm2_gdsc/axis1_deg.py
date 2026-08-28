"""轴 1：TCGA 中 TGM2 高表达 vs 低表达 的差异分析

数据：UCSC Xena TCGA HiSeqV2（log2(RSEM norm_count + 1)，20530 基因）
分组：仅原发肿瘤样本（barcode 后缀 -01），按 TGM2 表达上/下三分位（避免中位数附近噪声）
统计：Mann-Whitney U + log2FC（组均值差，数据已是 log2 尺度）+ BH FDR
阈值：|log2FC| >= 0.585 (FC 1.5) 且 FDR < 0.05
"""
import gzip, os, sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/store/zkyang/tgm2_gdsc/tcga"
OUT = "/store/zkyang/tgm2_gdsc/axis1"
os.makedirs(OUT, exist_ok=True)
LFC = 0.585
FDR = 0.05
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r)); q[i] = prev
    return q


def load(path):
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep="\t", index_col=0)


results = {}
for cancer in ["KIRC", "ESCA", "PAAD"]:
    log("")
    log("=" * 92)
    log(f"### TCGA-{cancer}  轴1：TGM2 高表达组 vs 低表达组")
    log("=" * 92)
    df = load(f"{BASE}/{cancer}_HiSeqV2.gz")
    log(f"原始矩阵：{df.shape[0]} 基因 x {df.shape[1]} 样本")

    # 只留原发肿瘤 -01
    tum = [c for c in df.columns if c.split("-")[-1].startswith("01")]
    df = df[tum]
    log(f"原发肿瘤样本：{len(tum)}")

    if "TGM2" not in df.index:
        log("! 矩阵中无 TGM2，跳过")
        continue
    tg = df.loc["TGM2"]
    log(f"TGM2 表达 log2(norm_count+1)：中位 {tg.median():.2f}，"
        f"范围 {tg.min():.2f} ~ {tg.max():.2f}")

    lo_cut, hi_cut = tg.quantile(1 / 3), tg.quantile(2 / 3)
    lo = tg[tg <= lo_cut].index
    hi = tg[tg >= hi_cut].index
    log(f"分组（上/下三分位）：TGM2 高组 n={len(hi)}（>= {hi_cut:.2f}），"
        f"低组 n={len(lo)}（<= {lo_cut:.2f}）")

    A = df[hi].to_numpy(float)
    B = df[lo].to_numpy(float)
    # 剔除低表达基因（两组都近乎不表达）
    keep = (np.mean(A > 0, axis=1) > 0.25) | (np.mean(B > 0, axis=1) > 0.25)
    genes = df.index[keep].to_numpy()
    A, B = A[keep], B[keep]
    log(f"过滤低表达后保留基因：{len(genes)}")

    lfc = A.mean(axis=1) - B.mean(axis=1)
    u, p = stats.mannwhitneyu(A, B, axis=1, alternative="two-sided")
    q = bh(p)

    res = pd.DataFrame({"gene": genes, "log2FC": lfc, "p": p, "fdr": q,
                        "mean_high": A.mean(axis=1), "mean_low": B.mean(axis=1)})
    res["direction"] = np.where(res.log2FC > 0, "up_in_TGM2high", "down_in_TGM2high")
    sig = res[(res.fdr < FDR) & (res.log2FC.abs() >= LFC)].copy()
    sig = sig.sort_values("log2FC", ascending=False)

    up = (sig.log2FC > 0).sum(); dn = (sig.log2FC < 0).sum()
    log("")
    log(f"差异基因（FDR<{FDR} 且 |log2FC|>={LFC}）：**{len(sig)}** 个"
        f"（TGM2高组上调 {up}，下调 {dn}）")

    res.to_csv(f"{OUT}/axis1_{cancer}_all.csv", index=False)
    sig.to_csv(f"{OUT}/axis1_{cancer}_DEG.csv", index=False)

    log("")
    log("  Top20 上调（TGM2 高组）")
    log(f"  {'基因':<14}{'log2FC':>9}{'FDR':>12}{'高组均值':>10}{'低组均值':>10}")
    for _, r in sig.head(20).iterrows():
        log(f"  {r.gene[:13]:<14}{r.log2FC:>9.2f}{r.fdr:>12.2e}"
            f"{r.mean_high:>10.2f}{r.mean_low:>10.2f}")
    log("")
    log("  Top20 下调（TGM2 高组）")
    for _, r in sig.tail(20).iloc[::-1].iterrows():
        log(f"  {r.gene[:13]:<14}{r.log2FC:>9.2f}{r.fdr:>12.2e}"
            f"{r.mean_high:>10.2f}{r.mean_low:>10.2f}")

    results[cancer] = set(sig.gene)

# 三癌种 DEG 交集（顺带给出 pan-cancer 稳健基因）
if len(results) >= 2:
    log("")
    log("=" * 92)
    log("三癌种 轴1 DEG 交集（TGM2 高低分组差异在多个癌种共有）")
    log("=" * 92)
    ks = list(results)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            log(f"  {ks[i]} ∩ {ks[j]}: {len(results[ks[i]] & results[ks[j]])}")
    if len(ks) == 3:
        tri = results[ks[0]] & results[ks[1]] & results[ks[2]]
        log(f"  三者交集: {len(tri)}")
        pd.Series(sorted(tri)).to_csv(f"{OUT}/axis1_三癌种交集.csv",
                                      index=False, header=["gene"])
        log("")
        log("  三癌种共有 DEG（前 60 个）：")
        log("  " + ", ".join(sorted(tri)[:60]))

open(f"{OUT}/axis1_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/axis1_report.txt")
