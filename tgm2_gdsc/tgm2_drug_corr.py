"""TGM2 表达 vs 药物 IC50 相关性（CCLE 表达 x GDSC 药敏）

目的：检验课题核心假设 —— TGM2 高表达是否真的对应耐药（IC50 升高）。
输出：pan-cancer 与 肾/食管/胰腺 组织内的 Spearman 相关，按显著性排序。
"""
import os, json, re, sys
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
import pandas as pd
from scipy import stats

OUT = "/store/zkyang/tgm2_gdsc"
HUB = "https://ucscpublic.xenahubs.net"
EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
TGM2_ID = "ENSG00000198959.7"

# ---------- 1. CCLE TGM2 表达 ----------
cache = f"{OUT}/ccle_TGM2_rpkm.json"
if os.path.exists(cache):
    expr = json.load(open(cache))
else:
    import xenaPython as xena
    samples = xena.dataset_samples(HUB, EXPR, None)
    vals = xena.dataset_fetch(HUB, EXPR, samples, [TGM2_ID])[0]
    expr = {s: v for s, v in zip(samples, vals)}
    json.dump(expr, open(cache, "w"))
print(f"[1] CCLE 细胞系数 {len(expr)}")


def norm(name):
    """细胞系名归一化：去掉组织后缀、非字母数字字符，转大写"""
    n = str(name).upper()
    n = n.split("_")[0] if "_" in n and not n[0].isdigit() else n
    return re.sub(r"[^A-Z0-9]", "", n)


# CCLE 名 XXX_TISSUE -> 归一名 + 组织
ccle = {}
for k, v in expr.items():
    if v is None or (isinstance(v, float) and np.isnan(v)):
        continue
    parts = k.split("_", 1)
    cl, tis = parts[0], (parts[1] if len(parts) > 1 else "")
    ccle[re.sub(r"[^A-Z0-9]", "", cl.upper())] = {"tgm2": float(v), "ccle_tissue": tis, "ccle_name": k}
print(f"[1] 有 TGM2 值的细胞系 {len(ccle)}  (log2(RPKM+1) 中位数 "
      f"{np.median([c['tgm2'] for c in ccle.values()]):.2f})")

# ---------- 2. GDSC 药敏 ----------
frames = []
for f, tag in [(f"{OUT}/GDSC2_fitted.xlsx", "GDSC2"), (f"{OUT}/GDSC1_fitted.xlsx", "GDSC1")]:
    if not os.path.exists(f):
        continue
    try:
        d = pd.read_excel(f)
        d["SRC"] = tag
        frames.append(d)
        print(f"[2] {tag}: {d.shape[0]} 条记录, {d['DRUG_NAME'].nunique()} 个药, "
              f"{d['CELL_LINE_NAME'].nunique()} 个细胞系")
    except Exception as e:
        print(f"[2] {tag} 读取失败: {e}")

if not frames:
    sys.exit("没有可用的 GDSC 药敏文件")
g = pd.concat(frames, ignore_index=True)
g["CL"] = g["CELL_LINE_NAME"].map(lambda x: re.sub(r"[^A-Z0-9]", "", str(x).upper()))
g["TGM2"] = g["CL"].map(lambda x: ccle.get(x, {}).get("tgm2", np.nan))
matched = g.dropna(subset=["TGM2"])
print(f"[2] GDSC x CCLE 匹配上的细胞系: {matched['CL'].nunique()} / {g['CL'].nunique()}")

# GDSC 自带组织注释
tissue_col = "TCGA_DESC" if "TCGA_DESC" in g.columns else None
print(f"[2] 组织注释列: {tissue_col}")

# ---------- 3. 相关性 ----------
MIN_N = 25


def corr_table(df, label):
    rows = []
    for (drug, src), sub in df.groupby(["DRUG_NAME", "SRC"]):
        sub = sub.dropna(subset=["TGM2", "LN_IC50"])
        sub = sub.drop_duplicates(subset=["CL"])
        if len(sub) < MIN_N:
            continue
        r, p = stats.spearmanr(sub["TGM2"], sub["LN_IC50"])
        if np.isnan(r):
            continue
        rows.append({"drug": drug, "src": src, "n": len(sub), "rho": r, "p": p,
                     "target": (sub["PUTATIVE_TARGET"].iloc[0] if "PUTATIVE_TARGET" in sub else ""),
                     "pathway": (sub["PATHWAY_NAME"].iloc[0] if "PATHWAY_NAME" in sub else "")})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # BH 校正
    t = t.sort_values("p").reset_index(drop=True)
    m = len(t)
    t["fdr"] = (t["p"] * m / (t.index + 1)).cummin()[::-1].cummin()[::-1] if m else np.nan
    t["cohort"] = label
    return t


results = {}
res_all = corr_table(matched, "PAN-CANCER")
results["PAN-CANCER"] = res_all

TISSUE_MAP = {
    "KIRC": ["KIRC"], "ESCA": ["ESCA"], "PAAD": ["PAAD"],
}
if tissue_col:
    for lab, keys in TISSUE_MAP.items():
        sub = matched[matched[tissue_col].isin(keys)]
        if sub.empty:
            print(f"[3] {lab}: GDSC 中无该 TCGA_DESC")
            continue
        n_cl = sub["CL"].nunique()
        print(f"[3] {lab}: {n_cl} 个细胞系")
        globals()["MIN_N"] = 0
        t = corr_table(sub, lab) if n_cl >= 8 else pd.DataFrame()
        results[lab] = t

# 组织内样本少，降低阈值重算
MIN_N = 8
if tissue_col:
    for lab, keys in TISSUE_MAP.items():
        sub = matched[matched[tissue_col].isin(keys)]
        if sub.empty:
            continue
        results[lab] = corr_table(sub, lab)

# ---------- 4. 输出 ----------
lines = []
lines.append("=" * 92)
lines.append("TGM2 表达 vs 药物 IC50 相关性  |  CCLE RNA-seq (log2 RPKM+1) x GDSC1/2 fitted LN_IC50")
lines.append("rho > 0 = TGM2 越高 IC50 越高 = 越耐药（支持课题假设）")
lines.append("=" * 92)

for lab, t in results.items():
    if t is None or t.empty:
        lines.append(f"\n### {lab}: 无足够数据\n")
        continue
    sig = t[t["fdr"] < 0.05]
    pos = (t["rho"] > 0).sum()
    lines.append("")
    lines.append("#" * 92)
    lines.append(f"### {lab}  |  可评估药物 {len(t)} 个  |  FDR<0.05 显著 {len(sig)} 个  "
                 f"|  rho>0(耐药方向) {pos}/{len(t)} = {100*pos/len(t):.0f}%")
    lines.append("#" * 92)
    lines.append("")
    lines.append(f"{'药物':<28}{'n':>5}{'rho':>8}{'P':>11}{'FDR':>11}  靶点/通路")
    lines.append("-" * 92)
    top = t.sort_values("rho", ascending=False).head(30)
    for _, r in top.iterrows():
        lines.append(f"{str(r['drug'])[:27]:<28}{r['n']:>5}{r['rho']:>8.3f}{r['p']:>11.2e}"
                     f"{r['fdr']:>11.2e}  {str(r['target'])[:20]} / {str(r['pathway'])[:26]}")
    lines.append("")
    lines.append("  --- 反方向（TGM2 高 = 更敏感）Top10 ---")
    for _, r in t.sort_values("rho").head(10).iterrows():
        lines.append(f"{str(r['drug'])[:27]:<28}{r['n']:>5}{r['rho']:>8.3f}{r['p']:>11.2e}"
                     f"{r['fdr']:>11.2e}  {str(r['target'])[:20]} / {str(r['pathway'])[:26]}")
    t.to_csv(f"{OUT}/tgm2_drug_corr_{lab}.csv", index=False)

txt = "\n".join(lines)
open(f"{OUT}/tgm2_drug_corr_report.txt", "w").write(txt)
print(txt)
