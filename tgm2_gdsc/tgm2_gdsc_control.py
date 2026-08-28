"""决定性对照：GDSC 全药谱上，TGM2 的广谱耐药关联是否只是 EMT/间质表型的替身？

三层对照：
  1. 基因对照 —— 同样的分析换成 EMT 标志(VIM/ZEB1/SNAI2/CDH2/FN1)、上皮标志(CDH1/EPCAM)、
     持家基因(GAPDH/ACTB, 阴性对照)、增殖(MKI67)，比较"耐药方向药物占比"和"中位 rho"。
     如果 TGM2 和 VIM 完全一样 -> TGM2 只是 EMT 替身；如果 TGM2 更强/更独立 -> 有自身贡献。
  2. 偏相关 —— 控制 VIM 后 TGM2 的偏 rho 是否保留。
  3. 阴性对照基线 —— 持家基因的"耐药方向占比"给出随机基线，判断 83% 是否真的偏离随机。
"""
import os, json, re, sys
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
import pandas as pd
from scipy import stats
import xenaPython as xena

OUT = "/store/zkyang/tgm2_gdsc"
HUB = "https://ucscpublic.xenahubs.net"
EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"

GENES = {
    "TGM2": "ENSG00000198959", "VIM": "ENSG00000026025", "ZEB1": "ENSG00000148516",
    "SNAI2": "ENSG00000019549", "CDH2": "ENSG00000170558", "FN1": "ENSG00000115414",
    "CDH1": "ENSG00000039068", "EPCAM": "ENSG00000119888",
    "GAPDH": "ENSG00000111640", "ACTB": "ENSG00000075624", "MKI67": "ENSG00000148773",
    "RPL13A": "ENSG00000142541", "TBP": "ENSG00000112592",
}
lines = []


def log(s=""):
    print(s); lines.append(s)


def ck(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).split("_", 1)[0].upper())


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r)); q[i] = prev
    return q


# ---------- 表达 ----------
cache = f"{OUT}/ccle_ctrl_genes.json"
if os.path.exists(cache):
    d = json.load(open(cache)); samples = d.pop("__samples__"); G = d
else:
    samples = xena.dataset_samples(HUB, EXPR, None)
    fields = list(xena.dataset_field(HUB, EXPR))
    base = {f.split(".")[0]: f for f in fields}
    ids, names = [], []
    for g, e in GENES.items():
        if e in base:
            ids.append(base[e]); names.append(g)
        else:
            log(f"  ! 未找到 {g} ({e})")
    mat = xena.dataset_fetch(HUB, EXPR, samples, ids)
    G = {n: [None if v is None else float(v) for v in row] for n, row in zip(names, mat)}
    json.dump({**G, "__samples__": samples}, open(cache, "w"))

E = {}
tissue = {}
for g in G:
    E[g] = {}
for j, s in enumerate(samples):
    k = ck(s)
    tissue[k] = s.split("_", 1)[1] if "_" in s else ""
    for g in G:
        v = G[g][j]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            E[g][k] = v
log(f"[0] 取到基因 {list(E.keys())}；CCLE 细胞系 {len(samples)}")

# ---------- GDSC ----------
gp = f"{OUT}/gdsc_merged.parquet"
if os.path.exists(gp):
    g = pd.read_parquet(gp)
else:
    fr = []
    for f, t in [(f"{OUT}/GDSC2_fitted.xlsx", "GDSC2"), (f"{OUT}/GDSC1_fitted.xlsx", "GDSC1")]:
        d = pd.read_excel(f)[["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50", "TCGA_DESC",
                              "PUTATIVE_TARGET", "PATHWAY_NAME"]]
        d["SRC"] = t
        fr.append(d)
    g = pd.concat(fr, ignore_index=True)
    g["CL"] = g["CELL_LINE_NAME"].map(ck)
    try:
        g.to_parquet(gp)
    except Exception:
        pass
if "CL" not in g.columns:
    g["CL"] = g["CELL_LINE_NAME"].map(ck)
log(f"[0] GDSC 记录 {len(g)}；药物 {g['DRUG_NAME'].nunique()}；细胞系 {g['CL'].nunique()}")

# 一个 (药, 来源) 一列，去重细胞系
g = g.dropna(subset=["LN_IC50"])
g["KEY"] = g["DRUG_NAME"].astype(str) + "@" + g["SRC"]
MIN_N = 50

log("")
log("=" * 104)
log("1. 基因对照：同一套 GDSC 药谱，换不同基因重算（rho>0 = 高表达更耐药）")
log("=" * 104)

groups = {k: sub.drop_duplicates("CL") for k, sub in g.groupby("KEY")}
groups = {k: v for k, v in groups.items() if len(v) >= MIN_N}
log(f"[1] 可评估 (药物 x 数据集) 组合：{len(groups)}")

summ = {}
per_gene_rho = {}
for gene in E:
    ev = E[gene]
    rhos, ps, names = [], [], []
    for k, sub in groups.items():
        v = sub["CL"].map(ev)
        m = v.notna()
        if m.sum() < MIN_N:
            continue
        r, p = stats.spearmanr(v[m], sub["LN_IC50"][m])
        if np.isnan(r):
            continue
        rhos.append(r); ps.append(p); names.append(k)
    if not rhos:
        continue
    q = bh(ps)
    rhos = np.array(rhos)
    summ[gene] = {
        "n_drug": len(rhos),
        "med_rho": float(np.median(rhos)),
        "frac_pos": float((rhos > 0).mean()),
        "n_sig": int((q < 0.05).sum()),
        "n_sig_pos": int(((q < 0.05) & (rhos > 0)).sum()),
        "n_sig_neg": int(((q < 0.05) & (rhos < 0)).sum()),
        "max_rho": float(rhos.max()),
    }
    per_gene_rho[gene] = dict(zip(names, rhos.tolist()))

log("")
log(f"{'基因':<9}{'类别':<14}{'可评估药':>9}{'中位rho':>10}{'耐药方向占比':>13}"
    f"{'FDR<.05':>9}{'其中耐药':>9}{'其中敏感':>9}{'最大rho':>9}")
log("-" * 104)
CAT = {"TGM2": "目标基因", "VIM": "EMT/间质", "ZEB1": "EMT/间质", "SNAI2": "EMT/间质",
       "CDH2": "EMT/间质", "FN1": "EMT/间质", "CDH1": "上皮", "EPCAM": "上皮",
       "GAPDH": "持家(阴性)", "ACTB": "持家(阴性)", "RPL13A": "持家(阴性)",
       "TBP": "持家(阴性)", "MKI67": "增殖"}
order = ["TGM2", "VIM", "ZEB1", "SNAI2", "CDH2", "FN1", "CDH1", "EPCAM",
         "MKI67", "GAPDH", "ACTB", "RPL13A", "TBP"]
for gene in order:
    if gene not in summ:
        continue
    s = summ[gene]
    log(f"{gene:<9}{CAT.get(gene,''):<14}{s['n_drug']:>9}{s['med_rho']:>10.3f}"
        f"{100*s['frac_pos']:>12.0f}%{s['n_sig']:>9}{s['n_sig_pos']:>9}"
        f"{s['n_sig_neg']:>9}{s['max_rho']:>9.3f}")

json.dump(summ, open(f"{OUT}/gene_control_summary.json", "w"), indent=1)

# ---------- 2. TGM2 与各对照基因 rho 谱的相似度 ----------
log("")
log("=" * 104)
log("2. 药谱相似度：TGM2 的 rho 谱 与 各基因 rho 谱 的相关（越接近1 = 越像同一个轴）")
log("=" * 104)
base = per_gene_rho.get("TGM2", {})
log("")
log(f"{'基因':<9}{'共同药物':>9}{'rho谱相关 r':>14}   解读")
log("-" * 70)
for gene in order:
    if gene == "TGM2" or gene not in per_gene_rho:
        continue
    common = set(base) & set(per_gene_rho[gene])
    if len(common) < 30:
        continue
    a = np.array([base[c] for c in common])
    b = np.array([per_gene_rho[gene][c] for c in common])
    r = np.corrcoef(a, b)[0, 1]
    tag = ("几乎同一个轴" if r > 0.8 else "高度重叠" if r > 0.6 else
           "部分重叠" if r > 0.3 else "基本独立" if r > -0.3 else "反向轴")
    log(f"{gene:<9}{len(common):>9}{r:>14.3f}   {tag}")

# ---------- 3. 偏相关：控制 VIM ----------
log("")
log("=" * 104)
log("3. 偏相关：控制 VIM 后，TGM2 与 LN_IC50 的独立关联（前 25 个原始 rho 最大的药）")
log("=" * 104)
tg, vm = E["TGM2"], E["VIM"]
rows = []
for k, sub in groups.items():
    a = sub["CL"].map(tg); b = sub["CL"].map(vm)
    m = a.notna() & b.notna()
    if m.sum() < MIN_N:
        continue
    x = stats.rankdata(a[m]); y = stats.rankdata(sub["LN_IC50"][m]); z = stats.rankdata(b[m])
    rxy = np.corrcoef(x, y)[0, 1]; rxz = np.corrcoef(x, z)[0, 1]; ryz = np.corrcoef(y, z)[0, 1]
    den = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if den <= 0:
        continue
    rp = (rxy - rxz * ryz) / den
    n = int(m.sum())
    t = rp * np.sqrt((n - 3) / max(1e-12, 1 - rp**2))
    rows.append({"drug": k, "n": n, "rho": rxy, "prho": rp,
                 "p": float(2 * stats.t.sf(abs(t), n - 3)),
                 "target": str(sub["PUTATIVE_TARGET"].iloc[0])[:22],
                 "pathway": str(sub["PATHWAY_NAME"].iloc[0])[:26]})
rows.sort(key=lambda r: -r["rho"])
log("")
log(f"{'药物':<26}{'n':>5}{'rho':>8}{'偏rho|VIM':>11}{'P偏':>11}{'保留':>7}  靶点/通路")
log("-" * 104)
for r in rows[:25]:
    keep = 100 * abs(r["prho"]) / max(1e-9, abs(r["rho"]))
    log(f"{r['drug'][:25]:<26}{r['n']:>5}{r['rho']:>8.3f}{r['prho']:>11.3f}"
        f"{r['p']:>11.2e}{keep:>6.0f}%  {r['target']} / {r['pathway']}")
if rows:
    kr = np.array([abs(r["rho"]) for r in rows]); kp = np.array([abs(r["prho"]) for r in rows])
    log("-" * 104)
    log(f"全 {len(rows)} 个药：中位 |rho| {np.median(kr):.3f} -> 控制 VIM 后 {np.median(kp):.3f}"
        f"（保留 {100*np.median(kp)/np.median(kr):.0f}%）；"
        f"偏相关 P<0.05 的 {sum(1 for r in rows if r['p']<0.05)}/{len(rows)}")
    pd.DataFrame(rows).to_csv(f"{OUT}/tgm2_partial_vim.csv", index=False)

open(f"{OUT}/tgm2_gdsc_control_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/tgm2_gdsc_control_report.txt")
