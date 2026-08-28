"""关键对照分析：TGM2 与广谱耐药的关联，是否只是 EMT / 间质表型的替身？

设计：
  1. 组织内分层：CCLE 中 KIDNEY / OESOPHAGUS / PANCREAS 细胞系单独算相关
  2. 基因对照：把 TGM2 换成 EMT 标志(VIM/ZEB1/SNAI2/CDH2)、上皮标志(CDH1/EPCAM)、
     持家基因(GAPDH/ACTB) 重算，看 TGM2 的效应量是否只是 EMT 轴的复制
  3. 偏相关：控制 VIM 后 TGM2 的偏相关是否还存在（TGM2 是否有 EMT 之外的独立贡献）
"""
import os, json, re
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
from scipy import stats
import xenaPython as xena

OUT = "/store/zkyang/tgm2_gdsc"
HUB = "https://ucscpublic.xenahubs.net"
EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
DRUG = "ccle/CCLE_NP24.2009_Drug_data_2015.02.24"

GENES = {
    "TGM2":  "ENSG00000198959.7",
    "VIM":   "ENSG00000026025.11",   # EMT / 间质
    "ZEB1":  "ENSG00000148516.16",
    "SNAI2": "ENSG00000019549.4",
    "CDH2":  "ENSG00000170558.6",
    "CDH1":  "ENSG00000039068.14",   # 上皮
    "EPCAM": "ENSG00000119888.9",
    "GAPDH": "ENSG00000111640.10",   # 持家（阴性对照）
    "ACTB":  "ENSG00000075624.9",
    "MKI67": "ENSG00000148773.9",    # 增殖（增殖速率混杂）
}
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def key(name):
    return re.sub(r"[^A-Z0-9]", "", str(name).split("_", 1)[0].upper())


def bh(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    q = np.empty(m); prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, ps[i] * m / (m - rank)); q[i] = prev
    return q


# ---- 取表达 ----
cache = f"{OUT}/ccle_multigene.json"
if os.path.exists(cache):
    G = json.load(open(cache))
    samples = G.pop("__samples__")
else:
    samples = xena.dataset_samples(HUB, EXPR, None)
    fields = set(xena.dataset_field(HUB, EXPR))
    ids, names = [], []
    for g, eid in GENES.items():
        if eid in fields:
            ids.append(eid); names.append(g)
        else:
            hit = [f for f in fields if f.split(".")[0] == eid.split(".")[0]]
            if hit:
                ids.append(hit[0]); names.append(g)
    mat = xena.dataset_fetch(HUB, EXPR, samples, ids)
    G = {n: [None if v is None else float(v) for v in row] for n, row in zip(names, mat)}
    json.dump({**G, "__samples__": samples}, open(cache, "w"))
log(f"[0] CCLE 样本 {len(samples)}；成功取到基因 {list(G.keys())}")

tissue = {}
E = {g: {} for g in G}
for j, s in enumerate(samples):
    k = key(s)
    tissue[k] = s.split("_", 1)[1] if "_" in s else ""
    for g in G:
        v = G[g][j]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            E[g][k] = v

# ---- 取药敏 ----
dm = json.load(open(f"{OUT}/ccle_drug_meta.json"))
dv = json.load(open(f"{OUT}/ccle_drug_values.json"))
dsamp, dfield = dm["samples"], dm["fields"]

# 只用 IC50 类字段（rho>0 = 耐药），语义清晰
IC = [(i, f) for i, f in enumerate(dfield) if "IC50" in f and f != "sampleID"]
log(f"[0] 使用 IC50 类药敏字段 {len(IC)} 个：{[f.split(':')[0] for _, f in IC]}")


def corr(gene, idx, keys=None):
    x, y = [], []
    for s, val in zip(dsamp, dv[idx]):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        k = key(s)
        if keys is not None and k not in keys:
            continue
        if k in E[gene]:
            x.append(E[gene][k]); y.append(float(val))
    if len(x) < 12:
        return None
    r, p = stats.spearmanr(x, y)
    return None if np.isnan(r) else (r, p, len(x))


# =========== 1. 泛癌：各基因 vs 各 IC50 ===========
log("")
log("=" * 100)
log("1. 泛癌对照：不同基因与 IC50 的 Spearman rho（rho>0 = 高表达更耐药）")
log("=" * 100)
hdr = f"{'药物(IC50)':<26}" + "".join(f"{g:>9}" for g in GENES if g in E)
log(hdr)
log("-" * len(hdr))
summary = {g: [] for g in GENES if g in E}
for idx, fname in IC:
    row = f"{fname.split(':')[0][:25]:<26}"
    for g in GENES:
        if g not in E:
            continue
        c = corr(g, idx)
        if c is None:
            row += f"{'-':>9}"
        else:
            row += f"{c[0]:>9.3f}"
            summary[g].append(c[0])
    log(row)
log("-" * len(hdr))
row = f"{'中位 rho':<26}"
for g in GENES:
    if g in E:
        row += f"{np.median(summary[g]):>9.3f}" if summary[g] else f"{'-':>9}"
log(row)
row = f"{'rho>0 个数':<26}"
for g in GENES:
    if g in E:
        row += f"{sum(1 for v in summary[g] if v>0)}/{len(summary[g]):<7}" if summary[g] else f"{'-':>9}"
log(row)

# =========== 2. 偏相关：控制 VIM ===========
log("")
log("=" * 100)
log("2. 偏相关：控制 VIM（间质表型）后，TGM2 与 IC50 的独立关联")
log("=" * 100)


def partial(g1, g2, idx):
    """Spearman 偏相关：g1 与药敏，控制 g2"""
    x, y, z = [], [], []
    for s, val in zip(dsamp, dv[idx]):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        k = key(s)
        if k in E[g1] and k in E[g2]:
            x.append(E[g1][k]); y.append(float(val)); z.append(E[g2][k])
    if len(x) < 20:
        return None
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if den == 0:
        return None
    rp = (r_xy - r_xz * r_yz) / den
    n = len(x)
    t = rp * np.sqrt((n - 3) / max(1e-12, 1 - rp**2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return r_xy, rp, p, n


log(f"{'药物(IC50)':<26}{'rho(TGM2)':>11}{'偏rho|VIM':>12}{'P偏':>11}{'n':>6}  衰减")
log("-" * 78)
keep = []
for idx, fname in IC:
    r = partial("TGM2", "VIM", idx)
    if r is None:
        continue
    r_xy, rp, p, n = r
    dec = (1 - abs(rp) / abs(r_xy)) * 100 if abs(r_xy) > 1e-9 else 0
    star = " *" if p < 0.05 else ""
    log(f"{fname.split(':')[0][:25]:<26}{r_xy:>11.3f}{rp:>12.3f}{p:>11.2e}{n:>6}  {dec:>5.0f}%{star}")
    keep.append((abs(r_xy), abs(rp), p))
if keep:
    med_raw = np.median([k[0] for k in keep])
    med_par = np.median([k[1] for k in keep])
    nsig = sum(1 for k in keep if k[2] < 0.05)
    log("-" * 78)
    log(f"中位 |rho| 原始 {med_raw:.3f} -> 控制VIM后 {med_par:.3f} "
        f"(衰减 {100*(1-med_par/med_raw):.0f}%)；偏相关 P<0.05 的药物 {nsig}/{len(keep)}")

# =========== 3. 组织内分层 ===========
log("")
log("=" * 100)
log("3. 组织内分层（对应 KIRC / ESCA / PAAD）")
log("=" * 100)
TIS = {"KIDNEY (~KIRC)": "KIDNEY", "OESOPHAGUS (~ESCA)": "OESOPHAGUS", "PANCREAS (~PAAD)": "PANCREAS"}
for lab, tkey in TIS.items():
    keys = {k for k, t in tissue.items() if t == tkey}
    with_drug = {key(s) for s, v in zip(dsamp, dv[IC[0][0]])
                 if v is not None and not (isinstance(v, float) and np.isnan(v))} & keys
    log("")
    log(f"--- {lab}: CCLE 该组织细胞系 {len(keys)} 个，有药敏数据 {len(with_drug)} 个 ---")
    if len(with_drug) < 12:
        log("    样本不足，跳过（这也是课题必须注意的：细胞系层面组织内样本量太小）")
        continue
    rows = []
    for idx, fname in IC:
        c = corr("TGM2", idx, keys)
        if c:
            rows.append((fname.split(":")[0], c[0], c[1], c[2]))
    if not rows:
        continue
    qs = bh([r[2] for r in rows])
    rows = [(a, b, c, d, q) for (a, b, c, d), q in zip(rows, qs)]
    rows.sort(key=lambda r: -r[1])
    log(f"    {'药物':<24}{'rho':>8}{'P':>11}{'FDR':>11}{'n':>6}")
    for a, b, c, d, q in rows:
        log(f"    {a[:23]:<24}{b:>8.3f}{c:>11.2e}{q:>11.2e}{d:>6}")
    pos = sum(1 for r in rows if r[1] > 0)
    log(f"    → rho>0(耐药方向) {pos}/{len(rows)}，中位 rho {np.median([r[1] for r in rows]):.3f}")

open(f"{OUT}/tgm2_control_report.txt", "w").write("\n".join(lines))
print("\n>>> 已写入", f"{OUT}/tgm2_control_report.txt")
