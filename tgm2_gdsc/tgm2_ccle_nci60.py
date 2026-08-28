"""TGM2 表达 vs 药敏：用 UCSC Xena 上的 CCLE 药敏(24药) 与 NCI60 DTP(大规模化合物)

检验课题核心假设：TGM2 高表达是否对应耐药（IC50/GI50 升高）。
"""
import os, json, re, sys
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
from scipy import stats
import xenaPython as xena

OUT = "/store/zkyang/tgm2_gdsc"
HUB = "https://ucscpublic.xenahubs.net"
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def key(name):
    s = str(name).upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    return re.sub(r"[^A-Z0-9]", "", s)


def bh(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        r = m - rank
        prev = min(prev, ps[i] * m / r)
        q[i] = prev
    return q


def cached(path, fn):
    if os.path.exists(path):
        return json.load(open(path))
    d = fn()
    json.dump(d, open(path, "w"))
    return d


# =========================================================
# A. CCLE: 表达(ENSG00000198959.7) x CCLE_NP24 药敏 (24 药)
# =========================================================
log("=" * 96)
log("A. CCLE  |  TGM2 (log2 RPKM+1) vs CCLE_NP24 药敏 ActArea/IC50")
log("=" * 96)

EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
DRUG = "ccle/CCLE_NP24.2009_Drug_data_2015.02.24"

expr = json.load(open(f"{OUT}/ccle_TGM2_rpkm.json"))
tgm2 = {}
tissue = {}
for k, v in expr.items():
    if v is None or (isinstance(v, float) and np.isnan(v)):
        continue
    p = k.split("_", 1)
    tgm2[key(p[0])] = float(v)
    tissue[key(p[0])] = p[1] if len(p) > 1 else ""
log(f"[A] TGM2 有值细胞系: {len(tgm2)}")

dmeta = cached(f"{OUT}/ccle_drug_meta.json",
               lambda: {"samples": xena.dataset_samples(HUB, DRUG, None),
                        "fields": xena.dataset_field(HUB, DRUG)})
dsamp, dfield = dmeta["samples"], dmeta["fields"]
log(f"[A] CCLE 药敏: {len(dsamp)} 细胞系, {len(dfield)} 个字段")

dvals = cached(f"{OUT}/ccle_drug_values.json",
               lambda: xena.dataset_fetch(HUB, DRUG, dsamp, dfield))

rows = []
for fi, fname in enumerate(dfield):
    if fname == "sampleID":
        continue
    v = np.array([np.nan if x is None else x for x in dvals[fi]], float)
    x, y = [], []
    for s, val in zip(dsamp, v):
        k = key(s.split("_", 1)[0])
        if k in tgm2 and not np.isnan(val):
            x.append(tgm2[k]); y.append(val)
    if len(x) < 25:
        continue
    r, p = stats.spearmanr(x, y)
    if np.isnan(r):
        continue
    rows.append({"field": fname, "n": len(x), "rho": float(r), "p": float(p)})

if rows:
    qs = bh([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["fdr"] = float(q)
    rows.sort(key=lambda r: -r["rho"])
    log("")
    log("  说明：CCLE 药敏字段含 ActArea(活性面积,越大=越敏感) 与 IC50/EC50。")
    log("        对 IC50 类字段 rho>0 = TGM2高更耐药；对 ActArea 类字段 rho<0 = TGM2高更耐药。")
    log("")
    log(f"  {'字段':<42}{'n':>5}{'rho':>8}{'P':>11}{'FDR':>11}")
    log("  " + "-" * 74)
    for r in rows:
        star = " *" if r["fdr"] < 0.05 else ""
        log(f"  {r['field'][:41]:<42}{r['n']:>5}{r['rho']:>8.3f}{r['p']:>11.2e}{r['fdr']:>11.2e}{star}")
    json.dump(rows, open(f"{OUT}/ccle_tgm2_drug_corr.json", "w"), indent=1)
else:
    log("[A] 无足够配对数据")

# =========================================================
# B. NCI60: 表达 x DTP GI50 Z-score（大规模化合物）
# =========================================================
log("")
log("=" * 96)
log("B. NCI60  |  TGM2 (Affy U133Plus2 RMA) vs DTP GI50 Z-score")
log("   注意 DTP Z-score 越大 = 越敏感，故 rho<0 = TGM2高更耐药")
log("=" * 96)

NEXPR = "NCI60/RNA_Affy_HG_U133_Plus_2.0_RMA.txt"
NDRUG = "NCI60/DTP_NCI60_ZSCORE.txt"

# TGM2 在 Affy HG-U133 Plus 2.0 上的探针组
TGM2_PROBES = ["201042_at", "211003_x_at", "211573_x_at"]

try:
    ns = xena.dataset_samples(HUB, NEXPR, None)
    nf = set(xena.dataset_field(HUB, NEXPR))
    cand = [p for p in TGM2_PROBES if p in nf]
    log(f"[B] NCI60 表达: {len(ns)} 细胞系; TGM2 探针命中 {cand}")
    if cand:
        mat = xena.dataset_fetch(HUB, NEXPR, ns, cand)
        arr = np.array([[np.nan if v is None else float(v) for v in row] for row in mat])
        # 多探针取均值
        ntg = {}
        for j, s in enumerate(ns):
            col = arr[:, j]
            col = col[~np.isnan(col)]
            ntg[key(s)] = float(col.mean()) if len(col) else np.nan
        log(f"[B] TGM2 表达可用细胞系 {sum(1 for v in ntg.values() if not np.isnan(v))}")

        ds2 = xena.dataset_samples(HUB, NDRUG, None)
        df2 = xena.dataset_field(HUB, NDRUG)
        log(f"[B] NCI60 DTP: {len(ds2)} 细胞系, {len(df2)} 个化合物")

        # 只测我们关心的临床药 + 全量扫描（分批）
        WANT = ["cisplatin", "carboplatin", "oxaliplatin", "gemcitabine", "fluorouracil", "5-fu",
                "paclitaxel", "docetaxel", "doxorubicin", "etoposide", "irinotecan", "topotecan",
                "sorafenib", "sunitinib", "everolimus", "temsirolimus", "pazopanib", "axitinib",
                "vinblastine", "vincristine", "cyclophosphamide", "mitomycin", "bleomycin",
                "methotrexate", "cytarabine", "dasatinib", "imatinib", "erlotinib", "bortezomib"]
        sel = [f for f in df2 if f != "sampleID"]
        want_hits = [f for f in sel if any(w in f.lower() for w in WANT)]
        log(f"[B] 全量扫描 {len(sel)} 个化合物（其中临床常用药 {len(want_hits)} 个）")

        res = []
        BATCH = 60
        for i in range(0, len(sel), BATCH):
            chunk = sel[i:i + BATCH]
            vv = xena.dataset_fetch(HUB, NDRUG, ds2, chunk)
            for fname, col in zip(chunk, vv):
                x, y = [], []
                for s, val in zip(ds2, col):
                    k = key(s)
                    if val is None:
                        continue
                    t = ntg.get(k, np.nan)
                    if np.isnan(t):
                        continue
                    x.append(t); y.append(float(val))
                if len(x) < 20:
                    continue
                r, p = stats.spearmanr(x, y)
                if np.isnan(r):
                    continue
                res.append({"drug": fname, "n": len(x), "rho": float(r), "p": float(p)})
        if res:
            qs = bh([r["p"] for r in res])
            for r, q in zip(res, qs):
                r["fdr"] = float(q)
            res.sort(key=lambda r: r["rho"])
            neg = sum(1 for r in res if r["rho"] < 0)
            nsig = sum(1 for r in res if r["fdr"] < 0.05)
            nsig_res = sum(1 for r in res if r["fdr"] < 0.05 and r["rho"] < 0)
            log("")
            log(f"  [B] 汇总：{len(res)} 个化合物可评估；方向为「TGM2高→耐药」的 {neg} 个"
                f"（{100*neg/len(res):.0f}%）；FDR<0.05 显著 {nsig} 个，"
                f"其中耐药方向 {nsig_res} 个")
            log("")
            log("  --- Top25「TGM2 高 → 耐药」（rho 最负）---")
            log(f"  {'化合物':<46}{'n':>5}{'rho':>8}{'P':>11}{'FDR':>11}")
            log("  " + "-" * 82)
            for r in res[:25]:
                star = " *" if r["fdr"] < 0.05 else ""
                log(f"  {r['drug'][:45]:<46}{r['n']:>5}{r['rho']:>8.3f}{r['p']:>11.2e}"
                    f"{r['fdr']:>11.2e}{star}")
            log("")
            log("  --- 临床常用药单独列出 ---")
            log(f"  {'化合物':<46}{'n':>5}{'rho':>8}{'P':>11}{'FDR':>11}  方向")
            log("  " + "-" * 92)
            for r in res:
                if not any(w in r["drug"].lower() for w in WANT):
                    continue
                d = "TGM2高→耐药" if r["rho"] < 0 else "TGM2高→敏感"
                star = " *" if r["fdr"] < 0.05 else ""
                log(f"  {r['drug'][:45]:<46}{r['n']:>5}{r['rho']:>8.3f}{r['p']:>11.2e}"
                    f"{r['fdr']:>11.2e}  {d}{star}")
            json.dump(res, open(f"{OUT}/nci60_tgm2_drug_corr.json", "w"), indent=1)
        else:
            log("[B] 无足够配对")
except Exception as e:
    import traceback
    log(f"[B] 失败: {e}")
    traceback.print_exc()

open(f"{OUT}/tgm2_drugsens_report.txt", "w").write("\n".join(lines))
print("\n>>> 已写入", f"{OUT}/tgm2_drugsens_report.txt")
