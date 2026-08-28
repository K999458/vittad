"""轴2 质控 + 交集富集

QC1：性别混杂 —— KIRC 交集上调里出现 RPS4Y1/KDM5D（Y染色体），下调里出现 XIST/TSIX，
      强烈提示耐药组与敏感组性别比例不平衡。直接查临床 gender 字段验证。
QC2：分期混杂 —— 耐药组是否更晚期（晚期本身就基质多、预后差）。
QC3：方向一致率的二项检验。
富集：核心交集（方向一致）基因的 KEGG/GO/Reactome/Hallmark。
"""
import os, json, gzip, collections
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
import pandas as pd
from scipy import stats
import xenaPython as xena
import requests, time

HUB = "https://tcga.xenahubs.net"
BASE = "/store/zkyang/tgm2_gdsc/tcga"
OUT = "/store/zkyang/tgm2_gdsc/axis2"
ENR = "https://maayanlab.cloud/Enrichr"
LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023",
        "Reactome_2022", "MSigDB_Hallmark_2020"]
RESIST = {"Progressive Disease", "Stable Disease"}
SENS = {"Complete Remission/Response", "Partial Remission/Response"}
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def decode(ds, samples, field):
    codes = xena.field_codes(HUB, ds, [field])[0].get("code")
    v = xena.dataset_fetch(HUB, ds, samples, [field])[0]
    if not codes:
        return v
    cl = codes.split("\t")
    out = []
    for x in v:
        try:
            i = int(float(x)); out.append(cl[i] if 0 <= i < len(cl) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def enrich(genes, label):
    genes = [g for g in genes if isinstance(g, str) and g and not g.startswith(("LOC", "C1orf", "CXorf"))]
    if len(genes) < 10:
        return {}
    r = requests.post(f"{ENR}/addList", files={"list": (None, "\n".join(genes)),
                                               "description": (None, label)}, timeout=90)
    uid = r.json()["userListId"]
    out = {}
    for lib in LIBS:
        for a in range(3):
            try:
                out[lib] = requests.get(f"{ENR}/enrich",
                                        params={"userListId": uid, "backgroundType": lib},
                                        timeout=90).json().get(lib, [])
                break
            except Exception:
                time.sleep(3)
                out.setdefault(lib, [])
        time.sleep(0.4)
    return out


log("=" * 96)
log("轴2 质控：混杂因素检查")
log("=" * 96)

QC = {}
for CA in ["KIRC", "PAAD", "ESCA"]:
    ds = f"TCGA.{CA}.sampleMap/{CA}_clinicalMatrix"
    csamp = xena.dataset_samples(HUB, ds, None)
    flds = set(xena.dataset_field(HUB, ds))
    resp = {f: decode(ds, csamp, f) for f in
            ["primary_therapy_outcome_success", "followup_treatment_success"] if f in flds}
    label = {}
    for i, s in enumerate(csamp):
        vals = [resp[f][i] for f in resp if resp[f][i]]
        if not vals:
            continue
        label[s] = "resistant" if any(v in RESIST for v in vals) else (
            "sensitive" if any(v in SENS for v in vals) else None)

    with gzip.open(f"{BASE}/{CA}_HiSeqV2.gz", "rt") as fh:
        cols = pd.read_csv(fh, sep="\t", index_col=0, nrows=1).columns
    tum = [c for c in cols if c.split("-")[-1].startswith("01")]
    R = [c for c in tum if label.get(c) == "resistant"]
    S = [c for c in tum if label.get(c) == "sensitive"]
    if len(R) < 10 or len(S) < 10:
        log(f"\n--- {CA}: 样本不足，跳过 ---")
        continue

    log("")
    log(f"--- {CA}  耐药 n={len(R)}  敏感 n={len(S)} ---")
    QC[CA] = {"n_resist": len(R), "n_sens": len(S)}

    for fld, name in [("gender", "性别"),
                      ("pathologic_stage", "病理分期"),
                      ("neoplasm_histologic_grade", "组织学分级"),
                      ("age_at_initial_pathologic_diagnosis", "年龄")]:
        if fld not in flds:
            continue
        v = dict(zip(csamp, decode(ds, csamp, fld)))
        vr = [v.get(c) for c in R if v.get(c) not in (None, "")]
        vs = [v.get(c) for c in S if v.get(c) not in (None, "")]
        if not vr or not vs:
            continue
        if fld == "age_at_initial_pathologic_diagnosis":
            try:
                ar = np.array([float(x) for x in vr]); asr = np.array([float(x) for x in vs])
                _, p = stats.mannwhitneyu(ar, asr)
                log(f"  {name:<8} 耐药 {ar.mean():.1f} 岁 vs 敏感 {asr.mean():.1f} 岁   P={p:.3f}"
                    + ("   ← 显著混杂" if p < 0.05 else ""))
                QC[CA][name] = {"resist": float(ar.mean()), "sens": float(asr.mean()), "p": float(p)}
            except Exception:
                pass
            continue
        cr = collections.Counter(vr); cs = collections.Counter(vs)
        keys = sorted(set(cr) | set(cs))
        tab = np.array([[cr.get(k, 0) for k in keys], [cs.get(k, 0) for k in keys]])
        tab = tab[:, tab.sum(axis=0) > 0]
        try:
            chi2, p, _, _ = stats.chi2_contingency(tab)
        except Exception:
            p = np.nan
        log(f"  {name}:  P={p:.4f}" + ("   ← 显著混杂" if p < 0.05 else ""))
        for k in keys:
            nr, ns = cr.get(k, 0), cs.get(k, 0)
            if nr + ns == 0:
                continue
            log(f"      {str(k)[:34]:<36} 耐药 {nr:>3} ({100*nr/max(1,len(vr)):>4.0f}%)"
                f"   敏感 {ns:>3} ({100*ns/max(1,len(vs)):>4.0f}%)")
        QC[CA][name] = {"p": float(p) if p == p else None}

# ---------- 方向一致率二项检验 ----------
log("")
log("=" * 96)
log("方向一致率二项检验（H0: 一致/矛盾 各 50%）")
log("=" * 96)
log("")
CONS = {"KIRC": (170, 183), "PAAD": (111, 111), "ESCA": (626, 651)}
for CA, (k, n) in CONS.items():
    p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    log(f"  {CA}: 一致 {k}/{n} = {100*k/n:.0f}%   二项检验 P = {p:.3e}")
log("")
log("  → 一致率远高于 50%，说明两条轴测到的是高度重叠的生物学过程，")
log("    你的双轴交集设计在数据上被验证是有效的。")
log("  ⚠ 但注意：若两条轴同时被同一个潜在因子驱动（如肿瘤纯度/基质含量），")
log("    方向一致率也会虚高。必须结合上面的混杂检查一起看。")

# ---------- 核心交集富集 ----------
log("")
log("=" * 96)
log("核心交集（方向一致）基因富集")
log("=" * 96)
ALL = {}
for CA in ["KIRC", "PAAD", "ESCA"]:
    f = f"{OUT}/交集_{CA}_核心_方向一致.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    up = d[d.log2FC_axis1 > 0].gene.tolist()
    dn = d[d.log2FC_axis1 < 0].gene.tolist()
    log("")
    log("#" * 96)
    log(f"### {CA}  核心交集 一致上调 {len(up)} / 一致下调 {len(dn)}")
    log("#" * 96)
    for tag, gs in [("一致上调", up), ("一致下调", dn)]:
        if len(gs) < 10:
            log(f"  [{tag}] 仅 {len(gs)} 个基因，跳过富集")
            continue
        res = enrich(gs, f"{CA}_{tag}")
        ALL[f"{CA}_{tag}"] = res
        log("")
        log(f"  ── {tag}（{len(gs)} 基因）──")
        for lib, terms in res.items():
            sig = [t for t in terms if t[6] < 0.05]
            if not sig:
                log(f"    [{lib}] 无 FDR<0.05")
                continue
            log(f"    [{lib}] FDR<0.05 共 {len(sig)}，Top8：")
            for t in sig[:8]:
                log(f"        {t[1][:60]:<62} FDR={t[6]:.2e} n={len(t[5])}")

json.dump({"qc": QC, "enrichment": ALL}, open(f"{OUT}/axis2_qc_enrich.json", "w"),
          ensure_ascii=False)
open(f"{OUT}/axis2_qc_enrich_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/axis2_qc_enrich_report.txt")
