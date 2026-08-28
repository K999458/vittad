"""轴 2（真实临床用药反应）+ 双轴方向一致性交集

关键升级：不用预测 IC50，直接用 TCGA 临床记录的治疗反应作为耐药轴。
  耐药 = Progressive Disease + Stable Disease
  敏感 = Complete Remission/Response + Partial Remission/Response
合并 primary_therapy_outcome_success 与 followup_treatment_success（任一记录到耐药即判耐药，
更保守的做法是只用 primary，两种都算并报告）。

交集按方向一致性拆四象限：
  同上调 = TGM2高组上调 ∩ 耐药组上调
  同下调 = TGM2高组下调 ∩ 耐药组下调
  方向矛盾的两格单独输出，用于讨论补偿机制。
"""
import os, json, gzip
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
import pandas as pd
from scipy import stats
import xenaPython as xena

HUB = "https://tcga.xenahubs.net"
BASE = "/store/zkyang/tgm2_gdsc/tcga"
A1 = "/store/zkyang/tgm2_gdsc/axis1"
OUT = "/store/zkyang/tgm2_gdsc/axis2"
os.makedirs(OUT, exist_ok=True)
LFC, FDRC = 0.585, 0.05
RESIST = {"Progressive Disease", "Stable Disease"}
SENS = {"Complete Remission/Response", "Partial Remission/Response"}
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r)); q[i] = prev
    return q


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


def deg(A, B, genes, min_frac=0.25):
    keep = (np.mean(A > 0, axis=1) > min_frac) | (np.mean(B > 0, axis=1) > min_frac)
    g, A, B = genes[keep], A[keep], B[keep]
    lfc = A.mean(axis=1) - B.mean(axis=1)
    _, p = stats.mannwhitneyu(A, B, axis=1, alternative="two-sided")
    return pd.DataFrame({"gene": g, "log2FC": lfc, "p": p, "fdr": bh(p),
                         "n_group1": A.shape[1], "n_group2": B.shape[1]})


summary = {}
for CA in ["KIRC", "PAAD", "ESCA"]:
    log("")
    log("=" * 96)
    log(f"### TCGA-{CA}  轴2：真实临床反应 耐药(PD+SD) vs 敏感(CR+PR)")
    log("=" * 96)

    ds = f"TCGA.{CA}.sampleMap/{CA}_clinicalMatrix"
    csamp = xena.dataset_samples(HUB, ds, None)
    flds = set(xena.dataset_field(HUB, ds))
    resp = {}
    for f in ["primary_therapy_outcome_success", "followup_treatment_success"]:
        if f in flds:
            resp[f] = decode(ds, csamp, f)

    # 合并：任一字段记录到 PD/SD -> 耐药；否则任一记录 CR/PR -> 敏感
    label = {}
    for i, s in enumerate(csamp):
        vals = [resp[f][i] for f in resp if resp[f][i]]
        if not vals:
            continue
        if any(v in RESIST for v in vals):
            label[s] = "resistant"
        elif any(v in SENS for v in vals):
            label[s] = "sensitive"
    log(f"临床反应可用样本：耐药 {sum(v=='resistant' for v in label.values())}，"
        f"敏感 {sum(v=='sensitive' for v in label.values())}"
        f"（合并 {list(resp)}）")

    with gzip.open(f"{BASE}/{CA}_HiSeqV2.gz", "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)
    tum = [c for c in expr.columns if c.split("-")[-1].startswith("01")]
    expr = expr[tum]

    res_ids = [c for c in expr.columns if label.get(c) == "resistant"]
    sen_ids = [c for c in expr.columns if label.get(c) == "sensitive"]
    log(f"与表达矩阵匹配后：耐药 n={len(res_ids)}，敏感 n={len(sen_ids)}")

    if len(res_ids) < 10 or len(sen_ids) < 10:
        log(f"! 样本量不足（需各 >=10），{CA} 轴2 跳过")
        summary[CA] = None
        continue

    # 先做合法性检查：耐药组 TGM2 是否更高
    tg_r = expr.loc["TGM2", res_ids].to_numpy(float)
    tg_s = expr.loc["TGM2", sen_ids].to_numpy(float)
    u, pv = stats.mannwhitneyu(tg_r, tg_s, alternative="two-sided")
    log("")
    log(f"【合法性检查】TGM2 表达 耐药组 {tg_r.mean():.3f} vs 敏感组 {tg_s.mean():.3f}"
        f"  差值 {tg_r.mean()-tg_s.mean():+.3f}  Mann-Whitney P = {pv:.4f}")
    log(f"   → {'支持' if (tg_r.mean()>tg_s.mean() and pv<0.05) else ('方向支持但不显著' if tg_r.mean()>tg_s.mean() else '方向不支持')}"
        f" 「TGM2 高 = 耐药」")

    d2 = deg(expr[res_ids].to_numpy(float), expr[sen_ids].to_numpy(float),
             expr.index.to_numpy())
    d2["direction"] = np.where(d2.log2FC > 0, "up_in_resistant", "down_in_resistant")
    sig2 = d2[(d2.fdr < FDRC) & (d2.log2FC.abs() >= LFC)]
    log("")
    log(f"轴2 DEG（FDR<{FDRC} 且 |log2FC|>={LFC}）：{len(sig2)} 个"
        f"（耐药组上调 {(sig2.log2FC>0).sum()}，下调 {(sig2.log2FC<0).sum()}）")
    # 样本量小时补充放宽阈值的版本
    loose = d2[(d2.p < 0.05) & (d2.log2FC.abs() >= LFC)]
    log(f"轴2 DEG（放宽：未校正 P<0.05 且 |log2FC|>={LFC}）：{len(loose)} 个"
        f"（上调 {(loose.log2FC>0).sum()}，下调 {(loose.log2FC<0).sum()}）")

    d2.to_csv(f"{OUT}/axis2_{CA}_all.csv", index=False)
    sig2.to_csv(f"{OUT}/axis2_{CA}_DEG_fdr.csv", index=False)
    loose.to_csv(f"{OUT}/axis2_{CA}_DEG_loose.csv", index=False)

    # ---------- 双轴交集（方向一致性四象限） ----------
    a1 = pd.read_csv(f"{A1}/axis1_{CA}_DEG.csv")
    for tag, b in [("FDR校正", sig2), ("放宽P", loose)]:
        m = a1.merge(b, on="gene", suffixes=("_axis1", "_axis2"))
        both_up = m[(m.log2FC_axis1 > 0) & (m.log2FC_axis2 > 0)]
        both_dn = m[(m.log2FC_axis1 < 0) & (m.log2FC_axis2 < 0)]
        up_dn = m[(m.log2FC_axis1 > 0) & (m.log2FC_axis2 < 0)]
        dn_up = m[(m.log2FC_axis1 < 0) & (m.log2FC_axis2 > 0)]
        log("")
        log(f"  ── 双轴交集（轴2 用「{tag}」阈值）──")
        log(f"     基因名重叠总数         : {len(m)}")
        log(f"     ✓ 一致上调（核心交集）  : {len(both_up)}")
        log(f"     ✓ 一致下调（核心交集）  : {len(both_dn)}")
        log(f"     ✗ 方向矛盾 高↑/耐药↓   : {len(up_dn)}")
        log(f"     ✗ 方向矛盾 高↓/耐药↑   : {len(dn_up)}")
        core = pd.concat([both_up, both_dn])
        if len(m):
            log(f"     → 方向一致率 {100*len(core)/len(m):.0f}%"
                f"（一致率显著高于 50% 才说明两条轴真的相关）")
        if tag == "放宽P":
            core.to_csv(f"{OUT}/交集_{CA}_核心_方向一致.csv", index=False)
            pd.concat([up_dn, dn_up]).to_csv(f"{OUT}/交集_{CA}_方向矛盾.csv", index=False)
            if len(both_up):
                log("")
                log(f"     一致上调 Top30（按轴1 log2FC）：")
                t = both_up.sort_values("log2FC_axis1", ascending=False).head(30)
                log("     " + ", ".join(t.gene.tolist()))
            if len(both_dn):
                log(f"     一致下调 Top30：")
                t = both_dn.sort_values("log2FC_axis1").head(30)
                log("     " + ", ".join(t.gene.tolist()))
        summary.setdefault(CA, {})[tag] = {
            "overlap": len(m), "both_up": len(both_up), "both_dn": len(both_dn),
            "conflict": len(up_dn) + len(dn_up),
            "tgm2_resist_mean": float(tg_r.mean()), "tgm2_sens_mean": float(tg_s.mean()),
            "tgm2_p": float(pv), "n_resist": len(res_ids), "n_sens": len(sen_ids),
        }

json.dump(summary, open(f"{OUT}/axis2_summary.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/axis2_intersect_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/axis2_intersect_report.txt")
