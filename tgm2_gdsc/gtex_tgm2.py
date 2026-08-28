"""用 TCGA + GTEx 统一处理的 Toil RSEM 数据补齐正常样本不足的癌种

TCGA 癌旁正常样本在 PAAD(4)、ESCA(11)、CESC(3)、PCPG(3)、OV(0)、SARC(2) 等极少，
Xena toil hub 的 TcgaTargetGtex_rsem_gene_tpm 把 TCGA/TARGET/GTEx 用同一流程重跑，
可以直接拿 GTEx 正常组织做对照（GEPIA2 用的就是这套逻辑）。
"""
import os, json, collections
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np
from scipy import stats
import xenaPython as xena
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUB = "https://toil.xenahubs.net"
OUT = "/store/zkyang/tgm2_gdsc/pancancer"
os.makedirs(OUT, exist_ok=True)
TGM2 = "ENSG00000198959"
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r)); q[i] = prev
    return q


import urllib.request
def xq(q):
    r = urllib.request.Request(HUB + "/data/", data=q.encode(),
                               headers={"Content-Type": "text/plain"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read().decode())


ds = xq('(map :name (query {:select [:name] :from [:dataset]}))')
expr_ds = [d for d in ds if "rsem_gene_tpm" in d or "RSEM_gene_tpm" in d]
pheno_ds = [d for d in ds if "pheno" in d.lower() or "phenotype" in d.lower()]
log(f"[0] toil hub 数据集 {len(ds)}")
log(f"    表达: {expr_ds}")
log(f"    表型: {pheno_ds}")

EX = "TcgaTargetGtex_rsem_gene_tpm"
PH = "TcgaTargetGTEX_phenotype.txt"

cache = f"{OUT}/toil_tgm2.json"
if os.path.exists(cache):
    D = json.load(open(cache))
else:
    samples = xena.dataset_samples(HUB, EX, None)
    fields = list(xena.dataset_field(HUB, EX))
    hit = [f for f in fields if f.split(".")[0] == TGM2]
    log(f"[1] 表达样本 {len(samples)}；TGM2 字段 {hit[:2]}")
    vals = xena.dataset_fetch(HUB, EX, samples, hit[:1])[0]
    # 表型
    psamp = xena.dataset_samples(HUB, PH, None)
    pf = list(xena.dataset_field(HUB, PH))
    log(f"[1] 表型字段: {pf}")
    want = ["_primary_site", "_sample_type", "_study", "detailed_category"]
    want = [f for f in want if f in pf]
    log(f"[1] 使用表型字段: {want}")
    P = {}
    for f in want:
        codes = xena.field_codes(HUB, PH, [f])[0].get("code")
        v = xena.dataset_fetch(HUB, PH, psamp, [f])[0]
        if codes:
            cl = codes.split("\t")
            o = []
            for x in v:
                try:
                    i = int(float(x)); o.append(cl[i] if 0 <= i < len(cl) else None)
                except (TypeError, ValueError):
                    o.append(None)
            P[f] = o
        else:
            P[f] = v
    D = {"samples": samples, "tgm2": vals, "psamples": psamp, "pheno": P}
    json.dump(D, open(cache, "w"))

samples, vals = D["samples"], D["tgm2"]
psamp, P = D["psamples"], D["pheno"]
log(f"[1] TGM2 取到 {sum(1 for v in vals if v is not None)} / {len(samples)} 个样本")

pidx = {s: i for i, s in enumerate(psamp)}
def ph(s, f):
    i = pidx.get(s)
    return P[f][i] if (i is not None and f in P) else None

FSITE = "_primary_site" if "_primary_site" in P else None
FTYPE = "_sample_type" if "_sample_type" in P else None
FSTUDY = "_study" if "_study" in P else None
FDIS = "detailed_category" if "detailed_category" in P else None
log(f"[2] 字段映射: site={FSITE} type={FTYPE} study={FSTUDY} disease={FDIS}")

E = {s: float(v) for s, v in zip(samples, vals) if v is not None}

# TCGA 项目 -> GTEx 组织 映射（按 primary_site）
MAP = {
    "PAAD": ("Pancreas", "Pancreas"),
    "ESCA": ("Esophagus", "Esophagus"),
    "KIRC": ("Kidney", "Kidney"),
    "KIRP": ("Kidney", "Kidney"),
    "STAD": ("Stomach", "Stomach"),
    "LIHC": ("Liver", "Liver"),
    "LUAD": ("Lung", "Lung"),
    "LUSC": ("Lung", "Lung"),
    "COAD": ("Colon", "Colon"),
    "READ": ("Colon", "Colon"),
    "THCA": ("Thyroid", "Thyroid"),
    "PRAD": ("Prostate", "Prostate"),
    "BRCA": ("Breast", "Breast"),
    "BLCA": ("Bladder", "Bladder"),
    "UCEC": ("Uterus", "Uterus"),
    "OV":   ("Ovary", "Ovary"),
    "CESC": ("Cervix", "Cervix"),
    "SKCM": ("Skin", "Skin"),
    "HNSC": ("Head and Neck region", "Esophagus"),
}

# 收集 TCGA 肿瘤 与 GTEx 正常
tcga_tumor = collections.defaultdict(list)
gtex_normal = collections.defaultdict(list)
tcga_normal = collections.defaultdict(list)

for s, v in E.items():
    study = ph(s, FSTUDY) or ""
    site = ph(s, FSITE) or ""
    styp = (ph(s, FTYPE) or "")
    dis = (ph(s, FDIS) or "")
    if "TCGA" in str(study):
        key = str(dis)
        if "Normal" in str(styp) or "normal" in str(styp):
            tcga_normal[key].append(v)
        else:
            tcga_tumor[key].append(v)
    elif "GTEX" in str(study).upper():
        gtex_normal[str(site)].append(v)

log("")
log(f"[3] TCGA 疾病类别 {len(tcga_tumor)} 种；GTEx 组织 {len(gtex_normal)} 种")
log(f"    GTEx 组织样例: {sorted(gtex_normal)[:14]}")
log(f"    TCGA 类别样例: {sorted(tcga_tumor)[:8]}")

DIS2SITE = {
    "pancreatic adenocarcinoma": "Pancreas",
    "esophageal carcinoma": "Esophagus",
    "kidney clear cell carcinoma": "Kidney",
    "kidney papillary cell carcinoma": "Kidney",
    "stomach adenocarcinoma": "Stomach",
    "liver hepatocellular carcinoma": "Liver",
    "lung adenocarcinoma": "Lung",
    "lung squamous cell carcinoma": "Lung",
    "colon adenocarcinoma": "Colon",
    "rectum adenocarcinoma": "Colon",
    "thyroid carcinoma": "Thyroid",
    "prostate adenocarcinoma": "Prostate",
    "breast invasive carcinoma": "Breast",
    "bladder urothelial carcinoma": "Bladder",
    "uterine corpus endometrioid carcinoma": "Uterus",
    "ovarian serous cystadenocarcinoma": "Ovary",
    "cervical & endocervical cancer": "Cervix",
    "head & neck squamous cell carcinoma": "Esophagus",
    "glioblastoma multiforme": "Brain",
    "brain lower grade glioma": "Brain",
    "acute myeloid leukemia": "Blood",
    "skin cutaneous melanoma": "Skin",
    "testicular germ cell tumor": "Testis",
    "adrenocortical cancer": "Adrenal Gland",
    "sarcoma": "Muscle",
}

rows = []
for dis, tv in sorted(tcga_tumor.items()):
    if len(tv) < 20:
        continue
    site = DIS2SITE.get(dis.lower())
    nv = gtex_normal.get(site, []) if site else []
    tn = tcga_normal.get(dis, [])
    combined = list(nv) + list(tn)
    if len(combined) < 5:
        continue
    a = np.array(tv); b = np.array(combined)
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"disease": dis, "site": site or "-",
                 "n_tumor": len(a), "n_gtex": len(nv), "n_tcga_normal": len(tn),
                 "tumor_median": float(np.median(a)), "normal_median": float(np.median(b)),
                 "log2FC": float(np.median(a) - np.median(b)), "p": float(p)})

if rows:
    qs = bh([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["fdr"] = float(q)
    rows.sort(key=lambda r: -r["log2FC"])
    log("")
    log("=" * 118)
    log("TGM2 表达：TCGA 肿瘤 vs (GTEx 正常 + TCGA 癌旁)  —— Toil 统一流程 log2(TPM+0.001)")
    log("=" * 118)
    log("")
    log(f"{'TCGA 疾病':<42}{'正常组织':<14}{'肿瘤n':>6}{'GTEx':>6}{'癌旁':>6}"
        f"{'肿瘤中位':>9}{'正常中位':>9}{'log2FC':>8}{'FDR':>11}  判定")
    log("-" * 118)
    for r in rows:
        if r["fdr"] < 0.05 and r["log2FC"] >= 1:
            v = "★ 显著高表达"
        elif r["fdr"] < 0.05 and r["log2FC"] > 0:
            v = "✓ 高表达"
        elif r["fdr"] < 0.05 and r["log2FC"] <= -1:
            v = "▼ 显著低表达"
        elif r["fdr"] < 0.05:
            v = "▽ 低表达"
        else:
            v = "— 无差异"
        log(f"{r['disease'][:41]:<42}{str(r['site'])[:13]:<14}{r['n_tumor']:>6}"
            f"{r['n_gtex']:>6}{r['n_tcga_normal']:>6}{r['tumor_median']:>9.2f}"
            f"{r['normal_median']:>9.2f}{r['log2FC']:>8.2f}{r['fdr']:>11.2e}  {v}")

    json.dump(rows, open(f"{OUT}/TGM2_TCGA_GTEx.json", "w"), ensure_ascii=False, indent=1)

    # 图
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    rs = sorted(rows, key=lambda r: r["log2FC"])
    cols = ["#e04b4b" if (r["fdr"] < 0.05 and r["log2FC"] > 0)
            else "#3b76c4" if (r["fdr"] < 0.05 and r["log2FC"] < 0) else "#bbb" for r in rs]
    ax.barh(range(len(rs)), [r["log2FC"] for r in rs], color=cols,
            edgecolor="k", linewidth=0.4)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([r["disease"][:38] for r in rs], fontsize=8)
    ax.axvline(0, c="k", lw=0.8); ax.axvline(1, ls="--", c="#888", lw=0.7)
    ax.axvline(-1, ls="--", c="#888", lw=0.7)
    ax.set_xlabel("TGM2 log2FC  (TCGA tumor vs GTEx normal + TCGA adjacent)", fontsize=11)
    ax.set_title("TGM2 pan-cancer differential expression (TCGA vs GTEx, Toil pipeline)",
                 fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/泛癌汇总_TGM2_TCGA_vs_GTEx.png", dpi=165)
    plt.close(fig)

open(f"{OUT}/泛癌_TGM2_GTEx补充报告.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/泛癌_TGM2_GTEx补充报告.txt")
