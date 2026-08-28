"""轴 1 DEG 的 GO / KEGG / Reactome 富集（Enrichr API）

对每个癌种的上调 / 下调 DEG 分别做富集，并对三癌种交集基因做富集。
"""
import os, json, time, io
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import requests
import pandas as pd

OUT = "/store/zkyang/tgm2_gdsc/axis1"
BASE = "https://maayanlab.cloud/Enrichr"
LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023",
        "Reactome_2022", "MSigDB_Hallmark_2020"]
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


def enrich(genes, libs=LIBS, label="x"):
    genes = [g for g in genes if isinstance(g, str) and g and not g.startswith("LOC")]
    if len(genes) < 10:
        return {}
    r = requests.post(f"{BASE}/addList",
                      files={"list": (None, "\n".join(genes)),
                             "description": (None, label)}, timeout=90)
    uid = r.json()["userListId"]
    out = {}
    for lib in libs:
        for attempt in range(3):
            try:
                q = requests.get(f"{BASE}/enrich",
                                 params={"userListId": uid, "backgroundType": lib},
                                 timeout=90)
                out[lib] = q.json().get(lib, [])
                break
            except Exception as e:
                time.sleep(3)
                if attempt == 2:
                    out[lib] = []
        time.sleep(0.4)
    return out


def show(res, title, topn=10):
    log("")
    log("-" * 96)
    log(f"  {title}")
    log("-" * 96)
    for lib, terms in res.items():
        # Enrichr 返回: [rank, term, pval, zscore, combined, genes, adjpval, ...]
        sig = [t for t in terms if t[6] < 0.05]
        if not sig:
            log(f"  [{lib}] 无 FDR<0.05 条目")
            continue
        log(f"  [{lib}]  FDR<0.05 条目 {len(sig)} 个，Top{min(topn,len(sig))}：")
        for t in sig[:topn]:
            log(f"      {t[1][:62]:<64} FDR={t[6]:.2e}  n={len(t[5])}")
        log("")


ALL = {}
for cancer in ["KIRC", "ESCA", "PAAD"]:
    f = f"{OUT}/axis1_{cancer}_DEG.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    up = d[d.log2FC > 0].sort_values("log2FC", ascending=False).gene.tolist()
    dn = d[d.log2FC < 0].sort_values("log2FC").gene.tolist()
    log("")
    log("=" * 96)
    log(f"### TCGA-{cancer}  轴1 DEG 富集（上调 {len(up)} / 下调 {len(dn)}）")
    log("=" * 96)
    r1 = enrich(up[:1000], label=f"{cancer}_up")
    show(r1, f"{cancer} — TGM2 高组【上调】基因富集")
    r2 = enrich(dn[:1000], label=f"{cancer}_down")
    show(r2, f"{cancer} — TGM2 高组【下调】基因富集")
    ALL[cancer] = {"up": r1, "down": r2}

# 三癌种交集
tri = f"{OUT}/axis1_三癌种交集.csv"
if os.path.exists(tri):
    g = pd.read_csv(tri).gene.tolist()
    log("")
    log("=" * 96)
    log(f"### 三癌种共有 轴1 DEG 富集（{len(g)} 个基因）")
    log("=" * 96)
    r = enrich(g, label="tri_intersect")
    show(r, "KIRC ∩ ESCA ∩ PAAD 共有 DEG", topn=15)
    ALL["TRI"] = r

json.dump(ALL, open(f"{OUT}/axis1_enrichment.json", "w"), ensure_ascii=False)
open(f"{OUT}/axis1_enrichment_report.txt", "w").write("\n".join(lines))
print("\n>>> 写入", f"{OUT}/axis1_enrichment_report.txt")
