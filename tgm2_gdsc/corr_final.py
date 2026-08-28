"""汇总出最终高置信共表达基因集，并做功能富集

高置信 = 同时满足：
  (1) 轴1 ∩ 轴2 交集（TGM2 高低差异 ∩ 耐药非耐药差异，方向一致）
  (2) TCGA 肿瘤内与 TGM2 相关，且纯度校正 + 成纤维评分校正后仍 rho>=0.3、FDR<0.05
  (3) CCLE 同谱系细胞系中同向复现（|rho|>=0.3、P<0.05）—— 独立队列、无间质
"""
import os, sys, json, time
import numpy as np
import pandas as pd

os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import requests

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt

OUT = "/store/zkyang/tgm2_gdsc/corr"
EN = "https://maayanlab.cloud/Enrichr"
LIBS = ["MSigDB_Hallmark_2020", "KEGG_2021_Human",
        "GO_Biological_Process_2023", "Reactome_2022"]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


ECACHE = f"{OUT}/enrich_cache.json"
_cache = json.load(open(ECACHE)) if os.path.exists(ECACHE) else {}


def enrich(genes, label):
    genes = [g for g in genes if isinstance(g, str) and g
             and not g.startswith("LOC")]
    if len(genes) < 8:
        return {}
    key = label + "|" + ",".join(sorted(genes))
    if key in _cache:
        return _cache[key]
    uid = None
    for a in range(6):
        try:
            r = requests.post(f"{EN}/addList",
                              files={"list": (None, "\n".join(genes)),
                                     "description": (None, label)}, timeout=90)
            uid = r.json()["userListId"]
            break
        except Exception as e:
            log(f"  Enrichr addList 第 {a+1} 次失败：{type(e).__name__}")
            time.sleep(5 * (a + 1))
    if uid is None:
        log("  ! Enrichr 不可用，跳过富集")
        return {}
    out = {}
    for lib in LIBS:
        for a in range(5):
            try:
                q = requests.get(f"{EN}/enrich",
                                 params={"userListId": uid,
                                         "backgroundType": lib}, timeout=90)
                out[lib] = q.json().get(lib, [])
                break
            except Exception:
                time.sleep(4 * (a + 1))
                if a == 4:
                    out[lib] = []
        time.sleep(0.5)
    _cache[key] = out
    json.dump(_cache, open(ECACHE, "w"))
    return out


final = {}
for CA, cn in [("PAAD", "胰腺癌"), ("KIRC", "肾透明细胞癌")]:
    R = pd.read_csv(f"{OUT}/相关性_{CA}_交集基因.csv")
    C = pd.read_csv(f"{OUT}/CCLE验证_{CA}.csv")
    core = set(pd.read_csv(f"{OUT}/核心共表达_{CA}_双校正后.csv").gene)
    rep = set(C[(np.sign(C.rho_ccle) == np.sign(C.rho_tcga)) &
                (C.rho_ccle.abs() >= 0.3) & (C.p_ccle < 0.05)].gene)
    hi = sorted(core & rep)

    T = R[R.gene.isin(hi)].merge(
        C[["gene", "rho_ccle", "p_ccle", "n_ccle"]], on="gene", how="left")
    T = T.sort_values("rho_adj_purity", ascending=False)
    T.to_csv(f"{OUT}/高置信共表达_{CA}.csv", index=False)

    log("")
    log("=" * 100)
    log(f"### {CA} {cn}   三重过滤后的高置信 TGM2 共表达基因：{len(hi)} 个")
    log("=" * 100)
    log(f"  交集基因 {len(R)} → 双校正存活 {len(core)} → 再要求 CCLE 复现 "
        f"{len(hi)}")
    log("")
    log(f"  {'基因':<12}{'TCGA rho':>9}{'纯度校正':>9}{'成纤维校正':>11}"
        f"{'CCLE rho':>9}{'轴1 log2FC':>11}{'轴2 log2FC':>11}")
    for _, r in T.iterrows():
        log(f"  {r.gene[:11]:<12}{r.rho:>+9.3f}{r.rho_adj_purity:>+9.3f}"
            f"{r.rho_adj_fibro:>+11.3f}{r.rho_ccle:>+9.3f}"
            f"{r.log2FC_axis1:>+11.2f}{r.log2FC_axis2:>+11.2f}")

    res = enrich(hi, f"{CA}_highconf")
    log("")
    for lib, terms in res.items():
        sig = [t for t in terms if t[6] < 0.05]
        if not sig:
            log(f"  [{lib}] 无 FDR<0.05 条目")
            continue
        log(f"  [{lib}] FDR<0.05 共 {len(sig)} 条，Top8：")
        for t in sig[:8]:
            log(f"      {t[1][:60]:<62} FDR={t[6]:.2e}  "
                f"命中 {len(t[5])}：{','.join(t[5][:8])}")
        log("")
    final[CA] = dict(cn=cn, genes=hi, n=len(hi), enrich=res, T=T)

# ---------- 图：高置信基因四指标 ----------
cas = [c for c in final if final[c]["n"] >= 3]
if cas:
    fig, axes = plt.subplots(1, len(cas),
                             figsize=(8.6 * len(cas),
                                      max(5, 0.36 * max(final[c]["n"]
                                                        for c in cas) + 3)))
    axes = np.atleast_1d(axes)
    nmax = max(final[c]["n"] for c in cas)
    for ax, CA in zip(axes, cas):
        d = final[CA]
        T = d["T"].sort_values("rho_adj_purity").reset_index(drop=True)
        y = np.arange(len(T))
        h = 0.28
        ax.barh(y + h, T.rho_adj_purity, height=h, color="#f58518",
                edgecolor="#333", lw=0.4, label="TCGA 纯度校正 rho")
        ax.barh(y, T.rho_adj_fibro, height=h, color="#54a24b",
                edgecolor="#333", lw=0.4, label="TCGA 成纤维校正 rho")
        ax.barh(y - h, T.rho_ccle, height=h, color="#b279a2",
                edgecolor="#333", lw=0.4, label="CCLE 细胞系 rho")
        ax.set_yticks(y)
        ax.set_yticklabels(T.gene, fontsize=9)
        # 两个面板用同一 y 轴跨度，避免基因少的癌种画出巨粗的条
        ax.set_ylim(-0.8, nmax - 0.2)
        ax.axvline(0, c="k", lw=0.9)
        ax.axvline(0.3, c="#999", ls="--", lw=0.8)
        ax.set_xlim(0, max(0.65, T[["rho_adj_purity", "rho_adj_fibro",
                                    "rho_ccle"]].to_numpy().max() * 1.12))
        ax.set_xlabel("与 TGM2 的 Spearman rho", fontsize=10.5)
        ax.set_title(f"{CA} {d['cn']}   高置信共表达基因 {d['n']} 个\n"
                     f"三重过滤：双轴交集 + 纯度/间质双校正 + CCLE 复现",
                     fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("TGM2 高置信共表达伙伴基因 —— 经组织与细胞系双重确认",
                 fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(f"{OUT}/高置信共表达基因.png", dpi=160)
    plt.close(fig)
    log("")
    log(f"[图] {OUT}/高置信共表达基因.png")

json.dump({k: dict(cn=v["cn"], n=v["n"], genes=v["genes"],
                   enrich={lb: [t[:7] for t in ts[:15]]
                           for lb, ts in v["enrich"].items()})
           for k, v in final.items()},
          open(f"{OUT}/高置信共表达_汇总.json", "w"),
          ensure_ascii=False, indent=1)
open(f"{OUT}/高置信共表达_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
