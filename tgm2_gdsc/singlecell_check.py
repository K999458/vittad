"""单细胞层面确认 TGM2 到底在哪类细胞里表达。

这是整套分析里唯一还没补上的短板：
  bulk RNA 说 PAAD 里 TGM2 高，但 TGM2 是 ECM/间质基因，
  校正 ECM signature 后效应消失；CCLE 细胞系说肿瘤细胞本身就高。
  两个证据方向相反，只有单细胞能裁决 —— 看 TGM2 是在恶性/上皮细胞里，
  还是只在 CAF / 内皮 / 平滑肌里。

数据源：CZ CELLxGENE Discover 的 WMG API（跨数据集汇总的 cell-type 表达）。
指标说明：
  pc = 表达该基因的细胞占比（percent expressing），这是判断"哪类细胞表达"的主指标
  me = 在表达细胞中的平均表达量，被 CELLxGENE 压缩到很窄的区间，单独看没意义
"""
import os, sys, json, urllib.request

os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import numpy as np

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/store/zkyang/tgm2_gdsc/pancancer"
API = "https://api.cellxgene.cziscience.com/wmg/v2/query"
TGM2 = "ENSG00000198959"
TISSUE = {"UBERON:0001264": "胰腺", "UBERON:0002113": "肾",
          "UBERON:0001043": "食管"}
MINN = 150
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def post(payload):
    r = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read().decode())


d = post({"filter": {"gene_ontology_term_ids": [TGM2],
                     "organism_ontology_term_id": "NCBITaxon:9606"},
          "is_rollup": True, "compare": "disease"})
ES = d["expression_summary"][TGM2]
LB = d["term_id_labels"]["cell_types"]
log(f"[0] CELLxGENE 覆盖组织 {len(ES)} 个")

import re

MAL = ("malignant", "neoplastic", "tumor cell", "cancer cell")
EPI = ("epitheli", "ductal", "acinar", r"\bbeta cell", r"\balpha cell",
       r"\bdelta cell", "pancreatic a cell", "pancreatic d cell", "podocyte",
       "nephron", "tubul", "duct cell", "secretory cell", "islet",
       "keratinocyte", r"\bbasal cell", "goblet", "enterocyte",
       "kidney cell", "endocrine cell")
STR = ("fibroblast", "stromal cell", "smooth muscle", "endotheli",
       "myofibro", "pericyte", "mesenchym", "stellate", r"\bmyocyte",
       "muscle cell", "collagen secreting")
IMM = (r"\bt cell", r"\bb cell", "macrophage", "monocyte", "leukocyte",
       r"\bnk cell", "natural killer", "dendritic", "mast cell",
       "plasma cell", "myeloid", "lymphocyte", "granulocyte", "neutrophil",
       "erythro", "microglial", "hematopoietic")


def klass(name):
    l = name.lower()
    # 恶性优先，"malignant cell" 不能被 "t cell" 之类的子串抢走
    for pats, tag in ((MAL, "恶性"), (STR, "间质"), (IMM, "免疫"),
                      (EPI, "上皮")):
        if any(re.search(p, l) for p in pats):
            return tag
    return "其他"


tables, dtables = {}, {}
for tid, cn in TISSUE.items():
    if tid not in ES:
        log(f"\n[{cn}] API 未覆盖该组织")
        continue
    agg, per_dz = [], []
    for ct, dd in ES[tid].items():
        if ct == "CL:0000000":
            continue
        lb = LB.get(tid, {}).get(ct, {})
        ctname = (lb.get("aggregated") or {}).get("name", ct)
        for key, a in dd.items():
            if not isinstance(a, dict):
                continue
            n = a.get("n", 0)
            if n < MINN:
                continue
            rec = (ctname, float(a.get("pc", 0)), float(a.get("me", 0)),
                   int(n), klass(ctname))
            if key == "aggregated":
                agg.append(rec)
            else:
                dzname = (lb.get(key) or {}).get("name", key)
                per_dz.append(rec + (dzname,))
    # 同名细胞类型去重，保留细胞数最多的
    ded = {}
    for r in agg:
        if r[0] not in ded or r[3] > ded[r[0]][3]:
            ded[r[0]] = r
    agg = sorted(ded.values(), key=lambda r: -r[1])
    tables[cn] = agg
    dtables[cn] = per_dz

    log("")
    log("=" * 96)
    log(f"{cn}  TGM2 各细胞类型阳性比例（所有疾病状态合并，细胞数 >= {MINN}）")
    log("=" * 96)
    log(f"  {'细胞类型':<42s} {'类别':<5s} {'阳性比例':>9s} {'细胞数':>9s}")
    log("  " + "-" * 90)
    for name, pc, me, n, k in agg[:24]:
        log(f"  {name[:42]:<42s} {k:<5s} {100*pc:>8.1f}% {n:>9d}")

    # 疾病维度：只看恶性 vs 正常
    dz_names = sorted({r[5] for r in per_dz})
    tumor_dz = [x for x in dz_names
                if any(k in x.lower() for k in
                       ("carcinoma", "cancer", "neoplas", "adenocarc",
                        "tumor", "sarcoma"))]
    log("")
    log(f"  该组织出现的疾病标签 {len(dz_names)} 个；其中恶性相关: "
        f"{tumor_dz if tumor_dz else '无'}")
    if tumor_dz:
        log("")
        log(f"  {'细胞类型':<38s} {'疾病':<28s} {'阳性%':>7s} {'n':>8s}")
        log("  " + "-" * 88)
        sel = [r for r in per_dz
               if r[5] in tumor_dz or r[5].lower() == "normal"]
        sel.sort(key=lambda r: -r[1])
        for name, pc, me, n, k, dz in sel[:26]:
            log(f"  {name[:38]:<38s} {dz[:28]:<28s} {100*pc:>6.1f}% {n:>8d}")

# ---------- 图 ----------
show = {k: v for k, v in tables.items() if v}
if show:
    fig, axes = plt.subplots(1, len(show), figsize=(8.6 * len(show), 7.8))
    axes = np.atleast_1d(axes)
    CMAP = {"恶性": "#c0392b", "上皮": "#e88b8b", "间质": "#f0b27a",
            "免疫": "#a9d18e", "其他": "#b8cfe0"}
    for ax, (cn, rows) in zip(axes, show.items()):
        rows = rows[:22]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [100 * r[1] for r in rows],
                color=[CMAP[r[4]] for r in rows], edgecolor="#333",
                height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r[0][:34]}  (n={r[3]})" for r in rows],
                           fontsize=8.5)
        for yy, r in zip(y, rows):
            ax.text(100 * r[1] + 0.6, yy, f"{100*r[1]:.0f}%", va="center",
                    fontsize=8.5)
        ax.set_xlabel("表达 TGM2 的细胞占比 (%)", fontsize=10)
        ax.set_title(f"{cn}", fontsize=13, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
    fig.legend(handles=[Patch(fc=CMAP[k], ec="#333", label=k) for k in CMAP],
               loc="lower center", ncol=5, fontsize=11)
    fig.suptitle("TGM2 单细胞表达谱：哪类细胞在表达 TGM2\n"
                 "CZ CELLxGENE Discover 跨数据集汇总",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    fig.savefig(f"{OUT}/单细胞_TGM2_细胞类型.png", dpi=160)
    plt.close(fig)
    log("")
    log(f"[图] {OUT}/单细胞_TGM2_细胞类型.png")

json.dump({"aggregated": {k: [list(r) for r in v] for k, v in tables.items()},
           "by_disease": {k: [list(r) for r in v] for k, v in dtables.items()}},
          open(f"{OUT}/单细胞_TGM2.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/单细胞验证_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
