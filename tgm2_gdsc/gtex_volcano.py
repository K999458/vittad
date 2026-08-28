"""目标癌种火山图（GTEx 正常组织做对照）

为什么要重做：
  TCGA 癌旁正常样本量极不均衡 —— PAAD 只有 4 例、ESCA 11 例、CHOL 9 例。
  用 4 例正常画火山图，几乎所有基因的 FDR 都过不了阈值（之前 PAAD 火山图
  是一团灰点，没有任何显著基因），这不是"TGM2 不高表达"，而是功效不足。
  Toil hub 的 TcgaTargetGtex_rsem_gene_tpm 把 TCGA + GTEx 用同一套流程
  （RSEM，log2(TPM+0.001)）重新定量，可以用 GTEx 的正常组织补足对照组。

基因集：取 TCGA HiSeqV2 的 20530 个 symbol 与 gencode v23 probemap 求交，
        得到 18214 个 symbol / 18682 个 ENSG，与之前 TCGA-only 火山图口径一致。
"""
import os, sys, json, time
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import gzip
import numpy as np
import pandas as pd
from scipy import stats
import xenaPython as xena

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401  设置中文字体
import matplotlib.pyplot as plt

HUB = "https://toil.xenahubs.net"
EX = "TcgaTargetGtex_rsem_gene_tpm"
PH = "TcgaTargetGTEX_phenotype.txt"
BASE = "/store/zkyang/tgm2_gdsc"
OUT = f"{BASE}/pancancer"
LFC, FDRC = 1.0, 0.05
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - r))
        q[i] = prev
    return q


def decode(ds, samples, field):
    c = xena.field_codes(HUB, ds, [field])[0].get("code")
    v = xena.dataset_fetch(HUB, ds, samples, [field])[0]
    if not c:
        return v
    cl = c.split("\t")
    o = []
    for x in v:
        try:
            i = int(float(x))
            o.append(cl[i] if 0 <= i < len(cl) else None)
        except (TypeError, ValueError):
            o.append(None)
    return o


# ---------- 基因列表 ----------
pm = {}
for i, l in enumerate(open(f"{BASE}/gencode_v23_probemap.tsv")):
    if i == 0:
        continue
    a = l.rstrip("\n").split("\t")
    pm.setdefault(a[1], []).append(a[0])

with gzip.open(f"{BASE}/tcga/PAAD_HiSeqV2.gz", "rt") as f:
    f.readline()
    hs_syms = [l.split("\t", 1)[0] for l in f]

sym_order = [s for s in hs_syms if s in pm]
ids, id2sym = [], {}
for s in sym_order:
    for g in pm[s]:
        ids.append(g)
        id2sym[g] = s
log(f"[0] 待取基因 {len(ids)} 个 ENSG / {len(sym_order)} 个 symbol")

# ---------- 样本分组 ----------
psamp = xena.dataset_samples(HUB, PH, None)
meta = {}
cols = {f: decode(PH, psamp, f) for f in
        ("_primary_site", "_sample_type", "_study", "detailed_category")}
for i, s in enumerate(psamp):
    meta[s] = (cols["_primary_site"][i], cols["_sample_type"][i],
               cols["_study"][i], cols["detailed_category"][i])

TASKS = {
    "PAAD": ("Pancreatic Adenocarcinoma", ["Pancreas"], "胰腺癌"),
    "KIRC": ("Kidney Clear Cell Carcinoma", ["Kidney"], "肾透明细胞癌"),
    "ESCA": ("Esophageal Carcinoma", ["Esophagus", "Stomach"], "食管癌"),
}

esamples = xena.dataset_samples(HUB, EX, None)
groups = {}
for CA, (dname, gsites, cn) in TASKS.items():
    tum, nor = [], []
    for s in esamples:
        m = meta.get(s)
        if not m:
            continue
        site, styp, study, dis = m
        norm = "Normal" in str(styp)
        if study == "TCGA" and dis == dname:
            (nor if norm else tum).append(s)
        elif study == "GTEX" and site in gsites:
            nor.append(s)
    groups[CA] = (tum, nor)
    log(f"[0] {CA} ({cn}) 肿瘤 {len(tum)}  正常 {len(nor)} "
        f"(GTEx {sum(1 for s in nor if meta[s][2]=='GTEX')} + "
        f"癌旁 {sum(1 for s in nor if meta[s][2]=='TCGA')})")

union = sorted({s for t, n in groups.values() for s in t + n})
uidx = {s: i for i, s in enumerate(union)}
log(f"[0] 需取表达的样本并集 {len(union)}")

# ---------- 取表达矩阵 ----------
CACHE = f"{OUT}/gtex_expr_cache.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    mat = z["mat"]
    cids = list(z["ids"])
    cun = list(z["samples"])
    if cids == ids and cun == union:
        log(f"[1] 命中缓存 {mat.shape}")
    else:
        log("[1] 缓存不匹配，重新下载")
        mat = None
else:
    mat = None

if mat is None:
    mat = np.full((len(ids), len(union)), np.nan, dtype=np.float32)
    B = 200
    t0 = time.time()
    nb = (len(ids) + B - 1) // B
    for bi, i in enumerate(range(0, len(ids), B)):
        chunk = ids[i:i + B]
        v = None
        for attempt in range(4):
            try:
                v = xena.dataset_fetch(HUB, EX, union, chunk)
                break
            except Exception as e:
                if attempt == 3:
                    log(f"    !! batch {bi} 失败 {e}")
                time.sleep(3 * (attempt + 1))
        if v is None:
            continue
        mat[i:i + len(chunk)] = np.array(
            [[np.nan if x is None else x for x in row] for row in v],
            dtype=np.float32)
        if bi % 10 == 0 or bi == nb - 1:
            el = time.time() - t0
            eta = el / (bi + 1) * (nb - bi - 1)
            log(f"    取表达 {bi+1}/{nb} 批  用时 {el:.0f}s  预计还需 {eta:.0f}s")
    np.savez_compressed(CACHE, mat=mat, ids=np.array(ids, object),
                        samples=np.array(union, object))
    log(f"[1] 表达矩阵 {mat.shape} 已缓存")

# ---------- 逐癌种差异分析 ----------
results = {}
for CA, (dname, gsites, cn) in TASKS.items():
    tum, nor = groups[CA]
    ti = [uidx[s] for s in tum]
    ni = [uidx[s] for s in nor]
    A = mat[:, ti]
    N = mat[:, ni]

    # symbol 层面聚合：同一 symbol 多个 ENSG 取表达量最高的那个
    rows = {}
    mean_all = np.nanmean(mat, axis=1)
    for k, g in enumerate(ids):
        s = id2sym[g]
        if s not in rows or (mean_all[k] > mean_all[rows[s]]):
            rows[s] = k
    keep = np.array([rows[s] for s in sym_order])
    syms = np.array(sym_order)
    A, N = A[keep], N[keep]

    ok = (~np.isnan(A)).sum(1) > 0.8 * len(tum)
    ok &= (~np.isnan(N)).sum(1) > 0.8 * len(nor)
    ok &= (np.nanmean(A, 1) > -5) | (np.nanmean(N, 1) > -5)
    A, N, syms = A[ok], N[ok], syms[ok]

    lfc = np.nanmean(A, 1) - np.nanmean(N, 1)
    _, p = stats.mannwhitneyu(A, N, axis=1, alternative="two-sided",
                              nan_policy="omit")
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    q = bh(p)
    D = pd.DataFrame({"gene": syms, "log2FC": lfc, "p": p, "fdr": q,
                      "tumor_mean": np.nanmean(A, 1),
                      "normal_mean": np.nanmean(N, 1)})
    D = D.sort_values("log2FC", ascending=False).reset_index(drop=True)
    D.to_csv(f"{OUT}/volcano_GTEx_{CA}.csv", index=False)
    results[CA] = (D, len(tum), len(nor), cn)

    up = int(((D.fdr < FDRC) & (D.log2FC >= LFC)).sum())
    dn = int(((D.fdr < FDRC) & (D.log2FC <= -LFC)).sum())
    t = D[D.gene == "TGM2"]
    log("")
    log(f"===== {CA} {cn}  肿瘤 {len(tum)} vs 正常 {len(nor)}  "
        f"基因 {len(D)} =====")
    log(f"  显著上调 {up}  显著下调 {dn}")
    if len(t):
        r = t.iloc[0]
        rk = int(t.index[0]) + 1
        pct = 100 * rk / len(D)
        verdict = ("★ 显著高表达" if r.fdr < FDRC and r.log2FC >= LFC else
                   "✓ 高表达" if r.fdr < FDRC and r.log2FC > 0 else
                   "▼ 显著低表达" if r.fdr < FDRC and r.log2FC <= -LFC else
                   "▽ 低表达" if r.fdr < FDRC and r.log2FC < 0 else "— 无差异")
        log(f"  TGM2: log2FC={r.log2FC:+.2f} (FC={2**r.log2FC:.1f}x)  "
            f"FDR={r.fdr:.2e}  上调排名 {rk}/{len(D)} (前 {pct:.1f}%)  {verdict}")
        log(f"        肿瘤均值 {r.tumor_mean:.2f}  正常均值 {r.normal_mean:.2f}"
            f"  (log2 TPM)")
    log(f"  上调 Top10: {', '.join(D.gene.head(10))}")
    log(f"  下调 Top10: {', '.join(D.gene.tail(10)[::-1])}")

# ---------- 画图 ----------
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.0))
axes = np.atleast_1d(axes)
for ax, (CA, (D, nt, nn, cn)) in zip(axes, results.items()):
    x = D.log2FC.to_numpy()
    y = -np.log10(np.clip(D.fdr.to_numpy(), 1e-300, None))
    c = np.full(len(D), "#cccccc", dtype=object)
    c[(D.fdr < FDRC) & (D.log2FC >= LFC)] = "#e04b4b"
    c[(D.fdr < FDRC) & (D.log2FC <= -LFC)] = "#3b76c4"
    ax.scatter(x, y, s=4, c=list(c), linewidths=0, alpha=0.55, rasterized=True)
    ax.axhline(-np.log10(FDRC), ls="--", lw=0.8, c="#888")
    ax.axvline(LFC, ls="--", lw=0.8, c="#888")
    ax.axvline(-LFC, ls="--", lw=0.8, c="#888")
    t = D[D.gene == "TGM2"]
    if len(t):
        tx = float(t.log2FC.iloc[0])
        ty = -np.log10(max(float(t.fdr.iloc[0]), 1e-300))
        ax.scatter([tx], [ty], s=170, facecolor="#ffd400", edgecolor="k",
                   linewidths=1.8, zorder=6)
        ax.annotate(f"TGM2\nlog2FC={tx:+.2f}\nFDR={float(t.fdr.iloc[0]):.1e}",
                    (tx, ty), textcoords="offset points", xytext=(14, 10),
                    fontsize=10, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff6c2",
                              ec="k", lw=0.9))
    up = int(((D.fdr < FDRC) & (D.log2FC >= LFC)).sum())
    dn = int(((D.fdr < FDRC) & (D.log2FC <= -LFC)).sum())
    ax.set_title(f"{CA} {cn}\n肿瘤 n={nt}  vs  正常 n={nn}"
                 f"（GTEx+癌旁）\n上调 {up} / 下调 {dn}", fontsize=11)
    ax.set_xlabel("log2 fold change（肿瘤 / 正常）", fontsize=10)
    ax.set_ylabel("-log10 FDR", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("TGM2 火山图 —— TCGA 肿瘤 vs GTEx 正常（Toil 统一流程 log2 TPM）",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/火山图_GTEx对照_目标癌种.png", dpi=170)
fig.savefig(f"{OUT}/火山图_GTEx对照_目标癌种.pdf")
plt.close(fig)
log("")
log(f"[图] {OUT}/火山图_GTEx对照_目标癌种.png")

open(f"{OUT}/火山图_GTEx对照_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
