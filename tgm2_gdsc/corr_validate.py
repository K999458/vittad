"""对相关性结果做两个必要的把关

问题：交集基因是按"TGM2 高三分位 vs 低三分位差异"筛出来的，
      再在同一批 TCGA 样本里算它们与 TGM2 的相关性，本来就一定显著。
      这是选择性偏倚（circularity），不能直接当结论。

检验一：随机基因零分布
  按表达量匹配抽 2000 个随机基因，算它们与 TGM2 的 rho。
  如果交集基因的 rho 只是略高于随机基因，说明相关性基本来自筛选偏倚。

检验二：外部独立验证（CCLE 细胞系）
  CCLE 是纯肿瘤细胞、独立队列、独立平台。
  在胰腺/肾癌细胞系里重算 TGM2 与交集基因的相关性。
  能在这里复现的基因，才是真正的 TGM2 共表达伙伴，
  而且因为没有间质，复现意味着共表达发生在肿瘤细胞内部。
"""
import sys, os, gzip, json
import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")

sys.path.insert(0, "/store/zkyang/tgm2_gdsc")
import cnfont  # noqa: F401
import matplotlib.pyplot as plt

BASE = "/store/zkyang/tgm2_gdsc"
OUT = f"{BASE}/corr"
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


CAS = [("PAAD", "胰腺癌", "PANCREAS"), ("KIRC", "肾透明细胞癌", "KIDNEY")]
rng = np.random.default_rng(20260828)

# ================= 检验一：随机基因零分布 =================
log("=" * 100)
log("检验一：交集基因的相关性有多少来自筛选偏倚？—— 表达量匹配的随机基因零分布")
log("=" * 100)

null_res = {}
for CA, cn, _ in CAS:
    R = pd.read_csv(f"{OUT}/相关性_{CA}_交集基因.csv")
    ig = set(R.gene)
    with gzip.open(f"{BASE}/tcga/{CA}_HiSeqV2.gz", "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")[1:]
        idx = np.array([i for i, s in enumerate(hdr) if s[13:15] == "01"])
        allg, allv = [], []
        for line in fh:
            g, rest = line.split("\t", 1)
            v = np.array([np.nan if x in ("", "NA") else float(x)
                          for x in rest.rstrip("\n").split("\t")])[idx]
            allg.append(g)
            allv.append(v)
    M = np.vstack(allv)
    gi = {g: i for i, g in enumerate(allg)}
    tg = M[gi["TGM2"]]
    mu = np.nanmean(M, axis=1)
    # 只在有表达的基因里抽，并按表达量分层匹配
    pool = [i for i, g in enumerate(allg)
            if g not in ig and g != "TGM2" and mu[i] > 1
            and np.nanstd(M[i]) > 0.1]
    ig_idx = [gi[g] for g in R.gene if g in gi]
    ig_mu = mu[ig_idx]
    # 分层：把交集基因的表达量分 10 层，每层按比例抽随机基因
    edges = np.quantile(ig_mu, np.linspace(0, 1, 11))
    pool_mu = mu[pool]
    picked = []
    per = 2000 // 10
    for a, b in zip(edges[:-1], edges[1:]):
        cand = [p for p, m in zip(pool, pool_mu) if a <= m <= b]
        if not cand:
            continue
        picked += list(rng.choice(cand, size=min(per, len(cand)),
                                  replace=False))
    picked = np.array(sorted(set(picked)))
    nr = np.array([stats.spearmanr(tg, M[i], nan_policy="omit")[0]
                   for i in picked])
    nr = nr[np.isfinite(nr)]
    obs = R.rho.to_numpy(float)
    log("")
    log(f"[{CA} {cn}]  交集基因 {len(obs)} 个，表达量匹配随机基因 {len(nr)} 个")
    log(f"  |rho| 中位数    交集 {np.median(np.abs(obs)):.3f}   "
        f"随机 {np.median(np.abs(nr)):.3f}   "
        f"倍数 {np.median(np.abs(obs))/max(1e-9,np.median(np.abs(nr))):.2f}x")
    log(f"  |rho|>=0.3 占比 交集 {np.mean(np.abs(obs)>=0.3):.0%}   "
        f"随机 {np.mean(np.abs(nr)>=0.3):.0%}")
    log(f"  |rho|>=0.4 占比 交集 {np.mean(np.abs(obs)>=0.4):.0%}   "
        f"随机 {np.mean(np.abs(nr)>=0.4):.0%}")
    u, p = stats.mannwhitneyu(np.abs(obs), np.abs(nr))
    log(f"  Mann-Whitney P = {p:.2e}")
    # 每个交集基因在零分布里的经验分位
    emp = np.array([np.mean(np.abs(nr) >= abs(o)) for o in obs])
    log(f"  交集基因中，|rho| 超过 95% 随机基因的：{np.mean(emp<0.05):.0%}")
    null_res[CA] = dict(cn=cn, obs=obs, null=nr, p=float(p),
                        med_obs=float(np.median(np.abs(obs))),
                        med_null=float(np.median(np.abs(nr))),
                        frac_beat95=float(np.mean(emp < 0.05)))

# ================= 检验二：CCLE 外部验证 =================
log("")
log("=" * 100)
log("检验二：CCLE 细胞系外部验证（纯肿瘤细胞、独立队列、独立平台）")
log("=" * 100)

CFILE = f"{BASE}/ccle_intersect_genes.json"
allwant = set()
for CA, _, _ in CAS:
    allwant |= set(pd.read_csv(f"{OUT}/相关性_{CA}_交集基因.csv").gene)
allwant.add("TGM2")

if os.path.exists(CFILE):
    cc = json.load(open(CFILE))
    log(f"[缓存] 读取 {CFILE}")
else:
    import xenaPython as xena
    HUB = "https://ucscpublic.xenahubs.net"
    EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
    samples = xena.dataset_samples(HUB, EXPR, None)
    fields = xena.dataset_field(HUB, EXPR)
    # CCLE 用带版本号的 Ensembl ID，且版本号与 gencode v23 不同，按去版本的
    # ENSG 主体匹配
    pm = pd.read_csv(f"{BASE}/gencode_v23_probemap.tsv", sep="\t")
    pm["base"] = pm.id.str.split(".").str[0]
    sym2base = {}
    for b, s in zip(pm.base, pm.gene):
        sym2base.setdefault(str(s), b)
    base2field = {f.split(".")[0]: f for f in fields}
    sym2field = {s: base2field[b] for s, b in sym2base.items()
                 if s in allwant and b in base2field}
    log(f"CCLE 样本 {len(samples)}；交集基因映射到 Ensembl 命中 "
        f"{len(sym2field)}/{len(allwant)}")
    field2sym = {v: k for k, v in sym2field.items()}
    hit = sorted(sym2field.values())
    data = {}
    B = 50
    for i in range(0, len(hit), B):
        ch = hit[i:i + B]
        for attempt in range(4):
            try:
                vv = xena.dataset_fetch(HUB, EXPR, samples, ch)
                for f_, v in zip(ch, vv):
                    data[field2sym[f_]] = v
                break
            except Exception as e:
                log(f"  批次 {i} 第 {attempt+1} 次失败：{e}")
        print(f"  取到 {len(data)}/{len(hit)}", flush=True)
    cc = dict(samples=samples, data=data)
    json.dump(cc, open(CFILE, "w"))
    log(f"[写入] {CFILE}")

csamp = cc["samples"]
cdat = {g: np.array([np.nan if v is None else float(v) for v in vs])
        for g, vs in cc["data"].items()}
log(f"CCLE 可用基因 {len(cdat)}，样本 {len(csamp)}")

ccle_res = {}
for CA, cn, LIN in CAS:
    R = pd.read_csv(f"{OUT}/相关性_{CA}_交集基因.csv")
    sel = np.array([i for i, s in enumerate(csamp) if s.endswith("_" + LIN)])
    if len(sel) < 15 or "TGM2" not in cdat:
        log(f"[{CA}] CCLE {LIN} 细胞系仅 {len(sel)} 个，跳过")
        continue
    tg = np.log2(cdat["TGM2"][sel] + 1)
    log("")
    log(f"[{CA} {cn}]  CCLE {LIN} 细胞系 {len(sel)} 个   "
        f"TGM2 RPKM 中位 {np.nanmedian(cdat['TGM2'][sel]):.1f}")
    rows = []
    for _, r in R.iterrows():
        v = cdat.get(r.gene)
        if v is None:
            continue
        y = np.log2(v[sel] + 1)
        ok = ~(np.isnan(tg) | np.isnan(y))
        if ok.sum() < 15 or np.nanstd(y[ok]) < 0.05:
            continue
        rr, pp = stats.spearmanr(tg[ok], y[ok])
        rows.append(dict(gene=r.gene, rho_tcga=float(r.rho),
                         rho_adj_purity=float(r.rho_adj_purity),
                         rho_ccle=float(rr), p_ccle=float(pp),
                         n_ccle=int(ok.sum())))
    C = pd.DataFrame(rows)
    if not len(C):
        continue
    C["fdr_ccle"] = bh(C.p_ccle)
    C = C.sort_values("rho_ccle", ascending=False)
    C.to_csv(f"{OUT}/CCLE验证_{CA}.csv", index=False)

    same = C[(np.sign(C.rho_ccle) == np.sign(C.rho_tcga))]
    rep = C[(np.sign(C.rho_ccle) == np.sign(C.rho_tcga)) &
            (C.rho_ccle.abs() >= 0.3) & (C.p_ccle < 0.05)]
    rr, rp = stats.spearmanr(C.rho_tcga, C.rho_ccle)
    log(f"  可测基因 {len(C)}   方向一致 {len(same)} ({len(same)/len(C):.0%})")
    log(f"  ★ CCLE 中复现（同向 + |rho|>=0.3 + P<0.05）：{len(rep)} 个 "
        f"({len(rep)/len(C):.0%})")
    log(f"  TCGA rho 与 CCLE rho 的一致性：Spearman rho={rr:+.3f} "
        f"(P={rp:.2e})")
    if len(rep):
        log(f"  复现基因（按 CCLE rho 排序）：")
        log("    " + ", ".join(
            f"{r.gene}({r.rho_ccle:+.2f})" for _, r in rep.head(30).iterrows()))
    ccle_res[CA] = dict(cn=cn, n=len(C), n_lines=int(len(sel)),
                        n_same=int(len(same)), n_rep=int(len(rep)),
                        consist_rho=float(rr), consist_p=float(rp),
                        rep_genes=rep.gene.tolist(), C=C)

# ================= 图 =================
n = len(null_res)
fig, axes = plt.subplots(2, n, figsize=(7.6 * n, 11))
axes = np.atleast_2d(axes)
if axes.shape[0] == 1:
    axes = axes.T
for j, (CA, d) in enumerate(null_res.items()):
    ax = axes[0, j]
    bins = np.linspace(0, 1, 41)
    ax.hist(np.abs(d["null"]), bins=bins, density=True, color="#bbbbbb",
            edgecolor="#777", lw=0.4, label=f"随机基因 n={len(d['null'])}")
    ax.hist(np.abs(d["obs"]), bins=bins, density=True, color="#d62728",
            alpha=0.62, edgecolor="#8b1a1a", lw=0.5,
            label=f"交集基因 n={len(d['obs'])}")
    ax.axvline(d["med_null"], c="#555", ls="--", lw=1.4)
    ax.axvline(d["med_obs"], c="#8b1a1a", ls="--", lw=1.4)
    ax.set_xlabel("与 TGM2 的 |Spearman rho|", fontsize=10.5)
    ax.set_ylabel("密度", fontsize=10.5)
    ax.set_title(f"{CA} {d['cn']}   交集基因 vs 表达量匹配随机基因\n"
                 f"中位 |rho|：{d['med_obs']:.3f} vs {d['med_null']:.3f}"
                 f"（{d['med_obs']/d['med_null']:.1f}x），P={d['p']:.1e}",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(ls=":", lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

for j, CA in enumerate(null_res):
    ax = axes[1, j]
    if CA not in ccle_res:
        ax.text(0.5, 0.5, "CCLE 数据不足", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        ax.axis("off")
        continue
    d = ccle_res[CA]
    C = d["C"]
    col = np.where((np.sign(C.rho_ccle) == np.sign(C.rho_tcga)) &
                   (C.rho_ccle.abs() >= 0.3) & (C.p_ccle < 0.05),
                   "#d62728", "#9ecae1")
    ax.scatter(C.rho_tcga, C.rho_ccle, s=26, c=col, alpha=0.8,
               edgecolors="#333", linewidths=0.35)
    ax.axhline(0, c="k", lw=0.8)
    ax.axvline(0, c="k", lw=0.8)
    ax.axhspan(0.3, 1.05, color="#d62728", alpha=0.05)
    lim = 1.02
    ax.plot([-lim, lim], [-lim, lim], ls=":", c="#888", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for _, r in C.head(6).iterrows():
        ax.annotate(r.gene, (r.rho_tcga, r.rho_ccle), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("TCGA 肿瘤组织中与 TGM2 的 rho", fontsize=10.5)
    ax.set_ylabel(f"CCLE {d['cn']}细胞系中与 TGM2 的 rho", fontsize=10.5)
    ax.set_title(f"{CA} 外部验证：{d['n_lines']} 个细胞系\n"
                 f"方向一致 {d['n_same']}/{d['n']} "
                 f"({d['n_same']/d['n']:.0%})，"
                 f"红点=复现 {d['n_rep']} 个   两队列 rho 一致性 "
                 f"{d['consist_rho']:+.2f}",
                 fontsize=11.5, fontweight="bold")
    ax.grid(ls=":", lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("相关性结果的两道把关：筛选偏倚零分布 + CCLE 细胞系外部验证",
             fontsize=14.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{OUT}/把关_零分布与CCLE验证.png", dpi=160)
plt.close(fig)
log("")
log(f"[图] {OUT}/把关_零分布与CCLE验证.png")

js = dict(
    null={k: {kk: vv for kk, vv in v.items() if kk not in ("obs", "null")}
          for k, v in null_res.items()},
    ccle={k: {kk: vv for kk, vv in v.items() if kk != "C"}
          for k, v in ccle_res.items()})
json.dump(js, open(f"{OUT}/把关_汇总.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/把关_报告.txt", "w").write("\n".join(lines))
print("\n>>> 完成")
