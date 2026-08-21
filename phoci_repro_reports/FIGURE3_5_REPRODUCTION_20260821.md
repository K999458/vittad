# Figure 3 / Figure 5 复现汇总（mns_maxspan seed1 官方模型）

- 生成时间：2026-08-21 (UTC+8)，由 CH-15 执行
- 项目：/storu/ysu/hiporec/deep-learning/Multi-EPI/phoci_epimci_literature_review_20260627/phoci_paper_aligned_v2_20260630
- 计算：node4 RTX 3090，conda 环境 vipg（Python 3.8），GPU 上限 12GB
- 产物目录：/store/zkyang/phoci_repro_reports/paper_figures_mns_maxspan/（ysu 项目目录对 zkyang 无写权限，故落本工作区）

## 背景与遇到的 bug

计划是在新的 mns_maxspan 模型（`outputs/mns_maxspan_models_seed1`，official 实现）上重做 Figure 3、Figure 5，口径对齐 8/20 已完成的 Figure 2 导出。

首次直接调用原脚本失败：`export_figure3_embeddings.py` 和 `export_figure5_myb_crispri.py` 的 `load_model` 用默认 `implementation="legacy"` 构建 `PHOCIModel`，无法加载 official 结构的 maxspan 权重，报 `Missing key(s): encoder.layers.*.self_linear.bias / encoder.output_norm.*`。

修复：把两个脚本拷到 `/store/zkyang/phoci_repro_reports/patched_scripts/`，各加一行从 checkpoint 的 `args` 读取 `model_implementation` 并传给 `PHOCIModel(..., implementation=...)`。这与 `export_figure2_eval_metrics.py` 里已有的加载方式一致。

## Figure 3（隐层嵌入 / 区室 / k-means 聚类）

- 状态：完成，status=0（23:35–23:41）
- 4 个 case 全部导出，每个 chr14、18172 节点、8 个聚类：
  - `gm12878_intra_chr14`、`k562_intra_chr14`、`comprehensive_gm12878_chr14`、`comprehensive_k562_chr14`
- 每 case 产物：embedding_nodes.tsv、embedding_arrays.npz、embedding_umap.png、contact_cluster.png/pdf、embedding_summary.json
- 汇总：`figure3_embeddings/figure3_embedding_panels.png`(+pdf)、`figure3_embedding_summary.json`、`figure3_cluster_counts.tsv`、`figure3_manifest.tsv`

## Figure 5（MYB Apriori 规则 + CRISPRi 面板）

- 状态：完成，status=0（23:41–23:43）
- 分片 `K562/train/chr6_0027.npz`（MYB TSS bin 27036 落在窗口 27011–28011），Comprehensive 模型，1,000,000 条 MYB-centered 预测，signal_min_score=0.3
- 规则/面板计数与旧 global 版本完全一致：tss_rule_count=3、pair_rule_count=12、experimental_rule_count=12、expression_rows=64、synergy_rows=36
- 产物：figure5a 三个规则 TSV、figure5a_b_myb_rule_panels.png/pdf、figure5c_h_crispri_panels.png/pdf、crispri_expression_summary.tsv、crispri_synergy_summary.tsv、supplementary_figure11_* 两个 TSV、figure5_myb_crispri_summary.json

## 与旧 global 版本对比

Figure 5 的全部规则计数与 CRISPRi 行数与 `outputs/paper_figures_global_mns_cap150k_main_mixed/figure5_myb_crispri/` 逐项一致；Figure 3 四个 case 结构一致（chr14、8 聚类），节点数因 maxspan 预计算分片覆盖略有差异（18172 vs 旧版 18409），属预期。

## 备注 / 未决

- 195MB 的 `myb_tss_random_walk_predictions.tsv` 是可重生成的中间产物，未纳入 git（脚本可用 `--reuse-predictions` 复用）。
- 后续按 PAPER_FIGURE_TABLE_REPRODUCTION_CHECKLIST：补 Supp Fig 2（CNS 打分）与 Supp Fig 9（sampling）的 maxspan 版本。
