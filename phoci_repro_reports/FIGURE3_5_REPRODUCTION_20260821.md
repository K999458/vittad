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

## Supplementary Figure 2（CNS 扰动打分 / 距离扫描 / 特征 delta）

- 状态：完成，status=0
- GM12878 模型 + GM12878 combined manifest，chr14 76–81Mb 区域，104,303 pairs，mean_score_delta=0.136、mean_span_delta=55.4
- 产物：supplementary_figure2a_c（CNS 分数）、d_f（距离扫描）、h_j（特征 delta）三组面板 png/pdf + cns_distance_scan.tsv + selected_override_examples.tsv + summary
- 41MB 的 cns_perturbation_pairs.tsv 是可重生成中间产物，已 gitignore

## Supplementary Figure 9（负例采样对比）

- 状态：完成，status=0（single-thread random-walk 采样，CPU 密集，约 5 分钟）
- Comprehensive 模型 + GM12878 test 全染色体（60 windows），每方法 2,000,000 候选（random_choice / random_walk 各 2M）
- 关键结论（与原文定性一致）：random_walk 候选 mean score 0.698 / max 0.991，random_choice 仅 0.140 / 0.989；过阈值比例 random_walk vs random_choice：>0.5 为 83.7% vs 2.44%，>0.88 为 18.7% vs 0.046%，>0.95 为 3.80% vs 0.008%——random-walk 采样产生高分多元互作候选的效率高出两个数量级
- 产物：supplementary_figure9c_d_pairwise_maps、9e_threshold_fraction 面板 png/pdf + sampling_threshold_summary.tsv + summary
- 705MB pairwise_counts.tsv 与 609MB predictions.tsv 是可重生成中间产物，已 gitignore

## 同类 bug 与补丁

`score_supplementary_figure2_cns.py`、`score_supplementary_figure9_sampling.py` 的 `load_model` 与 figure3/5 同样缺 `implementation` 参数，已在 `patched_scripts/` 一并修复（均从 checkpoint args 读 `model_implementation`）。

## 登录说明

`ssh ysu@node4` 已配好免密（经 telnet 用 `Ysu2024!` 首次登录后把本机公钥写入 ysu 的 authorized_keys），ysu 身份可直接读写项目目录；此前用的 ywjiang2 免密通道仍可用。

## Figure 2 / Supp Fig 3–8 画图面板（8/21 下午补齐）

- 基于 8/20 已导出的 maxspan 指标 TSV（1.585 亿打分行），用原 `plot_figure2_supp_metrics.py` 出图：figure2d_f / g_i / k(scatter+summary)、supp3–7 共 9 张，status=0。
- figure2j 与 supp8 两张频率面板被原脚本跳过：maxspan 分片里所有正例观测频次均为 1（旧 global 版有 1 和 2+ 两档），纯数字列被 pandas 读成 int 导致 `isin(["0","1","2+"])` 过滤为空。已用 `patched_scripts/plot_frequency_panels_maxspan.py`（bucket 按字符串读 + pointplot）补出这两张，面板标题注明单一频次档的口径。
- 11 张面板 png/pdf + plot_manifest 已拷贝到 `paper_figures_mns_maxspan/figure2_supp3_8/`（含 6 个指标 TSV 与 summary；21GB window_scores.tsv 留在项目目录不入库）。

## Figure 4 / Supp Fig 10（生成样本功能注释，8/21 下午补齐）

- 依赖的 suppfig9 `sampling_predictions.tsv` 此前出图后已清理，用相同种子（20260628）重生成（约 5 分钟，GPU），随后按旧 global 版参数原样注释：method=random_walk、top/low 各 5000、threshold=0.88、GM12878 ChromHMM(ENCFF671FDK)/cCRE 注释、expression_gene_bins 复用旧版（与模型无关）。
- 状态：完成，status=0。annotated_bins=34901、selected_predictions=10000（旧版 18170/5022，差异来自 maxspan 高分候选更多、去重后覆盖更广）。
- 产物：figure4c_f_generated_state_expression_summary.png/pdf、supplementary_figure10_generated_examples.png/pdf、generated_prediction_examples.tsv、generated_prediction_bin_annotations.tsv、figure4_supp10_manifest.tsv、summary JSON，均在 `paper_figures_mns_maxspan/figure4_functional/`。
- 重生成用的临时 `supplementary_figure9_sampling_regen/`（约 1.3GB）已删除。

## seed1 vs seed2 稳定性对照（8/21 下午补齐）

- seed2（20260713）三细胞系训练已全部收尾（GM12878 8/20 19:07、K562 20:38、Comprehensive 8/21 00:17）。
- 用 `patched_scripts/build_seed_comparison_table.py` 汇成 `paper_figures_mns_maxspan/tables/seed1_vs_seed2_metric_comparison.tsv`（27 行 = 3 模型 × train/valid/test × SNS/MNS/CNS）。
- 结论：max |ΔAUC| = 0.0042、max |ΔAP| = 0.0026，且都只出现在小样本 train 切片；test/valid 上 |ΔAUC| ≤ 0.0006——maxspan 结果对随机种子完全稳定。

## 备注 / 未决

- maxspan 版图目录至此与原文对齐：Figure 2（d–k）、Figure 3、Figure 4、Figure 5、Supp Fig 2–10 面板全部落盘；Figure 1 / Supp Fig 1 / Supp Fig 11 / 补充表与模型无关，无需随 maxspan 重做。
- 各大体积中间产物（predictions/pairs/npz/window_scores）均可用对应脚本按相同参数重生成。
