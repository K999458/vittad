# 收敛动力学对照实验汇总（batch 1024 vs 8192 × clip none/0.1/0.01）

- 生成时间：2026-08-21 10:45 (UTC+8)，由 CH-15 汇总
- 项目：/storu/ysu/hiporec/deep-learning/Multi-EPI/phoci_epimci_literature_review_20260627/phoci_paper_aligned_v2_20260630（zkyang 无写权限，报告落在本目录）
- 数据集：GM12878，mns_maxspan_porec 预计算窗口，官方模型实现，paper 负例平衡，official_static_pool
- 目的：钉死「一轮即收敛」是否由 batch 放大或裁剪失效造成

## 逐轮验证 AUC（GM12878 validation）

| 运行 | batch | clip（实际生效） | seed | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|---|---|---|
| mns_maxspan_train（seed1 基线） | 8192 | 无 | 20260712 | 0.8258 | 0.8256 | 0.8259 | 0.8258 | 0.8259 |
| mns_maxspan_train_seed20260713（seed2） | 8192 | 无 | 20260713 | 0.8258 | 0.8257 | 0.8257 | 0.8258 | 0.8259 |
| mns_maxspan_train_clip01 | 8192 | parameter ±0.1（生效，max_abs=0.1000） | 20260712 | 0.8256 | 0.8256 | 0.8258 | 0.8256 | 0.8257 |
| paper_regime_dynamics / A 臂 bs1024_noclip | 1024 | 无（max_abs=0.93） | 20260712 | **0.8260** | 0.8259 | 0.8261 | 0.8260 | — |
| paper_regime_dynamics / B 臂 bs1024_clip001 | 1024 | parameter ±0.01（生效，82~83% 参数饱和在边界） | 20260712 | 0.8150 | 0.8120 | 0.8097 | 0.8130 | — |

## 最终测试集（按负例类型，test AUC，A/B 臂）

| 负例类型 | A 臂 noclip | B 臂 clip±0.01 |
|---|---|---|
| SNS | 0.9372 | 0.9256 |
| MNS | 0.8767 | 0.8715 |
| CNS | 0.6475 | 0.6359 |

## 结论（钉死）

1. **一轮收敛与 batch 大小无关**：A 臂完全按原文 batch 1024、lr 1e-4、无裁剪，epoch 1 验证 AUC 即达 0.8260，与 8192 基线（0.8258）持平，且后续 3 轮完全平坦。「多轮才收敛」不能用我们放大 batch 来解释。
2. **裁剪 ±0.1 真生效也不改变结论**：该臂参数确被钳制（max_abs 恒为 0.1000），逐轮 AUC 与无裁剪基线相差 <0.0003，仍然一轮收敛。
3. **论文名义值 ±0.01 真生效时不是「变慢收敛」而是直接变差**：B 臂 82~83% 参数饱和在 ±0.01 边界，逐轮 AUC 0.8150→0.8120→0.8097→0.8130，震荡且始终低于基线约 1.1~1.6 个点，4 轮内看不到任何向基线爬升的趋势；最终测试三种负例类型全面变差。
4. **官方发布代码的裁剪是无效操作**（`param.clamp(-0.01,0.01)` 非原位调用，等价于没裁），因此作者实际训出的模型对应「无裁剪」方案；我们的无裁剪臂才是对官方行为的忠实复现。
5. 综合 1–4：官方口径下（有效上等价于无裁剪），本数据管线上模型一轮即收敛是稳健事实，与 batch、seed（20260712/20260713 两个种子一致）无关；「需要多轮」的说法在任何一个真实生效的配置里都复现不出来——真把 ±0.01 裁上去只会伤性能，也不会出现渐进收敛。

## 证据路径（均在项目目录下）

- A/B 臂日志：`logs/paper_regime_dynamics_20260820_021217/{bs1024_noclip,bs1024_clip001}_training.log`（A 臂 08-20 02:12–05:40，B 臂 05:40–08:10，均 status=0）
- A/B 臂产物：`outputs/paper_regime_dynamics/{bs1024_noclip,bs1024_clip001}/GM12878/`（含 sage_full_metrics_by_negative_type.tsv、模型权重、prediction_store）
- 基线日志：`logs/mns_maxspan_train/mns_maxspan_train_GM12878.log`、`logs/mns_maxspan_train_clip01_20260818_100904/GM12878_clip01_training.log`、`logs/mns_maxspan_train_seed20260713_20260820_164333/GM12878_training.log`
- 启动脚本（含实验设计注释）：`scripts/run_paper_regime_dynamics_gm12878.sh`

## 相关流水线状态（2026-08-21 10:45）

- seed2（20260713）三个模型（GM12878/K562/Comprehensive）已于 08-21 00:17 全部训完，status=0。
- Figure 2 / Supp 3–8 指标导出已于 08-20 20:42 完成（`outputs/paper_figures_mns_maxspan/figure2_supp3_8/`，7 个 TSV + summary JSON）。
- node4 当前在跑 `loo_head_stage1c`（bs8192、clip none、seed 20260820、5 epochs，08-20 19:04 起）。
- 后续：Figure 3 / Figure 5 / 补充图复现。
