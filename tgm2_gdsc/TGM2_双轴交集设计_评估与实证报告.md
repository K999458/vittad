# TGM2 耐药课题：双轴交集设计评估 + 公开药敏数据实证

生成时间：2026-08-28
工作目录：`/storu/ysu/nfcore/wenjie/`
分析脚本与原始结果：`/storu/ysu/nfcore/wenjie/drug_sensitivity/`

---

## 0. 你的设计（我的理解）

```
轴 1：TGM2 高表达组  vs  TGM2 低表达组      → DEG 集合 A
轴 2：耐药样本组      vs  非耐药样本组       → DEG 集合 B
                    A ∩ B  → 交集基因
                       ↓
        GO / KEGG / GSEA / WGCNA 富集 → 交集通路
                       ↓
        目标通路抑制剂药物分析 + TGM2 与交集基因相关性
```

结论先说：**这个设计是对的，而且比参考文献 PMC12215682 的单轴设计更能抵抗混杂**。但有三个必须处理的技术隐患，其中一个（数据源不同导致的批次效应）会直接决定文章能不能过审。下面逐条说，并附我用公开数据跑出来的实证结果。

---

## 1. 文献同构度：这个设计基本是白区

第四轮定向检索（在前三轮 553 篇基础上新增 194 篇，去重后共 **747 篇**），专门找"基因高/低表达 DEG × 耐药/敏感 DEG → 取交集"的同构文章：

| 检索条件 | 命中 |
|---|---|
| 同时具备两条轴（基因高低分组 + 耐药敏感分组） | **1 / 747** |
| 两条轴 + 明确写出取交集动作 | **0 / 747** |

唯一那一篇是 PMID 36506331（肺腺癌铂类耐药，Front Genet 2022），但它的"基因轴"是铂耐药基因集打分而非单基因高低分组，并没有做两个 DEG 集合的交集。

**含义**：摘要层面几乎没人做过你这个双轴交集。要注意检索局限 —— 很多文章的 Venn 图在正文不在摘要，所以"0 篇"应理解为"没有人把它当作卖点写进摘要"，而不是"绝对没人做过"。但即便如此，把双轴交集作为主图逻辑，新颖性是站得住的。

### 可直接照抄的分步模板

| 你的步骤 | 模板文章 | 说明 |
|---|---|---|
| 轴 1（基因高低分组 DEG + WGCNA + 交集） | **PMID 36504056**（南方医科大学学报 2022，卵巢癌 PD-L1 high/low） | 中文期刊，流程写得最细，直接对应你的轴 1 |
| 轴 2（耐药 vs 敏感 DEG，多数据集取交集） | **PMID 40224214**（Anal Cell Pathol 2025，奥希替尼/阿美替尼耐药细胞系） | 两个耐药细胞系 DEG 取 overlapping DEGs |
| 下游药物分析（通路抑制剂 + 分子对接） | **PMID 40587721**（Medicine 2025，小白菊内酯逆转卵巢癌顺铂耐药） | WGCNA + overlapping genes + 网络药理 + 对接，全套 |
| 药敏预测（oncoPredict/GDSC） | **PMID 40936892**（Front Immunol 2025，CEACAM6） | oncoPredict + GDSC + overlapping DEGs |
| 多组学整合 + 因果推断 | **PMID 42063729**（Front Oncol 2026，胃癌顺铂耐药 GSE14210） | 加了孟德尔随机化，可作为升级版 |

---

## 2. 三个必须处理的技术隐患

### 隐患 A（最致命）：两条轴来自不同数据源，交集被批次效应污染

你现在的默认方案大概是：
- 轴 1 用 TCGA（KIRC/ESCA/PAAD 肿瘤组织，n = 几百）
- 轴 2 用 GEO（耐药细胞系 vs 亲本细胞系，n = 3 vs 3）

这两个 DEG 列表在生物学上**不可比**：TCGA 的差异里混着免疫细胞、成纤维细胞、内皮细胞的成分（TGM2 本身在成纤维和巨噬细胞里高表达），而细胞系差异是纯癌细胞的。两者取交集，出来的很可能是通用应激基因（热激蛋白、代谢酶），审稿人一句"batch effect"就能否掉。

**解决方案：把两条轴放进同一个队列。**

用 oncoPredict（或 pRRophetic）在 TCGA 队列上对每个病人预测目标药物的 IC50，按中位数分成预测耐药组 / 预测敏感组：

- KIRC → sunitinib、sorafenib、pazopanib、axitinib（一线抗血管生成 TKI）
- ESCA → cisplatin、5-fluorouracil、paclitaxel、docetaxel
- PAAD → gemcitabine、paclitaxel、oxaliplatin、irinotecan

这样：
- 轴 1 = TGM2 high vs low（同一表达矩阵）
- 轴 2 = 预测耐药 vs 预测敏感（同一表达矩阵）
- 两条轴同批次、同平台、同样本，DEG 直接可比，交集有生物学意义
- GEO 的耐药细胞系数据**降级为外部验证**：验证交集基因在真实耐药模型里同向变化

而且这样能先做一个**合法性检查（生死判据）**：算 TGM2 表达与预测 IC50 的相关性。
- 如果 TGM2 高的病人预测 IC50 也高（更耐药）→ 整个课题逻辑立住，后面所有分析都有意义
- 如果不相关 → 立刻换药或换癌种，不要往下做

### 隐患 B：交集必须做方向一致性过滤

绝大多数人只取基因名重叠，这是错的。必须拆成四象限：

| | 耐药组上调 | 耐药组下调 |
|---|---|---|
| **TGM2 高组上调** | **一致上调 → 核心交集** | 方向矛盾，单独讨论 |
| **TGM2 高组下调** | 方向矛盾，单独讨论 | **一致下调 → 核心交集** |

只有对角线上的两格进入主线分析。方向矛盾的那两格不要丢掉，单独做一段"补偿机制"讨论，反而是加分项。这个过滤通常会把交集砍掉 40–60%，但剩下的才经得起问。

### 隐患 C：WGCNA 的正确用法不是再做一次差异

你提到要做 WGCNA。最优用法不是再跑一遍差异分析，而是**把它当第三条轴**：

在 TCGA 队列上做 WGCNA，把 **TGM2 表达量** 和 **预测 IC50** 同时作为性状（trait）导入，找出同时与两者显著相关的模块（`|MM| > 0.8 & |GS| > 0.2`）。

这样得到第三个基因集 C，做**三集 Venn（A ∩ B ∩ C）**。这是这类文章的经典主图，比双集 Venn 好看得多，而且逻辑上更硬 —— 交集基因必须同时满足"随 TGM2 变"、"随耐药变"、"在共表达网络里成模块"三个条件。

---

## 3. 实证部分：我用公开药敏数据直接检验了你的核心假设

因为"TGM2 高表达 = 耐药"是整个课题的地基，我没有停在文献层面，直接跑了数据。

**数据来源**
- 表达：CCLE RNA-seq（DepMap 18Q2 RPKM，1076 株细胞系），经 UCSC Xena 取 `TGM2 = ENSG00000198959`
- 药敏：GDSC1 + GDSC2 fitted dose response（release 8.5，`LN_IC50`），542 个药物、975 株细胞系
- 交集后可用：**622 株细胞系 × 664 个（药物 × 数据集）组合**
- 辅助验证：CCLE_NP24 药敏（24 药）、NCI60 DTP GI50 Z-score（263 化合物）

（`rho > 0` = TGM2 越高 IC50 越高 = 越耐药）

### 3.1 泛癌结果：假设成立，且效应量在全转录组前 2%

| 统计量 | TGM2 | 说明 |
|---|---|---|
| 呈耐药方向的药物占比 | **82.2%**（546/664） | |
| 中位 rho | **+0.210** | |
| 最大 rho | **+0.522** | AZD5991（MCL1 抑制剂） |

**与 KIRC 一线治疗直接相关的强命中：**

| 药物 | 靶点 | rho | P |
|---|---|---|---|
| **Sorafenib** | PDGFR/KIT/VEGFR/RAF | **+0.458** | 1.2e-32 |
| **Axitinib** | PDGFR/KIT/VEGFR | **+0.448** | 1.3e-32 |
| **Motesanib** | VEGFR/RET/KIT/PDGFR | **+0.446** | 6.0e-31 |

三个 VEGFR-TKI 全部指向"TGM2 高 → 耐药"，P 值在 1e-31 量级。**这是你课题最漂亮、最有临床落点的靶心。**

**其他机制上有意思的强命中：**

| 药物 | 靶点 | rho | 为什么重要 |
|---|---|---|---|
| AZD5991 | MCL1 | +0.522 | 抗凋亡，全药谱第一 |
| Venetoclax | BCL2 | +0.453 | 抗凋亡 |
| GSK2830371 / A | WIP1 / PPM1D | +0.462 / +0.442 | **p53 通路磷酸酶**，见 §4 |
| Vorinostat / Entinostat / ACY-1215 / Panobinostat | HDAC | +0.472 / +0.432 / +0.433 | HDAC 抑制剂整类耐药 |
| Veliparib | PARP1/2 | +0.429 | DNA 损伤修复 |
| GSK2256098C | FAK1 | +0.415 | **黏附/ECM**，见 §3.3 |
| EPZ004777 / EPZ5676 | DOT1L | +0.438 / +0.438 | 组蛋白甲基化 |

### 3.2 但是：我做了随机基因零分布，"83%"这个说法站不住

这一步很关键，是审稿人一定会问的。我随机抽 **400 个基因**（有效 371 个）跑完全相同的分析，得到经验零分布：

| 统计量 | TGM2 | 随机基因零分布 | TGM2 的位置 | 经验单侧 P |
|---|---|---|---|---|
| 耐药方向药物占比 | 82.2% | 均值 52.5%，SD 29.1%，P95 = 94.4% | 超过 77.4% 随机基因 | **0.23（不显著）** |
| 中位 rho | +0.210 | 均值 0.009，SD 0.084，P95 = 0.17 | 超过 97.8% 随机基因 | **0.024（显著）** |

**必须读懂这两行的差别：**

- **"83% 的药物呈耐药方向"这个说法没有统计意义。** 零分布的标准差高达 29%，四分之一的随机基因都能达到 80% 以上。原因是细胞系药理数据里存在一个主导性的全局轴（生长速率 / 谱系），任何与这个轴相关的基因都会呈现广谱方向偏倚。如果你在文章里写"TGM2 高表达与 83% 的 GDSC 药物耐药相关"，遇到懂行的审稿人，这句话会被置换检验直接打掉。
- **但效应量是真的。** 中位 rho = 0.21 落在全转录组前 2.2%（零分布 P95 只有 0.17）。加上 Sorafenib / Axitinib 这种 rho ≈ 0.45、P ≈ 1e-32 的单药命中，任何多重校正都扛得住。

**给你的写法建议**：不要写"广谱耐药"，要写**药物特异 / 通路特异**的表述：

> TGM2 高表达与 VEGFR-TKI（sorafenib / axitinib / motesanib）、HDAC 抑制剂、以及凋亡诱导剂（MCL1 / BCL2 抑制剂）的耐药显著相关，其效应量位于全转录组前 2%（随机基因置换检验）。

并且**把置换检验作为方法学亮点写进文章**。绝大多数单基因生信文章不做这一步，你做了反而是差异化优势。

### 3.3 TGM2 不是 EMT 的替身，但和 ECM/纤连蛋白轴高度重合

我换成 12 个对照基因跑同一套药谱：

| 基因 | 类别 | 耐药方向占比 | 中位 rho | 与 TGM2 的 rho 谱相关 |
|---|---|---|---|---|
| **TGM2** | 目标基因 | 83% | +0.212 | — |
| FN1 | EMT/间质 | 84% | +0.179 | **0.925（几乎同一个轴）** |
| SNAI2 | EMT/间质 | 78% | +0.130 | **0.877** |
| VIM | EMT/间质 | 36% | −0.034 | 0.677 |
| CDH2 | EMT/间质 | 83% | +0.101 | 0.667 |
| ZEB1 | EMT/间质 | 14% | −0.111 | **−0.126（基本独立）** |
| CDH1 | 上皮 | **95%** | **+0.241** | 0.418 |
| EPCAM | 上皮 | **95%** | **+0.232** | 0.322 |
| MKI67 | 增殖 | 6% | −0.120 | −0.617（反向） |
| GAPDH | 持家（阴性对照） | 52% | +0.004 | 0.759 |
| ACTB | 持家（阴性对照） | 44% | −0.010 | 0.685 |
| RPL13A | 持家（阴性对照） | 2% | −0.230 | −0.701 |
| TBP | 持家（阴性对照） | 7% | −0.195 | −0.847 |

**四个要点：**

1. **GAPDH（52%）和 ACTB（44%）是干净的随机基线**，TGM2 的 83% 明显偏离 —— 但见 §3.2，偏离幅度不足以单独成立。
2. **RPL13A（2%）和 TBP（7%）暴露了那个全局混杂**。核糖体蛋白和转录起始因子本该是阴性对照，却出现 600 / 531 个 FDR<0.05 的显著关联，方向几乎全是"高表达 → 敏感"。它们是增殖速率的代理（MKI67 也是 6%）。**结论：细胞系里生长快 = 大部分药 IC50 低。这个混杂必须在文章里明确控制。**
3. **CDH1（95%）和 EPCAM（95%）都超过 TGM2**。这两个是上皮标志，和 TGM2 的间质倾向相反，却更"广谱耐药"。这进一步证明广谱方向偏倚不是 TGM2 特有。
4. **偏相关：控制 VIM 后 TGM2 的效应保留 107%**（中位 |rho| 0.223 → 0.239，664 个药里 523 个偏相关 P<0.05）。也就是说 **TGM2 不是经典 EMT 的替身**，ZEB1 甚至和它基本独立（r = −0.126）。

   但 TGM2 的 rho 谱与 **FN1 相关 0.925**。FN1（纤连蛋白）正是 TGM2 的经典交联底物。

   **这一条直接指导你的机制叙事**：故事线应该是 **ECM 交联 / 整合素 / 黏附（TGM2–FN1–FAK 轴）**，而不是套用"TGM2 促 EMT"。FAK 抑制剂 GSK2256098C 的 rho = +0.415 也支持这条线。

### 3.4 组织内分层：细胞系层面样本量根本不够，必须回到病人队列

| 组织 | GDSC 细胞系数 | 可评估药物 | 耐药方向占比 | FDR<0.05 |
|---|---|---|---|---|
| KIRC | 13 | 596 | 69% | **0** |
| ESCA | 24 | 642 | 66% | **0** |
| PAAD | 26 | 605 | 71% | **0** |

方向趋势在三个癌种里都保住了（66–71%，都高于随机基线 ~50%），但 **n = 13~26，没有任何药物能过 FDR**。

**这是一条硬约束**：你想做的是癌种特异的结论，而细胞系层面在这三个癌种上做不了。必须走 **TCGA 病人队列 + oncoPredict** 的路线（KIRC n≈530，ESCA n≈160，PAAD n≈180）。这也再次说明 §2 隐患 A 的解决方案是唯一可行路径。

### 3.5 NCI60 数据不可用，别浪费时间

我也跑了 NCI60 DTP（263 化合物 × 59 细胞系），结果**自相矛盾，不能用**：同一个药物不同 NSC 编号给出相反方向 ——

| 化合物 | rho | 方向 |
|---|---|---|
| Paclitaxel_418145 | −0.440 | TGM2 高 → 耐药 |
| Paclitaxel_144075668 | **+0.493** | TGM2 高 → 敏感 |
| Vinblastine_101900 | −0.330 | 耐药 |
| Vinblastine_397349 | **+0.389** | 敏感 |
| Irinotecan_487652 | −0.154 | 耐药 |
| Irinotecan_144076208 | **+0.537** | 敏感 |

而且 262 个化合物里 161 个 FDR<0.05，其中 153 个都指向"TGM2 高 → 敏感"——这种一致性说明它测的是全局响应性混杂而非药物特异效应。n=59 的队列里这个混杂无法剥离。**结论：NCI60 不要用于这个课题。**

---

## 4. 癌种选择：KIRC 是最优解，且有意外的机制闭环

这一轮检索在 KIRC 上挖出一整条已发表的机制链（主要来自韩国 Soo-Youl Kim 团队十余年工作）：

| PMID | 年份/期刊 | 内容 |
|---|---|---|
| 24610445 | 2014 J Cancer Res Clin Oncol | TG2 抑制剂 **GK921** 单药逆转 RCC 移植瘤；明确写 "RCC is resistant to both radiation and chemotherapy" |
| 27031960 | 2016 Cell Death Dis | RCC 通过 **TG2 介导的 chaperoned autophagy 降解 p53** 逃避凋亡；TG2 与 HDM2 竞争结合 p53 |
| 30231606 | 2019 Biomol Ther | TG2 通过 p53 降解诱导 **LC3 / 自噬** |
| 30463244 | 2018 Cancers | 化合物库筛选得到 **streptonigrin** 稳定 p53 |
| 32560270 | 2020 Cells | **抑制 TG2 而非 MDM2** 对 RCC 有显著疗效；提到 MDM2 高表达"contribute to drug resistance" |
| 32708896 | 2020 IJMS | TG2 介导 p53 降解 → **HIF-1α–p300 结合增加 → 促血管生成** |
| 38154386 | 2024 Bioorg Chem | 新型苯并咪唑-4,7-二酮类 TG2 抑制剂，稳定 p53 |
| 25812656 | 2015 J Pathol Transl Med | **638 例 ccRCC** 的 TG2 表达与临床病理/预后 |
| 30736384 | 2019 Med Sci | 综述："TG2: The Maestro of the Oncogenic Mediators in RCC" |
| 32260198 | 2020 IJMS | 靶向 TG2 治疗 RCC 的精准策略 |

### 这条链和我跑出来的数据形成了闭环

三个独立证据互相印证：

1. **PMID 32708896**：TG2 → p53↓ → HIF-1α–p300↑ → 血管生成↑
   **我的数据**：Sorafenib（+0.458）、Axitinib（+0.448）、Motesanib（+0.446）三个 VEGFR-TKI 全部指向 TGM2 高 = 耐药
   → 机制与药敏在抗血管生成这条线上完全对上。**KIRC 一线就是抗血管生成 TKI，这是最好的临床落点。**

2. **PMID 32560270**："抑制 TG2 而非 MDM2 对 RCC 有效"
   **我的数据**：WIP1/PPM1D 抑制剂 GSK2830371（+0.462）、GSK2830371A（+0.442）耐药；CCLE 里 Nutlin-3（MDM2 抑制剂）IC50 rho = +0.151（FDR 2.4e-3）
   → TGM2 高的细胞对 p53 通路药物（MDM2、WIP1 抑制剂）耐药。因为在这些细胞里主导 p53 降解的是 TG2 而不是 MDM2，所以打 MDM2 没用。**生信结果独立复现了已发表的机制。**

3. **PMID 27031960 / 30231606**：TG2–自噬–p53
   **我的数据**：抗凋亡靶点药物 AZD5991（MCL1，+0.522，全药谱最强）、Venetoclax（BCL2，+0.453）耐药
   → 与"逃避凋亡"表型一致。

### 这是双刃剑，要想清楚怎么写

**优势**：生物学基础极强，审稿人会信；"目标通路抑制剂药物分析"这一步有大量现成 TG2 抑制剂可用（GK921、streptonigrin、cystamine、disulfiram/双硫仑、ZED1227、苯并咪唑二酮类），分子对接有充足配体。

**风险**：机制已知，纯生信"重新发现"会被说新颖性不足。

**建议定位**：不要写成"我们发现 TGM2 导致 KIRC 耐药"（这已经被做过），要写成
> **系统性刻画 TGM2 驱动的耐药下游程序** —— 已有工作停在 TG2–p53–autophagy 这一条通路，没有人在转录组层面把"TGM2 高表达"和"耐药表型"两个轴的共同下游程序完整画出来。

这样已发表的机制文献从"撞车"变成"你的生物学背书"，双轴交集设计正好是那个没人做过的方法学增量。

### ESCA 与 PAAD

**ESCA**：TGM2 + 耐药完全是白区，但机制证据很薄，只有三篇可用：
- **PMID 20874003**（Cancer Invest 2011）：ESCC 中 **GPR56 + TG2 + NF-κB 协作**驱动侵袭性 —— 这是 ESCA 里 TGM2 唯一的功能性证据，给了 **NF-κB** 这个通路锚点
- **PMID 24828664**（JTO 2014）：TGM2 是食管腺癌（EAC）细胞表面标志物，相对 Barrett 食管高表达
- **PMID 39375892**（Proteomics Clin Appl 2025）：ESCC 放疗抵抗 TMT 定量蛋白组 —— 值得下载补充材料查 TGM2 是否在差异蛋白里

新颖性最高，但需要更多自己的验证工作。适合作为第二篇或 KIRC 论文里的 pan-cancer 验证队列。

**PAAD**：**不建议作为主打**。前一轮已发现撞车严重（PMID 37246171、PMID 41469519），TGM2–吉西他滨耐药在胰腺癌已被充分覆盖。而且我这轮数据里 gemcitabine 在 NCI60 的 rho 只有 +0.018（无关联），GDSC 里也不在强命中之列。

---

## 5. 建议的最终方案

```
主队列：TCGA-KIRC（n≈530）
主药物：sunitinib / sorafenib / pazopanib / axitinib（抗血管生成 TKI，一线）

Step 0  合法性检查（生死判据）
        oncoPredict 预测每个病人的 TKI IC50
        → 检验 TGM2 表达 vs 预测 IC50 的相关性
        → 不显著就立刻停下换药/换癌种，别往下做

Step 1  轴 1：TGM2 high vs low（中位/最优 cutoff）→ DEG-A
Step 2  轴 2：预测耐药 vs 预测敏感（同队列同矩阵）→ DEG-B
Step 3  轴 3：WGCNA，TGM2 表达 + 预测 IC50 双性状 → 模块基因集 C
Step 4  三集 Venn（A ∩ B ∩ C）+ 方向一致性四象限过滤
Step 5  GO / KEGG / GSEA / GSVA 富集 → 交集通路
        预期锚点：ECM–受体互作、focal adhesion、HIF-1、p53、autophagy
Step 6  药物分析
        (a) 交集通路抑制剂（CMap / 网络药理）
        (b) TG2 抑制剂分子对接（GK921 / streptonigrin / ZED1227 / cystamine）
Step 7  TGM2 与交集基因相关性（Spearman + 共表达网络）
Step 8  验证
        (a) GEO 真实耐药细胞系数据集（外部验证交集基因同向）
        (b) ESCA / PAAD 作为 pan-cancer 验证队列
        (c) 随机基因置换检验（把 §3.2 的方法搬过来，作为方法学亮点）
```

### 三条不能省的对照（决定过审）

1. **随机基因置换零分布** —— 见 §3.2。任何"TGM2 与 N 个药物/通路相关"的陈述都要给经验 P 值。
2. **持家基因阴性对照** —— GAPDH / ACTB 必须做（RPL13A / TBP 不能当阴性对照，它们是增殖代理）。
3. **增殖速率校正** —— MKI67 或增殖评分作为协变量，或做偏相关。§3.3 已证明这个混杂是真实且强的。

### 机制叙事的调整

不要用"TGM2 促 EMT 导致耐药"。数据不支持：ZEB1 与 TGM2 的药谱基本独立（r = −0.126），VIM 只有 36% 耐药方向。

要用 **ECM 交联 / 整合素 / 黏附轴**：TGM2 的药谱与 FN1 相关 0.925，FAK 抑制剂也显著。叙事线为
> **TGM2–FN1 交联 → ECM 硬化 / 整合素–FAK 信号 → 药物渗透障碍 + 黏附介导的存活** ，在 KIRC 中叠加 **TG2–p53–HIF-1α** 通路导致抗血管生成 TKI 耐药。

---

## 5.5 我把轴 1 直接跑出来了（TCGA 真实数据）

既然设计已经确认，我用 UCSC Xena 的 TCGA HiSeqV2（log2(RSEM+1)，20530 基因）把**轴 1 完整跑了一遍**，结果可直接使用。

**方法**：仅原发肿瘤样本（barcode `-01`）；按 TGM2 表达**上/下三分位**分组（比中位数干净，避开分界噪声）；Mann-Whitney U + log2FC；BH 校正；阈值 `FDR<0.05 且 |log2FC|>=0.585`（FC 1.5）。

| 癌种 | 原发肿瘤 n | 高组 / 低组 n | DEG 总数 | 上调 | 下调 |
|---|---|---|---|---|---|
| **KIRC** | 533 | 178 / 178 | **2356** | 1834 | 522 |
| **ESCA** | — | — | 见 `axis1_report.txt` | | |
| **PAAD** | 178 | 60 / 60 | **1681** | 1282 | 399 |

三癌种 DEG 交集：KIRC∩ESCA = 1386，KIRC∩PAAD = 642，ESCA∩PAAD = 955，**三者交集 = 418**。

### KIRC 上调 Top 基因：整条纤维蛋白原/凝血通路

`FGB (+3.91)`、`FGA (+3.67)`、`FGG (+3.65)`、`F2` 凝血酶原 `(+2.62)`、`SAA1 (+3.45)`、`SAA2`、`LBP (+3.40)`、`SLPI`、`HP`、`CP`、`TGFBI (+2.62)`、`IL1R2`、`CXCL5`

**纤维蛋白原三条链（FGA/FGB/FGG）全部位列 Top3。** TGM2 是转谷氨酰胺酶，其经典底物就是**纤维蛋白**和**纤连蛋白**。这条结果在病人组织层面直接印证了 §3.3 从细胞系药谱推出的 ECM 交联轴（TGM2 与 FN1 药谱相关 0.925）。**两个完全独立的数据源指向同一个机制。**

### KIRC 下调：近端小管分化与脂肪酸氧化丢失

`HMGCS2 (−3.12)`、`SLC22A6`、`PAH`、`CYP4A11`、`SLC13A1`、`SLC22A12`、`REN`、`TMEM213/174/72`
富集于：缬氨酸/亮氨酸/异亮氨酸降解、**PPAR 信号**、脂肪酸降解、**近端小管碳酸氢盐重吸收**、过氧化物酶体 —— 即 ccRCC 经典的去分化 + 脂肪酸氧化丧失。

### 三癌种共有 418 基因的富集（最有价值的部分）

| 通路 | 来源 | FDR |
|---|---|---|
| **Extracellular Matrix Organization** | Reactome | 2.8e-16 |
| **Collagen Formation** | Reactome | 1.7e-15 |
| Collagen Biosynthesis And Modifying Enzymes | Reactome | 2.1e-13 |
| **Complement and coagulation cascades** | KEGG | 9.5e-13 |
| Collagen Chain Trimerization | Reactome | 1.5e-11 |
| Cell adhesion molecules | KEGG | 2.5e-9 |
| Regulation Of Complement Cascade | Reactome | 6.6e-9 |
| Neutrophil Degranulation | Reactome | 1.8e-8 |
| **Negative Regulation Of T Cell Activation** | GO BP | 7.0e-8 |
| Negative Regulation Of Lymphocyte Proliferation | GO BP | 7.0e-8 |

KIRC 单独的 Hallmark 富集里，**Epithelial Mesenchymal Transition FDR = 3.1e-40**（最强），其次 Coagulation 2.7e-13、Complement 2.3e-11、TNF-α via NF-κB 8.6e-8、**Hypoxia 9.7e-7**、IL-6/JAK/STAT3 4.2e-8。

**注意 Hallmark_EMT 与 §3.3 结论并不矛盾**：Hallmark_EMT 基因集本身是以 ECM/胶原/FN1/TGFBI 为主体的**基质重塑signature**，不是 EMT 转录因子程序。细胞系层面 ZEB1 与 TGM2 药谱基本独立（r = −0.126）而 FN1 高度一致（0.925），说的是同一件事：**是 EMT 的"基质臂"，不是"转录因子臂"**。论文里要把这个区分讲清楚，否则会被认为是套模板。

Hypoxia 富集（FDR 9.7e-7）同时印证了 PMID 32708896 的 TG2→p53↓→HIF-1α 通路。

### 这一轮跑出了一个新的、必须处理的问题

三癌种交集里免疫相关条目极多（Immune System n=106、中性粒细胞脱颗粒、补体、T 细胞活化负调控），KIRC 上调基因里 Reactome "Immune System" 命中 176 个基因。

**TGM2 在巨噬细胞和成纤维细胞中本身就高表达。**所以 bulk TCGA 里"TGM2 高组"很大程度上等于"基质/免疫浸润高的肿瘤"。如果不处理，审稿人会直接说：你发现的只是肿瘤纯度差异。

**必须加的两步**：
1. **肿瘤纯度校正** —— ESTIMATE 算 StromalScore / ImmuneScore / TumorPurity，作为协变量纳入差异分析（或用 limma 的 `~ group + purity`），并报告校正前后 DEG 的重叠。
2. **细胞来源溯源** —— CIBERSORT 或 xCell 估计免疫浸润；再用单细胞数据（KIRC 有多套公开 scRNA-seq，如 GSE159115、GSE171306）确认 TGM2 在肿瘤细胞 vs 巨噬细胞 vs 成纤维细胞中的分布。

这一步做扎实反而是加分项：可以明确写"TGM2 高表达的耐药程序同时包含肿瘤细胞内在的 ECM 交联和微环境重塑两个成分"，比单纯的 bulk DEG 讲得深。

**轴 1 的产出文件**（见 §6 文件清单）：
`axis1/axis1_KIRC_DEG.csv`、`axis1_ESCA_DEG.csv`、`axis1_PAAD_DEG.csv`（含 gene / log2FC / p / fdr / 两组均值 / 方向），
`axis1_*_all.csv`（全部 18000+ 基因，方便你自己调阈值），
`axis1_三癌种交集.csv`、`axis1_report.txt`、`axis1_enrichment_report.txt`、`axis1_enrichment.json`。

---

## 5.6 轴 2 也跑完了，而且我要**推翻 §4 的癌种建议**

我原本打算按 §2 的建议用 oncoPredict 预测 IC50 做轴 2。但在查 TCGA 临床矩阵时发现了更硬的东西：**TCGA 记录了真实的临床治疗反应**（`primary_therapy_outcome_success` 与 `followup_treatment_success`）。用真实反应做轴 2 远好过预测 IC50，所以我改用了它。

**分组**：耐药 = Progressive Disease + Stable Disease；敏感 = Complete Remission/Response + Partial Remission/Response。

### 生死判据的结果（§2 里说的那一步）

| 癌种 | 耐药 n | 敏感 n | 耐药组 TGM2 | 敏感组 TGM2 | 差值 | P | 判定 |
|---|---|---|---|---|---|---|---|
| KIRC | 27 | 118 | 13.239 | 12.927 | +0.313 | 0.448 | 方向对，不显著 |
| **PAAD** | **84** | **64** | **13.700** | **13.246** | **+0.454** | **0.045** | **✓ 支持** |
| ESCA | 16 | 93 | 11.562 | 11.196 | +0.366 | 0.418 | 方向对，不显著 |

**只有 PAAD 通过了生死判据。** 三个癌种方向全部一致（耐药组 TGM2 都更高），但只有 PAAD 的样本量支撑到统计显著。

### 双轴交集：方向一致率验证了你的设计

| 癌种 | 基因名重叠 | 一致上调 | 一致下调 | 方向矛盾 | 一致率 | 二项检验 P |
|---|---|---|---|---|---|---|
| KIRC | 183 | 133 | 37 | 13 | **93%** | 2.4e-36 |
| **PAAD** | 111 | 100 | 11 | **0** | **100%** | 3.9e-34 |
| ESCA | 651 | 557 | 69 | 25 | **96%** | 9.8e-152 |

如果两条轴无关，方向一致率应该在 50% 左右。实际是 93–100%。**这是你的双轴交集设计有效性的直接数据证明**，可以作为文章方法学部分的一个论证点。

（注意：轴 2 用 FDR 校正后三个癌种都是 0 个 DEG —— 真实临床耐药组样本量太小。上表用的是"未校正 P<0.05 且 |log2FC|≥0.585"。这个阈值必须在文中说明，并用方向一致率 + 外部验证来补偿统计功效不足。）

### ⚠ 质控揭穿了 KIRC：轴 2 是分期信号，不是耐药信号

| 混杂因素 | KIRC | PAAD | ESCA |
|---|---|---|---|
| 性别 | **P=0.014**（耐药组 85% 男 vs 敏感组 58%） | P=0.80 ✓ | P=0.24 |
| 病理分期 | **P<0.0001**（**Stage IV 占耐药组 44%，敏感组仅 2%**） | P=0.18 ✓ | **P=0.0014** |
| 组织学分级 | **P=0.0023**（G4 26% vs 5%） | P=0.42 ✓ | P=0.35 |
| 年龄 | P=0.79 ✓ | P=0.82 ✓ | P=0.54 ✓ |

**KIRC 的"耐药组"本质上就是"晚期高级别男性患者组"。** 这在交集基因里留下了直接指纹：一致上调里出现 `RPS4Y1`、`KDM5D`（Y 染色体基因），一致下调里出现 `XIST`、`TSIX`（X 失活相关）—— 这是纯粹的性别人为假象，不是生物学。KIRC 核心交集的富集也确实很弱（KEGG 无一条 FDR<0.05，GO 只有 3 条）。

**PAAD 是三个癌种里唯一四项混杂全部不显著的。**

### PAAD 核心交集（100 个一致上调基因）：干净且可直接用于下游药物分析

**基因**：`MUC16`、`ANXA8`、`CST6`、`GPR87`、`EREG`、`ZBED2`、`CGB5`、`LY6D`、`DKK1`、`FAM83A`、`KRT17`、`SCEL`、**`ITGB6`**、`VGLL1`、`KRT16`、`PSCA`、`KRT7`、`KLK6`、`HMGA2`、`GABRP`、`TRIM29`、`IVL`、`PADI1` …

这批基因是 PDAC **basal-like / squamous 亚型**的典型标志（KRT17/KRT16/KRT6A/S100A2/LY6D/FAM83A/TRIM29/GPR87）。而 basal/squamous 亚型本身就是公认的化疗抵抗、预后最差亚型。**故事线自洽：TGM2 高 → basal/squamous 亚型 → 化疗耐药。**

**富集（这就是你要的"交集通路"）**：

| 通路 | 来源 | FDR |
|---|---|---|
| **PI3K-Akt signaling pathway** | KEGG | **1.6e-05** |
| **ECM-receptor interaction** | KEGG | 2.3e-04 |
| **Focal adhesion** | KEGG | 2.0e-03 |
| Epidermis Development / Keratinocyte Differentiation | GO BP | 2.4e-07 / 2.7e-04 |
| Laminin Interactions | Reactome | 2.8e-04 |
| Anchoring Fibril Formation | Reactome | 2.8e-04 |
| Type I Hemidesmosome Assembly | Reactome | 2.8e-04 |
| Keratinization | Reactome | 2.8e-04 |
| **Epithelial Mesenchymal Transition** | Hallmark | **1.8e-05** |
| KRAS Signaling Up | Hallmark | 8.4e-04 |
| TNF-α Signaling via NF-κB | Hallmark | 2.2e-03 |
| p53 Pathway | Hallmark | 1.3e-02 |

**ECM-receptor interaction + Focal adhesion + Laminin + Anchoring Fibril + Hemidesmosome + ITGB6** —— 又是同一条 ECM/整合素/黏附轴，和 §3.3（FN1 相关 0.925）、§5.5（FGA/FGB/FGG）第三次独立汇合。

**PI3K-Akt（FDR 1.6e-5）是最好的药物分析落点**：现成抑制剂充足（alpelisib、buparlisib、capivasertib、ipatasertib）。而 **Focal adhesion 对应的 FAK 抑制剂 GSK2256098C 在我 §3.1 的 GDSC 分析里 rho = +0.415** —— 药敏数据和交集通路又对上了。

### 结论：癌种建议改为 PAAD 优先

| | KIRC | PAAD | ESCA |
|---|---|---|---|
| 生死判据（TGM2 耐药组更高） | 不显著 | **✓ P=0.045** | 不显著 |
| 轴2 样本量 | 27 vs 118 | **84 vs 64** | 16 vs 93 |
| 混杂 | 性别/分期/分级三重混杂 | **全部不显著** | 分期混杂 |
| 交集方向一致率 | 93% | **100%** | 96% |
| 核心交集富集质量 | 弱（KEGG 0 条） | **强（PI3K-Akt 1.6e-5）** | 中 |
| 文献撞车风险 | 低 | **高** | 最低 |

**PAAD 在数据上全面胜出，唯一的问题是撞车。** 这是可以处理的 —— 已发表的 PAAD TGM2 工作聚焦于吉西他滨耐药的单一机制，没有人做过双轴交集，也没有人把 TGM2 与 basal/squamous 亚型 + PI3K-Akt/焦点黏附程序联系起来。差异化定位建议：

> **TGM2 高表达标记 PDAC 的 basal/squamous 化疗抵抗亚型，其耐药程序汇聚于 PI3K-Akt 与 ECM–整合素–焦点黏附信号**

KIRC 如果还想做，必须**在分期和性别上做匹配（matched cohort）或作为协变量校正**，否则轴 2 拿不出手。ESCA 耐药组只有 16 例，建议只做验证队列。

---

## 6. 文件清单

`/storu/ysu/nfcore/wenjie/`

```
TGM2_耐药课题_文献检索报告.md            前三轮文献报告（撞车预警 / 模板 / GEO 数据集）
TGM2_双轴交集设计_评估与实证报告.md      本报告

literature_search/
  scripts/  lit_search.py  lit_search2.py  lit_search3.py  lit_search4.py
            lit_rank.py    lit_rank2.py    lit_rank3.py    lit_rank4.py
  raw/      lit_results.json  lit_results2.json  lit_results3.json  lit_results4.json
            geo_results.json  lit_report.txt  lit_report3.txt  lit_report4.txt

axis1/                                    轴1 已跑完的真实结果（见 §5.5）
  axis1_KIRC_DEG.csv    KIRC 2356 个 DEG（gene/log2FC/p/fdr/两组均值/方向）
  axis1_ESCA_DEG.csv    ESCA DEG
  axis1_PAAD_DEG.csv    PAAD 1681 个 DEG
  axis1_KIRC_all.csv    全部 18598 基因（自行调阈值用）
  axis1_ESCA_all.csv    axis1_PAAD_all.csv
  axis1_三癌种交集.csv   418 个共有 DEG
  axis1_report.txt              分组信息 + Top20 上下调
  axis1_enrichment_report.txt   KEGG/GO/Reactome/Hallmark 富集
  axis1_enrichment.json         富集原始结果
  scripts/  axis1_deg.py     轴1 差异分析
            axis1_enrich.py  Enrichr 富集

axis2/                                    轴2（真实临床反应）+ 双轴交集（见 §5.6）
  axis2_KIRC_DEG_loose.csv     轴2 DEG（P<0.05 & |log2FC|>=0.585）
  axis2_PAAD_DEG_loose.csv     axis2_ESCA_DEG_loose.csv
  axis2_*_DEG_fdr.csv          FDR 校正版（三个癌种均为空，保留以示透明）
  axis2_*_all.csv              全部基因
  交集_KIRC_核心_方向一致.csv    ★ 核心交集（方向一致），含两条轴的 log2FC/p/fdr
  交集_PAAD_核心_方向一致.csv    ★ PAAD 100 个一致上调 + 11 下调，最干净
  交集_ESCA_核心_方向一致.csv
  交集_*_方向矛盾.csv           方向矛盾基因（用于讨论补偿机制）
  axis2_intersect_report.txt    分组/生死判据/交集四象限统计
  axis2_qc_enrich_report.txt    ★ 混杂检查（性别/分期/分级/年龄）+ 交集富集
  axis2_summary.json  axis2_qc_enrich.json
  scripts/  axis2_intersect.py   轴2 + 双轴方向一致性交集
            axis2_qc_enrich.py   混杂质控 + 二项检验 + 富集

drug_sensitivity/
  scripts/  fetch_ccle.py           从 Xena 取 CCLE TGM2 表达
            tgm2_drug_corr.py       主分析：CCLE x GDSC1/2 全药谱相关 + 组织分层
            tgm2_ccle_nci60.py      CCLE_NP24 与 NCI60 DTP 辅助验证
            tgm2_gdsc_control.py    12 基因对照 + 药谱相似度 + 控制 VIM 偏相关
            tgm2_null.py            400 随机基因经验零分布
  results/  tgm2_drug_corr_report.txt       主结果（泛癌 + KIRC/ESCA/PAAD）
            tgm2_drug_corr_PAN-CANCER.csv   664 药物完整 rho/P/FDR
            tgm2_drug_corr_KIRC.csv
            tgm2_drug_corr_ESCA.csv
            tgm2_drug_corr_PAAD.csv
            tgm2_gdsc_control_report.txt    基因对照 + 偏相关
            tgm2_null_report.txt            零分布结论
            null_distribution.csv           372 个基因的 frac_pos / med_rho
            tgm2_partial_vim.csv            664 药物控制 VIM 后偏相关
            tgm2_drugsens_report.txt        CCLE_NP24 + NCI60
            gene_control_summary.json
```

复现方式：所有脚本走 `http://127.0.0.1:17895` 代理访问 UCSC Xena 与 Sanger。GDSC 原始 xlsx 从
`https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/` 下载（下载慢，建议挂后台重试）。

---

## 7. 一句话总结

**设计**：你的双轴交集在文献里几乎是白区（747 篇中 0 篇同构），且在数据上被验证有效 —— 三个癌种的方向一致率 93%/100%/96%（二项检验 P 最小 9.8e-152），远高于随机的 50%。

**核心假设**：在 622 株细胞系 × 664 个药物上成立，效应量位于全转录组前 2%，Sorafenib/Axitinib 命中 P ≈ 1e-32；但"83% 广谱耐药"过不了随机基因置换检验（P=0.23），必须改成药物特异表述，并补上随机基因零分布、持家基因阴性对照、增殖校正三条控制。

**癌种**（这一版结论已修正）：**PAAD 优先**。它是唯一通过生死判据的癌种（耐药组 TGM2 显著更高，P=0.045）、唯一四项混杂全部不显著的、方向一致率 100%、交集富集最强（PI3K-Akt FDR=1.6e-5）。KIRC 虽然机制文献最漂亮，但真实临床耐药组只有 27 例且被性别/分期/分级三重混杂（Stage IV 占耐药组 44% vs 敏感组 2%），必须做匹配队列才能用。ESCA 耐药组仅 16 例，只适合做验证。

**机制叙事**：从"EMT"改为 **ECM 交联–整合素–焦点黏附**。三条独立证据汇合：细胞系药谱与 FN1 相关 0.925（而与 ZEB1 独立，r=−0.126）；TCGA-KIRC 轴 1 上调 Top3 是纤维蛋白原 FGA/FGB/FGG；PAAD 核心交集富集于 ECM-receptor interaction、Focal adhesion、Laminin、Hemidesmosome 且含 ITGB6。

**下游药物落点**：PI3K-Akt（alpelisib/capivasertib/ipatasertib）与 FAK（defactinib/GSK2256098 —— 后者在我的 GDSC 分析里 rho=+0.415，药敏与交集通路自洽）。
