# 近五年 NILM 数据采集与数据 Pipeline 相关论文

检索时间：2026-06-10  
时间范围：2021-06-10 至 2026-06-10  
排序：从最近到最久  
筛选口径：优先选择与 NILM 数据采集硬件、公开数据集构建、数据清洗/标注/转换、实验复现工具链或端到端 pipeline 直接相关的论文。

## 1. Dataset: Device Activity Report with Complete Knowledge (DARCK) for NILM

- 时间：2025-11
- 作者：Justus Breyer, Kai Gützlaff, Leonardo Pompe, Klaus Wehrle
- 出处：Proceedings of the 12th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys 2025)
- DOI：<https://doi.org/10.1145/3736425.3771959>
- 链接：<https://www.comsys.rwth-aachen.de/publication/2025/2025_breyer_darck-dataset/>
- 相关性：提出 DARCK 数据集，覆盖德国两人公寓 6 个月、主表与 51 个设备的 1 Hz 功率读数；论文和数据说明都描述了测量设置、后处理、时间对齐、插值和数据处理步骤，和 NILM 数据采集及 pipeline 高度相关。

## 2. Capturing High-Frequency Harmonic Signatures for NILM: Building a Dataset for Load Disaggregation

- 时间：2025-07-25
- 作者：Farid Dinar, Sébastien Paris, Éric Busvelle
- 出处：Sensors, 25(15), 4601
- DOI：<https://doi.org/10.3390/s25154601>
- 链接：<https://pubmed.ncbi.nlm.nih.gov/40807766/>
- 相关性：围绕高频谐波 NILM 数据集构建，描述低成本、可扩展数据采集系统，采集聚合负载和单设备测量数据，并强调结构化数据集对实时能耗分解的支撑。

## 3. The Design, Creation, Implementation, and Study of a New Dataset Suitable for Non-Intrusive Load Monitoring

- 时间：2025-06-26
- 作者：Carlos Rodriguez-Navarro, Francisco Portillo, Francisco G. Montoya, Alfredo Alcayde
- 出处：Applied Sciences, 15(13), 7200
- DOI：<https://doi.org/10.3390/app15137200>
- 链接：<https://www.mdpi.com/2076-3417/15/13/7200>
- 相关性：介绍适用于 NILM 的新数据集设计、创建、实现与评估；包含新 converter、13 位时间戳、谐波数据、开源测量硬件和 NILMTK 指标评估，直接覆盖数据采集和数据转换 pipeline。

## 4. Fostering Non-Intrusive Load Monitoring for Smart Energy Management in Industrial Applications: An Active Machine Learning Approach

- 时间：2025-04-28
- 作者：Lukas Fabri, Daniel Leuthe, Lars-Manuel Schneider, Simon Wenninger 等
- 出处：Energy Informatics, 8, Article 54
- DOI：<https://doi.org/10.1186/s42162-025-00517-5>
- 链接：<https://link.springer.com/article/10.1186/s42162-025-00517-5>
- 相关性：面向工业 NILM 的真实数据和稀缺标签问题，基于 CRISP-DM 组织数据科学流程，并用主动学习减少标注数据需求；对工业场景的数据标注与训练 pipeline 有参考价值。

## 5. Transient Dataset of Household Appliances with Intensive Switching Events

- 时间：2024-05-14
- 作者：Dongyang Zhang, Xiaohu Zhang, Lei Hua 等
- 出处：Scientific Data, 11, Article 493
- DOI：<https://doi.org/10.1038/s41597-024-03310-3>
- 链接：<https://www.nature.com/articles/s41597-024-03310-3>
- 相关性：发布包含密集开关事件的家电瞬态数据集，使用继电器精确控制开关以降低机械开关干扰；适合事件型 NILM、负载识别和数据采集方案设计参考。

## 6. The Plegma Dataset: Domestic Appliance-Level and Aggregate Electricity Demand with Metadata from Greece

- 时间：2024-04-12
- 作者：Sotirios Athanasoulias, Fernanda Guasselli, Nikolaos Doulamis, Anastasios Doulamis, Nikolaos Ipiotis, Athina Katsari, Lina Stankovic, Vladimir Stankovic
- 出处：Scientific Data, 11, Article 376
- DOI：<https://doi.org/10.1038/s41597-024-03208-0>
- 链接：<https://www.nature.com/articles/s41597-024-03208-0>
- 相关性：采集希腊 13 户家庭一年期 10 秒级全屋聚合负载和设备级数据，并包含环境、建筑和社会人口学元数据；对 NILM 数据采集、元数据设计和区域泛化研究有价值。

## 7. A Non-Intrusive Load Identification Method Based on Novel Data Acquisition Terminals and Model Fusion

- 时间：2024
- 作者：Jian Zhuge, Guangzheng Lin, Hongfeng Fu, Licheng Zheng
- 出处：IEEE Access, 12, 146598-146609
- DOI：<https://doi.org/10.1109/ACCESS.2024.3474798>
- 链接：<https://dblp.org/rec/journals/access/ZhugeLFZ24>
- 相关性：提出低成本、高性能数据采集终端，并结合 FFT 频域特征与模型融合完成负载识别；重点在 NILM 采集终端和后续特征处理链路。

## 8. Development and Application of an Open Power Meter Suitable for NILM

- 时间：2024-01
- 作者：Carlos Rodríguez-Navarro, Francisco Portillo, Fernando Martínez, Francisco Manzano-Agugliaro, Alfredo Alcayde
- 出处：Inventions, 9(1), 2
- DOI：<https://doi.org/10.3390/inventions9010002>
- 链接：<https://www.mdpi.com/2411-5134/9/1/2>
- 相关性：提出 Open Multi Power Meter (OMPM)，面向 NILM 的开源硬件和固件方案；可用于聚合与分路测量、数据记录和开源可复现实验。

## 9. Unlocking the Full Potential of Neural NILM: On Automation, Hyperparameters & Modular Pipelines

- 时间：2022-09
- 作者：Hafsa Bousbiat, Anthony Faustine, Christoph Klemenjak, Lucas Pereira, Wilfried Elmenreich
- 出处：IEEE Transactions on Industrial Informatics
- DOI：<https://doi.org/10.1109/TII.2022.3206322>
- 链接：<https://www.researchgate.net/publication/363522364_Unlocking_the_Full_Potential_of_Neural_NILM_On_Automation_Hyperparameters_Modular_Pipelines>
- 相关性：提出 Deep-NILMTK，核心贡献是自动化、超参数搜索、实验模板和模块化 NILM pipeline；适合关注数据准备、训练、评估和复现实验链路。

## 10. Torch-NILM: An Effective Deep Learning Toolkit for Non-Intrusive Load Monitoring in Pytorch

- 时间：2022-04
- 作者：Nikolaos Virtsionis Gkalinikis, Christoforos Nalmpantis, Dimitris Vrakas
- 出处：Energies, 15(7), 2647
- DOI：<https://doi.org/10.3390/en15072647>
- 链接：<https://www.mdpi.com/1996-1073/15/7/2647>
- 相关性：开源 PyTorch NILM 工具包，兼容 NILMTK，提供标准化实验设置、基线模型和 benchmark 方法；属于 NILM 数据到模型评估的复现 pipeline 工具链。

## 备注

- 本列表已找到 10 篇符合近五年范围且与“数据采集/数据 pipeline”相关的文献，因此未继续扩大搜索。
- 部分文献更偏“数据集/采集硬件”，部分更偏“实验与训练 pipeline”；实际综述或方案设计时建议按这两类分组阅读。
