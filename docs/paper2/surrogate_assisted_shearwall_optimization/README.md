# Surrogate-assisted shear wall section optimization

本目录用于整理第二篇论文的研究主线、问题定义、方法路线与实验设置。内容定位是论文撰写前的结构化素材，不是实验日志，也不是完整论文初稿。

## 研究主线

本研究面向剪力墙结构截面与材料方案的自动优化问题。给定建筑平面布局、楼层数与抗震设防条件，目标是在满足结构分析与构件设计校核的前提下，搜索材料造价较低的墙厚、梁截面和混凝土等级组合。

完整研究路线可以概括为：

1. 基于 OpenSees 建立剪力墙结构的参数化建模、有限元分析、抗震性能评价与配筋设计工作流。
2. 基于 143 套剪力墙布局，随机或拉丁超立方采样截面、材料与设计条件参数，批量生成结构分析数据集。
3. 基于数据集训练两个代理模型：
   - 可行性筛选模型，用于识别明显不满足要求的候选方案；
   - 钢筋用量预测模型，用于估计材料造价中最难快速获得的钢筋部分。
4. 提出在线局部校准策略，利用优化过程中新增的真实 FEA 样本，对离线全局代理模型进行单布局残差校正。
5. 将代理模型嵌入优化算法，在不牺牲最终可行率和造价质量的前提下减少真实 FEA 调用次数，提高优化效率。

## 建议论文结构对应关系

| 论文部分 | 对应文档 | 主要内容 |
| --- | --- | --- |
| Problem formulation | [01_problem_formulation.md](01_problem_formulation.md) | 设计变量、约束、目标函数、优化任务定义 |
| Methodology: analysis workflow | [02_opensees_workflow.md](02_opensees_workflow.md) | OpenSees 参数化建模、响应分析、规范校核与配筋 |
| Methodology: dataset | [03_dataset_construction.md](03_dataset_construction.md) | 布局来源、参数采样、标签与数据划分 |
| Methodology: surrogate models | [04_surrogate_models.md](04_surrogate_models.md) | 可行性筛选模型、钢筋预测模型、图表征与基线 |
| Methodology: online calibration | [05_online_calibration.md](05_online_calibration.md) | 局部 Platt 校准、钢筋残差校准、在线更新机制 |
| Methodology: optimization | [06_surrogate_assisted_optimization.md](06_surrogate_assisted_optimization.md) | 代理筛选如何接入 GA/PSO/Random Search |
| Experiments setup | [07_experiment_setup.md](07_experiment_setup.md) | 实验组、对比方法、评价指标与可复现实验设置 |
| Writing notes | [08_paper_writing_notes.md](08_paper_writing_notes.md) | 论文叙述重点、创新点组织与避免过度声称 |

## 论文叙述原则

本研究不应被表述为“代理模型完全替代有限元分析”。更准确的定位是：

- 代理模型用于优化过程中的候选筛选和预算分配；
- 不确定或被保留的候选仍可进入真实 FEA；
- 最终最优方案必须由真实建模、分析和设计校核确认；
- 核心贡献是将结构工程校核流程、全局代理模型和在线局部校准整合成一个高效优化框架。

