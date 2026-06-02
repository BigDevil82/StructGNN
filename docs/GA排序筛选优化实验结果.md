# GA排序筛选优化实验结果

本文档整理 `outputs/result/optimization/ranking_overnight` 中的 overnight 实验结果。实验目标是验证：在剪力墙截面优化中，使用钢筋用量预测模型作为排序筛选器，是否能够减少真实 FEA 调用，并尽量保持优化解质量。

## 实验设置

主实验 `primary_large`：

- 布局数：10
- 随机种子：4
- 方法：Full GA-FEA、Random preselection、GNN steel ranking
- GA 设置：`population=16`，`generations=8`，`elite=2`
- GNN/Random 每代评估比例：`eval_ratio=0.5`
- GNN 额外随机探索比例：`random_ratio=0.125`

辅助实验：

- `ratio_025`：每代评估 25%，强调计算节省。
- `ratio_075`：每代评估 75%，强调解质量与可行率。
- `robust_i7`：设防烈度提高到 7.0，测试稳健性。

## 主实验结果

| 方法 | Runs | 可行率 | 平均目标值 | 平均材料成本 | 平均 FEA 调用 | FEA 调用中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Full GA-FEA | 40 | 0.85 | 697809 | 516523 | 118.95 | 119.5 |
| GNN ranking | 40 | 0.75 | 796026 | 512107 | 67.58 | 68.0 |
| Random preselection | 40 | 0.75 | 808212 | 524175 | 62.68 | 61.5 |

成对比较，即每个布局和 seed 与 full GA-FEA 对比：

| 方法 | 平均 FEA 减少率 | FEA 减少率中位数 | 可行率 | Full 可行率 | 平均目标比 |
|---|---:|---:|---:|---:|---:|
| GNN ranking | 43.2% | 43.5% | 75.0% | 85.0% | 1.153 |
| Random preselection | 47.3% | 47.2% | 75.0% | 85.0% | 1.176 |

注意：平均目标比包含不可行解罚函数，因此不能单独作为解质量结论。对双方都找到可行解的样本，结果更有解释意义：

| 方法 | 双方都可行样本数 | 目标比均值 | 目标比中位数 | 优于 full 的比例 |
|---|---:|---:|---:|---:|
| GNN ranking | 29 | 1.049 | 1.013 | 37.9% |
| Random preselection | 29 | 1.061 | 1.050 | 17.2% |

这说明：

- GNN ranking 在相似 FEA 节省量下，解质量明显优于 random baseline。
- 在双方都找到可行解时，GNN ranking 的中位目标损失约为 1.3%，random baseline 约为 5.0%。
- GNN ranking 并没有完全保持 full GA-FEA 的可行率：主实验中可行率从 85% 降到 75%。
- 因此当前最稳妥的论文表述不是“GNN ranking 优于 full GA-FEA”，而是“GNN ranking 提供了有效的 FEA 调用削减机制，在显著减少真实分析次数的同时保持较接近的优化解质量，并优于随机筛选基线”。

## 筛选比例消融

| 实验 | 方法 | 可行率 | 平均 FEA 减少率 | 目标比均值 |
|---|---|---:|---:|---:|
| ratio_025 | GNN ranking | 87.5% | 67.8% | 1.052 |
| ratio_025 | Random | 87.5% | 74.3% | 1.083 |
| ratio_075 | GNN ranking | 100.0% | 14.4% | 0.947 |
| ratio_075 | Random | 87.5% | 26.1% | 0.941 |

解释：

- `eval_ratio=0.25` 可以大幅减少 FEA，但双方都可行样本中 GNN 和 random 的目标值都明显变差，不适合作为默认优化策略。
- `eval_ratio=0.75` 的 GNN ranking 可行率达到 100%，目标值也较好，但 FEA 节省下降到约 14%。
- `eval_ratio=0.5` 是效率和质量之间较折中的设置；如果优化任务更重视可靠性，可以考虑 `0.75`。

## 高烈度稳健性

`robust_i7` 中：

- GNN ranking 可行率为 75%，与 full GA-FEA 持平。
- Random preselection 可行率下降到 62.5%。
- GNN ranking 平均减少约 40.9% FEA 调用，random 减少约 47.4%。
- 双方都可行时，GNN ranking 目标比中位数约 1.019，random 约 1.026。

这说明在更严格工况下，GNN ranking 比 random 更稳健。

## 建议图表

已生成图表在：

- `outputs/result/optimization/ranking_overnight/figures/primary_summary.png`
- `outputs/result/optimization/ranking_overnight/figures/primary_paired_distributions.png`
- `outputs/result/optimization/ranking_overnight/figures/eval_ratio_tradeoff.png`
- `outputs/result/optimization/ranking_overnight/figures/layout_level_gnn_tradeoff.png`

推荐论文或汇报中使用：

1. 主实验三指标柱状图：`primary_summary.png`
   - 展示 full、random、GNN 三种方法在 FEA 调用、可行率、目标值上的总体差异。
   - 用于说明 GNN ranking 的核心收益是减少真实 FEA 调用。

2. 成对分布图：`primary_paired_distributions.png`
   - 左图展示双方都可行时的目标比，右图展示 FEA 减少率。
   - 这是最关键的图，因为它避免了不可行罚函数对平均目标值的干扰。

3. 筛选比例权衡曲线：`eval_ratio_tradeoff.png`
   - 展示 `eval_ratio=0.25/0.5/0.75` 下，FEA 节省、可行率、目标质量之间的 trade-off。
   - 适合用来论证方法不是固定策略，而是可根据工程需求调节。

4. 布局级表现图：`layout_level_gnn_tradeoff.png`
   - 展示不同布局下 GNN ranking 的 FEA 节省与目标比。
   - 用于说明方法存在布局差异，也可以引出后续改进方向：难布局识别、自适应 eval ratio、置信度控制。

## 当前可写结论

当前结果已经能支撑一个“小论文级”的实验故事：

- 完整 FEA 优化成本较高，限制了智能优化在剪力墙截面设计中的实用性。
- 钢筋用量代理模型虽然不适合直接替代 FEA，但可以作为排序器嵌入 GA。
- 排序器不直接给出最终目标，而是决定每代哪些候选值得进行真实 FEA。
- 与随机筛选相比，GNN ranking 在相近 FEA 节省下能保持更好的解质量和稳健性。
- 与 full GA-FEA 相比，GNN ranking 可以显著减少 FEA 调用，但需要承认可行率和最优性存在一定损失。

建议论文中的方法定位：

> A surrogate-assisted preselection strategy for FEA-efficient shear wall section optimization.

不要把它表述为：

> A surrogate model replacing FEA.

## 后续建议

若还要继续增强论文说服力，下一步优先做两个实验：

1. 固定 FEA 调用预算比较。
   - 例如所有方法最多允许 70 次真实 FEA。
   - Full GA-FEA 只能跑较少代数，GNN ranking 可以跑更多代。
   - 这比“相同 GA 代数但不同 FEA 调用次数”的比较更公平，也更贴合工程效率。

2. 自适应筛选比例。
   - 初期用较大 `eval_ratio` 保证探索和可行性。
   - 中后期逐步降低 `eval_ratio` 提高效率。
   - 或者当连续多代可行率较低时自动提高 `eval_ratio`。

这两个方向都能自然强化“代理模型用于优化加速，而不是直接预测最终设计结果”的研究主线。
