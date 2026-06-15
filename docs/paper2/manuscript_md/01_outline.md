# 论文大纲初稿

## Title

CASCADE: Calibrated Adaptive Surrogate Candidate Assessment for FEA-Efficient Shear Wall Design Optimization

## Abstract

概括研究背景、核心问题、方法框架和主要实验结论。摘要中应突出本文不是用代理模型替代 FEA，而是通过代理模型和在线校准机制减少优化过程中的无效 FEA 调用。

## 1. Introduction

按“问题 - 现有解决方法 - 仍然存在的问题 - 本文改进 - 创新点概述”的逻辑组织。首先介绍剪力墙结构优化设计的工程需求，以及基于 FEA 的优化在候选方案数量较多时计算成本高的问题。然后综述已有的智能优化、代理模型辅助优化和深度学习预测方法，说明这些方法试图用数据驱动模型减少昂贵分析调用。接着指出关键矛盾：由于模型误差、训练数据覆盖范围、跨布局泛化能力和边界样本不确定性，代理模型很难被完全信任为最终分析结果；但如果只把代理模型看作不可靠的替代品，又无法充分发挥其在优化中的价值。由此引出本文的核心问题：在代理模型不能保证绝对准确的情况下，如何仍然有效利用它提升结构优化效率。最后概述本文的解决思路，即将全局代理模型、在线局部校准和真实 FEA 验证结合起来，使代理模型服务于候选筛选、排序和 FEA 预算分配，而不是直接给出最终结构评价结果。

## 2. Calibrated Surrogate-Assisted Optimization Framework

本节是方法部分。开头先用整体框架图说明从候选方案生成、代理筛选、局部校准、FEA 验证到优化器更新的完整闭环，然后再分别介绍问题定义、数据形成、代理模型和在线校准机制。

### Opening Framework Overview

用一张框架图和简短文字交代完整流程。重点说明全局代理模型、局部校准器和真实 FEA 的关系：代理模型用于筛选和排序，FEA 仍然是最终评价标准，FEA 结果会回流更新局部校准器。

### 2.1 Problem Formulation

定义给定剪力墙布局和设计条件下的截面与材料优化问题。说明设计变量、目标函数、约束条件和真实评价器。重点强调优化目标是材料造价，约束由 OpenSees 分析和规范校核确定。

### 2.2 Data Formulation for Surrogate Learning

说明代理模型训练数据如何形成。这里不展开具体数据规模和 split 细节，而是讲清楚一个样本如何从候选设计变成模型输入和监督标签。

#### 2.2.1 Parametric Structural Evaluation and Design Labels

介绍参数化结构评价流程中与论文相关的关键环节，而不是展开软件实现细节。需要说明输入参数包括几何截面参数、材料参数和设计条件；结构分析采用 OpenSees，剪力墙使用 MVLEM 类单元进行建模；分析工况包括用于结构响应和设计校核的主要荷载/地震作用；校核指标包括层间位移、构件内力需求、墙梁截面设计结果和最终综合可行性。重点解释为什么代理模型选择预测 `final_pass` 和 `material_steel_kg`：前者直接服务于优化中的不可行方案筛选；后者服务于候选方案造价排序。钢筋用量不能仅由几何快速得到，而是依赖多工况结构分析后的构件内力和截面设计结果，因此计算成本高；同时优化目标又面向材料造价，所以钢筋用量是代理模型中最关键的连续预测指标。

#### 2.2.2 Layout Graph and Design Parameter Representation

介绍模型输入表征。说明布局被表示为 room graph，同时保留全局布局统计特征；截面、材料和设计条件作为数值或类别参数输入模型。重点解释这种表征为何能够同时服务可行性判断和钢筋用量预测。

### 2.3 Global Surrogate Models

介绍两个全局代理模型。重点不是把模型写成复杂深度学习细节，而是说明它们共享统一架构，分别服务于保守可行性筛选和材料造价预排序。

#### 2.3.1 Unified LayoutParamGNN Architecture

描述统一的 LayoutParamGNN 架构：room graph 经过 GNN 编码，设计参数经过参数编码器，全局布局特征直接融合，最后由任务头输出预测结果。可配合模型结构图。

#### 2.3.2 Feasibility Screening Surrogate

说明可行性模型的任务是预测候选方案通过校核的概率。强调该模型在优化中被作为保守筛选器使用，阈值选择目标是尽量保持高 feasible recall，而不是追求普通分类阈值下的最高准确率。

#### 2.3.3 Steel Usage Surrogate and Cost Score

说明钢筋模型预测 `material_steel_kg`，其作用主要是候选排序和造价预估，而不是直接替代最终材料用量结果。混凝土用量通过几何和截面参数快速估计，钢筋预测与混凝土估计共同形成候选造价分数。

### 2.4 Online Local Calibration

介绍本文的关键机制：全局代理模型在所有布局上训练，但优化过程固定在单个布局和局部搜索区域内，因此可以利用优化中已经产生的 FEA 结果进行在线局部校准。

#### 2.4.1 Feasibility Probability Calibration

说明可行性概率校准的目的和方法。重点是修正全局模型在当前布局上的概率偏差，并根据局部样本得到更适合当前问题的保守筛选阈值。

#### 2.4.2 Steel Residual Calibration

说明钢筋残差校准的思想：不重新训练全局模型，而是学习真实钢筋用量与全局预测之间的残差。强调校准后的钢筋代理主要用于候选造价排序。

#### 2.4.3 Online Update During Optimization

说明在线使用流程。每一批候选经过代理筛选和排序后，部分候选进入真实 FEA；这些 FEA 结果再加入局部校准样本集，用于后续批次的概率和残差校准。

## 3. Experimental Setup

本节只交代实验设计，不讨论结果。目的是真实、清楚地说明数据来源、模型配置、优化设置和评价指标，使后续结果具有可复现性。

### 3.1 Dataset and Structural Evaluation Settings

说明布局数据来源、参数采样范围、OpenSees 分析与规范校核流程、训练/验证/测试划分方式。这里需要交代测试布局数量和布局类别，但不展开结果。

### 3.2 Surrogate Model Training Settings

说明全局可行性代理模型和钢筋代理模型的训练配置、输入表征、主要 baseline 或对比设置，局部校准模型的不同模型。

### 3.3 Optimization Benchmark Settings

说明优化算法、种群规模/迭代次数/随机种子、测试布局选择、full FEA baseline 和代理辅助方法的对比设置。强调最终方案必须经过真实 FEA 验证。

### 3.4 Evaluation Metrics

统一定义代理模型指标、局部校准指标和优化指标。代理模型包括分类、回归和排序指标；优化指标包括 FEA 调用次数、可行解成功率、首次可行解 FEA 次数和最终造价差异。

## 4. Results and Discussion

本节按照“代理模型是否可用、局部校准是否有效、优化效率是否提升”的顺序组织结果。讨论应服务于论文主线，避免堆砌所有探索性实验。

### 4.1 Global Surrogate Model Performance

展示两个全局代理模型的整体性能和局限性。可行性模型展示整体分类能力和布局级概率偏差；钢筋模型展示回归效果、高钢筋区间偏差和排序能力。结论应引出局部校准的必要性。

主要解读图 outputs\result\paper2\surrogate_model_performance\plots\main_3_2_surrogate_model_performance.png 中的四个子图：（a）ROC-AUC 和 PR-AUC 展示可行性模型的整体分类性能；(b) 不同布局下的模型预测的可行性概率与真实概率的对比，展示模型在不同布局下的预测偏差；Predicted vs True Steel Usage 展示钢筋模型的回归性能和高钢筋区间偏差；Steel Usage pred error by true steel usage quantile 展示钢筋模型在不同钢筋用量区间下的预测误差。

实验结果表明，两个全局代理模型在整体上具有一定的预测能力，但存在明显的局限性。可行性模型虽然在分类指标上表现尚可，但在不同布局上的概率预测存在系统偏差，导致难以直接用于可靠的筛选。钢筋模型在回归指标上表现一般，尤其在高钢筋用量区间存在较大偏差，虽然排序能力较好，但仍无法直接替代最终材料用量结果。这些结果表明全局代理模型不能完全信任，需要通过在线局部校准来修正其偏差并提升优化效率。

### 4.2 Online Local Calibration Performance

展示局部校准随样本数增加的效果，以及 layout-wise before/after 改善。重点说明局部校准能够减少概率校准误差、修正钢筋用量系统偏差，并保持较高候选排序准确性。

图outputs\result\paper2\local_calibration\plots\main_3_3_local_calibration_layout_improvement.png 展示了采用100个样本进行局部校准在不同布局上的改进效果。图（a）展示了局部校准后可行性概率预测的改善，图（b）展示了局部校准后钢筋用量预测的改善。可行性模型在Brier score较大的布局上效果显著，明显降低了预测误差，然而仍不能完全消除偏差，这也解释了为什么不能直接依赖代理模型的结果进行结构优化。钢筋用量预测模型经局部校准后在高钢筋用量区间的偏差得到明显修正，在不同的quantile区间的MAE基本相近且较低，说明局部校准有效提升了模型在当前布局上的预测性能。

main_3_3_local_calibration_sample_efficiency 这张图则展示了随着局部校准样本数增加，概率校准误差和钢筋用量预测误差的变化趋势。结果表明，可行性模型仅需25个样本就能显著降低概率校准误差，且随着样本数增加，误差持续下降但边际效益递减；reject rate在25个样本已达到约50%，说明局部校准能够快速提升代理模型的筛选能力。钢筋用量预测模型校准的样本效率极高，不同的样本数达到的MAE和bias水平相近。且校准后bias接近0，说明局部校准成功修正了钢筋用量预测的系统偏差。

从下面这个表可以看出，局部校准在可行性模型的 Brier score 和 ECE 上分别提升了 42% 和 63%，在筛选指标上 reject rate 提升了 16%，虽然 screen recall 有轻微下降，但仍保持在非常高的水平（99.39%）。钢筋用量预测模型在 MAE、RMSE、R2 和 MAPE 上分别提升了 82%、78%、17% 和 82%，说明局部校准显著提升了模型的回归性能。

task	metric	global	calib	improvement
Feasibility	Brier	0.07 	0.04 	42%
Feasibility	ECE	0.09 	0.03 	63%
Feasibility	Screen reject rate	0.47 	0.55 	16%
Feasibility	Screen recall	0.9990 	0.9939 	-1%
Steel	MAE	11479.89 	2044.75 	82%
Steel	RMSE	12300.64 	2755.98 	78%
Steel	R2	0.85 	0.99 	17%
Steel	MAPE	0.09 	0.02 	82%


### 4.3 Optimization Efficiency and Solution Quality

展示代理辅助优化相对 full FEA 的核心结果。重点比较 FEA 调用次数、最终可行率、材料造价质量和不同优化算法下的一致性，证明方法减少 FEA 开销但不明显损害解质量。

相关图片在 outputs\result\paper2\optim_efficiency
main_01_distribution_summary_3x3 对比了GA，PSO、Random Search三种不同优化算法三种筛选模式下FEA调用次数、造价差距和找到首个可行解所需FEA调用数的分布情况。结果表明，不同优化算法，可行性筛选+造价排序筛选均可以显著降低FEA调用次数，同时保持较低的造价差距。这也就是说，代理辅助优化在不牺牲解质量的前提下，能够有效减少结构分析的计算开销。同时，经过筛选后，找到首个可行解所需的FEA调用数也降低，说明代理模型在优化初期就能够有效引导搜索过程远离明显不可行的设计区域。此外，Random Search即使仅通过代理模型进行筛选，也可以降低FEA调用次数并保持较好的解质量，说明代理模型的确筛出了大量不可行的设计方案。基于两种代理辅助筛选的优化方法在不同优化算法下表现出较好的一致性，说明该方法具有较好的鲁棒性。

main_02_tolerance_success_curve 这张图展示在仅考虑 full FEA baseline 能够找到可行解的 cases 上，进一步比较代理辅助方法是否能够在减少 FEA 调用的同时保持接近的优化质量。横轴表示相对于 full FEA 最优造价允许增加的容差，纵轴表示满足“找到可行解、FEA 调用次数减少、最终造价不超过给定容差”三个条件的 case 比例。可以看到，随着容差从 0% 增加到约 5%，三类优化算法下的合格比例均快速上升，之后趋于平稳，说明代理辅助方法在大多数可解 case 中能够以较小造价损失换取 FEA 调用次数减少。相比仅使用造价代理的设置，结合可行性筛选与造价预排序的方法在 GA、PSO 和 random search 中整体取得更高的合格比例，表明可行性筛选能够有效避免低造价但不可行的候选方案占用评价预算，从而提升代理辅助优化的稳定性。需要注意的是，不同算法子图中的样本基数不同，因此图中标注的合格 case 数不能直接横向比较；该图主要用于说明在各自 full-feasible cases 内，代理辅助方法保持解质量并减少 FEA 调用的能力。

### 4.4 Discussion: Mechanism and Practical Implications of Surrogate-Assisted Optimization

前述结果表明，校准代理辅助方法能够在保持可行率和最终造价基本稳定的同时显著降低真实 FEA 调用次数。为了理解这一现象，需要从优化过程本身出发，观察候选方案在代理筛选、造价预排序和真实 FEA 验证之间如何流动。因此，本节从三个层面展开讨论：首先分析候选方案在代理辅助流程中的分流比例，以说明 FEA 调用减少的直接来源；随后通过代表性优化轨迹说明可行性代理和造价代理的互补作用；最后从布局层面分析不同方法的成功率差异，讨论方法的适用范围和困难案例的来源。

在线校准的贡献可以从图outputs\result\paper2\discussion\supp_03_screening_funnel.png反映出来。该图展示了代理辅助策略在优化过程中对候选方案的分流作用。堆叠区域表示每一代候选方案中被可行性代理模型提前筛除、通过可行性筛选但因造价预排序被跳过、以及最终进入真实 FEA 的平均比例。可以看到，约 60% 的候选方案在进入 FEA 前被代理流程过滤，说明代理模型有效减少了真实分析调用。绿色曲线表示全部候选方案中最终经过 FEA 验证且确认为可行的比例。绿色曲线低于红色区域说明，进入 FEA 的候选中仍有一部分未能通过真实校核。这是因为可行性筛选采用保守策略，目标是尽量避免误删潜在可行方案，因此会有意保留部分不确定或边界候选进入 FEA 验证。

outputs\result\paper2\discussion\main_04_process_scatter_examples.png 这张图，选取了三个具有代表性的 GA 优化过程，分别对应较易、中等和困难的布局案例，用于展示代理模型在候选筛选和搜索轨迹中的作用。Full FEA 方法对所有候选方案进行真实分析，因此可以观察到大量高造价候选和不可行候选被反复评估。引入钢筋用量代理后，优化过程优先保留预测材料造价较低的候选方案，使高造价、偏保守的设计明显减少，搜索轨迹更集中于较低造价区域。然而，仅依赖造价代理仍无法充分区分低造价但不满足规范要求的方案，因此在 Cost surrogate 列中仍存在较多灰色不可行点。进一步加入可行性筛选后，Screen + cost 方法显著减少了进入 FEA 的不可行候选，剩余评价点以绿色可行方案为主，同时得到的最优可行造价与 Full FEA 和 Cost surrogate 基本接近。这说明两个代理模型在优化中承担了互补作用，即钢筋用量代理主要压缩高造价候选的评价空间，可行性代理则进一步减少低价值的不可行候选，从而在不明显损害最终解质量的前提下降低真实 FEA 调用次数。

对于第三个困难案例，三种方法均未能发现可行解。值得注意的是，Full FEA 和 Screen + cost 都尝试了较宽范围的材料造价水平，其中也包括相对较高造价的截面方案，但所有候选仍未通过校核。这表明该案例的失败并非由代理筛选过度激进造成，而更可能源于布局本身或当前截面材料搜索空间的限制。换言之，对于部分结构布置，仅通过截面尺寸和材料等级优化可能不足以获得满足规范要求的方案，后续需要回到布局层面调整剪力墙布置或扩大设计变量范围。

outputs\result\paper2\discussion\main_03_success_heatmap.png 该热力图展示了不同优化算法和代理辅助策略在各测试布局上的可行解发现率。行按照各方法从左到右的成功率模式排序，因此图中从上到下大致反映了布局优化难度由低到高的变化。可以看到，部分布局在 GA、PSO 和 random search 下均能稳定找到可行解，说明这些布局在当前截面与材料搜索空间内具有较大的可行域；而底部若干布局即使在 full FEA baseline 下成功率也较低，表明其困难主要来自布局或搜索空间本身，而非代理筛选造成。相比仅使用钢筋造价代理的设置，加入可行性筛选与局部校准后的 `Screen + cost` 方法在 GA 和 PSO 中整体表现更接近 full FEA，并在多个布局上高于仅使用 cost surrogate 的方法，说明可行性筛选有助于避免低造价但不可行的候选方案占用有限评价预算。Random search 中三种方法差异较小，说明在缺少有效搜索更新机制时，代理筛选主要减少计算开销，而对可行解发现率的提升受到随机采样质量限制。

Taken together, these process-level analyses clarify the role of the proposed surrogate-assisted framework. The funnel analysis shows how FEA calls are reduced, the trajectory examples reveal the complementary effects of cost-based ranking and feasibility screening, and the layout-level heatmap highlights the dependence of optimization success on layout difficulty. These observations support the practical interpretation of the method: calibrated surrogates are most useful as decision-support tools for allocating expensive FEA evaluations, while final design acceptance must remain tied to physics-based analysis and code checking.

## 5. Conclusions

总结本文提出的方法、主要发现和工程意义。强调代理模型用于 FEA 预算分配而非替代最终验算，并指出后续可扩展方向，例如更真实的工程布局、更丰富的设计变量和多目标优化。

## Acknowledgements

如需要，放置资助、数据或软件工具致谢。

## References

按目标期刊格式整理相关文献，包括剪力墙优化、代理模型辅助优化、图神经网络、可靠性/校准方法和 OpenSees 结构分析等方向。
