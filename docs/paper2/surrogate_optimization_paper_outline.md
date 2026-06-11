# Surrogate-Assisted Shear Wall Design Optimization Paper Outline

本文档用于梳理论文中 **Methodology** 与 **Experiments** 两部分的章节结构。目标不是写正文，而是明确论文叙事顺序、每节应回答的问题、应放入的关键内容和图表。

## 1. Paper and Methodology Section Titles

标题需要体现三个核心亮点：

- calibration：不是静态代理，而是利用在线 FEA 样本进行局部校准；
- FEA-efficient optimization：方法目的在于减少高保真有限元分析调用。
- surrogate-assisted design optimization：重点是结构设计优化流程，而不是单个预测模型。

### 1.1 Paper title candidates

可选文章标题：

1. **CASCADE: Calibrated Adaptive Surrogate Candidate Assessment for FEA-Efficient Shear Wall Design Optimization**
2. **FEA-Efficient Shear Wall Design Optimization via Calibrated Surrogate-Assisted Candidate Screening**
3. **Calibrated Surrogate-Assisted Design Optimization for Shear Wall Structures with Reduced FEA Calls**
4. **Adaptive Surrogate Screening and Calibration for FEA-Efficient Shear Wall Design Optimization**

推荐优先考虑：

```text
CASCADE: Calibrated Adaptive Surrogate Candidate Assessment for FEA-Efficient Shear Wall Design Optimization
```

理由：

- "CASCADE" 有一个相对自然的 acronym，含义上也符合候选方案逐级筛选、排序、进入 FEA 的流程；
- "Calibrated" 体现在线局部校准；
- "Surrogate Candidate Assessment" 体现代理模型用于候选评估和 FEA 预算分配；
- "FEA-Efficient" 直接指向工程收益；
- "Shear Wall Design Optimization" 比 "section optimization" 更宽，不把文章限制在截面尺寸这一单点上。

如果觉得 acronym 略显刻意，则第 2 个标题更稳妥：

```text
FEA-Efficient Shear Wall Design Optimization via Calibrated Surrogate-Assisted Candidate Screening
```

### 1.2 Methodology section title

methodology 大节标题建议短一些，不必把所有输入信息都写进去。

推荐：

```text
Calibrated Surrogate-Assisted Optimization Framework
```

这个标题聚焦于方法本身的两个核心：

- surrogate-assisted optimization；
- online/local calibration。

`Layout-Parameter` 不建议放在大标题里。代理模型接收布局和设计参数是自然需求，不是最需要抢占读者注意力的亮点；它可以在模型小节中说明。

## 2. Methodology

建议该节标题后不要立刻进入子小节，而是先用 2-4 段文字配合一张整体框架图介绍完整流程。

### Opening Framework Description

这一部分写在 methodology 标题之后、正式子小节之前。

需要传达的核心逻辑：

1. 给定剪力墙布局和设计条件，优化算法生成候选截面方案。
2. 全局 layout-parameter surrogate 对候选进行可行性概率预测和钢筋用量预测。
3. 在线局部校准器根据当前优化过程中已有的 FEA 样本修正全局预测。
4. 明显不可行候选被保守筛除。
5. 其余候选根据预测材料造价和可行性风险分数排序。
6. 仅排名靠前的候选进入真实 OpenSees FEA 和规范校核。
7. 真实 FEA 结果回流为局部校准样本。
8. 最终优化结果仍由真实 FEA 评价确定。

整体框架图建议包含以下模块：

```text
Candidate designs from optimizer
        |
        v
LayoutParamGNN feasibility surrogate
LayoutParamGNN steel surrogate
        |
        v
Online local calibration
        |
        v
Feasibility screening + cost preselection
        |
        v
Selected candidates for OpenSees FEA
        |
        v
Design check + material cost
        |
        v
Update optimizer and local calibrator
```

图中应明确两条反馈回路：

- FEA result -> optimizer；
- FEA result -> local calibration model。

### 2.1 Problem Formulation

这一节回答：优化问题是什么，为什么 FEA 昂贵，代理模型到底服务哪个环节。

应包含内容：

- 给定布局 `L` 和设计条件 `q`，优化截面与材料参数 `x`；
- 目标函数为材料造价：

```text
minimize C(x; L, q)
```

- 材料造价包括混凝土成本和钢筋成本；
- 约束由 OpenSees 分析与规范校核得到；
- 综合可行性记为：

```text
feasible(x; L, q) in {0, 1}
```

- 高保真评价器可写为：

```text
H(x; L, q) -> {feasible, material_cost, steel_kg, constraint_metrics}
```

需要强调：

- 代理模型不改变最终评价标准；
- 代理模型只减少高保真评价器 `H` 的调用次数；
- 被推荐为最优的方案必须经过真实 FEA 验证。

### 2.2 Data Formulation for Surrogate Learning

这一节用于解释代理模型的训练样本如何定义：输入是什么、监督标签如何得到、为什么这些数据能支撑后面的两个代理任务。

建议标题可选：

1. **Data Formulation for Surrogate Learning**
2. **Surrogate Learning Data Formulation**
3. **Design Candidate Representation and Labels**

推荐使用：

```text
Data Formulation for Surrogate Learning
```

理由：

- 标题不强行把“参数化建模”和“数据表征”用 `and` 拼接；
- 共同主题是“代理学习所需的数据如何形成”；
- 可以自然承接 problem formulation，并引出后续 surrogate model。

建议这一节按以下逻辑写：

1. 一个候选设计样本可以表示为：

```text
s = (L, x, q)
```

其中 `L` 是剪力墙布局，`x` 是设计变量，`q` 是设计条件。

2. 对每个样本运行参数化建模、OpenSees 分析和规范校核，得到监督信号：

```text
y_f = final_pass
y_s = material_steel_kg
```

3. 将同一个样本整理成代理模型输入：

```text
layout graph + graph-level features + design parameters + design conditions
```

这样本节的逻辑是：

```text
candidate design -> physics-based evaluation -> surrogate labels
candidate design -> graph/parameter encoding -> surrogate inputs
```

而不是把“建模分析”和“数据表征”作为两个并列内容硬放在一起。

应包含内容：

- 参数化剪力墙结构建模；
- OpenSees 有限元分析；
- 构件设计与规范校核；
- 得到 `final_pass` 和 `material_steel_kg` 两个关键监督信号；
- 每个样本由三类信息组成：
  - 设计变量；
  - 设计条件；
  - 布局表达。

布局表达部分建议重点写 room graph：

- node：房间或空间单元；
- edge：空间邻接关系；
- node features：几何特征、约束向量、剪力墙分布向量；
- edge features：邻接边界特征；
- graph-level features：图规模、面积、墙分布统计等。

参数表达：

- 数值参数标准化；
- 类别参数使用 embedding；
- 包括墙厚、梁截面、混凝土强度、烈度、场地类别、地震分组等。

注意写法：

- 不需要在 methodology 详细展开数据集规模和 train/val/test split；
- 这些放在 `Experimental setup`；
- 这里只说明代理模型的输入与标签如何形成。

### 2.3 Unified Surrogate Model

这一节是方法部分的核心模型小节。重点是两个代理模型共享统一架构，而不是分别孤立介绍。

建议结构：

#### 2.3.1 LayoutParamGNN Architecture

介绍统一网络：

```text
Layout graph -> GNN encoder
Design parameters -> ParamEncoder
Graph-level features -> direct fusion
Fused representation -> task head
```

可以写成公式：

```text
h_G = Readout(GNN(G_L))
h_p = ParamEncoder(x, q)
z = concat(h_G, h_edge, f_L, h_p)
y_hat = MLP(z)
```

说明：

- GNN encoder 使用 GraphSAGE/GINE message passing；
- node readout 使用 mean pooling 和 max pooling；
- edge features 通过 MLP 编码并池化；
- 参数编码器处理数值参数和类别参数；
- 最终拼接图表示、边表示、全局布局特征和参数表示。

#### 2.3.2 Feasibility Screening Surrogate

任务：

```text
p_f = P(feasible | L, x, q)
```

训练：

- target：`final_pass`；
- loss：weighted BCE；
- 类别不平衡通过 `pos_weight` 处理。

部署阈值：

```text
maximize reject_rate
subject to feasible_recall >= r_target
```

需要强调：

- 该模型不是普通二分类器；
- 它是保守筛选器；
- 不确定候选保留给 FEA；
- 目标是在极低误删可行方案风险下筛掉明显不可行方案。

#### 2.3.3 Steel Usage Surrogate

任务：

```text
steel_kg_hat = f_s(L, x, q)
```

训练：

- target：`material_steel_kg`；
- target transform：`standardize(log1p(steel_kg))`；
- loss：Smooth L1。

部署：

- 预测钢筋用量；
- 混凝土用量由几何和截面快速计算；
- 二者组合成材料造价预测。

#### 2.3.4 Cost-Aware Candidate Score

代理造价：

```text
C_hat = C_concrete_fast + C_steel_hat
```

考虑可行性风险：

```text
score = C_hat + lambda * max(0, tau - p_f)
```

说明：

- `C_hat` 用于排序，不作为最终目标；
- 真实材料造价仍由 FEA 和设计校核后确定；
- 风险惩罚避免低成本但明显不可行的候选被过度优先。

### 2.4 Online Local Calibration

这一节体现文章亮点：全局代理模型不是静态使用，而是在优化过程中适配当前布局。

#### 2.4.1 Motivation

应说明：

- 全局代理模型在所有布局上训练；
- 但优化过程固定在单个布局和单个设计条件附近；
- 优化过程中产生的 FEA 样本天然构成局部数据；
- 局部校准利用这些样本修正全局模型。

#### 2.4.2 Feasibility Probability Calibration

方法：

```text
z = log(p_f / (1 - p_f))
P_local = sigmoid(a z + b)
```

即 Platt-style calibration。

阈值仍按保守筛选原则设置：

```text
threshold = quantile(P_local among feasible samples) * threshold_scale
```

需要强调：

- 样本不足或类别单一时不启用；
- 启用后仍以高 feasible recall 为目标；
- 不鼓励激进筛选。

#### 2.4.3 Steel Residual Calibration

残差定义：

```text
r = steel_true - steel_global
steel_calibrated = steel_global + r_hat
```

局部模型：

- 小样本：Ridge；
- 中样本：Gaussian Process；
- 大样本：LightGBM。

输入：

- 当前设计变量；
- 全局钢筋预测；
- log 全局钢筋预测。

#### 2.4.4 Online Update Mechanism

用一个 batch-level algorithm 描述：

1. 全局代理预测；
2. 局部校准修正；
3. 筛选与排序；
4. 对选中候选做真实 FEA；
5. 将真实结果加入局部样本集；
6. 下一批候选使用更新后的校准器。

需要强调：

- 校准样本来自优化过程本身；
- 不引入额外 FEA 数据；
- 校准模型轻量，计算成本相对 FEA 可忽略。

## 3. Experiments

实验部分保持粗粒度结构，重点让读者先理解实验设置，再看代理模型，再看优化效果。

### 3.1 Experimental Setup

这一节统一交代数据、模型训练、优化实验和评价指标。

建议包含四部分。

#### Dataset Construction and Labels

- 143 个剪力墙布局；
- 参数化采样截面、材料和设计条件；
- OpenSees 分析；
- 规范校核；
- 得到：
  - `final_pass`；
  - `material_steel_kg`；
  - 材料造价和约束指标；
- train/validation/test split；
- 测试布局族：L17、L27、L1L28。

#### Surrogate Training Setup

- room graph cache；
- LayoutParamGNN 配置；
- 可行性模型训练目标；
- 钢筋模型训练目标；
- 主要 artifact 路径；
- 阈值选择策略。

#### Optimization Setup

算法：

- GA；
- PSO；
- Random Search。

方法：

- Full FEA；
- Cost surrogate；
- Screen + cost surrogate。

说明每种方法：

- Full FEA：所有候选真实评价；
- Cost surrogate：用造价代理预选；
- Screen + cost surrogate：先保守可行性筛选，再造价预选。

#### Evaluation Metrics

代理模型指标：

- feasibility ROC-AUC、PR-AUC、F1；
- screening reject rate；
- feasible recall；
- false reject count；
- steel MAE、RMSE、R2、MAPE。

优化指标：

- FEA calls；
- first feasible FEA calls；
- final material cost gap；
- feasible success rate；
- qualified case ratio under cost tolerance。

### 3.2 Surrogate Model Performance

这一节证明全局代理模型具备可用性。

建议内容：

#### Feasibility surrogate

表格展示：

- ROC-AUC；
- PR-AUC；
- F1；
- screening threshold；
- reject rate；
- feasible recall；
- false reject count。

重点解释：

> The feasibility model is evaluated primarily as a conservative screening model rather than a standard classifier.

#### Steel surrogate

表格展示：

- MAE；
- RMSE；
- R2；
- MAPE。

可补充：

- predicted vs true scatter；
- error by true steel usage quantile；
- ranking accuracy 或 Spearman correlation，如果后续补充。

#### Model representation comparison

可以用一个简洁表格呈现：

- parameter-only；
- parameter + layout statistics；
- room graph LayoutParamGNN。

注意不要写成探索流水账。目标是说明最终选择统一架构是为了：

- 支持布局图；
- 支持设计参数；
- 同时用于分类与回归；
- 能嵌入优化流程。

### 3.3 Online Calibration Evaluation

这一节证明在线校准不是装饰，而是提高当前布局适配能力。

建议实验：

#### Sample-efficiency analysis

横轴：

```text
number of local FEA samples
```

纵轴可选：

- feasibility Brier / ECE；
- reject rate at fixed feasible recall；
- steel MAE；
- steel residual error；
- ranking accuracy。

比较：

- global surrogate；
- calibrated surrogate。

#### Layout-wise improvement

每个 layout 一条线或一个点：

```text
global error -> calibrated error
```

分别用于：

- feasibility calibration；
- steel residual calibration。

需要强调：

- 在线校准样本来自优化过程中已有 FEA；
- 不额外增加高保真数据成本。

### 3.4 Optimization Efficiency and Solution Quality

这一节是最终方法有效性的主结果。

#### Overall distributions

对应当前图：

```text
main_01_distribution_summary_3x3.png
```

展示：

- FEA calls；
- cost gap；
- first feasible FEA calls。

这一图回答：

- 是否减少 FEA 调用；
- 是否保持最终造价质量；
- 是否影响找到可行解的效率。

#### Cost-tolerance qualified ratio

对应当前图：

```text
main_02_tolerance_success_curve.png
```

定义：

```text
qualified = feasible
          and FEA calls reduced
          and cost gap <= tolerance
```

展示不同造价容忍度下，有多少 case 同时满足效率和质量要求。

#### Layout-level success heatmap

对应当前图：

```text
main_03_success_heatmap.png
```

展示每个布局在不同算法和方法下的成功率。

重点解释：

- 部分布局 full FEA 也难找到可行解；
- 不能简单把失败归因于代理筛选；
- 方法应在可解布局上保持接近 full 的能力。

#### Representative optimization trajectories

对应当前图：

```text
main_04_process_scatter_examples.png
```

展示：

- 候选材料造价分布；
- feasible / infeasible 点；
- best feasible trajectory；
- 代理方法如何改变 FEA 预算分配。

#### Candidate flow

作为补充图：

```text
supp_03_screening_funnel.png
```

展示：

- feasibility rejected；
- cost skipped；
- FEA evaluated；
- FEA feasible。

### 3.5 Ablation and Discussion

这一节解释方法边界与组件贡献。

建议包含：

#### Online calibration contribution

如果已有或后续补充实验，建议对比：

- without local calibration；
- with local calibration。

指标：

- FEA calls；
- success rate；
- final cost gap；
- first feasible calls。

#### Failure cases and data limitations

需要诚实讨论：

- 部分布局来自图像数据集前处理，不一定是合理工程布置；
- 梁由规则自动生成；
- 因此 full FEA 也可能找不到可行解；
- 这些 case 反映的是布局/搜索空间难度，不完全是代理方法失败。

#### Role of GNN

建议表述：

- GNN 不是为了替代所有表格特征模型；
- 它提供统一的 layout-parameter 表达；
- 本文贡献重点是代理模型、在线校准和优化流程的系统集成。

## 4. Recommended Main Figures and Tables

建议主文图表：

1. Overall framework figure：方法整体流程图；
2. LayoutParamGNN architecture figure：统一代理模型结构；
3. Surrogate model performance table：两个代理模型关键指标；
4. Online calibration sample-efficiency figure；
5. `main_01_distribution_summary_3x3.png`；
6. `main_02_tolerance_success_curve.png`；
7. `main_03_success_heatmap.png`；
8. `main_04_process_scatter_examples.png`。

补充图：

1. `supp_03_screening_funnel.png`；
2. reliability diagram；
3. steel prediction residual plot；
4. calibration before/after layout-level improvement。
