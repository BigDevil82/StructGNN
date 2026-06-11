# Surrogate Model Recap

本文档用于快速恢复当前剪力墙截面优化中代理模型部分的整体认识。它只保留最终论文中值得说明的主线，不记录所有探索性分支。

## 1. 代理模型在优化中的定位

当前代理模型不是用于替代最终有限元验算，而是用于在优化过程中减少不必要的 FEA 调用。

最终优化流程中保留了两个关键离线代理模型：

1. **可行性筛选模型**：预测给定布局、截面参数和设计条件下结构是否可能满足校核要求。
2. **钢筋用量预测模型**：预测 `material_steel_kg`，再结合快速计算的混凝土用量得到材料造价，用于候选方案预排序。

此外，优化过程中会启用一个 **在线局部校准层**：

- 用已经真实 FEA 过的候选样本校准可行性概率；
- 用真实钢筋用量校准全局钢筋预测误差；
- 不重新训练 GNN，只训练轻量局部校正模型。

因此，论文中更准确的表述是：

> 全局代理模型提供跨布局先验，在线局部校准使其适配当前布局与当前设计条件，二者共同作为 FEA 预算分配器，而不是作为最终验算替代品。

## 2. 输入数据与特征

训练数据来自参数化建模与 OpenSees 分析结果。每个样本包含：

- `layout_id`：剪力墙布局；
- 截面与材料参数；
- 地震设防条件；
- FEA 与规范校核得到的综合可行标签；
- 钢筋用量等材料指标。

参数特征定义在：

```text
src/surrogate/features/consts.py
```

主要包括：

- 层数：`N`
- 墙厚：`tw_bot`, `tw_mid`, `tw_top`
- 主梁/次梁截面：`hb_main`, `bb_main`, `hb_sec`, `bb_sec`
- 混凝土强度：`conc_bot`
- 设防烈度、场地类别、地震分组：`intensity`, `site_class`, `seismic_group`

布局全局特征也保留在同一文件中，例如：

- 平面包围盒、长宽比、面积；
- 墙总长度、墙数量、墙方向比例；
- 梁墙长度比、梁墙数量比；
- 节点数、平均度、连通分量数；
- 墙体惯性矩、边界墙比例等。

## 3. 图表示：room graph

最终保留的图表示是 **room graph**，构建代码位于：

```text
src/surrogate/gnn/graph_data.py
```

该表示将房间或空间单元作为图节点：

- node：房间/空间单元；
- edge：房间之间的邻接关系；
- node feature：空间几何特征、约束向量、剪力墙分布向量；
- edge feature：邻接边界相关特征；
- graph feature：节点数、边数、图密度、面积统计、墙分布统计等。

room graph 通过 `build_graph_from_dxf` 从 DXF 构建，与前期布局生成研究中的空间图表达一致。它比“构件作为节点”的 member graph 更适合描述建筑平面空间组织，因此是论文中更值得保留的图表达。

需要注意：后续探索中发现，仅靠 GNN 图结构带来的增益并不总是显著，参数特征和布局统计特征已经具有很强预测能力。但在论文叙事中，room graph 仍然提供了统一表达布局拓扑与空间约束的接口。

## 4. 统一 GNN 架构

两个离线代理模型都使用同一个主体网络：

```text
src/surrogate/gnn/model.py
LayoutParamGNN
```

模型结构如下：

1. 节点特征投影到 hidden dimension；
2. 使用多层 GraphSAGE 或 GINE 进行 message passing；
3. 对节点表示做 global mean pooling 和 global max pooling；
4. 对边特征做 MLP 编码并池化；
5. 用 `ParamEncoder` 编码设计参数；
6. 拼接：

```text
node_mean_pool + node_max_pool + edge_mean_pool + graph_feat + param_embedding
```

7. 通过 MLP head 输出一个标量。

两个任务的区别只在训练目标和输出解释：

- 可行性任务：输出 logit，经过 sigmoid 得到可行概率；
- 钢筋任务：输出标准化后的 log 钢筋用量。

## 5. 可行性筛选模型

训练入口：

```text
scripts/surrogate/train_gnn_final_pass.py
src/surrogate/gnn/train.py
```

默认优化中使用的模型：

```text
data/parametric/ckpt/baseline_gnn_room_hybrid_h256_screen995_v1/gnn_final_pass.pt
```

训练目标：

```text
target = final_pass
```

其中 `final_pass` 表示结构在当前截面、材料和设计条件下是否通过综合校核。

训练损失：

```text
BCEWithLogitsLoss(pos_weight=neg/pos)
```

这样处理样本类别不平衡问题。

### 关键点：不是普通二分类阈值

优化中使用这个模型时，真正重要的不是 0.5 阈值下的分类准确率，而是保守筛选能力。

筛选阈值在验证集上按如下目标选择：

```text
maximize reject_rate
subject to feasible_recall >= screening_target_recall
```

当前默认目标：

```text
screening_target_recall = 0.995
```

即尽量保留真实可行样本，只筛掉模型非常确定不可行的候选。

当前 artifact 中的关键测试集指标：

```text
test ROC-AUC = 0.9693
test PR-AUC  = 0.8593
test F1      = 0.7688

screening_threshold = 0.004
test reject_rate = 0.4736
test feasible_recall = 0.9998
test false_reject_count = 3
```

因此，该模型在论文中的主要作用应表述为：

> 在极低误删可行方案风险下，提前筛掉接近一半明显不可行候选。

优化中调用位置：

```text
src/shearwall_optimization/surrogate_screening.py
GNNFeasibilityScreener
```

若候选满足：

```text
predicted_probability < screening_threshold
```

则直接赋予一个很大的惩罚目标值，不进入真实 FEA。

## 6. 钢筋用量预测模型

训练入口：

```text
scripts/surrogate/train_gnn_steel.py
src/surrogate/gnn/train_steel.py
```

默认优化中使用的模型：

```text
data/parametric/ckpt/steel_gnn_room_lr5e4_b512/gnn_steel.pt
```

训练目标：

```text
target = material_steel_kg
```

目标变换：

```text
y = standardize(log1p(material_steel_kg))
```

训练损失：

```text
SmoothL1Loss
```

推理时反变换：

```text
steel_kg = expm1(y_pred * std + mean)
```

当前 artifact 中的关键测试集指标：

```text
test MAE  = 11482.8 kg
test RMSE = 15674.7 kg
test R2   = 0.9239
test MAPE = 0.0926
```

该模型不是直接决定最终设计，而是用于候选预排序。优化目标是材料造价：

```text
material_cost = concrete_cost + steel_cost
```

其中：

- 钢筋用量由 GNN 预测；
- 混凝土用量根据几何、截面和层高快速计算；
- 钢筋价格和混凝土价格由优化问题配置给出。

优化中调用位置：

```text
src/shearwall_optimization/surrogate_cost.py
GNNMaterialCostEstimator
```

成本预筛分数为：

```text
score = predicted_material_cost
      + feasibility_penalty_cost * max(0, hinge_target - feasible_probability)
```

这个分数避免低造价但高不可行风险的候选过度靠前。

## 7. 在线局部校准

在线校准实现位于：

```text
src/shearwall_optimization/local_calibration.py
OnlineLocalCalibrator
```

它使用优化过程中自然产生的 FEA 结果，不需要额外采样。

每个真实 FEA 样本会记录：

- 设计变量；
- 真实可行性；
- 全局可行性概率；
- 全局预测钢筋用量；
- 真实钢筋用量。

### 7.1 可行性概率校准

可行性校准采用 Platt-style logistic calibration：

```text
z = log(p / (1 - p))
P(local feasible | p) = sigmoid(a z + b)
```

当局部样本数足够，且已同时观察到可行/不可行样本时，模型开始启用。

局部筛选阈值仍然按高召回思想选取：

```text
threshold = quantile(local_feasible_probabilities, 1 - target_recall)
threshold = threshold * screening_threshold_scale
```

默认：

```text
screening_target_recall = 0.995
screening_threshold_scale = 0.1
```

因此局部校准不会激进地扩大筛选，而是保守修正全局概率。

### 7.2 钢筋残差校准

钢筋校准不直接重新预测钢筋总量，而是预测全局模型残差：

```text
residual = true_steel_kg - global_predicted_steel_kg
calibrated_steel_kg = global_predicted_steel_kg + predicted_residual
```

局部特征包括：

- 当前设计变量；
- 全局预测钢筋量；
- `log1p(global_predicted_steel_kg)`。

局部模型随样本数自动切换：

| 局部样本规模 | 校准模型 |
| --- | --- |
| 较少样本 | Ridge |
| 中等样本 | Gaussian Process |
| 较多样本 | LightGBM |

当前默认阈值：

```text
steel_min_samples = 25
steel_gpr_min_samples = 50
steel_lgbm_min_samples = 200
```

这可以避免在样本很少时使用过高容量模型。

## 8. 优化中的完整调用顺序

统一代理评估器：

```text
src/shearwall_optimization/surrogate_candidate.py
SurrogateCandidateEvaluator
```

每一批候选方案的处理顺序：

1. 修复候选变量，使其满足基本离散取值约束；
2. 可行性 GNN 批量预测可行概率；
3. 在线校准可行概率和筛选阈值；
4. 明确不可行的候选直接跳过，不进入 FEA；
5. 对剩余候选预测钢筋用量；
6. 快速估算混凝土用量与材料造价；
7. 在线校准钢筋预测残差；
8. 按材料造价和可行性风险分数排序；
9. 只对排名靠前的一部分候选进行真实 FEA；
10. 将真实 FEA 结果回流到在线校准器；
11. 优化算法继续下一代或下一批搜索。

也就是说，代理模型参与的是：

- **是否值得做 FEA**；
- **哪些候选优先做 FEA**。

最终最优方案仍然来自真实 FEA 评估结果。

## 9. 论文中建议保留的叙事

建议论文中不要把所有探索性模型都展开，而是按以下逻辑组织：

1. **全局代理模型**
   - 基于参数、布局全局特征和 room graph；
   - 分别学习可行性概率与钢筋用量。

2. **保守筛选原则**
   - 可行性模型按高 feasible recall 选择阈值；
   - 目标是尽量少误删可行候选，而不是最大化普通分类准确率。

3. **材料成本预筛**
   - 钢筋用量由代理模型预测；
   - 混凝土用量由几何快速计算；
   - 二者组合成材料造价分数，用于分配 FEA 预算。

4. **在线局部校准**
   - 全局模型适应当前布局；
   - FEA 样本被复用为局部校准数据；
   - 校准层轻量、在线、无需额外离线数据。

5. **代理辅助优化**
   - 代理模型不输出最终设计；
   - 它减少无效或低价值 FEA 调用；
   - 真实 FEA 始终作为最终评价标准。

## 10. 最容易混淆的点

### GNN 是否一定比表格模型强？

不一定。之前实验显示，参数特征和布局统计特征已经承担了很强预测能力，GNN 图结构增益未必非常大。因此论文重点不应写成“提出一个显著优于所有模型的 GNN”，而应写成：

> 构建了可用于结构优化流程的布局-参数融合代理模型，并将其与在线校准和 FEA 预算分配机制结合。

### 钢筋模型是否直接替代 FEA？

不是。钢筋模型用于排序与预筛，不用于最终验算。被选中的候选仍需要真实 FEA，真实钢筋用量也会用于更新局部校准模型。

### 可行性模型是否可以大胆筛选？

不应大胆筛选。它的论文定位是保守筛选器。核心指标是：

```text
在 feasible_recall 很高时能拒绝多少不可行候选
```

而不是普通 accuracy。

### 在线校准是否增加额外计算成本？

基本不增加 FEA 成本。它使用优化过程中已经产生的 FEA 结果，只增加轻量机器学习模型的拟合和预测开销。

