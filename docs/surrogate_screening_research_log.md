# 剪力墙结构可行性筛选模型研究记录

本文记录 surrogate 可行性预测模型从二分类 GNN 到优化预筛选模型的实验过程、问题诊断、改进路径和当前推荐策略。

## 1. 任务目标

原始任务是预测一个剪力墙结构方案在给定布局、截面、材料和抗震条件下是否满足规范校核，即 `final_pass` / `feasible` 类二分类任务。

后续结合优化使用场景后，目标调整为：

- 在结构优化阶段快速筛掉明显不可行的截面方案；
- 尽量减少真实可行方案被误筛掉的概率；
- 允许模型保留不确定样本进入 FEM 或更精细校核；
- 评估重点从 F1 转为高可行解召回率下的过滤率。

因此最终更关心：

- `feasible_recall`：真实可行样本中被保留的比例；
- `false_reject_count` / `false_reject_rate`：被误筛掉的可行样本数量和比例；
- `reject_rate`：被模型提前过滤掉的样本比例；
- `reject_infeasible_precision`：被过滤样本中真实不可行的比例。

## 2. 既有表格模型基线

已有的手工布局特征 + kfold MLP ensemble 在测试集上的表现为：

```json
{
  "roc_auc": 0.9717500304600232,
  "pr_auc": 0.879501366693711,
  "f1": 0.7845195681144442,
  "precision": 0.7224866002103892,
  "recall": 0.8582054028323218,
  "balanced_accuracy": 0.8993784820793223,
  "brier": 0.0634453287931932
}
```

LightGBM/GBDT 给出的重要特征包括：

- `beam_wall_length_ratio`
- `N`
- `site_class_I0`
- `tw_bot`
- `seismic_group_1`
- `site_class_IV`
- `intensity_6.0`
- `beam_wall_count_ratio`
- `boundary_wall_ratio`
- `wall_xy_ratio`

这些特征具有明确结构工程含义，说明手工统计特征已经捕获了大量决定性信息。

## 3. 房间图 GNN 表征尝试

最初的 GNN 图表征是将结构构件作为图节点。后来新增了房间图表征：

- 节点：房间；
- 边：相邻房间共享边；
- 节点特征：房间归一化几何、可布置约束、剪力墙分布向量；
- 边特征：共享边长度、重合位置、相对方向等；
- 图来源：`data\dxf\cad_json_data\fem_raw` 中 json 文件名映射到 `data\dxf\fem_raw` 中同名 dxf 文件，再通过 `build_graph_from_dxf` 构图。

相关提交：

```text
b601c62 Add room graph representation for surrogate GNN
```

初次训练结果异常差，F1 约 0.25。排查发现不是房间图本身无效，而是全局图特征尺度错误。

## 4. 关键问题一：图全局特征尺度爆炸

初版房间图缓存中 `graph_feat` 包含原始毫米尺度和面积：

```text
scale max      ~= 8.0e4
total_area max ~= 1.94e9
mean_area max  ~= 1.21e8
```

这些特征直接拼接到分类器输入，没有标准化，导致第一轮训练 loss 爆炸：

```text
epoch=1 train_loss=52756
```

修正方式：

- 将 `graph_feat` 改为无量纲比例特征；
- 使用节点数比例、边数比例、连接密度、归一化面积、归一化剪力墙量；
- 给 room graph cache 加 `feature_version=room_v2`，旧缓存自动重建。

相关提交：

```text
c2bde8d Normalize room graph global features
```

修正后房间图 GNN 回到正常训练区间。

## 5. GNN 分类实验迭代

### 5.1 Room + SAGE

修正特征后，room 图 + SAGE 的结果：

```text
val F1     ~= 0.701
test F1    ~= 0.753
test PR-AUC~= 0.824
```

虽然明显好于错误版本，但仍低于表格 kfold MLP。

### 5.2 Edge-aware GINE

考虑房间图边特征包含共享边长度、相对方位等重要信息，新增了 GINE 分支，让边特征进入 message passing。

相关提交：

```text
c9eb91d Add edge-aware GNN option
```

实验结果：

```text
test F1 ~= 0.736
```

GINE 不如 SAGE。推测原因是当前布局图数量只有 143 个，边特征进入 message passing 后更容易过拟合或训练不稳定；而边特征的全局平均已经提供了一部分信息。

### 5.3 按 F1 选择 checkpoint

原训练流程按验证集 PR-AUC 保存 checkpoint，但用户关心的是可行性分类效果，因此增加 `monitor_metric=f1`。

相关提交：

```text
4360fc7 Allow F1-based GNN checkpoint selection
```

结果小幅提升：

```text
test F1     ~= 0.763
test PR-AUC ~= 0.840
```

### 5.4 学习率调度

新增可选 ReduceLROnPlateau。

相关提交：

```text
f0636c1 Add optional GNN learning rate scheduler
```

结果：

```text
test F1     ~= 0.763
test PR-AUC ~= 0.851
```

F1 基本持平，PR-AUC 有所提升。

## 6. 关键问题二：纯图没有同等信息量

纯 GNN 只使用：

- 参数特征 `PARAM_FEATURES`；
- 房间图缓存中的几何/邻接特征。

但表格模型还使用了 `LAYOUT_FEATURES` 中的大量强布局统计特征，例如：

- `beam_wall_length_ratio`
- `beam_wall_count_ratio`
- `wall_xy_ratio`
- `boundary_wall_ratio`
- `Ixx_wall`
- `Iyy_wall`
- `J_wall`

这些特征是结构工程先验的浓缩表达。让纯 GNN 只靠 143 个不同布局图重新学出这些物理统计量并不现实。

因此新增 hybrid GNN：房间图 embedding + 参数特征 + 手工布局统计特征。

相关提交：

```text
f266d8c Add layout scalar features to surrogate GNN
```

调参后较好结果：

```text
test F1     ~= 0.7799
test PR-AUC ~= 0.8677
test ROC-AUC~= 0.9713
```

已经接近 kfold MLP ensemble，但仍没有稳定超过它。

## 7. 关键诊断：二分类目标不适合边界样本

对测试集 margin 进行分析发现，样本大量集中在规范边界附近：

- `min_margin < 0` 的样本大多明显不可行；
- `min_margin >= 0` 但接近 0 的样本最难区分；
- 二分类标签把刚好过和刚好不过硬切开，导致模型在边界附近天然不稳定。

典型诊断结果：

```text
margin[-0.05, 0)  : 真实不可行，误差约 2%
margin[0, 0.05)   : 贴边可行/不可行混合，误差约 13%
```

这说明继续追求全局 F1 并不一定服务于优化使用场景。更合理的方式是设置灰区：

- 明显不可行：直接筛掉；
- 明显可行或不确定：保留进入 FEM 或更精细模型。

## 8. 评估目标切换：从 F1 到预筛选效用

新增筛选评估脚本：

```text
scripts/surrogate/evaluate_screening.py
```

相关提交：

```text
687b351 Add feasibility screening evaluation script
```

该脚本在验证集上搜索阈值，使可行解召回率不低于目标值，同时最大化过滤率，然后在测试集上评估：

- `reject_rate`
- `keep_rate`
- `feasible_recall`
- `false_reject_rate`
- `false_reject_count`
- `reject_infeasible_precision`
- `kept_feasible_rate`

这个指标更符合优化预筛选目标。

## 9. Oracle 上界分析

用真实 `min_margin = min(margin_drift, margin_torsion, margin_shear, margin_stiffness)` 做 oracle 筛选：

```text
规则：min_margin < 0 直接筛掉
test reject_rate     ~= 55.6%
test feasible_recall = 100%
false_reject_count   = 0
```

这给出了当前数据上的安全筛选上界：如果只筛掉真实负 margin 样本，最多大约能过滤 55% 左右，同时不损失可行解。

因此一个模型若能在高可行解召回率下过滤 45%-53% 样本，已经接近这个目标的有效上界。

## 10. 筛选目标训练

新增 `monitor_metric=screen_reject`，训练时不再按 F1/PR-AUC 保存模型，而是在验证集上：

1. 搜索满足 `feasible_recall >= target_recall` 的筛选阈值；
2. 选择 `reject_rate` 最大的 checkpoint；
3. 将 `screening_threshold` 保存到模型 artifact。

相关提交：

```text
dd7e6c4 Select GNN checkpoints by screening utility
```

同时推理接口直接输出筛选决策：

```text
pred_final_pass_prob
pred_final_pass
screening_threshold
pred_screen_reject
```

相关提交：

```text
7ec6b13 Expose GNN screening decisions in inference
```

## 11. 当前推荐模型

### 11.1 安全默认版：screen995

模型目录：

```text
data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1
```

训练配置要点：

```text
graph_repr=room
conv_type=sage
hidden_dim=256
batch_size=512
lr=0.0002
dropout=0.35
weight_decay=0.0003
monitor_metric=screen_reject
screening_target_recall=0.995
```

测试集筛选结果：

```text
screening_threshold          = 0.004
test reject_rate             = 47.36%
test feasible_recall         = 99.982%
test false_reject_count      = 3 / 16806
test false_reject_rate       = 0.018%
reject_infeasible_precision  = 99.994%
```

该模型适合作为优化阶段默认安全预筛选器：过滤接近一半候选方案，同时几乎不损失真实可行解。

### 11.2 更激进版：screen99

模型目录：

```text
data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen99_v1
```

测试集筛选结果：

```text
screening_threshold          = 0.012
test reject_rate             = 52.71%
test feasible_recall         = 99.935%
test false_reject_count      = 11 / 16806
test false_reject_rate       = 0.065%
reject_infeasible_precision  = 99.981%
```

该模型适合更看重优化效率、允许极少量可行方案被错筛的场景。

## 12. 使用方式

推荐先使用安全默认版：

```powershell
python scripts\surrogate\predict_gnn_final_pass.py `
  --artifact-path data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt `
  --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
  --output-path data\parametric\surrogate_dataset\screening_predictions.parquet `
  --batch-size 512
```

输出中：

```text
pred_screen_reject = True
```

表示该方案可作为明显不可行方案提前筛掉。

在优化流程中建议使用：

```text
if pred_screen_reject:
    跳过 FEM
else:
    进入 FEM 或后续精细校核
```

不要直接把 `pred_final_pass` 当最终结构可行性判定。

## 13. 当前结论

1. 单纯追求二分类 F1 不适合当前任务。
   数据大量集中在规范边界附近，硬二分类会放大边界噪声。

2. 纯 GNN 没有稳定超过手工特征模型。
   主要原因是真实图拓扑只有 143 个布局，图样本数量有限；而手工布局统计特征已经很强。

3. Hybrid GNN 是更合理方向。
   图结构提供布局拓扑信息，手工特征提供物理统计先验，两者结合后效果接近强表格基线。

4. 对优化预筛选来说，`screen_reject` 目标比 F1 更合适。
   当前模型已经能在几乎不损失可行解的前提下过滤 47%-53% 候选方案。

5. 当前最推荐部署 `screen995_v1`。
   它测试集只错筛 3 个真实可行样本，风险更低。

## 14. 后续研究方向

### 14.1 多任务 margin 预测

后续最值得继续推进的是多任务模型：

- 分类头：`final_pass`
- 回归头：`margin_drift`
- 回归头：`margin_shear`
- 可选回归头：`margin_stiffness`
- 谨慎处理：`margin_torsion`

已有 LightGBM 回归结果显示：

- `margin_drift`、`margin_shear` 可预测性较强；
- `margin_torsion` R2 很差，不适合简单取最小值参与硬筛选。

因此多任务模型应对不同 margin 使用不同权重，避免 torsion 噪声拖垮筛选。

### 14.2 主动学习

优化过程中应记录：

- 被模型保留但 FEM 失败的样本；
- 被模型低置信判断的样本；
- 接近筛选阈值的样本；
- 特定难布局中的样本。

将这些样本补充 FEM 后增量训练，会比随机新增样本更有效。

### 14.3 布局级困难样本分析

已有错误集中布局包括：

- `L1L28_30`
- `L1L28_74`
- `L1L28_25`
- `L17_226`
- `L1L28_47`

这些布局可能存在特殊拓扑、异常边界或样本分布偏移，应单独可视化和诊断。

### 14.4 保守集成策略

简单概率平均不适合高召回筛选，因为低概率尾部校准不稳定。

更适合的集成方式是保守 AND 策略：

```text
只有多个模型都判定明显不可行时才筛掉
```

不过当前两个 seed 的集成没有明显超过最佳单模型，后续需要更多 seed 或不同模型族验证。

## 15. 实验提交列表

```text
b601c62 Add room graph representation for surrogate GNN
c2bde8d Normalize room graph global features
c9eb91d Add edge-aware GNN option
4360fc7 Allow F1-based GNN checkpoint selection
f0636c1 Add optional GNN learning rate scheduler
f266d8c Add layout scalar features to surrogate GNN
687b351 Add feasibility screening evaluation script
dd7e6c4 Select GNN checkpoints by screening utility
7ec6b13 Expose GNN screening decisions in inference
```

