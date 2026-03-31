## Conditional Generation of Shear Wall Layouts Using Graph Neural Networks on Architectural Space Representations

---

## Abstract (150-200 words)

### 结构模板：
1. **Background (1-2句)**: 剪力墙结构对高层建筑抗震性能的重要性，传统设计依赖人工经验
2. **Problem (1句)**: 现有基于图像或几何图元的方法在捕捉空间语义和条件化控制方面存在局限
3. **Method (3-4句)**:
   - 提出基于建筑空间（房间）的图表示方法
   - 设计结合FiLM条件化机制的GATv2网络
   - 提出双流训练策略增强条件响应能力
4. **Results (2-3句)**:
   - 测试集上达到的主要指标（IoU、F1等）
   - 消融实验验证各组件有效性
   - 与baseline对比的优势
5. **Conclusion (1句)**: 本方法为AI辅助结构设计提供了新的空间级建模范式

### 关键词 (5-6个):
Shear wall layout; Graph neural networks; Conditional generation; Architectural space representation; Structural design automation

---

## 1. Introduction

### 1.1 Background (1段, ~200 words)

**内容要点：**
- 剪力墙结构是高层建筑（尤其是高层住宅）的主要抗侧力体系
- 合理的剪力墙布置直接影响建筑的抗震性能、承载能力和经济性
- 引用1-2篇权威文献说明剪力墙设计的重要性

**参考句式：**

> Shear wall systems are widely adopted in high-rise residential buildings due to their excellent lateral load resistance and seismic performance [ref]. The arrangement of shear walls significantly influences the structural stiffness distribution, load transfer paths, and construction costs [ref].

### 1.2 Problem Statement (1段, ~200 words)

**内容要点：**
- 传统剪力墙设计流程：工程师依据规范和经验手动布置 → 结构分析验证 → 迭代修改
- 存在的问题：
  - 效率低下：多次迭代消耗大量时间
  - 方案受限：难以穷举探索最优布置
  - 难以优化：人工方法无法同时兼顾多目标

**参考句式：**
> The conventional shear wall design process heavily relies on engineers' experience and iterative trial-and-error, which is time-consuming and may not yield optimal solutions. Moreover, different design conditions (e.g., seismic intensity levels) require different wall densities, further complicating the design space exploration.

### 1.3 Existing Methods and Limitations (1-2段, ~300 words)

**内容要点：**

**基于图像的方法：**

- 简述方法：CNN将建筑平面图作为图像输入，输出剪力墙分布图，实现img2img的约束映射
- 局限性：
  - 像素级表示无法显式学习构件间拓扑关系
  - 模型参数量大、信息冗余
  - 输出需要后处理才能得到精确几何位置
- 引用代表性论文

**基于几何图元的GNN方法（重点对比对象）：**

- 简述方法：以墙体线段为节点构建图，使用GNN预测每条边是否为剪力墙
- 局限性：
  - 节点为构件级元素，无法捕捉建筑空间的功能语义
  - 图规模与墙体数量成正比，大型建筑计算复杂度高
- **重点引用**：陆新征团队的相关论文（主要baseline）

**参考句式：**
> Recent studies have explored deep learning approaches for automated shear wall layout prediction. Image-based methods [ref] treat floor plans as images and use convolutional neural networks to generate heatmaps of wall distributions. However, these methods suffer from the loss of topological information and require extensive post-processing to extract precise wall geometries.
>
> Graph-based approaches [ref] represent wall segments as graph nodes and leverage graph neural networks to predict structural layouts. While these methods preserve geometric relationships, they operate at the component level (geometric primitives) and fail to capture the higher-level spatial semantics of architectural spaces. Furthermore, the graph size scales with the number of wall segments, leading to computational inefficiency for large buildings.

### 1.4 Our Approach and Contributions (1段, ~250 words)

**内容要点：**
- 核心思想：以**建筑空间（房间）**而非几何图元作为图的节点
- 简述方法框架：Room-based Graph → Conditional GNN → Dual-stream Training
- Bullet points列出主要贡献（3-4条）

**贡献列表：**
1. **Room-based graph representation**: We propose a novel graph construction approach that uses architectural spaces (rooms) as nodes, inherently preserving spatial semantics and reducing graph complexity.

2. **Conditional generation with FiLM**: We introduce Feature-wise Linear Modulation (FiLM) mechanism to enable the model to generate different wall layouts according to design conditions (seismic intensity levels).

3. **Dual-stream training strategy**: We propose a training scheme that combines supervised learning with condition-constrained generation, using fake conditions to explicitly enforce density differentiation across design categories.

4. **Comprehensive evaluation**: We conduct extensive experiments including ablation studies and baseline comparisons, demonstrating the effectiveness of each proposed component.

### 1.5 Paper Organization (1段, ~50 words)

**内容要点：**
- 简述论文结构
- Section 2: Related Work
- Section 3: Methodology
- Section 4: Experiments
- Section 5: Conclusion

---

## 2. Related Work

### 2.1 AI-based Structural Design (~300 words)

**内容要点：**

- 综述AI/ML在结构工程中的应用
- 分类讨论：
  - 结构分析加速（代理模型）
  - 结构优化
  - 结构布置/生成
- 指出本研究属于"结构布置自动化"范畴

**需要引用的方向：**
- 深度学习在结构分析中的应用
- 生成式设计在建筑/结构中的应用
- 特别是剪力墙布置相关的工作

### 2.2 Graph Neural Networks in AEC (~250 words)

**内容要点：**

- GNN的基本原理和优势（捕捉拓扑关系）
- GNN在建筑/工程/施工(AEC)领域的应用：
  - 建筑能耗预测
  - 施工进度预测
  - 建筑布局生成
  - **结构布置预测**（重点）
- 引出基于几何图元的GNN方法及其局限

**需要引用的论文：**

- GATv2原始论文
- GNN在建筑领域应用的代表性论文
- 陆新征团队的剪力墙GNN论文（核心对比对象）

### 2.3 Conditional Generation in Design (~200 words)

**内容要点：**
- 条件化生成的概念：根据输入条件控制生成结果
- 在设计领域的应用：
  - 条件图像生成
  - 条件布局生成
- FiLM机制的介绍及其在其他领域的成功应用
- 引出本研究如何将条件化生成引入结构设计

**需要引用的论文：**
- FiLM原始论文
- 条件化生成在设计/建筑领域的应用

### 2.4 Research Gap (~100 words)

**内容要点：**
- 总结现有方法的Gap：
  1. 缺乏空间语义层面的图表示
  2. 缺乏对设计条件的显式控制能力
  3. 缺乏保证条件响应的训练策略
- 引出本研究如何填补这些Gap

---

## 3. Methodology

### 3.1 Problem Formulation (~200 words)

**内容要点：**
- 数学化定义问题
- 输入：建筑平面图（DXF格式）→ 提取房间信息 → 构建图 G = (V, E)
- 条件：设计条件向量 c ∈ R³ (one-hot编码的抗震等级)
- 输出：每个房间的剪力墙分布向量 y ∈ [0,1]^(N×16)
- 目标：学习映射函数 f: (G, c) → y

**公式示例：**
```
Given: G = (V, E, X_v, X_e), c ∈ {low, medium, high}
Predict: Y = {y_1, y_2, ..., y_N}, where y_i ∈ [0,1]^16
```

### 3.2 Graph Representation (~500 words)

**3.2.1 Graph Construction Rationale**

**内容要点：**

- 解释为什么选择房间作为节点（与几何图元对比）
- 动机1：剪力墙布置受建筑功能分区约束，墙体自然位于房间边界
- 动机2：房间级抽象保留了空间语义
- 动机3：图规模可控（房间数 << 墙体数）
- 动机4：可扩展性：梁同样位于房间边界，可用同一框架

**参考句式：**
> Unlike geometric primitive-based approaches that treat wall segments as isolated elements, our room-based representation inherently preserves architectural semantics. Shear wall placement is fundamentally constrained by architectural functional requirements—walls are placed along room boundaries, not arbitrary locations.

**3.2.2 Node Features (25-dimensional)**

**内容要点（用表格呈现）：**

| 特征类别 | 维度 | 具体内容 |
|----------|------|----------|
| 几何特征 | 9 | 中心坐标(2), 面积(1), 宽高比(1), 边界框(4), 周长(1) |
| 约束特征 | 16 | 四条边的可布置区域信息 (4边×2段×2端点) |

- 解释约束特征的含义：门窗位置→不可布置区域
- 说明特征归一化方式

**3.2.3 Edge Features (8-dimensional)**

**内容要点（用表格呈现）：**

| 特征 | 维度 | 说明 |
|------|------|------|
| 共享边长度 | 1 | 两房间共享边界的长度 |
| 相对距离 | 2 | 房间中心的相对位置(dx, dy) |
| 相对方位 | 4 | One-hot编码 (上/下/左/右) |
| 共享比例 | 1 | 共享边占各自边界的比例 |

**3.2.4 Output Representation (16-dimensional)**

**内容要点：**
- 解释16维向量的物理意义
- 用图示说明：4条边 × 每边2段 × 每段2端点 = 16
- 每个值表示墙体占该半边长度的比例（0-1）
- 说明为什么采用这种表示而非其他方式

**建议配图：**
- Figure: Illustration of the 16-dimensional output vector mapping to room boundaries

### 3.3 Conditional GNN Model (~600 words)

**3.3.1 Model Architecture Overview**

**内容要点：**
- 提供架构图（Figure）
- 整体流程描述：编码 → 图卷积 → 条件调制 → 解码

**建议配图：**
- Figure: Overall architecture of the proposed conditional GNN model

**3.3.2 Feature Encoders**

**内容要点：**
- 节点编码器：Linear(25 → 256)
- 边编码器：Linear(8 → 256)
- 条件编码器：MLP(3 → 32 → 32)

**公式：**
```
h_v^(0) = ReLU(W_node · x_v)
h_e = ReLU(W_edge · x_e)
z_c = MLP(c)
```

**3.3.3 GATv2 Backbone**

**内容要点：**
- 介绍GATv2的原理和优势
- 与原始GAT的区别：动态注意力
- 本研究配置：3层、4头、hidden_dim=256
- 残差连接的使用

**公式（注意力计算）：**
```
α_ij = softmax(a^T · LeakyReLU(W · [h_i || h_j || e_ij]))
h_i' = Σ_j α_ij · W · h_j
```

**3.3.4 FiLM Conditioning Mechanism**

**内容要点：**
- 解释FiLM的原理：特征级线性调制
- 在每个GATv2层后应用FiLM
- 使条件信息能够调控特征变换

**公式：**
```
γ, β = Linear(z_c)
h' = h ⊙ (1 + γ) + β
```

**参考句式：**
> Feature-wise Linear Modulation (FiLM) provides an effective mechanism for conditioning neural network computations [ref]. By applying learned scaling (γ) and shifting (β) parameters derived from the condition vector, FiLM enables the model to dynamically adjust its feature representations according to design requirements.

**3.3.5 Dual-Head Output**

**内容要点：**
- 分类头：预测墙体存在性（Binary）
- 回归头：预测墙体长度比例（Continuous）
- 最终输出 = 分类概率 × 回归比例

**公式：**
```
p = Sigmoid(MLP_cls(h))  ∈ [0,1]^16
r = Sigmoid(MLP_reg(h))  ∈ [0,1]^16
y_pred = p ⊙ r
```

### 3.4 Training Strategy (~500 words)

**3.4.1 Dual-Stream Training**

**内容要点：**
- 解释双流训练的动机：单纯监督学习可能忽略条件信息
- 监督流：标准的真实数据训练
- 约束流：使用假条件强制密度差异化

**建议配图：**
- Figure: Illustration of the dual-stream training strategy

**伪代码/流程图：**

```
for each batch:
    # Supervised Stream
    pred_real = model(batch, real_condition)
    loss_supervised = hybrid_loss(pred_real, gt)
    loss_density_real = density_loss(pred_real, real_condition)

    # Constraint Stream (after warmup)
    if epoch > warmup_epochs:
        fake_condition = random_sample(conditions)
        pred_fake = model(batch, fake_condition)
        loss_density_fake = density_loss(pred_fake, fake_condition)

    loss_total = loss_supervised + λ_real * loss_density_real + λ_fake * loss_density_fake
```

**3.4.2 Hybrid Loss Function**

**内容要点：**

- 各项损失的数学定义
- 各项损失的作用解释

**公式和表格：**

| 损失项 | 公式 | 权重 | 作用 |
|--------|------|------|------|
| BCE | BCE(p, y>0) | 1.0 | 分类准确性 |
| MSE | MSE(r, y) on y>0 | 2.0 | 回归精度 |
| IoU | 1 - VectorIoU(r, y) | 2.0 | 向量级重叠 |
| Consistency | MSE(shared_edges) | 0.5 | 邻接一致性 |
| Density | MSE(avg_density, target) | 0.05/w | 条件控制 |

**3.4.3 Data Augmentation**

**内容要点：**
- 6种几何变换：原图、flip_x、flip_y、rot_90、rot_180、rot_270
- 为什么这些增广有效：剪力墙布置规律与绝对方位无关
- 增广后的数据量

**3.4.4 Implementation Details**

**内容要点：**
- 优化器：AdamW (lr=1e-3, weight_decay=1e-4)
- Batch size：1（因为图规模不同）
- Epochs：100
- K-Fold：5折交叉验证
- Warmup：前20个epoch不使用约束流
- 硬件：GPU型号

---

## 4. Experiments

### 4.1 Experimental Setup (~300 words)

**4.1.1 Dataset**

**内容要点：**
- 数据来源：真实高层住宅建筑CAD图纸
- 数据格式：DXF文件
- 数据规模：N张图纸（说明具体数量）
- 标注内容：房间边界、剪力墙位置、设计条件
- 数据划分：训练集/测试集比例

**建议表格：**

| 项目 | 训练集 | 测试集 |
|------|--------|--------|
| 原始图纸数 | X | Y |
| 增广后样本数 | X×6 | Y |
| 平均房间数/图 | ~Z | ~Z |

**4.1.2 Baseline Methods**

**内容要点：**
- Baseline 1：基于几何图元的GNN方法（陆新征团队）
- 简述baseline实现细节
- 说明对比的公平性（相同数据、相同评估方式）

**4.1.3 Implementation Details**

**内容要点：**
- 开发框架：PyTorch + PyTorch Geometric
- 硬件配置：GPU型号、显存
- 训练时间：约X分钟/epoch

### 4.2 Evaluation Metrics (~200 words)

**4.2.1 Quantitative Metrics**

**内容要点（用表格呈现）：**

| 指标 | 公式/定义 | 评估目标 |
|------|----------|----------|
| Vector IoU | IoU(pred_vector, gt_vector) | 位置和长度的综合准确性 |
| Precision | TP / (TP + FP) | 预测为有墙的正确率 |
| Recall | TP / (TP + FN) | 实际有墙的检出率 |
| F1 Score | 2×P×R / (P+R) | 分类综合指标 |
| MAE | Mean|pred - gt| on valid | 长度预测误差 |

**4.2.2 Conditional Generation Score (CGS)**

**内容要点：**

- 密度一致性：预测密度与条件目标的匹配度
- 条件区分度：不同条件下预测的差异性
- 综合评分计算方式

**4.2.3 Qualitative Evaluation**

**内容要点：**

- 工程师问卷评估（如果有）
- 可视化案例分析
- 工程可行性分析（建模分析）

### 4.3 Main Results (~300 words)

**内容要点：**

- 测试集上的综合评估结果
- 与baseline的对比

**建议表格：**

| Method | IoU | Precision | Recall | F1 | MAE |
|--------|-----|-----------|--------|-----|-----|
| Baseline (Geometric GNN) | X.XX | X.XX | X.XX | X.XX | X.XX |
| **Ours (Room-based GNN)** | **X.XX** | **X.XX** | **X.XX** | **X.XX** | **X.XX** |

**讨论要点：**
- 主要指标的提升幅度
- 分析提升的原因（空间语义、条件控制等）

### 4.4 Ablation Study (~400 words)

**4.4.1 Backbone Comparison**

**内容要点：**

- 对比GATv2、GCN、GraphSAGE、GIN
- 验证GATv2的选择合理性

**建议表格：**

| Backbone | IoU | F1 | CGS |
|----------|-----|----|-----|
| GCN | X.XX | X.XX | X.XX |
| GraphSAGE | X.XX | X.XX | X.XX |
| GIN | X.XX | X.XX | X.XX |
| **GATv2** | **X.XX** | **X.XX** | **X.XX** |

**4.4.2 Conditioning Method Comparison**

**内容要点：**

- 对比FiLM、Concat-Early、Concat-Late、None
- 验证FiLM的有效性

**建议表格：**

| Conditioning | IoU | CGS |
|--------------|-----|-----|
| None | X.XX | X.XX |
| Concat-Early | X.XX | X.XX |
| Concat-Late | X.XX | X.XX |
| **FiLM** | **X.XX** | **X.XX** |

**4.4.3 Loss Function Ablation**

**内容要点：**
- 各项损失的贡献分析
- 移除某项损失后性能变化

**建议表格：**

| Configuration | IoU | F1 |
|---------------|-----|----|
| Full Model | X.XX | X.XX |
| w/o BCE Loss | X.XX | X.XX |
| w/o MSE Loss | X.XX | X.XX |
| w/o IoU Loss | X.XX | X.XX |
| w/o Consistency Loss | X.XX | X.XX |
| w/o Density Loss | X.XX | X.XX |

**4.4.4 Training Strategy Ablation**

**内容要点：**

- 对比双流训练 vs 仅监督学习
- Warmup的影响
- 加权采样的影响

### 4.5 Qualitative Analysis (~300 words)

**4.5.1 Visualization Examples**

**内容要点：**

- 选取代表性案例
- 对比展示：GT vs Ours vs Baseline
- 分析预测质量

**建议配图：**

- Figure: Qualitative comparison of shear wall predictions
  - 子图1：建筑平面图
  - 子图2：Ground Truth
  - 子图3：Our Prediction
  - 子图4：Baseline Prediction

**4.5.2 Conditional Generation Demonstration**

**内容要点：**

- 同一建筑在不同条件下的预测结果
- 展示条件化控制能力

**建议配图：**

- Figure: Predictions under different design conditions
  - 子图1：Low seismic intensity
  - 子图2：Medium seismic intensity
  - 子图3：High seismic intensity

**4.5.3 Engineering Case Study (可选)**

**内容要点：**

- 选取典型工程案例
- 将预测结果建模并进行结构分析
- 验证方案的工程可行性

---

## 5. Conclusion

### 5.1 Summary (~150 words)

**内容要点：**

- 简要重述研究目标和方法
- 强调核心创新：房间级图表示、FiLM条件化、双流训练

**参考句式：**

> In this paper, we proposed a novel graph neural network approach for conditional shear wall layout generation. Unlike existing methods that operate on geometric primitives, our method constructs graphs based on architectural spaces (rooms), preserving spatial semantics while reducing computational complexity. We introduced FiLM conditioning mechanism and dual-stream training strategy to enable controllable generation according to design requirements.

### 5.2 Key Findings (~150 words)

**内容要点（Bullet points）：**

1. 房间级图表示相比几何图元方法在IoU上提升X%
2. FiLM条件化机制使模型能够生成差异化密度的布置方案
3. 双流训练策略显著增强条件响应能力（CGS提升X%）
4. 各消融实验验证了每个组件的必要性

### 5.3 Limitations (~100 words)

**内容要点：**
- 条件控制粒度粗糙：仅使用3类抗震等级，无法进行更精细控制
- 空间形状限制：当前仅处理矩形房间，无法直接处理L形、多边形空间
- 未显式考虑结构性能：模型仅从数据学习，未引入力学约束

### 5.4 Future Work (~100 words)

**内容要点：**
- 更精细的条件控制：使用连续的建筑高度、烈度值
- 异形空间扩展：处理非矩形房间
- 结构性能约束：引入位移、应力等分析结果作为训练约束
- 梁柱布置扩展：利用相同框架预测梁柱布置

---

## References

**需要引用的关键论文（分类整理）：**

### 剪力墙/结构设计相关
- 陆新征团队的GNN剪力墙论文（核心baseline）
- 其他AI辅助结构设计的论文

### GNN相关
- GATv2原始论文
- GCN原始论文
- Message Passing Neural Networks

### 条件化生成相关
- FiLM原始论文
- Conditional generation综述

### 建筑/AEC领域AI应用
- GNN在建筑领域应用的论文
- 生成式设计相关论文

---

## Appendix (可选)

### A. Dataset Details
- 数据预处理流程详细说明
- 特征提取的具体实现

### B. Additional Results
- 更多可视化案例
- 不同超参数的敏感性分析

---

## 写作建议

### 写作顺序
1. **第一步**：Experiments (4) - 先有数据再写
2. **第二步**：Methodology (3) - 解释代码逻辑
3. **第三步**：Related Work (2) - 有针对性地综述
4. **第四步**：Introduction (1) - 知道做了什么再写intro
5. **第五步**：Conclusion (5) - 总结全文
6. **第六步**：Abstract - 最后提炼

### 图表清单
1. Figure 1: Overall framework of the proposed method
2. Figure 2: Room-based graph construction illustration
3. Figure 3: Model architecture diagram
4. Figure 4: Dual-stream training strategy
5. Figure 5: Qualitative comparison of predictions
6. Figure 6: Conditional generation demonstration
7. Table 1: Dataset statistics
8. Table 2: Main results comparison with baseline
9. Table 3: Ablation study - backbone comparison
10. Table 4: Ablation study - conditioning method
11. Table 5: Ablation study - loss functions

### 目标期刊格式
- Engineering Structures: 单栏，Word或LaTeX
- Advanced Engineering Informatics: 双栏，LaTeX preferred
- 下载目标期刊模板，直接在模板中写作

### 字数估计
- Abstract: 150-200 words
- Introduction: ~1000 words
- Related Work: ~800 words
- Methodology: ~1800 words
- Experiments: ~1500 words
- Conclusion: ~500 words
- **Total: ~6000 words** (符合工程类期刊标准)



