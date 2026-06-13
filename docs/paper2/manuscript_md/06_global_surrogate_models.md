# 2.3 Global Surrogate Models

本文使用两个全局代理模型分别近似结构综合可行性和钢筋用量。两个模型采用相同的输入表征和 LayoutParamGNN 主体架构，仅在输出任务和损失函数上不同。给定样本 $z=\{G,q,p\}$，其中 $G$ 为 room graph，$q$ 为全局布局特征，$p$ 为截面、材料和设计条件参数，模型首先学习布局参数联合表示 $h$，再通过任务头输出可行性概率或钢筋用量估计。

## 2.3.1 Unified LayoutParamGNN Architecture

建筑布局被表示为 room graph $G=(V,E)$。图节点 $v_i \in V$ 对应平面中的房间或空间单元，图边 $e_{ij}\in E$ 表示空间单元之间的相邻关系。每个节点携带局部几何和空间属性，边特征描述相邻空间之间的关系。与图结构并行，模型还接收一组布局级统计特征 $q$，用于补充全局尺度、墙体分布、梁墙关系和整体拓扑摘要。

图编码器首先将节点特征投影到隐空间：

$$
h_i^{(0)} = W_x x_i + b_x .
$$

随后通过 $K$ 层图卷积更新节点表示。本文实现中可采用 GraphSAGE 或 GINE 作为消息传递算子，其一般形式可写为

$$
\tilde{h}_i^{(k)} =
\mathrm{GNN}^{(k)}\left(h_i^{(k-1)}, \{h_j^{(k-1)}, e_{ij}: j\in \mathcal{N}(i)\}\right),
$$

$$
h_i^{(k)} = \sigma\left(\mathrm{BN}(\tilde{h}_i^{(k)}) + h_i^{(k-1)}\right),
$$

其中 $\mathcal{N}(i)$ 为节点 $i$ 的邻居集合，$\mathrm{BN}$ 表示批归一化，残差连接用于稳定多层图编码。完成节点更新后，采用 mean pooling 和 max pooling 得到图级节点表示：

$$
g_{\mathrm{mean}}=\frac{1}{|V|}\sum_{i\in V}h_i^{(K)}, \quad
g_{\mathrm{max}}=\max_{i\in V}h_i^{(K)} .
$$

边特征通过独立的多层感知机编码，并在图级进行平均池化：

$$
g_e=\frac{1}{|E|}\sum_{e_{ij}\in E}\psi_e(e_{ij}).
$$

设计参数 $p$ 由数值参数和类别参数组成。数值参数经过标准化处理，类别参数通过 embedding 映射为连续向量。二者拼接后输入参数编码器：

$$
h_p=\psi_p([p_{\mathrm{num}}, \mathrm{Emb}(p_{\mathrm{cat}})]) .
$$

最终，模型将节点池化表示、边表示、全局布局特征和参数表示拼接，形成候选方案的联合表示：

$$
h=\phi([g_{\mathrm{mean}},g_{\mathrm{max}},g_e,q,h_p]).
$$

其中 $\phi$ 为融合网络。该联合表示同时包含布局拓扑、全局几何统计和当前设计参数，因此可作为可行性分类和钢筋用量回归的共同输入。

// 正式英文稿中这一小节可以配模型结构图。图中应显示 room graph encoder、parameter encoder、global layout features 和 fusion head 的数据流，并在图注中说明 node mean/max pooling、edge pooling 和 task-specific heads。

## 2.3.2 Feasibility Screening Surrogate

可行性代理模型以联合表示 $h$ 为输入，输出候选方案通过综合校核的概率：

$$
\hat{p}_f=\sigma(w_f^\top h+b_f),
$$

其中 $\sigma(\cdot)$ 为 sigmoid 函数。监督标签为 $y_f\in\{0,1\}$，表示候选方案是否通过分析收敛性、结构响应限值和构件设计校核。由于可行样本和不可行样本通常不平衡，训练时采用带类别权重的 binary cross entropy loss：

$$
\mathcal{L}_{f}
=-\omega y_f \log \hat{p}_f -(1-y_f)\log(1-\hat{p}_f),
$$

其中 $\omega$ 根据训练集中正负样本比例确定，用于提高模型对少数类样本的学习权重。

在优化过程中，可行性代理模型作为筛选器使用。给定阈值 $\tau_f$，若 $\hat{p}_f<\tau_f$，候选方案被判定为低可行性方案，可跳过真实有限元分析或降低其进入分析队列的优先级。该阈值不按普通分类任务中的默认阈值选取，而是在验证集上根据筛选目标确定：

$$
\max_{\tau_f} \quad R_{\mathrm{reject}}(\tau_f)
\quad
\mathrm{s.t.}\quad
R_{\mathrm{feasible}}(\tau_f)\ge r_0 ,
$$

其中 $R_{\mathrm{reject}}$ 表示被代理模型筛掉的候选比例，$R_{\mathrm{feasible}}$ 表示真实可行样本中被保留的比例，$r_0$ 为目标可行召回率。该策略将模型从普通分类器转化为保守筛选器，使其主要承担减少明显不可行候选 FEA 调用的作用。

// 实验部分应报告两类指标：分类指标反映模型本身性能，如 ROC-AUC、PR-AUC、F1；筛选指标反映优化使用价值，如 reject rate、screen recall 或 feasible recall、false reject count。方法部分只需定义这些量的用途，不必提前给结果。

## 2.3.3 Steel Usage Surrogate and Cost Score

钢筋用量代理模型使用相同的联合表示 $h$，输出钢筋用量估计。由于钢筋用量分布具有明显偏态，训练时对目标进行对数变换和标准化：

$$
y_s=\frac{\log(1+m_s)-\mu_s}{\sigma_s},
$$

其中 $\mu_s$ 和 $\sigma_s$ 由训练集统计得到。模型输出标准化空间中的预测值：

$$
\hat{y}_s=w_s^\top h+b_s.
$$

训练损失采用 Smooth L1 loss：

$$
\mathcal{L}_s=\mathrm{SmoothL1}(\hat{y}_s,y_s).
$$

推理时，预测结果通过反变换得到钢筋用量：

$$
\hat{m}_s=\exp(\hat{y}_s\sigma_s+\mu_s)-1 .
$$

钢筋代理模型不直接给出最终设计结果，而是用于有限元分析前的候选排序。混凝土用量 $\tilde{m}_c$ 可由布局几何、楼层数和截面尺寸快速估计，因此候选方案的代理材料造价可写为

$$
\hat{C}=p_c\tilde{m}_c+p_s\hat{m}_s .
$$

为了在排序中同时考虑经济性和可行性风险，本文进一步构造候选评分

$$
S=\hat{C}+\eta \rho(\hat{p}_f),
$$

其中 $\rho(\hat{p}_f)$ 为由可行性概率得到的风险项，$\eta$ 为风险惩罚权重。优化算法在一批候选方案中优先选择评分较低者进入真实有限元分析。经过真实分析的候选方案再以其真实可行性和材料造价更新优化器；未进入真实分析的候选方案不作为最终设计结果。

// 如果正式论文中篇幅允许，可以在这里进一步定义 $\rho(\hat{p}_f)$ 的具体形式，例如 $\max(0, r_h-\hat{p}_f)$ 或 $1-\hat{p}_f$。若不同实验使用了局部校准后的概率和钢筋残差，则在 2.4 中再给出校准后的 $\hat{p}_f^{cal}$ 和 $\hat{m}_s^{cal}$，避免 2.3 与 2.4 内容重复。
