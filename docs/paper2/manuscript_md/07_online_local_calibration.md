# 2.4 Online Local Calibration

全局代理模型在多个布局和参数组合上训练，能够提供跨样本空间的初始预测能力。然而，结构优化过程通常固定在单个布局 $L$ 和给定设计条件 $c$ 下进行，搜索区域也会随着优化迭代逐渐集中。此时，优化过程中已经完成的真实有限元分析结果构成了一组局部样本，可用于修正全局代理模型在当前布局和当前搜索区域中的偏差。

本文引入在线局部校准器 $\mathcal{C}_t$。在第 $t$ 次迭代前，局部校准样本集记为

$$
\mathcal{D}_t^{loc}
=\{(x_i,\hat{p}_{f,i},\hat{m}_{s,i},y_{f,i},m_{s,i})\}_{i=1}^{n_t},
$$

其中 $x_i$ 为已经真实评价过的候选方案，$\hat{p}_{f,i}$ 和 $\hat{m}_{s,i}$ 分别为全局代理模型给出的可行性概率和钢筋用量预测，$y_{f,i}$ 和 $m_{s,i}$ 为真实有限元分析与设计校核得到的标签。局部校准器不替代全局代理模型，而是在全局预测结果之上学习当前优化问题中的概率偏差和钢筋残差。

## 2.4.1 Feasibility Probability Calibration

可行性校准用于修正全局可行性代理模型在当前布局上的概率偏差。设全局模型对候选方案 $x$ 输出概率 $\hat{p}_f$，先将其转换为 logit 特征：

$$
\ell(\hat{p}_f)=\log \frac{\hat{p}_f}{1-\hat{p}_f}.
$$

当局部样本数量足够且同时包含可行与不可行样本时，校准器在 $\mathcal{D}_t^{loc}$ 上拟合一个 Platt-style logistic model：

$$
\hat{p}_f^{cal}
=\sigma(a\ell(\hat{p}_f)+b),
$$

其中 $a$ 和 $b$ 由局部样本学习得到。校准后的概率 $\hat{p}_f^{cal}$ 用于后续候选筛选。

局部筛选阈值同样由局部样本确定。令 $\mathcal{P}_t^+=\{\hat{p}_{f,i}^{cal}: y_{f,i}=1\}$ 表示局部可行样本的校准概率集合。给定目标可行召回率 $r_0$，原始阈值可取为

$$
\tau_t^{raw}=Q_{1-r_0}(\mathcal{P}_t^+),
$$

其中 $Q_{1-r_0}$ 表示分位数函数。为保持筛选保守性，最终阈值可进一步缩放为

$$
\tau_t^{cal}=\alpha \tau_t^{raw},
$$

其中 $\alpha \in (0,1]$ 为阈值缩放系数。对于新候选方案，若 $\hat{p}_f^{cal}<\tau_t^{cal}$，则其被视为低可行性候选，可跳过真实有限元分析或降低优先级。当局部样本不足或仅包含单一类别时，校准器退回使用全局模型概率和全局筛选阈值。

## 2.4.2 Steel Residual Calibration

钢筋校准用于修正全局钢筋代理模型在当前布局和局部搜索区域中的系统误差。本文不重新训练全局 GNN，而是在局部样本上学习真实钢筋用量与全局预测之间的残差：

$$
\Delta m_s = m_s-\hat{m}_s .
$$

局部残差模型以候选设计参数和全局钢筋预测为输入：

$$
\widehat{\Delta m}_s
=r_t(x,\hat{m}_s).
$$

校准后的钢筋用量估计为

$$
\hat{m}_s^{cal}
=\max(\hat{m}_s+\widehat{\Delta m}_s,0).
$$

这种残差校准思想与 residual learning 和 gradient boosting 中的逐步误差修正类似。全局钢筋代理模型提供基准预测，局部残差模型只学习当前布局和局部搜索区域中的剩余误差，因此不需要重新学习完整的钢筋用量映射。残差模型可以采用较简单的回归器实现，其输入主要由设计变量和全局钢筋预测组成。校准后的钢筋预测主要用于候选造价排序和 FEA 预算分配，而不作为最终材料用量结果。

结合快速估计的混凝土用量 $\tilde{m}_c$，校准后的候选造价分数可写为

$$
\hat{C}^{cal}=p_c\tilde{m}_c+p_s\hat{m}_s^{cal}.
$$

若同时考虑可行性风险，则用于排序的评分为

$$
S^{cal}=\hat{C}^{cal}+\eta \rho(\hat{p}_f^{cal}),
$$

其中 $\rho(\hat{p}_f^{cal})$ 为低可行性风险惩罚项。该评分只决定候选方案进入真实有限元分析的优先级。

## 2.4.3 Online Update During Optimization

在线局部校准嵌入候选评估流程中。对于优化器在第 $t$ 轮生成的一批候选方案 $\mathcal{B}_t$，首先由全局代理模型计算 $\hat{p}_f$ 和 $\hat{m}_s$。若局部校准器已经可用，则进一步得到 $\hat{p}_f^{cal}$ 和 $\hat{m}_s^{cal}$。随后，可行性校准概率用于筛掉明显不可行候选，钢筋残差校准结果与混凝土估计共同形成候选排序分数。

经过筛选和排序后，仅有一部分候选方案 $\mathcal{S}_t\subseteq \mathcal{B}_t$ 被送入真实有限元分析和设计校核：

$$
(y_f,m_s,m_c,r)=\mathcal{E}(L,c,x), \quad x\in \mathcal{S}_t .
$$

真实评价结果用于更新优化器状态，并被加入局部校准样本集：

$$
\mathcal{D}_{t+1}^{loc}
=\mathcal{D}_{t}^{loc}
\{(x,\hat{p}_f,\hat{m}_s,y_f,m_s):x\in \mathcal{S}_t\}.
$$

未进入真实有限元分析的候选方案只影响当前批次的预算分配，不被作为真实训练标签加入局部校准集。通过这种在线更新机制，代理模型在优化早期主要依赖全局预测；随着优化过程不断产生新的 FEA 样本，局部校准器逐步提升可行性概率和钢筋残差预测在当前布局上的准确性。更准确的局部校准进一步改善后续批次的筛选和排序，使有限元预算更集中地分配给高价值候选方案。优化搜索和局部校准由此形成相互促进的闭环：优化过程提供局部真实样本，局部校准提高代理辅助决策质量，改进后的代理决策又提升后续优化效率。

// 正式论文中可以把 2.4.3 配成一个简短算法框：Input: global surrogates, optimizer, local buffer; for each generation: predict, calibrate, screen/rank, evaluate selected candidates, update optimizer and local buffer。算法框应强调 skipped candidates are not accepted as final evaluations。
