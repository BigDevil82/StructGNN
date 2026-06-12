# 1. Introduction

剪力墙结构是高层建筑和抗震结构设计中常用的主要抗侧力体系，其结构性能与墙体布置、截面尺寸、材料强度以及抗震设防条件密切相关。工程设计通常需要在满足承载力、层间位移、轴压比、构造要求和抗震性能约束的前提下，尽量降低混凝土和钢筋等材料用量。随着建筑平面形式和设计约束日益复杂，仅依靠经验调整截面和材料方案往往难以系统比较大量设计可能性，因此，面向剪力墙结构的自动化建模、分析与优化设计逐渐成为结构工程智能设计中的重要方向。这里需要引用剪力墙结构设计优化、抗震设计自动化、结构智能设计相关文献。

既有研究首先关注如何将结构优化算法引入建筑结构设计问题。遗传算法、粒子群优化、模拟退火、贝叶斯优化以及多目标进化算法等方法已被用于梁柱截面优化、框架结构优化、剪力墙布置优化和结构材料用量控制等任务。这类研究通常将结构设计问题转化为带约束的离散或混合变量优化问题，并通过惩罚函数、可行性规则、多目标排序或启发式搜索策略处理规范约束和经济性目标。相关工作说明，智能优化算法能够在复杂搜索空间中发现较优方案，但其效果高度依赖候选解评估的效率和可靠性。当每个候选方案都需要完整有限元分析和规范校核时，优化过程会迅速受到计算成本限制。这里需要引用结构优化算法综述、GA/PSO/BO 在结构工程中的应用、剪力墙或高层结构优化设计相关文献。

为降低结构优化中的计算开销，大量研究进一步引入代理模型或响应面模型。传统代理模型包括多项式响应面、Kriging/Gaussian process、径向基函数、支持向量回归和随机森林等，它们常被用于近似结构响应、可靠度指标或优化目标。近年来，深度神经网络、梯度提升树和图神经网络等模型也被用于预测结构响应、构件设计需求、损伤状态或结构可行性。与传统表格特征模型相比，图神经网络能够显式利用构件连接关系、结构拓扑或空间布局关系，因此在结构体系具有明显拓扑特征的问题中具有潜在优势。这里需要引用 surrogate-assisted structural optimization、response surface methodology、Kriging/RBF surrogate、deep learning structural response prediction、graph neural networks for structural engineering 等方向的文献。

然而，在工程优化场景中，代理模型的作用并不应简单理解为完全替代有限元分析。结构设计样本通常具有强烈的布局依赖性和参数耦合性，同一截面变化在不同平面布置、不同设防烈度和不同场地条件下可能产生不同影响。对于接近规范限值的候选方案，微小参数变化可能导致可行性标签发生翻转，二分类代理模型难以保证绝对可靠。材料用量预测也存在类似问题，钢筋用量往往依赖多工况内力包络和后续截面设计过程，在高用量区间可能出现系统性偏差。已有研究中常通过主动学习、序贯采样、信赖域、在线更新、模型校准或不确定性估计缓解代理模型误差，但在具体结构优化流程中，如何在代理模型不完全准确的情况下仍然稳定地减少有限元调用，仍是一个需要进一步讨论的问题。这里需要引用 surrogate uncertainty、active learning、trust-region surrogate optimization、model calibration、constrained surrogate-assisted optimization、classification under uncertainty 等文献。

因此，本文围绕代理模型误差条件下的有限元预算分配问题展开研究。代理模型如果能够识别明显不可行的候选方案，或能够对剩余候选方案的相对优劣给出足够有效的排序，就可以帮助优化算法减少无效有限元分析，将真实分析优先分配给更有希望的设计区域。与直接使用代理模型给出最终设计结果相比，这种思路保留了最终方案的真实有限元验证和规范校核，因此更符合工程设计对可靠性的要求。

基于上述思路，本文提出一种校准代理辅助的剪力墙结构优化框架。该框架基于参数化建模、OpenSees 有限元分析和规范校核流程生成训练数据，分别训练结构可行性代理模型和钢筋用量代理模型；在优化过程中，利用新产生的真实有限元结果对全局代理模型进行在线局部校准，并将校准后的可行性概率和材料造价估计用于候选方案筛选与优先排序。最终候选方案仍通过真实有限元分析和规范校核确认，从而在控制代理模型误差风险的同时降低优化过程中的有限元调用次数。

本文的主要贡献体现在以下几个方面。

1. 建立了面向剪力墙结构优化的数据生成与评估流程，将参数化建模、有限元分析、规范校核和材料用量计算整合为可批量运行的工作流。
2. 构建了面向剪力墙布局和设计参数的全局代理模型，用于结构可行性筛选和钢筋用量预测，为优化过程提供候选方案预评估信息。
3. 提出利用优化过程中已有有限元结果进行在线局部校准的方法，修正全局代理模型在特定布局和局部搜索区域中的偏差。
4. 将校准后的代理模型嵌入结构优化流程，在保持最终有限元验证的前提下减少无效候选方案分析，从而提高剪力墙结构优化效率。

本文其余部分组织如下。第 2 节介绍校准代理辅助优化框架，包括问题定义、数据表征、全局代理模型和在线局部校准方法。第 3 节说明实验设置，包括数据集、模型训练、优化算法和评价指标。第 4 节给出代理模型、局部校准和优化效率实验结果，并讨论方法适用性和局限性。第 5 节总结全文并展望后续研究方向。

## 文献补充位置与检索关键词

这一部分是后续补文献时的写作备注，不属于最终论文正文。

1. 第一段需要支撑剪力墙结构优化设计的工程背景。可检索关键词包括：`shear wall structural optimization`，`reinforced concrete shear wall design optimization`，`seismic design optimization of shear wall structures`，`automated structural design reinforced concrete buildings`，`performance-based seismic design optimization`。

2. 第二段需要综述结构优化算法在土木工程中的应用，并指出 FEA 调用成本是瓶颈。可检索关键词包括：`genetic algorithm structural optimization reinforced concrete`，`particle swarm optimization structural design`，`Bayesian optimization structural engineering`，`metaheuristic structural optimization finite element analysis`，`constrained structural optimization code design`。

3. 第三段需要综述代理模型辅助结构优化，从传统响应面过渡到机器学习和深度学习。可检索关键词包括：`surrogate-assisted structural optimization`，`response surface structural optimization`，`Kriging surrogate structural reliability optimization`，`Gaussian process surrogate finite element structural analysis`，`machine learning surrogate model structural engineering`，`deep learning structural response prediction`。

4. 第三段末尾可补充图神经网络用于结构工程或拓扑表征的研究。可检索关键词包括：`graph neural network structural engineering`，`graph neural networks structural response prediction`，`GNN building structure analysis`，`graph representation structural topology optimization`，`graph neural network finite element surrogate`。

5. 第四段需要支持“代理模型不能完全替代 FEA”的论点，重点找代理误差、不确定性、边界样本和泛化问题。可检索关键词包括：`surrogate model uncertainty structural optimization`，`surrogate model generalization structural engineering`，`active learning surrogate structural optimization`，`trust region surrogate optimization engineering design`，`surrogate-assisted constrained optimization uncertainty`。

6. 第四段还可补充模型校准和在线更新相关研究，为本文局部校准做铺垫。可检索关键词包括：`online surrogate model update optimization`，`local surrogate model calibration`，`residual learning surrogate model`，`model calibration machine learning structural engineering`，`adaptive surrogate modeling structural optimization`。

7. 第五段需要形成本文问题定位，可以引用少量关于“代理模型用于筛选、排序、预评估，而非完全替代仿真”的文献。可检索关键词包括：`surrogate screening optimization`，`preselection surrogate-assisted evolutionary algorithm`，`surrogate-assisted candidate selection expensive optimization`，`ranking surrogate model optimization`，`simulation budget allocation surrogate model`。
