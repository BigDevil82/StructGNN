# 1. Introduction

剪力墙结构是高层建筑和抗震结构设计中常用的主要抗侧力体系，其结构性能与墙体布置、截面尺寸、材料强度以及抗震设防条件密切相关。工程设计通常需要在满足承载力、层间位移、轴压比、构造要求和抗震性能约束的前提下，尽量降低混凝土和钢筋等材料用量。随着建筑平面形式和设计约束日益复杂，仅依靠经验调整截面和材料方案往往难以系统比较大量设计可能性，因此，面向剪力墙结构的自动化建模、分析与优化设计逐渐成为结构工程智能设计中的重要方向。这里需要引用剪力墙结构设计优化、抗震设计自动化、结构智能设计相关文献。

既有研究首先关注如何将结构优化算法引入建筑结构设计问题。遗传算法、粒子群优化、模拟退火、贝叶斯优化以及多目标进化算法等方法已被用于梁柱截面优化、框架结构优化、剪力墙布置优化和结构材料用量控制等任务。这类研究通常将结构设计问题转化为带约束的离散或混合变量优化问题，并通过惩罚函数、可行性规则、多目标排序或启发式搜索策略处理规范约束和经济性目标。相关工作说明，智能优化算法能够在复杂搜索空间中发现较优方案，但其效果高度依赖候选解评估的效率和可靠性。当每个候选方案都需要完整有限元分析和规范校核时，优化过程会迅速受到计算成本限制。这里需要引用结构优化算法综述、GA/PSO/BO 在结构工程中的应用、剪力墙或高层结构优化设计相关文献。

为降低结构优化中的计算开销，大量研究进一步引入代理模型或响应面模型。传统代理模型包括多项式响应面、Kriging/Gaussian process、径向基函数、支持向量回归和随机森林等，它们常被用于近似结构响应、可靠度指标或优化目标。近年来，深度神经网络、梯度提升树和图神经网络等模型也被用于预测结构响应、构件设计需求、损伤状态或结构可行性。与传统表格特征模型相比，图神经网络能够显式利用构件连接关系、结构拓扑或空间布局关系，因此在结构体系具有明显拓扑特征的问题中具有潜在优势。对于剪力墙结构而言，代理建模的主要困难不仅来自有限元响应本身的非线性，还来自布局依赖性和设计参数耦合：同一组截面与材料参数在不同平面布置、不同建筑高度和不同设防条件下可能对应完全不同的受力状态和设计裕度。这里需要引用 surrogate-assisted structural optimization、response surface methodology、Kriging/RBF surrogate、deep learning structural response prediction、graph neural networks for structural engineering 等方向的文献。

上述特征使得结构优化中的代理模型难以被简单视为完整有限元分析的替代品。对于接近规范限值的候选方案，微小参数变化可能导致可行性标签发生翻转，二分类代理模型难以保证绝对可靠。材料用量预测也存在类似问题，钢筋用量往往依赖多工况内力包络和后续截面设计过程，在高用量区间可能出现系统性偏差。已有研究中常通过主动学习、序贯采样、信赖域、在线更新、模型校准或不确定性估计缓解代理模型误差，但这些方法通常仍将主要目标放在提高代理模型对真实分析结果的近似精度上。对于规范约束强、误删可行方案代价高的结构优化问题，更关键的问题是：当代理模型存在不可避免的跨布局偏差时，如何仍然安全、稳定地减少有限元调用。这里需要引用 surrogate uncertainty、active learning、trust-region surrogate optimization、model calibration、constrained surrogate-assisted optimization、classification under uncertainty 等文献。

因此，本文将研究视角从“用代理模型替代有限元分析”转向“用代理模型分配有限元分析预算”。代理模型如果能够识别明显不可行的候选方案，或能够对剩余候选方案的相对优劣给出足够有效的排序，就可以帮助优化算法减少无效有限元分析，将真实分析优先分配给更有希望的设计区域。与直接使用代理模型给出最终设计结果相比，这种思路保留了最终方案的真实有限元验证和规范校核，因此更符合工程设计对可靠性的要求。

基于上述思路，本文提出一种校准代理辅助的剪力墙结构优化框架。该框架基于参数化建模、OpenSees 有限元分析和规范校核流程生成训练数据，分别训练结构可行性代理模型和钢筋用量代理模型；在优化过程中，利用新产生的真实有限元结果对全局代理模型进行在线局部校准，并将校准后的可行性概率和材料造价估计用于候选方案筛选与优先排序。最终候选方案仍通过真实有限元分析和规范校核确认，从而在控制代理模型误差风险的同时降低优化过程中的有限元调用次数。

本文的主要贡献体现在以下几个方面。

1. 基于约 715,000 个参数化 FEA 与规范校核样本，构建了面向剪力墙布局和设计参数的统一代理建模框架，分别用于结构可行性筛选和钢筋用量预测，为优化过程提供候选方案预评估信息。
2. 提出利用优化过程中已有有限元结果进行在线局部校准的方法，修正全局代理模型在特定布局和局部搜索区域中的概率偏差和钢筋用量残差。
3. 将校准后的代理模型嵌入结构优化流程，在保持最终有限元验证的前提下减少无效候选方案分析，从而提高剪力墙结构优化效率。

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

# 2. Calibrated Surrogate-Assisted Optimization Framework

本节介绍本文提出的校准代理辅助剪力墙结构优化框架。该框架面向这样一类设计问题：在给定剪力墙布局、建筑几何条件和抗震设防条件后，优化墙、梁等关键构件的截面尺寸和材料参数，使结构满足有限元分析和规范校核要求，并尽量降低材料造价。完整的有限元分析能够提供可靠评价，但在优化算法反复生成候选方案时，逐一调用完整分析会带来较高计算开销。本文方法的核心是在保留真实有限元验证的前提下，利用代理模型对候选方案进行预筛选和优先排序，从而减少明显低价值候选方案占用有限元计算预算。

图 2.1 可用于展示本文框架的整体流程。优化器首先根据当前搜索状态生成一批候选设计方案，每个候选方案由布局信息、截面与材料参数以及设计条件共同定义。候选方案随后进入代理评估模块。可行性代理模型估计其通过结构校核的概率，用于识别明显不满足要求的方案；钢筋用量代理模型估计其钢筋需求，并结合可由几何信息快速计算的混凝土用量形成材料造价分数。经过代理模型筛选和排序后，只有一部分更有希望的候选方案进入真实有限元分析和规范校核流程。

真实有限元分析在框架中仍然承担最终评价标准的作用。被选中的候选方案通过 OpenSees 完成结构分析，并根据规范校核流程得到综合可行性、钢筋用量、混凝土用量和材料造价等结果。这些真实结果一方面用于更新优化算法的种群、粒子或采样策略，另一方面被回流到局部校准模块。随着优化过程推进，局部校准模块逐步积累当前布局和当前搜索区域内的真实分析样本，用于修正全局代理模型在该具体问题上的概率偏差和钢筋用量残差。

这种闭环流程使代理模型在优化过程中扮演辅助决策角色，而不是直接替代结构分析。全局代理模型提供跨布局、跨参数空间的初始判断能力，使优化过程在早期也能进行候选筛选；局部校准模型利用优化过程中已经产生的真实分析结果，提升代理判断对当前布局的适应性；有限元分析则持续为优化器提供可靠反馈，并防止代理误差直接决定最终设计结果。三者共同构成一个“生成候选、代理筛选、真实验证、局部校准、优化更新”的迭代闭环。

// 后续小节按照这一闭环流程展开。第 2.1 节首先形式化定义剪力墙结构优化问题，包括设计变量、目标函数、约束条件和真实评价器。第 2.2 节说明代理学习数据如何由参数化结构评价流程形成，并介绍布局图和设计参数的输入表征。第 2.3 节描述两个全局代理模型，包括统一的布局参数融合架构、可行性筛选代理和钢筋用量代理。第 2.4 节进一步介绍在线局部校准方法，说明如何利用优化过程中新增的有限元结果校准可行性概率和钢筋用量残差。

// 图 2.1 建议设计为“优化闭环 + 代理增强模块”的组合式框架图，而不是把代理模型、优化算法和 FEA 三个部分画成相同权重的并列模块。主体可以采用从左到右的流程：左侧是 optimization algorithm，表示 GA/PSO/random search 等优化器根据历史评价结果生成 candidate designs；中间是本文核心的 surrogate-assisted candidate assessment，建议占据图中最大面积，并在内部进一步拆成 global surrogates、online local calibration、candidate filtering/ranking 三个子模块；右侧是 high-fidelity FEA and code checking，面积略小但用深色或边框强调其 final verification role。数据流上，候选方案从优化器进入代理评估模块，先由 feasibility surrogate 给出可行概率并筛掉明显不可行方案，再由 reinforcement surrogate 加上快速混凝土估计形成 calibrated cost score，对剩余候选进行排序和预算分配；只有被选中的候选进入 FEA。FEA 输出 feasible reinforcement usage、concrete volume、material cost，这些结果一条箭头返回优化器用于更新搜索状态，另一条箭头返回 local calibration buffer，用于更新 probability calibration 和 reinforcement residual calibration。图中可以用两种线型区分代理路径和真实验证路径，例如实线表示 FEA-verified feedback，虚线表示 surrogate prediction。视觉重点应放在两个创新点上：一是 global surrogate 经 online local calibration 后再参与决策，二是 calibrated surrogate 不直接给出最终设计结果，而是通过 screening and ranking 减少 FEA 调用。图中不需要展开 OpenSees 建模或网络层细节，这些内容在后续小节说明即可。

# 2.1 Problem Formulation

本文研究给定剪力墙布局下的结构设计优化问题。对于布局 $L$ 和设计条件 $c$，优化算法需要确定截面与材料方案 $x$，使结构满足分析与设计约束，并最小化材料造价。布局 $L$ 在优化过程中保持固定，本文关注的是在既定布置基础上的截面和材料优化，而非墙体位置或平面布局生成。

设计变量 $x$ 由离散的截面尺寸和材料参数组成，可概括为

$$
x \in \mathcal{X},
$$

其中 $\mathcal{X}$ 表示由墙厚、梁截面尺寸和混凝土强度等级等参数构成的可行设计空间。设计条件 $c$ 包括楼层数、层高、抗震设防烈度、场地类别和地震分组等影响结构分析和设计校核的外部条件。需要说明的是，$x$ 和 $c$ 的区分取决于具体优化任务：在单次优化中，楼层数和抗震设防条件通常作为给定项目条件固定；在代理模型数据集构建阶段，为提高模型覆盖范围，这些项目条件也会被采样并作为输入特征参与训练。因此，表 1 中列出的 $N$ 等参数表示数据生成时的采样维度，而不意味着其在每一次优化任务中都作为可调设计变量。

给定 $(L,c,x)$ 后，真实评价器 $\mathcal{E}$ 通过参数化建模、有限元分析和规范校核得到候选方案的结构性能与材料需求。该过程可表示为

$$
\mathcal{E}(L,c,x) \rightarrow (y_f, m_s, m_c, r),
$$

其中 $y_f \in \{0,1\}$ 表示综合可行性，$m_s$ 和 $m_c$ 分别表示钢筋用量和混凝土用量，$r$ 表示结构响应和设计校核指标。综合可行性同时考虑分析收敛性、结构响应是否满足控制限值以及构件设计是否通过。

结构约束可表示为一组关于响应指标的函数 $g_j(L,c,x)$。当所有约束均满足时，候选方案被视为可行。为便于优化算法比较不可行方案，约束违反程度可进一步写成非负形式 $v_j(L,c,x)$，并合成为总违反度 $V(L,c,x)$。这种标量化违反度仅用于优化搜索中的候选排序，不改变最终可行性仍由真实分析与校核判定的原则。

优化目标为材料造价。设 $p_c$ 和 $p_s$ 分别表示混凝土和钢筋的单位价格，则候选方案的材料造价可写为

$$
C(L,c,x)=p_c m_c(L,c,x)+p_s m_s(L,c,x).
$$

在该目标中，混凝土用量主要由布局几何、楼层数和截面尺寸决定，计算相对直接；钢筋用量依赖结构分析后的内力需求和后续截面设计结果，是优化流程中更昂贵且更难直接估计的关键量。

由此，给定布局和设计条件下的剪力墙结构优化问题可写为

$$
\begin{aligned}
\min_{x \in \mathcal{X}} \quad & C(L,c,x) \\
\text{s.t.} \quad & y_f(L,c,x)=1 .
\end{aligned}
$$

在算法实现中，常将约束违反度并入一个用于搜索排序的标量目标，例如

$$
\tilde{C}(L,c,x)=C(L,c,x)(1+V(L,c,x))+\lambda \mathbb{I}[y_f(L,c,x)=0],
$$

其中 $\lambda$ 为不可行惩罚系数，$\mathbb{I}[\cdot]$ 为指示函数。该目标用于引导优化算法减少对明显不可行方案的选择；最终报告的最优方案仍以真实有限元分析和规范校核确认后的可行性与材料造价为准。

由于真实评价器 $\mathcal{E}$ 包含有限元分析和设计校核，其调用次数构成优化过程中的主要计算成本。本文将减少 $\mathcal{E}$ 的调用次数作为代理模型辅助优化的核心目标，同时要求最终可行性和材料造价质量与完整真实评价流程保持一致。

# 2.2 Data Formulation for Surrogate Learning

代理模型训练数据来源于参数化结构评价流程。每个样本对应一个给定布局、设计条件和截面材料方案，其输入由布局表征和设计参数组成，监督标签由真实结构分析与设计校核得到。与直接记录优化过程中的搜索结果不同，本文的数据构建过程在较大的参数空间内随机采样候选方案，使代理模型能够学习不同布局、设计条件和截面材料组合下的可行性与材料需求变化。

对于第 $i$ 个样本，输入可表示为

$$
z_i = \{G_i, q_i, p_i\},
$$

其中 $G_i$ 表示布局图结构，$q_i$ 表示全局布局统计特征，$p_i$ 表示截面、材料和设计条件参数。对应的监督标签为

$$
y_i = \{y_{f,i}, m_{s,i}\},
$$

其中 $y_{f,i}$ 为综合可行性标签，$m_{s,i}$ 为钢筋用量。这样构建的数据既可用于训练可行性分类代理模型，也可用于训练钢筋用量回归代理模型。

## 2.2.1 Parametric Structural Evaluation and Design Labels

参数化结构评价流程将候选设计方案转化为可监督学习样本。对于每个剪力墙布局，候选方案由几何截面参数、材料参数和设计条件共同定义。几何截面参数描述墙体和梁构件的主要尺寸；材料参数描述混凝土强度等级等设计选择；设计条件包括楼层数、层高、抗震设防烈度、场地类别和地震分组等影响结构响应和规范校核的外部条件。

结构分析基于 OpenSees 完成。剪力墙采用 MVLEM 类单元表征其受力行为，梁构件和整体结构体系按照参数化建模流程自动生成。分析工况覆盖用于结构响应计算和设计校核的主要竖向荷载与地震作用。完成分析后，评价流程进一步提取层间位移、扭转响应、周期特征、构件内力需求以及墙梁截面设计结果等指标。

综合可行性标签 `final_pass` 由分析收敛性、结构响应限值和截面设计结果共同确定。只有当候选方案完成有效分析，并且主要响应指标和构件设计均满足要求时，该样本被标记为可行。该标签直接对应优化过程中的筛选需求，因为明显不可行的方案可以在进入真实有限元分析前被代理模型识别并跳过。

钢筋用量标签 `material_steel_kg` 来自结构分析后的构件内力需求和截面设计结果。与混凝土用量不同，钢筋用量难以仅由几何尺寸快速确定，而是依赖多工况分析、内力包络和后续配筋设计过程。由于优化目标面向材料造价，钢筋用量又是造价中需要真实分析才能可靠获得的重要组成部分，因此本文将其作为连续代理模型的主要预测目标。该模型在后续优化中用于候选方案的造价预估和优先排序，而最终材料用量仍由真实分析与设计校核确认。

## 2.2.2 Layout Graph and Design Parameter Representation

代理模型输入需要同时描述结构布局和设计参数。布局部分采用 room graph 表征，即将建筑平面中的房间或空间单元视为图节点，并根据相邻关系构造边。与直接以构件作为节点相比，room graph 更接近建筑平面组织方式，能够从空间划分层面表达剪力墙布置对整体结构行为的影响。图中节点和边用于编码局部空间关系，全局图读出结果用于形成布局级表示。

除图结构外，模型还保留一组全局布局统计特征 $q$，用于描述图结构难以直接表达或需要全局聚合的信息。这类特征包括布局尺度、墙体数量和长度分布、墙体方向比例、边界墙比例、梁墙数量或长度关系以及图连通性等。它们为代理模型提供更稳定的全局几何和拓扑摘要。

设计参数 $p$ 包括截面尺寸、材料等级和设计条件。连续或有序参数以数值形式输入模型，类别参数通过嵌入或编码方式表示。由此，代理模型能够同时感知三个层面的信息：布局图结构提供空间拓扑关系，全局布局特征提供整体几何摘要，设计参数提供当前候选方案的截面、材料和抗震条件。

这种输入表征能够同时服务于可行性判断和钢筋用量预测。结构是否可行取决于布局、截面刚度、材料强度和地震作用之间的共同影响；钢筋用量也由相同因素决定，只是输出形式从离散标签变为连续材料需求。因此，本文后续采用统一的布局参数表征作为两个代理任务的输入基础，并通过不同任务头分别预测 `final_pass` 和 `material_steel_kg`。

// 这一节可以配一张较小的数据形成示意图，也可以并入图 2.1。若单独画图，建议展示“layout + design parameters -> parametric analysis/design check -> labels -> surrogate dataset”的流程，不需要画数据规模和划分方式。

# 2.3 Global Surrogate Models

本文使用两个全局代理模型分别近似结构综合可行性和钢筋用量。两个模型采用相同的输入表征和 LayoutParamGNN 主体架构，仅在输出任务和损失函数上不同。给定样本 $z=\{G,q,p\}$，其中 $G$ 为 room graph，$q$ 为全局布局特征，$p$ 为截面、材料和设计条件参数，模型首先学习布局参数联合表示 $h$，再通过任务头输出可行性概率或钢筋用量估计。

## 2.3.1 Unified LayoutParamGNN Architecture

建筑布局被表示为 room graph $G=(V,E)$。图节点 $v_i \in V$ 对应平面中的房间或空间单元，图边 $e_{ij}\in E$ 表示空间单元之间的相邻关系。该表征沿用了前序 RC-GNN 布局研究中的房间节点建模思想，其优势在于能够直接描述剪力墙布置对空间划分和整体平面组织的影响。相比以墙、梁等构件为节点的图，room graph 更接近建筑平面层面的拓扑关系，也避免图规模随构件细分程度过度变化。每个节点携带局部几何和空间属性，边特征描述相邻空间之间的关系。与图结构并行，模型还接收一组布局级统计特征 $q$，用于补充全局尺度、墙体分布、梁墙关系和整体拓扑摘要。

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

其中 $\rho(\hat{p}_f)$ 为由可行性概率得到的风险项，$\eta$ 为风险惩罚权重。本文采用 hinge-type 风险惩罚

$$
\rho(\hat{p}_f)=\max(0, p_h-\hat{p}_f),
$$

其中 $p_h$ 为排序阶段的可行性概率参考值。当候选方案的可行概率低于 $p_h$ 时，其评分会随不可行风险增加而上升；当 $\hat{p}_f\ge p_h$ 时，排序主要由材料造价估计决定。优化算法在一批候选方案中优先选择评分较低者进入真实有限元分析。经过真实分析的候选方案再以其真实可行性和材料造价更新优化器；未进入真实分析的候选方案不作为最终设计结果。

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

上述 Platt-style model 给出了最基本的概率校准形式。实际实现中，$\mathcal{C}_t$ 也可以采用 isotonic regression、logistic calibration、局部概率残差修正或集成校正等轻量模型，只要其输入为全局可行性概率及少量设计参数特征，输出为当前布局下校准后的可行概率。实验部分将比较这些候选校准器在不同局部样本规模下的表现。

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

这种残差校准思想与 residual learning 和 gradient boosting 中的逐步误差修正类似。全局钢筋代理模型提供基准预测，局部残差模型只学习当前布局和局部搜索区域中的剩余误差，因此不需要重新学习完整的钢筋用量映射。方法设计上并不限定残差模型的具体类型，Ridge、Gaussian process、LightGBM、KNN 或小型 MLP 等轻量回归器均可用于实现 $r_t(\cdot)$；实验部分将比较不同局部样本数量下的候选校准器。校准后的钢筋预测主要用于候选造价排序和 FEA 预算分配，而不作为最终材料用量结果。

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
\cup \{(x,\hat{p}_f,\hat{m}_s,y_f,m_s):x\in \mathcal{S}_t\}.
$$

未进入真实有限元分析的候选方案只影响当前批次的预算分配，不被作为真实训练标签加入局部校准集。通过这种在线更新机制，代理模型在优化早期主要依赖全局预测；随着优化过程不断产生新的 FEA 样本，局部校准器逐步提升可行性概率和钢筋残差预测在当前布局上的准确性。更准确的局部校准进一步改善后续批次的筛选和排序，使有限元预算更集中地分配给高价值候选方案。优化搜索和局部校准由此形成相互促进的闭环：优化过程提供局部真实样本，局部校准提高代理辅助决策质量，改进后的代理决策又提升后续优化效率。

**Algorithm 1. Online calibrated surrogate-assisted optimization**

**Input:** layout $L$, design condition $c$, optimizer $\mathcal{O}$, global feasibility surrogate $f_g$, global steel surrogate $s_g$, true evaluator $\mathcal{E}$, initial local buffer $\mathcal{D}_0^{loc}=\emptyset$.

**For** optimization iteration $t=0,1,\ldots,T-1$:

1. Generate a candidate batch $\mathcal{B}_t$ from optimizer $\mathcal{O}$.
2. For each $x\in\mathcal{B}_t$, compute global predictions $\hat{p}_f=f_g(L,c,x)$ and $\hat{m}_s=s_g(L,c,x)$.
3. If $\mathcal{D}_t^{loc}$ contains sufficient FEA-verified samples, fit or update local calibration models and obtain $\hat{p}_f^{cal}$ and $\hat{m}_s^{cal}$; otherwise use the global predictions.
4. Reject candidates with low calibrated feasible probability, and rank the remaining candidates by the calibrated cost score $S^{cal}$.
5. Select a subset $\mathcal{S}_t\subseteq\mathcal{B}_t$ for true FEA evaluation. Skipped candidates are assigned low priority in the current iteration but are not treated as verified design results.
6. Evaluate each $x\in\mathcal{S}_t$ using $\mathcal{E}$ to obtain feasibility labels, steel usage, concrete usage and material cost.
7. Update optimizer $\mathcal{O}$ with the verified evaluation results, and update $\mathcal{D}_{t+1}^{loc}$ by adding the new FEA-verified samples.

**Output:** the best feasible design verified by true FEA and code checking.

# 3. Experimental Setup

本节介绍实验数据、代理模型训练与优化对比的设置。与第 2 节的方法定义不同，本节的重点是说明实验对象如何构成、各类模型在什么数据上训练和验证，以及后续结果将围绕哪些设置展开。具体数值结果和现象分析放在第 4 节讨论。

## 3.1 Dataset and Structural Evaluation Settings

实验数据基于 143 张剪力墙结构布局图纸构建。这些布局给出了剪力墙和梁的布置位置，来源于建筑设计院工程师提供的结构方案，因此可以作为具有工程合理性的初始结构布置。按照前序布局生成研究中的条件划分方式，样本可根据设防烈度和建筑高度组合分为三类设计条件。Group7-H1 对应设防烈度 7 度、PGA 为 0.10 g 且建筑高度 $H \le 50$ m 的低密度需求组；Group7-H2 对应相同设防烈度但建筑高度 $H>50$ m 的中等需求组；Group8 对应设防烈度 8 度、PGA 为 0.20 g 的高需求组。该分组反映了抗震需求和结构高度对剪力墙布置及截面设计的共同影响。

布局图纸只确定了结构几何和构件布置。为了形成可用于监督学习的数据，还需要为每个布局指定截面、材料和设计条件参数。本文采用拉丁超立方采样在参数空间内生成候选设计方案，每个布局采样约 5000 个样本。采样参数覆盖墙厚、梁截面、混凝土强度等级、楼层数和抗震设计条件等关键因素，具体范围见表 1。为控制变量数量并聚焦截面与材料优化，本文将层高固定为 2.9 m；对于存在竖向分区的墙厚参数，采样后保持上部墙厚不大于下部墙厚，以符合常规结构设计逻辑。

| Parameter | Symbol | Unit | Sampling values |
| --- | --- | --- | --- |
| Number of stories | $N$ | - | 18-33 |
| Story height | $h_{story}$ | m | 2.9 |
| Bottom wall thickness | $t_{w,b}$ | mm | 200, 250, 300, 350, 400 |
| Middle wall thickness | $t_{w,m}$ | mm | 160, 180, 200, 250, 300 |
| Top wall thickness | $t_{w,t}$ | mm | 160, 180, 200, 250 |
| Main beam depth | $h_{b,main}$ | mm | 400, 500, 550, 600, 650, 700 |
| Main beam width | $b_{b,main}$ | mm | 200, 250, 300, 350 |
| Secondary beam depth | $h_{b,sec}$ | mm | 300, 400, 450, 500 |
| Secondary beam width | $b_{b,sec}$ | mm | 200, 250, 300 |
| Concrete grade | $f_c$ | - | C30, C35, C40, C45, C50 |
| Seismic intensity | - | - | 6.0, 7.0, 7.5, 8.0 |
| Site class | - | - | I0, I, II, III, IV |
| Seismic group | - | - | 1, 2, 3 |

对于每个布局和参数组合，本文使用 OpenSeesPy 自动建立结构分析模型并执行结构评价。剪力墙采用 MVLEM 类单元建模，分析过程包括静力分析和模态分析。静力分析用于获得主要竖向和水平作用下的结构响应及构件内力，模态分析用于提取周期等整体动力特性。随后，根据结构响应指标和构件内力完成墙、梁构件的设计校核与配筋计算。荷载工况组合可在附录中以表格形式列出，正文中仅保留与代理标签构造有关的说明。

上述结构评价流程为代理模型提供两个核心监督标签。第一个标签为 `final_pass`，表示候选结构是否同时满足分析收敛性、整体响应限值和构件设计要求。第二个标签为 `material_steel_kg`，表示该结构经分析和设计后得到的总钢筋用量。前者直接对应优化过程中的不可行方案筛选，后者对应材料造价预排序中最难快速获得的连续变量。

最终数据集包含约 715,000 个真实有限元分析样本。模型训练时采用 layout-level split，即以布局为单位划分训练集、验证集和测试集，避免同一布局的参数变体同时出现在不同数据子集中。当前实验划分为 100 个训练布局、21 个验证布局和 22 个测试布局，对应约 0.70:0.15:0.15 的布局比例。测试布局覆盖 Group7-H1、Group7-H2 和 Group8 三类设计条件，用于评估代理模型对未见布局的泛化能力。

// 这里我按当前代码和数据写为 0.70:0.15:0.15。如果论文必须采用 0.75:0.15:0.15，需要重新生成 split 并同步更新后续实验。
// 表 1 后续英文稿可以改成正式三线表。荷载组合表建议放附录，正文只引用 “load combinations listed in Appendix A”。

## 3.2 Surrogate Model Training Settings

代理模型实验分为全局代理模型比较和局部校准模型比较。全局代理模型包括可行性分类模型和钢筋用量回归模型，二者均在训练布局上训练，并在未见测试布局上评估。为验证布局信息和图结构表征的作用，本文比较三类模型。第一类模型仅使用截面、材料和设计条件参数作为输入，通过多层感知机进行预测；第二类模型在上述设计参数基础上进一步加入全局布局统计特征，同样采用多层感知机作为预测器；第三类模型为本文采用的 room-graph LayoutParamGNN，它同时利用房间图结构、全局布局特征和设计参数。上述对比用于区分设计参数本身、人工统计布局特征以及图结构表征对代理预测性能的贡献。

局部校准实验在测试布局内进行。对于每个测试布局，随机抽取不同数量的 FEA 样本作为局部校准集，其余样本作为布局内验证集，以模拟优化过程中逐步积累真实分析结果的过程。可行性概率校准比较 Platt scaling、isotonic regression、local logistic calibration、概率残差修正和 ensemble vote correction。钢筋残差校准比较 Ridge regression、Gaussian process regression、LightGBM、KNN 和小型 MLP。每种设置重复多次随机抽样，以减小局部样本选择带来的偶然性。

## 3.3 Optimization Benchmark Settings

优化实验在测试布局上进行。主实验从 Group7-H1、Group7-H2 和 Group8 中分别选取若干布局，并对每个布局采用多个随机种子重复运行。对于同一布局和随机种子，所有对比方法使用相同的设计条件、搜索空间和候选生成预算，以保证比较只反映评价策略的差异。

本文首先以 full FEA optimization 作为基准方法，即所有候选方案均进入真实结构分析与设计校核。在此基础上，比较两种代理辅助设置：一种仅使用钢筋用量代理模型进行候选造价预排序，另一种同时使用可行性筛选、钢筋造价预排序和在线局部校准。主优化器采用遗传算法。为进一步验证代理辅助筛选思想在不同搜索策略下的适用性，实验还引入 PSO 和 random search 作为补充优化器。

所有优化方法均以真实 FEA 和设计校核结果确定最终最优方案。代理模型只用于减少候选评价前的无效 FEA 调用，不直接提供最终可行性或材料造价结果。

// 具体 GA/PSO/random search 的 population、generation、seed 列表和候选保留比例建议放入实验设置表，不要在正文中逐项堆叠。

## 3.4 Evaluation Metrics

全局代理模型评价包括可行性分类和钢筋用量回归两个任务。对于可行性模型，采用 PR-AUC 评价类别不平衡条件下的概率排序能力，并采用 reject rate 和 feasible recall 评价其作为保守筛选器的表现。reject rate 表示模型可提前排除的候选比例，feasible recall 表示真实可行候选被保留的比例。对于钢筋模型，采用 MAE 和 $R^2$ 评价回归精度，并采用 ranking accuracy 评价候选钢筋用量相对大小的预测能力。

局部校准模型评价分为概率校准和残差校准。可行性概率校准采用 Brier score 衡量概率误差，并结合 screen recall 评估校准后筛选策略的安全性。钢筋残差校准采用 MAE 和 bias 衡量局部误差修正效果，并继续采用 ranking accuracy 评估其对候选排序的影响。

优化实验评价包括计算效率和解质量两个方面。计算效率以真实 FEA 调用次数和 FEA reduction 衡量；解质量以可行解成功率、首次发现可行解所需 FEA 次数和最终材料造价衡量。对于相同布局和随机种子，代理辅助方法与 full FEA baseline 进行成对比较，以判断 FEA 调用减少是否伴随可行率或最终造价的明显损失。

// 正式论文中可将指标整理为 “task - metric - purpose” 表，使土木领域读者更容易理解每个指标的作用。

# 4. Results and Discussion

本节按照代理模型性能、在线局部校准效果和代理辅助优化结果的顺序展开。首先评估两个全局代理模型是否具备作为候选预评估器的基本能力，并分析其局限性；随后考察局部校准是否能够利用少量当前布局样本修正全局模型偏差；最后将校准代理模型嵌入优化流程，比较其相对于 full FEA baseline 的计算效率和解质量。

## 4.1 Global Surrogate Model Performance

图 4.1 展示了 LayoutParamGNN 作为全局代理模型时的整体表现和主要误差形态。为了检验布局表征对可行性判断的作用，表 4.1 进一步比较了三种输入表征下的可行性代理模型。仅使用截面、材料和设计条件参数时，模型在测试集上的 PR-AUC 为 0.6137，F1 为 0.5927；加入全局布局统计特征后，PR-AUC 提升至 0.8163，F1 提升至 0.7370。采用 room-graph LayoutParamGNN 后，ROC-AUC 和 PR-AUC 进一步达到 0.9693 和 0.8593，F1 提升至 0.7688。该结果表明，可行性判别不仅依赖设计参数，也明显受布局几何与拓扑条件影响；相比全局统计特征，room graph 表征能够提供额外的判别信息。

在面向优化筛选的应用中，模型的关键作用在于以较高可行解召回率拒绝明显不可行的候选，而普通分类阈值下的最高准确率并非主要目标。LayoutParamGNN 在保守筛选阈值下能够提前拒绝约 47.36% 的候选样本，同时保持 0.9998 的 feasible recall，仅误拒 3 个真实可行样本。相比之下，参数模型虽然也能拒绝相近比例的样本，但误拒数量明显更高；加入布局统计特征后误拒数量降至 10 个，但整体 PR-AUC 和 F1 仍低于 LayoutParamGNN。因此，后续优化实验采用 room-graph LayoutParamGNN 作为全局可行性筛选模型。

| Model | ROC-AUC | PR-AUC | F1 | Reject rate | Feasible recall | False reject count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Design-parameter MLP | 0.9117 | 0.6137 | 0.5927 | 0.4787 | 0.9946 | 90 |
| Design-parameter + layout-statistics MLP | 0.9547 | 0.8163 | 0.7370 | 0.4447 | 0.9994 | 10 |
| Room-graph LayoutParamGNN | 0.9693 | 0.8593 | 0.7688 | 0.4736 | 0.9998 | 3 |

然而，图 4.1(b) 显示该模型在布局层面仍存在明显概率偏差。图中每个点对应一个测试布局，横轴为该布局中真实可行样本比例，纵轴为模型预测的平均可行性概率。理想情况下，各布局应接近对角线；实际结果中，部分布局的预测概率明显高于真实可行率。例如若干 Group8 高需求布局具有较低真实可行率，但模型给出的平均可行概率偏高。这说明全局分类模型虽然能够在样本层面进行排序和筛选，但其准确性在不同布局间存在较大差异。若直接将该概率作为可靠的结构可行性判断，可能导致过度乐观的筛选决策。

钢筋用量代理模型呈现出相似的规律，其整体回归趋势见图 4.1(c)，不同输入表征的定量对比见表 4.2。仅使用截面、材料和设计条件参数的多层感知机模型误差较大，测试 MAE 为 32.3 t，$R^2$ 仅为 0.351。加入全局布局统计特征后，MAE 降至 13.8 t，$R^2$ 提升至 0.891。进一步采用 room-graph LayoutParamGNN 后，MAE 降至 11.5 t，$R^2$ 提升至 0.924。可行性与钢筋用量两个任务上的对比结果共同说明，布局信息是截面优化代理建模中的必要输入，而图表征能够在人工布局统计特征之外进一步提高模型性能。

| Model | MAE (t) | RMSE (t) | $R^2$ | MAPE | Bias (t) | Pairwise acc. | Spearman | Top-10% recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Design-parameter MLP | 32.30 | 45.77 | 0.351 | 0.247 | -13.89 | 0.9586 | 0.9547 | 0.8974 |
| Design-parameter + layout-statistics MLP | 13.83 | 18.75 | 0.891 | 0.112 | -1.66 | 0.9652 | 0.9628 | 0.9142 |
| Room-graph LayoutParamGNN | 11.48 | 15.67 | 0.924 | 0.093 | -5.02 | 0.9787 | 0.9779 | 0.9483 |

从优化应用角度看，钢筋代理模型的排序能力同样重要。LayoutParamGNN 的 pairwise ranking accuracy 为 0.9787，Spearman 相关系数为 0.9779，top-10% recall 为 0.9483，说明模型能够较可靠地比较同一问题中候选方案的相对钢筋需求。这与本文将其用于候选预排序的目标相吻合。尽管如此，图 4.1(d) 显示其误差并非均匀分布。在最高 20% 钢筋用量区间，MAE 增加到约 20.7 t，bias 达到 -17.4 t，表明模型对高钢筋需求样本存在系统性低估。该偏差会影响候选造价排序，尤其在复杂布局或高需求设计条件下更为明显。

总体而言，两个全局代理模型均具有可用的初始预测能力，但也暴露出典型的跨布局泛化问题。可行性模型在布局层面存在概率校准偏差，钢筋模型在高用量区间存在系统性误差。因此，全局代理模型不适合直接替代 FEA 或规范校核，但可以作为后续局部校准和代理辅助优化的基础模型。

## 4.2 Online Local Calibration Performance

图 4.2 和图 4.3 展示了在线局部校准对全局代理模型的修正效果。局部校准实验在测试布局内进行，即从同一布局中抽取少量已知 FEA 样本作为校准集，并在剩余样本上评估校准后的预测性能。这一设置模拟了优化过程中逐步积累真实 FEA 结果的情形。

为避免将局部校准结果绑定到单一模型形式，本文在不同校准样本数量下比较了多种轻量校准器，并选择验证表现最优者作为对应样本规模下的代表结果。可行性概率校准的候选方法包括 Platt scaling、isotonic regression、logistic calibration、残差修正和集成投票等；钢筋残差校准的候选方法包括 Ridge、Gaussian process、LightGBM 和 KNN 等。附表 A1 和附表 A2 汇总了不同局部样本数量下的最优校准器及其主要指标。可以看到，较小样本数下简单模型更稳定，而随着局部样本增加，残差随机森林、Gaussian process 或 LightGBM 等模型能够进一步降低校准误差。

**Appendix Table A1. Best feasibility probability calibrator under different local sample sizes.**

| calib. samples | best feasibility calibrator | Brier | ECE | screen reject | screen recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25 | Platt scaling | 0.0505 | 0.0455 | 47.4% | 0.9476 |
| 50 | residual random forest | 0.0464 | 0.0406 | 51.6% | 0.9485 |
| 100 | residual random forest | 0.0420 | 0.0325 | 52.3% | 0.9487 |
| 200 | residual random forest | 0.0381 | 0.0261 | 59.1% | 0.9481 |
| 500 | residual random forest | 0.0344 | 0.0201 | 63.9% | 0.9441 |

**Appendix Table A2. Best steel residual calibrator under different local sample sizes.**

| calib. samples | best steel residual model | MAE (kg) | RMSE (kg) | $R^2$ | pair acc. |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25 | Ridge | 2395.9 | 3162.8 | 0.9924 | 0.9769 |
| 50 | Gaussian process | 2229.1 | 3003.6 | 0.9932 | 0.9785 |
| 100 | Gaussian process | 2044.7 | 2756.0 | 0.9943 | 0.9797 |
| 200 | LightGBM | 1741.3 | 2386.0 | 0.9957 | 0.9821 |
| 500 | LightGBM | 1444.8 | 1992.6 | 0.9970 | 0.9851 |

图 4.2 比较了使用 100 个局部样本进行校准前后的布局级表现。对于可行性模型，局部校准在 Brier score 较大的布局上带来更明显的改善，说明全局概率偏差可以通过少量当前布局样本得到有效修正。不过，校准后仍存在少量偏离理想关系的布局，表明概率校准并不能完全消除边界样本和困难布局带来的不确定性。因此，可行性代理模型在优化中仍应作为保守筛选器使用，而不应直接作为最终可行性判定器。

对于钢筋用量模型，局部残差校准的改善更加显著。校准前，全局模型在高钢筋用量区间存在明显低估；校准后，不同钢筋用量分位区间的 MAE 更加接近，bias 也接近于零。这说明局部残差模型能够捕捉当前布局下全局钢筋预测的系统性偏差，并将其转化为更适合候选排序的局部估计。

图 4.3 进一步给出了局部校准的样本效率。对于可行性模型，仅使用 25 个局部样本即可显著降低概率校准误差，并获得约 50% 的筛选拒绝率。随着校准样本数量增加，Brier score 和 ECE 继续下降，但改善幅度逐渐减小，呈现明显的边际收益递减。对于钢筋模型，不同局部样本数量下的 MAE 和 bias 差异较小，说明残差校准对样本数量不敏感，少量布局内样本已经足以修正主要系统偏差。

表 4.3 汇总了采用 100 个局部校准样本时的代表性结果。可行性模型的 Brier score 从 0.0749 降至 0.0437，ECE 从 0.0895 降至 0.0333，分别提升约 42% 和 63%。筛选拒绝率从 0.4708 提高到 0.5481，而 screen recall 从 0.9990 轻微下降至 0.9939，仍保持在较高水平。钢筋模型的改善更为显著：MAE 从 11.48 t 降至 2.04 t，RMSE 从 12.30 t 降至 2.76 t，$R^2$ 从 0.847 提升到 0.994，MAPE 从 9.26% 降至 1.64%。同时，bias 从 -5.02 t 几乎修正为 0，说明局部残差校准成功消除了主要系统偏差。

| Task | Metric | Global | Calibrated | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Feasibility | Brier | 0.0749 | 0.0437 | 42% |
| Feasibility | ECE | 0.0895 | 0.0333 | 63% |
| Feasibility | Screen reject rate | 0.4708 | 0.5481 | 16% |
| Feasibility | Screen recall | 0.9990 | 0.9939 | -1% |
| Steel | MAE | 11.48 t | 2.04 t | 82% |
| Steel | RMSE | 12.30 t | 2.76 t | 78% |
| Steel | $R^2$ | 0.847 | 0.994 | 17% |
| Steel | MAPE | 9.26% | 1.64% | 82% |

上述结果表明，全局代理模型在跨布局预测中不可避免存在偏差，但优化过程中自然产生的少量 FEA 样本可以用于快速修正当前布局上的概率和残差误差。经过校准后，代理模型更适合用于筛选和排序，但最终结构评价仍需由真实 FEA 完成。

## 4.3 Optimization Efficiency and Solution Quality

图 4.4 比较了 GA、PSO 和 random search 三类优化算法下不同评价策略的 FEA 调用次数、最终造价差异和首次发现可行解所需 FEA 次数。总体来看，代理辅助方法在三类优化算法中均显著减少真实 FEA 调用。对于 GA，`Cost surrogate` 和 `Screen + cost` 的平均 FEA 调用次数分别为 279 和 223，较 full FEA baseline 的 650 次分别减少约 57% 和 66%。对于 PSO，二者分别减少约 54% 和 62%；对于 random search，二者分别减少约 50% 和 66%。这说明 FEA 调用减少并非依赖某一种特定优化器，而是来自代理模型对候选评价预算的重新分配。

在解质量方面，代理辅助方法的可行解发现率整体接近 full FEA baseline。以 GA 为例，full baseline 的可行率为 68.0%，`Screen + cost` 为 65.3%；PSO 中二者分别为 61.3% 和 58.7%；random search 中二者分别为 68.0% 和 66.7%。虽然代理辅助方法略有下降，但其以约 60% 以上的 FEA 减少换取了相对较小的可行率损失。在最终造价方面，`Screen + cost` 相对于 full baseline 的平均目标比在 GA、PSO 和 random search 中分别约为 1.047、1.056 和 1.019，说明在多数可解 case 中，代理筛选并未造成显著的造价劣化。

不同优化器对代理筛选的响应存在差异。GA 的种群更新机制会在每一代产生一批多样候选，代理模型能够在批量候选之间进行筛选和排序，因此更充分地发挥预算分配作用。PSO 的粒子位置更新较容易在早期向局部区域集中，若尚未积累足够局部校准样本，代理筛选对搜索方向的影响会受到粒子多样性限制。Random search 缺少历史反馈驱动的搜索更新机制，因此代理模型主要体现为减少 FEA 调用，而对可行解发现率和最终造价的改善更依赖随机样本本身的质量。

图 4.4 还显示，代理辅助方法通常能够降低首次发现可行解所需的 FEA 调用次数。这表明可行性筛选不仅减少了总计算量，也在优化早期引导搜索避开明显不可行区域，使有限的真实分析预算更快接触到可行设计。对于 random search，即使没有复杂的搜索更新机制，代理筛选仍能降低 FEA 调用并保持较好的解质量，进一步说明代理模型确实过滤了大量低价值候选方案。

图 4.5 从 full-feasible cases 的角度进一步考察代理辅助方法的稳定性。横轴表示相对于 full FEA 最优造价允许增加的容差，纵轴表示同时满足“找到可行解、减少 FEA 调用、最终造价不超过给定容差”的 case 比例。随着容差从 0% 增加到约 5%，三类优化算法下的合格比例均快速上升，随后趋于平稳。这说明代理辅助方法在大多数可解 case 中能够以较小造价损失换取显著 FEA 节省。相比仅使用钢筋造价代理的设置，结合可行性筛选与造价预排序的 `Screen + cost` 在 GA、PSO 和 random search 中整体取得更高的合格比例，表明可行性筛选能够避免低造价但不可行的候选方案占用评价预算。

需要注意的是，不同算法对应的 full-feasible case 数量并不完全相同，因此图 4.5 中不同子图的合格 case 数不能直接横向比较。该图主要用于说明在各自可解案例内，代理辅助方法保持解质量并减少 FEA 调用的能力。

## 4.4 Discussion: Mechanism and Practical Implications of Surrogate-Assisted Optimization

前述结果表明，校准代理辅助方法能够在基本保持可行率和最终造价质量的同时显著降低真实 FEA 调用次数。为了理解这一结果，需要进一步观察候选方案在代理筛选、造价预排序和真实 FEA 验证之间的流动方式。

图 4.6 展示了代理辅助流程中的候选分流比例。堆叠区域表示每一代候选方案中被可行性代理模型提前筛除、通过可行性筛选但因造价预排序被跳过、以及最终进入真实 FEA 的平均比例。结果显示，约 60% 的候选方案在进入 FEA 前被代理流程过滤，这是 FEA 调用减少的直接来源。与此同时，绿色曲线表示全部候选方案中最终经过 FEA 验证且确认为可行的比例。该曲线低于进入 FEA 的区域，说明进入真实分析的候选仍有一部分未能通过规范校核。这一现象是合理的，因为可行性筛选采用保守策略，优先避免误删潜在可行方案，因此会保留部分不确定或边界候选进入真实验证。

图 4.7 选取三个具有代表性的 GA 优化过程，分别对应较易、中等和困难的布局案例。full FEA 方法对所有候选方案进行真实分析，因此可以观察到大量高造价候选和不可行候选被反复评估。引入钢筋用量代理后，优化过程优先保留预测材料造价较低的候选方案，使评价点更集中于低造价区域。然而，仅依赖造价代理仍无法充分区分低造价但不满足规范要求的方案，因此仍存在较多不可行评价点。进一步加入可行性筛选后，`Screen + cost` 显著减少了进入 FEA 的不可行候选，剩余评价点以可行方案为主，同时最优可行造价与 full FEA 和 cost surrogate 结果基本接近。

这一过程说明两个代理模型在优化中具有互补作用。钢筋用量代理主要用于压缩高造价候选的评价空间，使优化器更关注经济性较好的设计；可行性代理则进一步减少低造价但不可行候选占用 FEA 预算。二者结合后，真实 FEA 更集中地用于验证高价值候选，从而在不明显损害最终解质量的前提下降低计算成本。

对于第三个困难案例，三种方法均未能发现可行解。值得注意的是，full FEA 和 `Screen + cost` 都尝试了较宽范围的材料造价水平，其中也包括相对较高造价的截面方案，但所有候选仍未通过校核。这表明该案例的失败并非由代理筛选过度激进造成，而更可能源于布局本身或当前截面材料搜索空间的限制。对于这类结构，仅通过截面尺寸和材料等级优化可能不足以获得满足规范要求的方案，后续需要回到布局层面调整剪力墙布置，或扩大设计变量范围。

这一现象也说明，代理辅助框架并不改变优化问题本身的可解性。对于在当前布局和设计变量空间内存在较充分可行域的问题，代理模型能够将有限 FEA 预算集中到更有希望的候选区域，从而提高搜索效率；对于即使 full FEA 也难以找到可行解的问题，代理模型只能减少无效尝试，不能替代布局调整或搜索空间扩展。

图 4.8 给出了不同优化算法和代理辅助策略在各测试布局上的可行解发现率。热力图中的行按照不同方法的成功率模式排序，因此从上到下大致反映了布局优化难度由低到高的变化。部分布局在 GA、PSO 和 random search 下均能稳定找到可行解，说明这些布局在当前截面与材料搜索空间内具有较大的可行域。底部若干布局即使在 full FEA baseline 下成功率也较低，说明困难主要来自布局或搜索空间本身，而不是代理筛选造成。

相比仅使用钢筋造价代理的设置，加入可行性筛选与局部校准后的 `Screen + cost` 方法在 GA 和 PSO 中整体更接近 full FEA，并在多个布局上优于 cost surrogate。Random search 中不同方法的成功率差异较小，说明在缺少有效搜索更新机制时，代理筛选主要体现为计算节省，而对可行解发现率的提升受到随机采样质量限制。

上述讨论表明，代理辅助优化的关键不在于用模型直接替代结构分析，而在于改变有限元分析预算的使用方式。钢筋代理模型将搜索重点从明显高造价候选中转移出来，可行性代理模型进一步降低低造价但不可行候选进入真实分析的概率，局部校准则使这两个判断逐步适应当前布局。对于可行域较充分的布局，这种机制能够以较小造价损失换取显著 FEA 节省；对于 full FEA 也难以找到可行解的布局，代理模型的作用则主要体现为减少无效尝试，而不能弥补布局或设计变量空间本身的不足。因此，本文方法更适合作为面向工程优化的 decision-support mechanism，用于分配昂贵的物理分析预算，并保持最终设计验算仍由 FEA 和规范校核完成。

从适用边界看，本文方法在三类条件下收益更明显：第一，单次 FEA 和规范校核成本较高，使减少无效调用能够直接转化为优化效率提升；第二，可行域并非极小，保守筛选能够剔除大量明显不可行候选而不显著损害可行解发现率；第三，候选方案之间存在明显的造价和可行性差异，使代理模型的排序信息能够有效区分评价优先级。当可行域极小、设计变量范围无法覆盖可行方案，或新布局与训练布局的空间组织差异过大且少量局部样本无法纠正全局偏差时，代理筛选的收益会明显降低。此时，优化失败更可能反映问题定义或搜索空间本身的限制，而不是单纯的评价预算分配问题。

// 图编号需要在最终英文稿中统一调整。当前文本暂以 Figure 4.1-4.8 组织：4.1 surrogate performance；4.2 layout calibration improvement；4.3 sample efficiency；4.4 distribution summary；4.5 tolerance success curve；4.6 screening funnel；4.7 process examples；4.8 success heatmap。

# 5. Conclusions

本文围绕剪力墙结构截面与材料优化中的高计算成本问题，提出了一种校准代理辅助的结构优化框架。该框架将全局代理模型、在线局部校准和真实 FEA 验证结合起来，将代理模型用于识别低价值候选、引导有限元计算预算分配，并在优化过程中根据当前布局的真实分析结果持续修正预测偏差。主要结论如下：

实验结果表明，布局图表征能够有效提升代理模型对剪力墙结构性能的表征能力。采用 room-graph LayoutParamGNN 后，可行性模型在测试集上取得 0.9693 的 ROC-AUC 和 0.8593 的 PR-AUC；钢筋用量模型取得 0.924 的 $R^2$ 和 0.9787 的 pairwise ranking accuracy。与仅使用设计参数或全局布局统计特征的模型相比，图表征在可行性判断和钢筋用量排序两个任务上均表现出更好的综合性能，说明剪力墙布局的几何与拓扑信息对截面优化具有实质影响。

在线局部校准结果说明，少量当前布局上的 FEA 样本足以显著改善全局代理模型在单个优化任务中的适应性。采用 100 个局部样本后，可行性模型的概率校准误差明显降低，Brier score 从 0.0749 降至 0.0437；钢筋用量模型的 MAE 从 11.48 t 降至 2.04 t。该结果表明，即使全局代理模型无法在所有布局上保持绝对准确，优化过程中逐步积累的真实分析结果仍可以被有效转化为局部校准信息。

在 15 个测试布局和 5 个随机种子的优化实验中，结合可行性筛选与造价预排序的代理辅助策略平均减少约 64.6% 的真实 FEA 调用。与此同时，其平均可行解发现率为 63.6%，接近 full FEA baseline 的 65.8%；在双方均找到可行解的案例中，最终材料造价平均仅增加约 1.19%。这些结果表明，所提出方法能够在基本保持优化解质量的同时显著降低计算开销，并且这一收益在 GA、PSO 和 random search 三类优化算法中均能观察到。

本文工作的主要意义在于提供了一种更符合结构工程可靠性要求的代理模型使用方式。代理模型被用于筛选和排序候选方案，而最终设计仍由真实有限元分析和规范校核确认。对于需要反复评估大量候选方案的剪力墙结构优化任务，这种策略能够在不明显牺牲设计质量的前提下提高优化效率，从而增强智能优化方法在工程设计流程中的可用性。

本文仍存在若干局限。当前优化变量主要集中在截面尺寸和材料等级，尚未考虑墙体位置调整、边缘构件细化和更完整的施工构造约束。其次，本文测试布局虽然覆盖多组工程方案，但对与训练布局平面组织差异显著的新类型结构，例如超高层核心筒结构或特殊公共建筑平面，room graph 表征和全局 GNN 的泛化能力仍需进一步验证。再次，在线局部校准依赖优化过程中逐步积累的真实 FEA 样本，在优化早期样本极少时，筛选决策仍主要依赖全局代理模型，其可靠性和保守阈值选择需要更系统的不确定性分析。后续研究可进一步引入更丰富的工程设计变量，构建面向实际项目的连续优化与多目标优化框架，并发展带不确定性估计的代理筛选策略，以提升代理辅助结构优化的可靠性和适用范围。
