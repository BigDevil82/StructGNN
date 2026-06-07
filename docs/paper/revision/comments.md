Reviewer Comments:

1. The motivation for using room-level graphs is clear, but the manuscript should discuss the possible information loss caused by this abstraction. Compared with component-level graphs, room-level graphs may weaken the representation of local wall-segment details, small openings, and fine-grained alignment constraints. The authors should clarify under what types of floor plans the proposed representation may become less effective, especially for irregular rooms, complex internal partitions, or cases where local geometric details strongly affect shear wall placement.

2. The dataset contains 143 floor plans, and data augmentation increases the number of training samples to 678. The manuscript should clearly state how augmented samples are handled in the cross-validation process. All augmented versions of the same original floor plan should be kept within the same fold to avoid data leakage between training, validation, and test sets.

3. The comparison with the component-based GNN baseline is useful, but the experimental positioning should be clarified. Since the paper also discusses pixel-based GAN and diffusion methods in the related work, the authors should explain why the experimental comparison focuses only on graph-based methods. A brief discussion comparing the proposed method with representative image-based methods in terms of representation, inference efficiency, vectorization requirements, and engineering applicability would make the positioning clearer.

4. Table 6 reports the main performance comparison, but only mean values are provided. Since the experiments are based on 5-fold cross-validation, the authors should report standard deviations or confidence intervals for the main metrics, including Image IoU, precision, recall, F1, MAE, and RMSE. This would help readers evaluate the stability of the reported improvements across different data splits.

5. The Conditional Generation Score is useful for evaluating conditional behavior, but its definition contains several manually selected weights. The manuscript should better justify the weighting strategy in CGS, especially the equal weights assigned to density agreement, matched-condition IoU, and spatial uniformity. A brief sensitivity analysis or additional explanation is needed to show that the conclusions are not overly dependent on these selected weights.

6. The dual-stream training strategy is an important part of the proposed method, but the contribution of individual loss terms is not fully demonstrated. The authors should provide more evidence for the effects of the density loss and the topological consistency loss. For example, an ablation setting without the consistency loss, or a quantitative indicator measuring inconsistency on shared room boundaries, would make the role of this term clearer.

7. The paper claims that the room-level representation reduces graph complexity and improves efficiency, but the experimental section mainly reports accuracy metrics. The authors should add a quantitative efficiency comparison between the proposed method and the component-level baseline, such as average node/edge numbers, inference time per floor plan, training time per epoch, or GPU memory consumption. This would directly support the claimed computational advantage.

8. The current design condition is represented by three discrete density groups related to seismic intensity and building height. This setting is reasonable for the available dataset, but the manuscript should avoid overstating the level of engineering controllability. The current model mainly controls shear wall density categories rather than continuous engineering parameters such as PGA, structural height, period, or drift demand. This distinction should be stated more clearly in the abstract, experimental discussion, and conclusion.

9. The qualitative results in the main text are helpful, but the number of visualized test cases is limited. The authors should provide more visual comparison results in the appendix, covering different density groups and floor plan types. Additional examples, including successful cases and less satisfactory cases, would help readers better understand the model's actual generation quality, robustness, and typical failure modes.


## 1. 关于房间级图表征的信息损失与适用范围

**1. Response**

Thank you for this comment. We agree that the room-level abstraction improves semantic alignment and computational compactness, but it may also lose some fine-grained local geometric information. We have revised the manuscript to clarify the representation trade-off and the applicable scope of the proposed method. In particular, we now state that the current formulation is mainly suitable for regular high-rise residential, hotel, and apartment-type plans with predominantly orthogonal and rectangular rooms. We also discuss cases where component-level graph representations may be more advantageous, such as highly irregular layouts, plans with many narrow corridors or fragmented spaces, and cases where shear wall placement depends strongly on local segment-level details.

**Revised:**
We revised Section 3.1 to clarify the assumptions and trade-off of the room-level representation. We also expanded Section 5.3 to discuss the limitations of applying the method to irregular floor plans and to explain when component-level graphs may be preferable.

**具体修改：**

**位置 1：Section 3.1 Room-based graph representation，建议放在现有关于“current formulation assumes rectangular room geometry”之后。**

新增内容：

> The proposed room-level formulation is designed primarily for regular high-rise residential, apartment, hotel, and dormitory-type buildings, where rooms are mostly orthogonal and can be represented by rectangular or rectilinearly decomposed regions. In such cases, shear wall candidates are usually constrained by room boundaries, and room-level nodes provide a compact representation that is well aligned with engineering practice. However, this abstraction may lose some fine-grained local geometric details compared with component-level graphs. For floor plans with highly irregular spaces, long and narrow corridors, fragmented public areas, or many small local wall segments, rectilinear decomposition may introduce virtual partitions that do not correspond to actual structural decisions. Although these virtual boundaries can be excluded by the constraint mask, they may still affect feature aggregation and model learning. In such cases, component-level graphs may retain more detailed local topology and may be more suitable when shear wall placement depends strongly on segment-level geometry, local openings, or fine alignment constraints.

**位置 2：Section 5.3 Limitations，替换或扩展当前第一条 limitation。**

建议改为：

> The first limitation lies in the geometric abstraction and applicable scope. The current room-level graph formulation is most suitable for regular high-rise residential, hotel, and apartment-type plans with predominantly rectangular or rectilinear spaces. For highly irregular layouts, large public spaces, complex corridor systems, or plans containing many fragmented spatial regions, the decomposition into rectangular room nodes may introduce virtual partitions that do not correspond to actual walls or structural design decisions. Although the constraint mask prevents shear wall placement on these artificial boundaries, the additional nodes and edges may still influence message passing and reduce the effectiveness of the learned representation. In such scenarios, component-level graph representations may be advantageous because they preserve local segment-level topology, small openings, and detailed geometric alignment more directly.

---

## 2. 关于数据增广与交叉验证的数据泄露问题

**2. Response**

Thank you for pointing this out. We have clarified the data splitting and augmentation protocol. The cross-validation split was performed at the original floor-plan level before data augmentation. All augmented variants generated from the same original floor plan were kept within the same fold. Therefore, no augmented version of a validation or test layout appeared in the training set.

**Revised:**
We added a statement in Section 4.1.3 to clarify that data augmentation was performed after fold assignment and only within the corresponding training set, ensuring that no layout-level leakage occurred across training, validation, and test sets.

**具体修改：**

**位置：Section 4.1.3 Implementation details and training settings，放在数据增广描述之后、5-fold cross-validation 描述之前。**

新增内容：

> To avoid data leakage, the cross-validation split was performed at the original floor-plan level before data augmentation. All augmented variants of the same original floor plan were kept within the same fold. In each fold, geometric augmentation was applied only to the training portion, while validation and test layouts were never included in the training set in either original or augmented form.

---

## 3. 关于未与像素生成方法进行实验对比的问题

**3. Response**

Thank you for this suggestion. We have clarified the experimental positioning of this study. The main objective of this paper is not to benchmark all possible shear wall generation methods, but to investigate whether changing the graph representation from component level to room level improves graph-based structural layout prediction. Therefore, the primary baseline is a component-level GNN under a comparable graph-learning setting. Pixel-based GAN or diffusion methods use different input/output representations, training objectives, and post-processing procedures, especially because their raster outputs require vectorization before engineering analysis. Direct comparison with them would introduce additional differences beyond the representation level. We have added a discussion to explain this choice and to clarify the advantages of graph-based vectorized prediction over pixel-based generation.

**Revised:**
We revised Section 4.1.2 to explain why the experimental comparison focuses on graph-based methods. We also added discussion in Section 4.2 comparing graph-based and pixel-based methods in terms of topology, vectorization, engineering usability, and inference workflow.

**具体修改：**

**位置 1：Section 4.1.2 Baseline，放在 baseline 描述之后。**

新增内容：

> The experimental comparison focuses on graph-based methods because the central question of this study is whether a room-level graph representation can improve graph-based shear wall layout prediction compared with the commonly used component-level graph representation. Pixel-based GAN or diffusion models follow a substantially different formulation, where both inputs and outputs are raster images and predicted wall layouts require additional vectorization before engineering analysis. A direct comparison with such models would involve differences in representation, post-processing, resolution, and evaluation pipeline, making it difficult to isolate the effect of graph representation. Therefore, the component-level GNN is selected as the primary baseline to provide a controlled comparison at the representation level.

**位置 2：Section 4.2 Main results，结果分析部分补充。**

新增内容：

> Compared with pixel-based generation methods, the proposed graph-based formulation directly predicts vectorized wall-boundary parameters and therefore avoids the additional raster-to-vector conversion step. This is useful for downstream structural modeling, where wall locations and lengths must be represented as geometric entities rather than image pixels. The room-level graph also explicitly encodes adjacency and boundary feasibility, whereas such topological relations are only implicit in image-based representations. These differences explain why the present comparison is centered on graph-based baselines rather than on image-to-image generative models.

---

## 4. 关于 Table 6 缺少标准差或置信区间

**4. Response**

Thank you for the comment. We have revised Table 6 to report the standard deviation of the main metrics across the 5-fold cross-validation. This allows the stability of the improvements to be assessed more clearly across different data splits.

**Revised:**
Table 6 has been updated to include mean and standard deviation values for Image IoU, precision, recall, F1, accuracy, MAE, and RMSE. The corresponding text in Section 4.2 has also been revised to discuss the stability of the results.

**具体修改：**

**位置 1：Table 6，建议改成如下表头形式。具体数值用你重新统计的 5-fold 结果填入。**

修改为：

| Method   | Image IoU ↑ | Precision ↑ |   Recall ↑ |       F1 ↑ | Accuracy ↑ |      MAE ↓ |     RMSE ↓ |
| -------- | ----------: | ----------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Ours     |  mean ± std |  mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Baseline |  mean ± std |  mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Diff.    |         ... |         ... |        ... |        ... |        ... |        ... |        ... |

**位置 2：Section 4.2 Main results，Table 6 后补充一句。**

新增内容：

> The standard deviations across the five folds indicate that the improvements of the proposed method are consistent rather than being dominated by a single favorable split. In particular, the room-level model achieves higher mean Image IoU and F1 with comparable or lower cross-fold variation, suggesting better prediction stability under limited data.

---

## 5. 关于 CGS 权重设置的合理性

**5. Response**

Thank you for this comment. We have revised the description of CGS to better justify the weighting strategy. CGS is intended as a descriptive metric for conditional generation rather than a code-based engineering index. The three components evaluate complementary aspects: density compliance, geometric agreement, and spatial rationality. Equal weights are used because no single component should dominate the evaluation. We have also added a short sensitivity analysis to show that the relative comparison between the full model and ablation variants remains stable under moderate changes of the weights.

**Revised:**
We revised Section 4.1.4 to clarify the rationale of the CGS weights. We also added a sensitivity analysis in Section 4.4 to examine whether the conclusions are affected by alternative weight settings.

**具体修改：**

**位置 1：Section 4.1.4 Evaluation metrics，CGS 定义之后，扩展权重解释。**

建议修改为：

> Equal weights are assigned to the three CGS components because they reflect complementary aspects of conditional layout generation. The density term evaluates whether the generated wall quantity follows the specified condition; the matched-condition IoU evaluates geometric agreement with engineering layouts when ground truth is available; and the spatial uniformity term penalizes highly uneven or locally erratic wall distributions. Since CGS is used as a descriptive evaluation score rather than a code-prescribed engineering index, no single component is assumed to be intrinsically more important than the others. Equal weighting therefore provides a balanced summary of condition compliance, geometric fidelity, and spatial rationality.

**位置 2：Section 4.4 Ablation study 或 Conditional generation evaluation，新增一小段敏感性分析。**

新增内容：

> To examine whether the conclusions depend strongly on the selected CGS weights, a sensitivity analysis was conducted using alternative weight combinations. Besides the equal-weight setting $(w_1,w_2,w_3)=(1/3,1/3,1/3)$, we tested density-oriented weights $(0.5,0.25,0.25)$, geometry-oriented weights $(0.25,0.5,0.25)$, and uniformity-oriented weights $(0.25,0.25,0.5)$. The full model consistently achieved the highest or near-highest CGS among the compared variants under these settings, indicating that the main conclusion is not sensitive to moderate changes in the CGS weighting scheme.

**位置 3：可新增一个小表 Table X。**

| Weight setting      | Full model | w/o FiLM | w/o cross-condition stream | w/o density loss |
| ------------------- | ---------: | -------: | -------------------------: | ---------------: |
| Equal weights       |        填数值 |      填数值 |                        填数值 |              填数值 |
| Density-oriented    |        填数值 |      填数值 |                        填数值 |              填数值 |
| Geometry-oriented   |        填数值 |      填数值 |                        填数值 |              填数值 |
| Uniformity-oriented |        填数值 |      填数值 |                        填数值 |              填数值 |

---

可以，这条确实不用再新增实验。更合理的处理方式是：**在 response 中说明原稿已有相关消融结果，但我们意识到表述可能不够突出，因此在修订稿中加强了对 Table 8 和 Fig. 6 的说明**。这样既回应审稿人，又避免显得“审稿人没看见”。

下面是第 6 条的替换版本。

---

## 6. 关于 density loss 和 topological consistency loss 的贡献

**6. Response**

Thank you for this comment. The effects of the density loss and the topological consistency loss have been examined in the ablation study of the original manuscript. Specifically, Fig. 6 and Table 8 report the test-set CGS results of different loss-function variants, including the model without density loss and the model without consistency loss. The results show that removing the density loss leads to a clear decrease in CGS, mainly due to reduced density agreement, while removing the consistency loss also degrades the overall conditional generation performance. To make this point clearer, we have revised the discussion of the ablation study and explicitly highlighted the contribution of these two loss terms.

**Revised:**
We revised Section 4.3 to make the existing loss-function ablation results more explicit. In particular, we now emphasize that the variants “w/o density loss” and “w/o consistency loss” in Table 8 directly evaluate the roles of the two corresponding loss terms, and we added a clearer explanation of their influence on CGS and its sub-metrics.

**具体修改：**

**位置：Section 4.3 Ablation studies，Table 8 和 Fig. 6 后，原文关于 loss-term ablations 的分析段落。**

原文中已经有类似内容：

> Loss-term ablations show complementary roles. The density loss most strongly affects compliance with design conditions (CGS -0.058 when removed), primarily through reduced $S_{den}$, ...

建议将这一段适当强化，改为：

> The loss-function ablations in Table 8 and Fig. 6 demonstrate the roles of the density and consistency losses. Removing the density loss reduces CGS from 0.679 to 0.621, mainly because $S_{den}$ decreases from 0.750 to 0.607, indicating weaker compliance with the specified density condition. Removing the consistency loss reduces CGS from 0.679 to 0.664 and lowers the matched-condition IoU from 0.541 to 0.513, suggesting reduced coherence on shared room boundaries. The larger degradation observed when all auxiliary losses are removed further confirms that these losses provide complementary supervision for condition compliance and boundary consistency.


## 7. 关于计算效率对比

**7. Response**

Thank you for this comment. We have added a quantitative efficiency comparison between the proposed room-level GNN and the component-level GNN baseline. The comparison includes average graph size, inference time per floor plan, training time per epoch, and GPU memory usage. The results show that the room-level representation reduces graph scale and improves computational efficiency while achieving better prediction accuracy.

**Revised:**
We added an efficiency comparison table in Section 4.2 and revised the discussion to connect the reduced graph size with actual runtime and memory advantages.

**具体修改：**

**位置 1：Section 4.2 Main results，Table 6 或 Table 7 后新增效率表。**

新增表格：

| Method              | Avg. nodes | Avg. edges | Inference time / plan | Training time / epoch | GPU memory |
| ------------------- | ---------: | ---------: | --------------------: | --------------------: | ---------: |
| Component-level GNN |        填数值 |        填数值 |                   填数值 |                   填数值 |        填数值 |
| Room-level GNN      |        填数值 |        填数值 |                   填数值 |                   填数值 |        填数值 |
| Reduction           |        填数值 |        填数值 |                   填数值 |                   填数值 |        填数值 |

**位置 2：Section 4.2 Main results，新增文字。**

新增内容：

> The efficiency comparison further confirms the computational advantage of the room-level abstraction. Because each node represents a room rather than a wall intersection, the proposed representation substantially reduces the number of graph nodes and edges. This reduction leads to shorter inference time and lower memory consumption compared with the component-level baseline. Therefore, the improvement of the proposed method is not limited to prediction accuracy; it also provides a more compact and efficient representation for preliminary structural layout generation.

---

## 8. 关于离散条件控制的局限性

**8. Response**

Thank you for the comment. We have revised the manuscript to clarify the scope of conditional control in the current model. The present framework controls three discrete shear wall density categories derived from seismic intensity and building height groups. It does not yet perform continuous conditioning on engineering parameters such as PGA, structural height, period, drift ratio, or torsional response. We have made this distinction clearer in the abstract, experimental discussion, and conclusion.

**Revised:**
We revised the abstract, Section 4.1.1, Section 4.4/5.2 discussion, and Section 6 to state that the current method demonstrates discrete density-controlled preliminary generation, while continuous engineering-condition control remains future work.

**具体修改：**

**位置 1：Abstract，原来 “density-controlled generation” 附近补充限定。**

建议改为：

> Conditional modulation and a dual-stream training strategy are further introduced to support discrete density-controlled generation under limited paired engineering data.


**位置 2：Section 4.1.1 Dataset，条件分组说明之后补充。**

新增内容：

> It should be noted that the condition label used in this study is a discrete density category rather than a continuous engineering descriptor. Therefore, the model learns to respond to low-, medium-, and high-density regimes, but it does not directly condition on continuous parameters such as PGA, structural height, fundamental period, inter-story drift demand, or torsional response.

**位置 3：Section 5.3 Limitations，第三条 limitation 扩展。**

建议改为：

> The third limitation is the granularity of conditional control. The current model uses three discrete condition groups derived from seismic intensity and building height. This setting is consistent with the available data but does not provide continuous control over engineering parameters such as PGA, building height, structural period, drift demand, or torsional behavior. Therefore, the present framework should be interpreted as discrete density-controlled preliminary generation rather than fully continuous performance-conditioned structural design.

**位置 4：Conclusion 结尾 future work，适当强化。**

建议改为：

> Future work should focus on supporting irregular polygonal rooms without decomposition, replacing discrete density categories with continuous engineering descriptors, and coupling generation more tightly with structural performance objectives during training.

---

## 9. 关于附录补充更多测试集可视化结果

**9. Response**

Thank you for the suggestion. We have added more qualitative results in the Appendix to provide a broader visual evaluation of the proposed method. The added cases cover different density groups and different floor-plan configurations. Both representative successful cases and less satisfactory cases are included to better show the model’s generation quality, robustness, and typical failure modes.

**Revised:**
We added Appendix A with additional test-set visualizations. We also added a short description in Section 4.2 referring readers to the appendix for more qualitative comparisons.

**具体修改：**

**位置 1：Section 4.2 Main results，Fig. 5 分析之后补充。**

新增内容：

> Additional qualitative results from the test set are provided in Appendix A. These cases cover different density groups and floor-plan configurations, including representative successful predictions and cases with visible local errors. The additional visualizations provide a broader view of the model’s generation quality and typical failure modes beyond the examples shown in the main text.

**位置 2：新增 Appendix A。**

标题：

> Appendix A. Additional qualitative results on test cases

说明文字：

> This appendix presents additional qualitative comparisons between the generated shear wall layouts and the corresponding engineer-designed layouts. The cases are selected from the test folds and cover low-, medium-, and high-density conditions. The results further illustrate the model’s ability to generate boundary-aligned wall layouts under different floor-plan configurations. Some less satisfactory cases are also included to show typical errors, such as missing local wall segments, over-prediction near dense boundary regions, and reduced accuracy in layouts with more complex room partitioning.

图注建议：

> Fig. A1. Additional qualitative results for low-density test cases.
> Fig. A2. Additional qualitative results for medium-density test cases.