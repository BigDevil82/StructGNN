Reviewer #2: The manuscript proposes a room-level conditional graph neural network for shear wall layout generation in high-rise residential buildings. The topic is relevant to Automation in Construction, and the room-level graph formulation is a meaningful improvement over component-level representations. The proposed FiLM-based conditioning and dual-stream training strategy also address the practical issue that most engineering datasets provide only one finalized layout for each architectural plan. The experimental results show clear improvements over the component-based GNN baseline, and the finite-element case studies further support the practical potential of the method. The manuscript is generally well organized and technically sound, but several issues should be addressed before publication.

1. The motivation for using room-level graphs is clear, but the manuscript should discuss the possible information loss caused by this abstraction. Compared with component-level graphs, room-level graphs may weaken the representation of local wall-segment details, small openings, and fine-grained alignment constraints. The authors should clarify under what types of floor plans the proposed representation may become less effective, especially for irregular rooms, complex internal partitions, or cases where local geometric details strongly affect shear wall placement.

Response:
Thank you for this important comment. We agree that the room-level abstraction improves semantic alignment and computational compactness, but may also weaken fine-grained local geometric information. We therefore revised Section 3.1 to clarify the intended application scope and representation trade-off, and expanded Section 5.3 to discuss cases where component-level or hybrid representations may be more appropriate.

Revision:
In Section 3.1:
The proposed room-level formulation is therefore designed primarily for regular high-rise residential, apartment, hotel, and dormitory-type buildings, where rooms are mostly orthogonal and can be represented by rectangular or rectilinearly decomposed regions. In such cases, shear wall candidates are usually constrained by room boundaries, and room-level nodes provide a compact representation that is aligned with engineering practice. However, this abstraction may lose fine-grained local geometric details compared with component-level graphs. For floor plans with highly irregular spaces, long and narrow corridors, fragmented public areas, or many small local wall segments, rectilinear decomposition may introduce virtual partitions that do not correspond to actual structural decisions. Although these virtual boundaries can be excluded from feasible wall placement, they may still affect feature aggregation and model learning.

In Section 5.3:
Broader deployment would therefore require additional validation on more diverse typologies and, where necessary, hybrid or polygon-based representations that combine room-level semantics with selected component-level details.

2. The dataset contains 143 floor plans, and data augmentation increases the number of training samples to 678. The manuscript should clearly state how augmented samples are handled in the cross-validation process. All augmented versions of the same original floor plan should be kept within the same fold to avoid data leakage between training, validation, and test sets.

Response:
Thank you for pointing this out. We agree that augmentation must be handled carefully to avoid data leakage. We have clarified that cross-validation splitting is performed at the original floor-plan level before augmentation, and that all augmented variants of the same plan remain within the corresponding subset.

Revision:
In Section 4.1.3:
To avoid data leakage, the dataset split was performed at the original floor-plan level before data augmentation. All augmented variants of the same original floor plan were kept within the same subset. In each subset, geometric augmentation was applied only to the training portion, while validation and test layouts were never included in the training set in either original or augmented form.

3. The comparison with the component-based GNN baseline is useful, but the experimental positioning should be clarified. Since the paper also discusses pixel-based GAN and diffusion methods in the related work, the authors should explain why the experimental comparison focuses only on graph-based methods. A brief discussion comparing the proposed method with representative image-based methods in terms of representation, inference efficiency, vectorization requirements, and engineering applicability would make the positioning clearer.

Response:
Thank you for this suggestion. We agree that the experimental positioning should be made clearer, especially because image-based methods are discussed in the related work. We clarified that the main comparison is designed to isolate the effect of graph representation under a controlled graph-learning setting, and we added discussion on the differences between graph-based vector prediction and pixel-based generation in terms of representation, post-processing, and engineering usability.

Revision:
In Section 4.1.2:
The experimental comparison focuses on graph-based methods because the central question of this study is whether a room-level graph representation can improve graph-based shear wall layout prediction compared with the commonly used component-level graph representation. Pixel-based GAN or diffusion models follow a substantially different formulation, where both inputs and outputs are raster images and predicted wall layouts require additional vectorization before engineering analysis. A direct comparison with such models would involve differences in representation, post-processing, resolution, and evaluation pipeline, making it difficult to isolate the effect of graph representation. Therefore, the component-level GNN is selected as the primary baseline to provide a controlled comparison at the representation level.

In Section 4.2:
In the room-based graph, message passing occurs over room adjacencies and all candidate walls are attached to room boundaries from the outset. The model is therefore biased toward boundary-consistent predictions and is less likely to place isolated wall fragments in regions that do not correspond to valid room interfaces. This is important in engineering practice because fewer false positives mean lower unnecessary construction cost and fewer conflicts with architectural constraints.

4. Table 6 reports the main performance comparison, but only mean values are provided. Since the experiments are based on 5-fold cross-validation, the authors should report standard deviations or confidence intervals for the main metrics, including Image IoU, precision, recall, F1, MAE, and RMSE. This would help readers evaluate the stability of the reported improvements across different data splits.

Response:
Thank you for the comment. We agree that cross-fold variation is important for evaluating the stability of the reported improvements. We therefore updated the main comparison table to report mean and standard deviation across the 5-fold cross-validation and revised the accompanying discussion to interpret the stability of the results.

Revision:
In Section 4.2:
Across the five fold checkpoints, the room-level model achieves an Image IoU of 0.565 +/- 0.005 and an F1 score of 0.765 +/- 0.001, compared with 0.454 +/- 0.021 and 0.630 +/- 0.012 for the component-level baseline. The smaller cross-fold variation of the proposed method suggests that the improvement is not dominated by a single favorable split.

In Table 7:
Image IoU: 0.565 +/- 0.005 (ours) vs. 0.454 +/- 0.021 (baseline).
Precision: 0.798 +/- 0.001 (ours) vs. 0.677 +/- 0.013 (baseline).
Recall: 0.746 +/- 0.001 (ours) vs. 0.598 +/- 0.011 (baseline).
F1: 0.765 +/- 0.001 (ours) vs. 0.630 +/- 0.012 (baseline).
MAE: 0.093 +/- 0.003 (ours) vs. 0.182 +/- 0.013 (baseline).
RMSE: 0.193 +/- 0.003 (ours) vs. 0.353 +/- 0.008 (baseline).

5. The Conditional Generation Score is useful for evaluating conditional behavior, but its definition contains several manually selected weights. The manuscript should better justify the weighting strategy in CGS, especially the equal weights assigned to density agreement, matched-condition IoU, and spatial uniformity. A brief sensitivity analysis or additional explanation is needed to show that the conclusions are not overly dependent on these selected weights.

Response:
Thank you for this comment. We agree that the manually selected CGS weights should be justified more clearly. We revised the metric description to explain the role of each component and added a sensitivity analysis with alternative weight settings. The added results show that the main ablation trends are not overly dependent on the specific weighting choice.

Revision:
In Section 4.1.4:
In the main ablation analysis, the CGS weights are set to (w1,w2,w3)=(0.4,0.4,0.2). This setting gives comparable emphasis to density compliance and matched-condition geometric agreement, while assigning a smaller but non-negligible weight to spatial uniformity. The rationale is that the first two terms directly measure whether the generated layout follows the specified design condition and matches the engineer-designed layout under the true condition, whereas S_uni acts mainly as a regularity check that penalizes highly uneven or locally erratic distributions. CGS is therefore used as a descriptive composite score rather than as a code-prescribed engineering index.

In Table 6:
Equal weights (1/3,1/3,1/3): 0.708 (full model), 0.700 (w/o FiLM), 0.692 (w/o dual-stream), 0.651 (w/o density loss).
Density-oriented (0.5,0.25,0.25): 0.713, 0.713, 0.702, 0.640.
Geometry-oriented (0.25,0.5,0.25): 0.661, 0.652, 0.650, 0.624.
Uniformity-oriented (0.25,0.25,0.5): 0.729, 0.715, 0.725, 0.690.

6. The dual-stream training strategy is an important part of the proposed method, but the contribution of individual loss terms is not fully demonstrated. The authors should provide more evidence for the effects of the density loss and the topological consistency loss. For example, an ablation setting without the consistency loss, or a quantitative indicator measuring inconsistency on shared room boundaries, would make the role of this term clearer.

Response:
Thank you for this comment. We agree that the roles of the density loss and consistency loss should be stated more explicitly. The revised ablation discussion now directly interprets the variants without density loss and without consistency loss, and explains how their changes in CGS and sub-metrics reflect condition compliance and shared-boundary coherence.

Revision:
In Section 4.3:
The loss-function ablations in Table 9 and Fig. 6 demonstrate the roles of the density and consistency losses. Removing the density loss reduces CGS from 0.679 to 0.621, mainly because S_den decreases from 0.750 to 0.607, indicating weaker compliance with the specified density condition. Removing the consistency loss reduces CGS from 0.679 to 0.664 and lowers the matched-condition IoU from 0.541 to 0.513, suggesting reduced coherence on shared room boundaries. The larger degradation observed when all auxiliary losses are removed further confirms that these losses provide complementary supervision for condition compliance and boundary consistency.

7. The paper claims that the room-level representation reduces graph complexity and improves efficiency, but the experimental section mainly reports accuracy metrics. The authors should add a quantitative efficiency comparison between the proposed method and the component-level baseline, such as average node/edge numbers, inference time per floor plan, training time per epoch, or GPU memory consumption. This would directly support the claimed computational advantage.

Response:
Thank you for this comment. We agree that the claimed computational advantage should be supported quantitatively rather than only by accuracy results. We added an efficiency comparison between the room-level and component-level graphs, covering graph size, inference time per plan, training time per epoch, and GPU memory usage.

Revision:
In Section 4.2:
Table 8 provides a quantitative comparison of graph complexity and computational efficiency. The room-level representation reduces the average number of nodes from 244 to 26 and the average number of edges from 444 to 71, corresponding to reductions of 89.3% and 84.0%, respectively. This compact representation also improves efficiency in the measured implementation: inference time per plan decreases from 0.12 s to 0.06 s, training time per epoch decreases from 103 s to 86 s, and GPU memory usage decreases from 981 MB to 589 MB. These results support the claim that representing rooms rather than low-level wall components reduces graph complexity while lowering the computational cost of training and inference.

8. The current design condition is represented by three discrete density groups related to seismic intensity and building height. This setting is reasonable for the available dataset, but the manuscript should avoid overstating the level of engineering controllability. The current model mainly controls shear wall density categories rather than continuous engineering parameters such as PGA, structural height, period, or drift demand. This distinction should be stated more clearly in the abstract, experimental discussion, and conclusion.

Response:
Thank you for the comment. We agree that the controllability demonstrated in this study should not be overstated. We revised the abstract, experimental setup, discussion, and conclusion to clarify that the current model controls discrete density categories derived from available labels, rather than continuous engineering parameters such as PGA, structural height, period, or drift demand.

Revision:
In the Abstract:
Conditional modulation and a dual-stream training strategy are further introduced to support discrete density-controlled generation under limited paired engineering data.

In Section 4.1.1:
It should be noted that the condition label used in this study is a discrete density category rather than a continuous engineering descriptor. Therefore, the model learns to respond to low-, medium-, and high-density regimes, but it does not directly condition on continuous parameters such as PGA, structural height, fundamental period, inter-story drift demand, or torsional response.

In Section 5.3:
As a result, the condition input should be interpreted as a coarse design intent rather than a continuous engineering command.

In Section 6:
Future work should focus on three directions that follow directly from the current limits of the framework: supporting irregular polygonal rooms without decomposition, replacing discrete density categories with continuous engineering descriptors, and coupling generation more tightly with structural-performance objectives during training.

9. The qualitative results in the main text are helpful, but the number of visualized test cases is limited. The authors should provide more visual comparison results in the appendix, covering different density groups and floor plan types. Additional examples, including successful cases and less satisfactory cases, would help readers better understand the model's actual generation quality, robustness, and typical failure modes.

Response:
Thank you for the suggestion. We agree that additional visual cases help readers assess the actual generation quality and typical failure modes. We added Appendix A with more test-set qualitative comparisons across density groups, and referenced these additional examples in the main results section.

Revision:
In Section 4.2:
The qualitative comparisons in Fig. 5 provide visual evidence consistent with the quantitative results. Further illustrative examples are shown in Fig. A.1.

In Appendix A:
Fig. A.1 presents additional qualitative comparisons on test cases. These additional cases further show that the proposed room-based method generally produces more coherent and boundary-aligned shear wall layouts than the edge-based baseline. They also reveal remaining local errors in some predictions, such as overly short shear wall segments or missing walls in detailed regions, indicating that fine-grained local refinement remains a direction for future improvement.
