Reviewer #1: The manuscript proposes a room-level conditional graph neural network for shear wall layout generation. The topic is relevant to automated structural design, and the overall workflow is interesting. The manuscript is generally well organized and technically sound. However, several minor issues should be addressed before acceptance. The reviewer would like to invite the authors to revise the manuscript considering the comments below.

1. Since the proposed method relies on room-level graph construction, please provide more details on how rooms, openings, feasible wall boundaries, and irregular spatial regions are extracted from the original floor plans. A brief algorithmic description or additional explanation in the methodology section would be helpful.

Response:
Thank you for this suggestion. We revised Section 3.1.1 to explain the CAD preprocessing and graph construction steps more explicitly, including room extraction, opening mapping, feasible-boundary identification, and the treatment of irregular regions.

Revision:
In Section 3.1.1:
Given an architectural floor plan P in CAD format, the room adjacency graph G is constructed through the following procedure. The preprocessing stage converts CAD primitives into room polygons, opening-aware boundary constraints, and graph connectivity.

Step 1: Space Partitioning. Wall, door and window entities are first parsed from the CAD drawing according to their layer types. Double-line wall boundaries are converted into wall centerlines, and door/window embedded segments are projected onto their host walls as openings. The wall centerlines are then calibrated by closing small drafting gaps, merging nearly collinear segments, and adjusting intersections within a geometric tolerance, producing a connected floor-plan skeleton. Polygonizing this calibrated skeleton yields closed spatial regions. To process complex architectural layouts, a rectilinear decomposition algorithm is applied to partition any non-rectangular regions (e.g., L-shaped or T-shaped spaces) into a set of axis-aligned rectangular sub-regions. Corridors and circulation spaces follow the same rule: rectangular regions remain unchanged, whereas irregular connected regions are decomposed into rectangular sub-regions so that the fixed boundary-slot representation introduced in Section 3.1.2 remains applicable.

Step 2: Room Node Registration. Openings are associated with the corresponding room-boundary segments according to geometric overlap, so that wall placement is prohibited where doors or windows interrupt a boundary.

2. Please clarify whether the main results are averaged over all folds or obtained from a representative split. If possible, please report mean and standard deviation for the main evaluation metrics to better support the robustness of the comparison.

Response:
Thank you for this comment. We clarified that the main results are based on 5-fold cross-validation and revised the main comparison table to report mean and standard deviation.

Revision:
In Section 4.2:
The proposed room-based representation yields higher fold-averaged Image IoU and F1 than the component-based baseline under the same backbone family and training protocol. Across the five fold checkpoints, the room-level model achieves an Image IoU of 0.565 +/- 0.005 and an F1 score of 0.765 +/- 0.001, compared with 0.454 +/- 0.021 and 0.630 +/- 0.012 for the component-level baseline. The smaller cross-fold variation of the proposed method suggests that the improvement is not dominated by a single favorable split. The ensemble result remains higher (Image IoU = 0.597), but Table 7 reports fold-level statistics to make model stability explicit.

In Table 7:
Image IoU: 0.565 +/- 0.005 (ours) vs. 0.454 +/- 0.021 (baseline).
Precision: 0.798 +/- 0.001 (ours) vs. 0.677 +/- 0.013 (baseline).
Recall: 0.746 +/- 0.001 (ours) vs. 0.598 +/- 0.011 (baseline).
F1: 0.765 +/- 0.001 (ours) vs. 0.630 +/- 0.012 (baseline).
MAE: 0.093 +/- 0.003 (ours) vs. 0.182 +/- 0.013 (baseline).
RMSE: 0.193 +/- 0.003 (ours) vs. 0.353 +/- 0.008 (baseline).

3. Please explain how the low-, medium-, and high-density groups are determined, and whether the density targets used for training and evaluation are computed independently within each training split. This would help readers better understand the controllability evaluation.

Response:
Thank you for this comment. We clarified the definition of the three condition groups and stated how the density targets are computed and used in training and evaluation.

Revision:
In Section 4.1.1:
Following the condition taxonomy introduced in prior shear wall studies, which groups projects by the combined effects of seismic intensity and structural height, samples are categorized into three design-condition groups: the low-density group (Group7-H1), corresponding to seismic intensity 7 (PGA = 0.10 g) and buildings with height H <= 50 m; the medium-density group (Group7-H2), sharing the same seismic intensity but covering buildings with H > 50 m; and the high-density group (Group8), corresponding to seismic intensity 8 (PGA = 0.20 g). This grouping is used because the available dataset does not provide a sufficiently dense sampling of continuous design parameters for direct regression. The three groups are defined before model training and are used as discrete condition inputs throughout the experiments.

In Section 4.1.4:
In our implementation, the target densities used in both the density loss and CGS evaluation are fixed group-level averages computed from the complete set of available layouts after condition grouping, rather than being recomputed separately within each cross-validation fold.

4. Please consider adding a small table comparing the generated and engineer-designed schemes in terms of key indicators, such as maximum inter-story drift ratio, code limit, vertical displacement, and wall density.

Response:
Thank you for this suggestion. The requested indicators are reported in the finite-element validation figures, and we revised Section 4.5 to make this explicit and summarize their implications.

Revision:
In Section 4.5:
The comparison focuses on key preliminary-design indicators shown in Figs. 8 and 9, including the predicted and engineer-designed layouts, inter-story drift ratios, the code limit, and maximum vertical displacement.

As shown in Fig. 8, the predicted schemes in both cases exhibit drift-ratio profiles close to those of the engineer-designed schemes, and all story drift ratios remain below the code limit of 1/1000. This result indicates that, for these two projects, the generated layouts maintain lateral stiffness at a level comparable to the reference schemes. Fig. 9 provides a consistent picture: the displacement cloud patterns of the predicted and engineer-designed schemes are similar, and the differences in maximum floor vertical displacement remain small.

5. Please briefly discuss the limitations of the proposed method, such as the limited dataset size, dependence on preprocessing quality, use of discrete density conditions, and the need for subsequent structural analysis and engineering verification before practical application.

Response:
Thank you for this comment. We expanded the limitations in Section 5.3 and added a concise limitation statement in Section 6 to clarify the current scope of the method.

Revision:
In Section 5.3:
Despite the promising results, the limitations of the current framework are best understood in terms of their impact on engineering use. The first limitation concerns the transferability of a room-level abstraction beyond the building types represented in the present data. The current dataset contains 143 high-rise residential layouts, so broader generalization remains constrained by both data scale and typological diversity.

The second limitation concerns optimization target mismatch. The model learns from existing engineering layouts and is regularized by density-related constraints, but it is not trained to optimize structural response indicators such as drift ratio, period, or torsional behavior. In addition, the method depends on reliable preprocessing of CAD drawings into room graphs and buildable masks; ambiguous room partitions or inaccurate opening extraction can directly affect the prediction space.

The third limitation is the resolution of conditional control. The experimental setup in Section 4.1.1 defines the available labels as a small number of density regimes, which is sufficient for testing whether the model responds to different design demands but not for prescribing a target performance value. As a result, the condition input should be interpreted as a coarse design intent rather than a continuous engineering command.

In Section 6:
The current framework still has several limitations, corresponding to the issues discussed above. Its validation is limited to regular high-rise residential layouts, so broader typological transfer requires further testing, especially for plans with highly irregular regions or strong segment-level constraints. The method also depends on reliable CAD preprocessing and does not directly optimize structural response indicators during training. In addition, the condition input is represented by discrete density groups, and the present validation assumes a repeated standard-floor layout rather than explicitly modeling vertical layout variation.
