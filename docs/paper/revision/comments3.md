## 1. 关于 CAD 图纸过滤与轮廓检测过程说明不足

**1. Response**

Thank you for this comment. We have revised the methodology section to describe the CAD preprocessing procedure more clearly. The revised manuscript now explains how key architectural elements, including walls, doors, and windows, are extracted from CAD layers, how double-line walls are converted into wall centerlines, how door and window embedded lines are identified, and how the floor-plan skeleton is calibrated to form closed room regions. We also clarify that irregular spatial regions are further decomposed into rectangular sub-regions before graph construction.

**Revised:**
We revised Section 3.1.1 by adding a more explicit description of the CAD preprocessing workflow, including layer-based element extraction, wall centerline extraction, opening embedding-line extraction, skeleton calibration, room-contour identification, and rectangular decomposition of irregular regions.

**具体修改：**

**位置：Section 3.1.1 Graph construction from floor plans，替换或扩展原 Step 1。**

建议修改为：

> **Step 1: CAD element extraction and skeleton construction.** The original CAD floor plan is first processed according to layer information. Key architectural elements, including walls, doors, and windows, are extracted from their corresponding CAD layers. Since architectural walls are usually represented by double-line boundaries, a centerline extraction algorithm is applied to convert double-line wall representations into wall axis lines. Meanwhile, doors and windows are projected onto their host walls, and their embedded line segments on the wall axis are extracted. These wall axis lines and opening embedded lines jointly form the skeleton of the architectural layout.
>
> After skeleton extraction, a geometric calibration step is applied to ensure that adjacent line segments are properly connected. Small gaps caused by drafting errors are closed, nearly collinear segments are merged, and intersections are adjusted within a predefined tolerance. This calibrated skeleton is then used to identify closed room contours. For irregular closed regions, rectilinear decomposition is further applied to partition them into rectangular sub-regions before room-level graph construction.

---

## 2. 关于走廊和交通空间在房间级图中的处理方式

**2. Response**

Thank you for this comment. We have clarified how corridors and circulation spaces are handled in the current room-level graph. Since the proposed output representation uses a fixed 16-dimensional boundary parameterization, each spatial region needs to be represented as a rectangular or rectilinearly decomposed room node. Therefore, irregular corridors and connected circulation spaces are decomposed into a minimum set of rectangular sub-regions. Virtual boundaries introduced by this decomposition are marked in the constraint mask and are excluded from both loss computation and final wall placement. We have also added a discussion that this strategy works for layouts with limited irregular regions, but more general representations are needed when irregular corridors or circulation spaces dominate the floor plan.

**Revised:**
We revised Section 3.1.1 and Section 5.3 to explain the treatment of corridors and circulation spaces. The revised text clarifies that irregular circulation areas are decomposed into rectangular sub-regions, while virtual boundaries are masked out during both training and inference.

**具体修改：**

**位置 1：Section 3.1.1 Graph construction from floor plans，放在不规则区域矩形分割说明之后。**

新增内容：

> Corridors and circulation spaces are processed using the same rule as other spatial regions. If a corridor or connected circulation area is rectangular, it is directly registered as one room node. If it is irregular, it is decomposed into a minimum set of axis-aligned rectangular sub-regions so that the fixed 16-dimensional boundary representation can be applied. The internal boundaries introduced by this decomposition are treated as virtual boundaries rather than physical walls. Their corresponding entries in the constraint mask are set to zero, so they do not contribute to the supervised loss during training and cannot be selected as shear wall locations during inference.

**位置 2：Section 5.3 Limitations，增加一句关于走廊和不规则区域的限制。**

新增内容：

> This decomposition-based treatment is effective when irregular regions and corridor spaces are limited, as in most regular residential layouts considered in this study. However, if a floor plan contains many highly irregular circulation spaces or large connected public areas, the decomposition may introduce many virtual boundaries and reduce the semantic clarity of the room-level graph. In such cases, a more general polygon-based or hybrid room-component representation may be more appropriate.

---

## 3. 关于剪力墙有效长度的定义标准

**3. Response**

Thank you for this comment. We have clarified the criterion used to define the effective length of shear walls during label construction and density calculation. In the revised manuscript, the effective length is defined as the projected overlap length between an extracted structural wall segment and a feasible room-boundary slot. Wall segments located on openings or virtual decomposition boundaries are excluded by the constraint mask. Very short overlaps caused by drafting or mapping noise are discarded using a geometric tolerance. When multiple wall segments correspond to the same boundary slot, their projected lengths are merged before computing the coverage ratio.

**Revised:**
We revised Section 3.1.1 and Section 3.1.4 to explain how shear wall segments from structural drawings are mapped to room-boundary slots, how their effective lengths are calculated, and how the resulting length ratios are used for the 16-dimensional ground-truth label and density score.

**具体修改：**

**位置 1：Section 3.1.1 Graph construction from floor plans，Step 4 / Label mapping 部分补充。**

新增内容：

> During label mapping, shear wall segments are extracted from the structural drawing and projected onto the corresponding room-boundary slots. The effective length of a shear wall is defined as the projected overlap length between the structural wall segment and a feasible room-boundary slot. Wall segments located on openings, non-buildable boundaries, or virtual decomposition boundaries are not counted because the corresponding mask entries are set to zero. Very short overlaps caused by drafting noise or geometric tolerance are discarded. If multiple wall segments overlap the same boundary slot, their projected intervals are merged before the effective length is calculated.

**位置 2：Section 3.1.4 Output parameterization，16 维输出定义后补充。**

新增内容：

> For each boundary slot, the ground-truth coverage ratio is calculated by dividing the effective shear wall length within that slot by the slot length. Therefore, a value of 1.0 indicates that the slot is fully covered by shear wall, while an intermediate value indicates partial wall coverage. These coverage ratios form the 16-dimensional ground-truth vector and are also used to compute the density score in the condition-aware loss.

---

## 4. 关于 16 维 constraint mask 的含义

**4. Response**

Thank you for this suggestion. We have revised the manuscript to describe the 16-dimensional constraint mask more explicitly. The constraint mask has the same dimension and ordering as the 16-dimensional output vector. Each component indicates whether the corresponding room-boundary slot is feasible for shear wall placement. A value of 1 means that the slot is a physical, buildable boundary, while a value of 0 indicates that the slot is occupied by an opening, belongs to a virtual decomposition boundary, or is otherwise infeasible for shear wall placement. We have also revised the illustration of the room-level output parameterization to show the correspondence between the output vector and the mask vector.

**Revised:**
We revised Section 3.1.2 and Section 3.1.4 to explicitly define each component of the 16-dimensional mask. We also updated the illustration of the room-level output parameterization to include the mask-vector correspondence.

**具体修改：**

**位置 1：Section 3.1.2 Node feature design，constraint mask 说明处扩展。**

建议修改为：

> The constraint information is represented by a 16-dimensional buildable mask $m_i \in {0,1}^{16}$, which has the same ordering as the 16-dimensional output vector $y_i$. The four room boundaries are first ordered as top, right, bottom, and left. Each boundary is divided into two half-segments, and each half-segment is represented by two directional coverage entries, resulting in 16 entries in total. Therefore, each mask component $m_{ij}$ corresponds one-to-one to the $j$-th output component $y_{ij}$. If $m_{ij}=1$, the corresponding boundary slot is a feasible location for shear wall placement. If $m_{ij}=0$, the slot is infeasible because it is occupied by a door, window, other opening, or because it corresponds to a virtual boundary introduced by rectilinear decomposition.

**位置 2：Section 3.1.4 Output parameterization，Fig. 3 附近增加说明。**

新增内容：

> The same ordering is used for both the output vector and the constraint mask. During training, the mask is used to exclude infeasible slots from regression and density-related losses. During inference, the predicted layout is multiplied by the mask so that shear walls cannot be placed on openings or virtual boundaries.

**位置 3：新增了 Figure 3(c)。**

我已在 “Illustration of room-level output parameterization” 中补充一个子图

图名建议改为：

> Figure 3. Illustration of the 16-dimensional output parameterization and the corresponding constraint mask. 

---

## 5. 关于 Figure 2 中部分区域没有与周围边界形成图边的问题

**5. Response**

Thank you for pointing this out. We have clarified the graph-construction rule used in Fig. 2. In the dataset preprocessing, room partitions were manually checked to ensure the quality of training labels. When a region enclosed by surrounding rectangles did not correspond to an independent room or did not introduce additional physical wall boundaries, it was not registered as an additional room node. This avoids repeatedly representing the same physical wall boundary through multiple overlapping rectangles, which could otherwise introduce inconsistent labels and unstable training. We have added this explanation to the caption and methodology description.

**Revised:**
We revised the description around Fig. 2 and added an explanatory note in Section 3.1.1. The revised text clarifies that not every visually enclosed background region is necessarily registered as a room node; only valid room or decomposed spatial regions that contribute meaningful physical boundaries are used for graph construction.

**具体修改：**

**位置 1：Section 3.1.1 Graph construction from floor plans，room node registration 后补充。**

新增内容：

> During dataset preparation, the room partitions were manually checked to ensure stable label mapping. If a visually enclosed region was already covered by the surrounding rectangular room partitions and did not correspond to an independent room or an additional physical wall boundary, it was not registered as a separate room node. This rule avoids duplicating the same physical boundary in multiple overlapping room nodes, which could lead to inconsistent predictions for the same shear wall segment during training.

---

## 6. 关于 Figure 1 中 cross-condition stream 作用说明不足

**6. Response**

Thank you for this comment. We have revised the explanation of the cross-condition stream. In practical engineering datasets, each floor plan is usually associated with only one design condition and one finalized shear wall layout. This makes it difficult for a conditional model to learn how the same layout should change under different conditions. The cross-condition stream addresses this issue by feeding the same floor-plan graph with a randomly sampled mismatched condition during training. Since no ground-truth layout is available for this synthetic pairing, the model is supervised by a density-alignment loss with the target density of the sampled condition. This provides condition-related gradients and encourages the model to generate different wall densities for the same layout under different conditions.

**Revised:**
We revised Section 3.4.1 and the caption of Fig. 1 to more clearly explain the role of the cross-condition stream, the reason for using fake conditions, and how density regularization enables conditional generation under sparse paired supervision.

**具体修改：**

**位置 1：Section 3.4.1 Dual-stream training framework，重写或扩展 cross-condition stream 说明。**

建议修改为：

> The cross-condition stream is introduced to address the lack of multi-condition paired labels. In real engineering projects, one architectural floor plan is typically associated with only one design condition and one finalized shear wall layout. Therefore, the training data do not directly show how the shear wall layout of the same plan should vary under different seismic or height-related conditions. If only the supervised stream is used, the model may learn to reconstruct the observed layout mainly from geometry and may ignore the condition input.
>
> To provide condition-related supervision, the same graph is also fed into the model with a randomly sampled mismatched condition $c_{fake}$ during training. Since no ground-truth layout exists for this synthetic plan-condition pair, the generated layout is not compared with a paired label. Instead, it is constrained by the density-alignment loss, which compares the predicted wall density with the target density of $c_{fake}$. In this way, the cross-condition stream encourages the model to adjust the generated wall quantity according to the specified condition while still learning geometric layout patterns from the supervised stream.

---

## 7. 关于是否考虑剪力墙沿建筑高度方向的竖向连续性

**7. Response**

Thank you for this comment. We have added a discussion on vertical continuity. In the current study, the model is applied to a typical standard floor plan. After the shear wall layout is generated for the standard floor, the same layout is used for all stories in the structural model. Therefore, vertical continuity of shear walls is naturally maintained in the finite-element case studies. For buildings with multiple standard floors or changing layouts along the height, additional constraints are needed. A practical strategy is to enforce the upper-floor shear wall layout as a subset of the lower-floor layout, so that upper-story walls do not become discontinuous or unsupported. This issue has now been discussed as a limitation and future extension.

**Revised:**
We revised Section 4.5 and Section 5.3 to clarify that the current FE validation uses one standard-floor layout repeated along the building height. We also added a discussion on how vertical continuity could be enforced for buildings with multiple standard floors.

**具体修改：**

**位置 1：Section 4.5 Finite-element case studies，结构建模说明处补充。**

新增内容：

> In the FE case studies, the generated layout corresponds to one standard floor plan, and the same shear wall layout is assigned to all stories of the building. Therefore, vertical continuity of shear walls is maintained in the analyzed structural models, and no upper-story wall is suspended without a corresponding wall below.
