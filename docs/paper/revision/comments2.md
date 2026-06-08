
3. Please explain how the low-, medium-, and high-density groups are determined, and whether the density targets used for training and evaluation are computed independently within each training split. This would help readers better understand the controllability evaluation.

4. Please consider adding a small table comparing the generated and engineer-designed schemes in terms of key indicators, such as maximum inter-story drift ratio, code limit, vertical displacement, and wall density.

5. Please briefly discuss the limitations of the proposed method, such as the limited dataset size, dependence on preprocessing quality, use of discrete density conditions, and the need for subsequent structural analysis and engineering verification before practical application.


## 3. 关于低、中、高密度组及 density targets 的确定方式

**3. Response**

Thank you for this comment. We have clarified how the low-, medium-, and high-density groups are defined and how the corresponding density targets are computed. The three groups are determined according to seismic intensity and building height following the grouping strategy used in prior shear wall studies. After grouping the layouts, the density target of each group is computed from the shear wall ratio encoded by the room-boundary representation. Specifically, for each room, the shear wall coverage ratios along the four boundaries are summed over the 16 output entries, and the graph-level density is obtained by averaging over all rooms. Since the maximum value depends on the number of active boundary slots, this density score is not normalized to [0,1] and can theoretically approach 16. In the revised manuscript, we also clarify that the three density targets are precomputed from all available layouts and then fixed during training and evaluation, rather than being recomputed independently within each fold.

**Revised:**
We revised Section 4.1.1 and Section 3.4.2 to clarify the definition of density groups and density targets. We now explicitly state that the condition groups are determined by seismic intensity and building height, while the density targets are calculated from the average room-level shear wall coverage of all layouts in each group and fixed throughout the experiments.

**具体修改：**

**位置 1：Section 4.1.1 Dataset，原文已有 Group7-H1、Group7-H2、Group8 定义，可在该段之后补充。**

新增内容：

> The three condition groups are therefore defined before model training according to seismic intensity and building height: Group7-H1 is treated as the low-density group, Group7-H2 as the medium-density group, and Group8 as the high-density group. These labels are used as discrete condition inputs for conditional generation.


**位置 3：Section 4.1.4 Evaluation metrics，CGS 中 $S_{den}$ 解释后补充。**

新增内容：

> In all experiments, the density targets used in both the density loss and CGS evaluation are the fixed group-level averages computed from the complete set of available layouts after condition grouping. They are not recomputed separately for each fold.

---

## 4. 关于是否增加 FE 关键指标对比表

**4. Response**

Thank you for this suggestion. The requested structural indicators have already been reported and visually compared in the finite-element validation section. In Fig. 8 and Fig. 9, we compare the generated and engineer-designed schemes in terms of maximum inter-story drift ratio, code limit, vertical displacement, and shear wall density. To avoid duplicating the same information in both figures and tables, we did not add an additional table. Instead, we revised the text to explicitly point out that these key indicators are included in Fig. 8 and Fig. 9 and to summarize their main implications.

**Revised:**
We revised Section 4.5 to more clearly describe the structural indicators shown in Fig. 8 and Fig. 9, including maximum inter-story drift ratio, code limit, vertical displacement, and wall density. We also added a short summary explaining that the generated schemes remain close to the engineer-designed schemes in these indicators.

**具体修改：**

**位置：Section 4.5 Finite-element case studies，Fig. 8 和 Fig. 9 前后补充说明。**

新增内容：

> The FE validation compares the generated and engineer-designed schemes using several key structural indicators, including maximum inter-story drift ratio, the corresponding code limit, vertical displacement, and shear wall density. These indicators are visualized in Fig. 8 and Fig. 9 for the two case-study buildings. The results show that the generated schemes remain close to the engineer-designed layouts in terms of global lateral deformation and vertical displacement, while maintaining comparable wall density. Therefore, the FE case studies provide case-based evidence that the generated layouts are structurally plausible for preliminary design, although final engineering verification is still required before practical application.

---

## 5. 关于方法局限性的补充讨论

**5. Response**

Thank you for this comment. We have expanded the limitations section to more explicitly discuss the limited dataset size, dependence on preprocessing quality, use of discrete density conditions, and the need for subsequent structural analysis and engineering verification. These revisions clarify the current scope of the method and avoid overstating its readiness for direct engineering application.

**Revised:**
We revised Section 5.3 by expanding the discussion of limitations. The revised text now covers four aspects: dataset size and diversity, preprocessing dependence, discrete density-based conditioning, and the requirement of structural analysis and engineer review before practical use.

**具体修改：**

**位置：Section 5.3 Limitations。原文已有 geometric abstraction、optimization target mismatch、available data 三方面限制，可在此基础上整合扩展。**

因为limitation已经改了很多了，可以简单补充一两句或基本不修改。