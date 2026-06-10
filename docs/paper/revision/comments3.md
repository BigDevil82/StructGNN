Reviewer #3: The manuscript presents an interesting and timely contribution to the automation of preliminary shear wall layout design using room-level conditional graph neural networks. The paper is well written, clearly organized, and addresses a relevant problem for structural design automation. In my opinion, the manuscript deserves publication after the authors address the following minor comments.

1. The filtering and contour detection procedure from CAD drawings should be explained more clearly.

Response:
Thank you for this comment. We agree that the original manuscript did not describe the CAD-to-graph preprocessing in sufficient detail. We revised Section 3.1.1 to clarify how CAD entities are filtered by layer, simplified into a wall skeleton, calibrated geometrically, and converted into closed spatial regions for room-level graph construction.

Revision:
In Section 3.1.1:
Wall, door and window entities are first parsed from the CAD drawing according to their layer types. Double-line wall boundaries are converted into wall centerlines, and door/window embedded segments are projected onto their host walls as openings. The wall centerlines are then calibrated by closing small drafting gaps, merging nearly collinear segments, and adjusting intersections within a geometric tolerance, producing a connected floor-plan skeleton. Polygonizing this calibrated skeleton yields closed spatial regions.

2. The authors should clarify how corridors and circulation spaces between rooms are considered in the room-level graph.

Response:
Thank you for this comment. We have clarified how corridors and circulation spaces are represented in the proposed room-level graph. In the current fixed boundary-slot formulation, rectangular circulation regions are kept as room nodes, while irregular connected regions are decomposed into rectangular sub-regions. The virtual boundaries introduced by this decomposition are not treated as feasible shear-wall locations and are excluded through the feasibility mask.

Revision:
In Section 3.1.1:
Corridors and circulation spaces follow the same rule: rectangular regions remain unchanged, whereas irregular connected regions are decomposed into rectangular sub-regions so that the fixed boundary-slot representation introduced in Section 3.1.2 remains applicable.

In Section 3.1.2:
Each mask component $m_ij$ corresponds one-to-one to the output component $y_ij$: $m_ij=1$ indicates that the slot is a feasible physical boundary for shear wall placement, while $m_ij=0$ indicates that the slot is blocked by an opening or corresponds to a virtual decomposition boundary. During training and inference, the mask excludes infeasible slots from loss computation, density calculation, and final wall placement.

3. The criterion used to define or select the effective length of the shear walls should be better explained.

Response:
Thank you for this comment. We agree that the definition of effective wall length is important for understanding the label construction and density calculation. We added an explicit description of how structural wall segments are projected onto feasible room-boundary slots, how multiple intervals on the same slot are merged, and how the resulting effective length is converted into the 16-dimensional coverage-ratio label.

Revision:
In Section 3.1.1:
Ground-truth shear wall locations are extracted from structural engineering drawings and mapped to the same boundary-slot representation. During this mapping, the effective length of a shear wall is defined as the projected overlap length between an extracted structural wall segment and a feasible room-boundary slot. Multiple wall intervals on the same slot are merged before the final coverage ratio is computed.

In Section 3.1.2:
For each boundary slot, the ground-truth coverage ratio is obtained by dividing the effective shear wall length within that slot by the slot length; therefore, a value of 1.0 indicates full coverage and an intermediate value indicates partial coverage. These ratios form the ground-truth vector and are also used in density-related losses and scores.

4. The 16-dimensional constraint mask should be described more explicitly, including the meaning of each component.

Response:
Thank you for this suggestion. We revised both the text and the illustration to make the constraint mask more explicit. The revised manuscript now states that the mask has the same ordering and dimension as the 16-dimensional output vector, and that each mask entry determines whether the corresponding boundary slot is buildable during loss computation, density calculation, and final wall placement. In particular, Fig. 3c was added to visually distinguish buildable physical boundaries from door/window openings and virtual decomposition boundaries.

Revision:
In Section 3.1.2:
The same boundary-slot ordering is used to define a 16-dimensional buildable mask $m_i$ for each room. Each mask component $m_ij$ corresponds one-to-one to the output component $y_ij$: $m_ij=1$ indicates that the slot is a feasible physical boundary for shear wall placement, while $m_ij=0$ indicates that the slot is blocked by an opening or corresponds to a virtual decomposition boundary. Fig. 3c visualizes this correspondence between output slots and buildable-mask entries. During training and inference, the mask excludes infeasible slots from loss computation, density calculation, and final wall placement.

In Fig. 3c:
The orange boundary segments indicate buildable room-boundary slots with mask value 1, whereas slots interrupted by doors/windows or introduced as virtual decomposition boundaries are assigned mask value 0 and excluded from shear-wall prediction.

5. In some cases, such as Figure 2, some rooms do not seem to form edges with all surrounding walls or boundaries, particularly where doors are present. This should be clarified.

Response:
Thank you for pointing this out. We clarified the room-node registration rule to avoid ambiguity in Fig. 2. Some visually enclosed background regions do not correspond to independent rooms or additional physical wall boundaries after manual room-partition annotation; registering them as separate nodes would duplicate boundaries and could introduce inconsistent labels. We therefore added an explanation of why these regions are not included as graph nodes.

Revision:
In Section 3.1.1:
During dataset preparation, room partitions were manually annotated; visually enclosed regions that were already covered by surrounding room partitions and did not introduce an independent room or additional physical boundary were not registered as separate nodes, as shown in the white regions in Fig. 2b. Partition boundaries introduced during the decomposition step are treated as virtual edges that carry no structural meaning and are therefore excluded from feasible wall placement.

6. The cross-condition stream shown in Figure 1 should be explained more clearly, especially its role during training.

Response:
Thank you for this comment. We agree that the role of the cross-condition stream is central to the proposed training strategy and should be explained more clearly. We revised Section 3.4.1 to state that the stream compensates for the lack of multi-condition paired labels by applying a mismatched condition to the same floor-plan graph and supervising the result with density alignment rather than reconstruction loss.

Revision:
In Section 3.4.1:
The cross-condition stream is introduced to compensate for the absence of multi-condition paired labels. During training, the same floor-plan graph is also evaluated under a randomly sampled mismatched condition $c_fake$, which asks the model how the wall quantity should change for the same geometry under another seismic or height-related demand. Because this synthetic plan-condition pair has no paired engineering layout, it is not supervised by reconstruction loss; instead, it is constrained by the density-alignment loss associated with $c_fake$. This creates a missing gradient signal that couples the output distribution to the condition input, discouraging the model from collapsing to condition-invariant predictions.

7. The authors should discuss whether vertical continuity of shear walls along the building height is considered or enforced.

Response:
Thank you for this comment. We clarified how vertical continuity is handled in the current FE validation. Since one generated standard-floor layout is repeated along the building height, vertical continuity is naturally maintained in the analyzed cases. We also added a limitation explaining that buildings with multiple standard floors or changing layouts would require additional inter-story constraints.

Revision:
In Section 4.5:
The generated layout corresponds to one standard floor plan, and the same shear wall layout is assigned to all stories in each FE model. Therefore, vertical continuity is maintained in the analyzed cases, and no upper-story wall is suspended without a corresponding wall below.

In Section 5.3:
The fourth limitation concerns vertical layout variation. The current FE validation uses one generated standard-floor layout repeated along the building height, so vertical continuity is naturally satisfied in the tested models. Buildings with multiple standard floors or changing architectural layouts would require additional inter-story constraints, such as enforcing upper-story shear walls to be supported by corresponding lower-story walls or treating the multi-story layout as a coupled generation problem.
