## Editorial comments and responses

- 5 highlights are allowed, yet each highlight can only count 85 characters, white spaces included. Please revise.

Revised. The manuscript now includes five concise highlights, each within the 85-character limit.

Revision:
Highlights were revised as:
Room-level graphs encode spatial semantics and boundary feasibility.
FiLM conditioning enables discrete density-controlled wall generation.
Dual-stream training improves compliance with sparse paired labels.

- Avoid orphan headings without context: these are headers that are immediately followed by the next header, without any text in between (e.g. line 89-90, 601-603). At least add a brief introduction to explain what will be covered in the section.

Revised. Brief introductory sentences were added after the headings for Related work, Experiments, Experimental setup.

Revision:
In Section 2:
This section reviews prior work from three perspectives: automated structural layout generation, graph-based representations, and conditional generation under limited paired data.

In Section 4:
This section evaluates the proposed method in terms of prediction accuracy, conditional controllability, ablation behavior, computational efficiency, and engineering feasibility.

In Section 4.1:
The experimental setup first describes the dataset and condition grouping, then introduces the baseline, implementation settings, and evaluation metrics used for comparison.

- It is strongly supported and actively requested to share data as part of the article publication process. Links to available data sets and/or code can be supplied in the Data Availability Statement. Even if not all data can be provided, publication of a limited dataset in support of the work done is strongly encouraged, as it allows proving the claims made in the article, and it supports replicability of research results. Recommendations on the sharing of research data can be found here: https://www.sciencedirect.com/journal/automation-in-construction/publish/guide-for-authors#writing-and-formatting-research-data

Revised. The Data Availability Statement now clarifies that the full drawing dataset cannot be publicly released due to confidentiality restrictions, while source code and processed non-confidential materials are available from the corresponding author upon reasonable request.

Revision:
In the Data Availability Statement:
The full drawing dataset is not publicly released because it contains proprietary engineering drawings and project information subject to confidentiality restrictions. The source code and processed non-confidential materials supporting this study are available from the corresponding author upon reasonable request.

- Limitations need to be included inside the Conclusion section. They are an important part of the critical reflection that is expected in the conclusion.

Revised. A concise limitations paragraph was added to the Conclusion section, covering dataset scope, preprocessing dependence, discrete condition labels, and the need for structural analysis and engineering review.

Revision:
In Section 6:
The current framework still has several limitations, corresponding to the issues discussed above. Its validation is limited to regular high-rise residential layouts, so broader typological transfer requires further testing, especially for plans with highly irregular regions or strong segment-level constraints. The method also depends on reliable CAD preprocessing and does not directly optimize structural response indicators during training. In addition, the condition input is represented by discrete density groups, and the present validation assumes a repeated standard-floor layout rather than explicitly modeling vertical layout variation.

- Check the references for completeness in details. For example, reference [1] and [2] lack an ISBN number and publisher information, [16] and [22] lacks article number, [18], [44] and [33] lack DOI. Consider replacing ref [28], [40-41] with a peer-reviewed and published version, instead of the un-peer-reviewed arXiv version. Ref [31] and [48] are incomplete.

Revised. The reference list has been checked and updated for completeness. Missing publisher information, available book identifiers, article numbers, and DOIs have been supplemented where applicable. The arXiv references noted by the editor have been replaced or updated with peer-reviewed/published versions, and we have further checked the other references and revised several formatting issues for consistency.

Revision:
The reference list was checked and updated. Representative revised entries include:

Ref. [1]:
Ministry of Housing and Urban-Rural Development of the People's Republic of China, GB 50011--2010. Code for Seismic Design of Buildings, China Architecture & Building Press, Beijing, China, 2016. Standard book number: 15112.28896.

Ref. [2]:
J. Qian, Z. Zhao, X. Ji, L. Ye, Design of Tall Building Structures, 3rd ed., China Architecture & Building Press, Beijing, China, 2018. ISBN: 978-7-112-21684-0.

The arXiv references noted by the editor were replaced or updated with peer-reviewed/published conference versions where available. For example:
N. Nauata, S. Hosseini, K.-H. Chang, H. Chu, C.-Y. Cheng, Y. Furukawa, House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13632--13641. doi: 10.1109/CVPR46437.2021.01342.

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, Rethinking the Inception Architecture for Computer Vision, 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2016, pp. 2818--2826. doi: 10.1109/cvpr.2016.308.
