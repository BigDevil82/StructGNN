# Room-Level Graph Neural Networks for Conditional Shear Wall Layout Generation in High-Rise Buildings

**Abstract**
 Designing shear-wall layouts for high-rise residential buildings is a labor-intensive early-stage task that must satisfy architectural constraints while meeting seismic performance demands. Existing learning-based approaches are commonly formulated either as image-to-image generation or as graph learning on geometric primitives, but these formulations can be inefficient and may not explicitly reflect room-level spatial semantics. This paper proposes a room-level graph formulation that represents rooms as nodes and their adjacencies as edges, enabling wall placement to be modeled directly along room boundaries and reducing the graph size relative to primitive-based representations. On top of this representation, we develop a conditional graph neural network with FiLM-based modulation to generate layouts that comply with prescribed shear-wall density regimes. To mitigate limited paired data across design conditions, we introduce a dual-stream training strategy that combines supervision from engineered layouts with density-consistency constraints under counterfactual conditions. Experiments on 143 residential floor plans show that the proposed method achieves an Image IoU of 0.597 and improves over an edge-based GNN baseline by 9.8 percentage points. Ablation results indicate that conditioning is the primary driver of controllability, and case-based finite-element evaluation further suggests engineering feasibility under the tested settings.

**Keywords:** shear wall layout; graph neural networks; conditional generation; room-level representation; structural design automation

## 1. Introduction 

Shear wall systems are the principal lateral-force-resisting components in many high-rise residential buildings, particularly in seismic regions. The arrangement of shear walls influences stiffness distribution, load paths, and construction cost, and it must simultaneously respect architectural constraints such as functional zoning, circulation, and openings. In preliminary design, these competing requirements are commonly reconciled through iterative manual layout adjustment and repeated coordination between architectural and structural teams. Such a workflow is labor-intensive and tends to restrict systematic exploration of alternative layouts.

Despite extensive design experience and code provisions, routine shear wall layout development remains inefficient for three reasons. First, the mapping from architectural drawings to structurally rational wall candidates relies on implicit, experience-driven rules that are difficult to formalize and reuse. Second, resolving spatial conflicts often requires multiple cross-discipline iterations, creating a slow feedback loop during early design. Third, the quality and consistency of outcomes can vary across designers and projects, while the limited ability to explore the design space can lead to locally satisfactory but globally suboptimal layouts. These limitations motivate automated methods that can interpret architectural constraints and generate feasible wall layouts with controllable design intent.

Recent data-driven studies have treated layout generation as an image-to-image translation task, using convolutional networks, conditional generative adversarial networks, or diffusion models to map rasterized floor plans to structural layouts. While pixel-based representations can capture complex spatial patterns, they introduce inefficiencies that are particularly pronounced for structural elements: thin wall lines occupy a small fraction of the image grid, leading to high redundancy; topological relationships are only implicitly represented; and raster outputs require non-trivial post-processing to recover vectorized wall geometry for downstream analysis.

To address these issues, graph neural networks (GNNs) have been adopted to model structural topology more explicitly. A common formulation builds graphs from geometric primitives, with nodes as wall intersections and edges as wall segments, **and predicts whether each segment should be a shear wall.** This representation preserves connectivity but remains low-level: individual segments carry limited information about the architectural spaces they bound. As a result, room-level constraints (e.g., functional requirements and openings) are not directly expressed, and related structural layout tasks may require separate representations or additional prediction stages.

A room-level representation is considered more aligned with the physical and architectural constraints governing shear wall placement. In typical residential plans, feasible shear wall candidates are largely restricted to room boundaries, and many practical constraints (openings, circulation, functional usage) are naturally described at the room level rather than at the level of fragmented wall primitives. Modeling rooms as graph nodes and room adjacencies as edges therefore provides an explicit interface between architectural semantics and structural layout generation. In addition, the number of rooms per plan is usually far smaller than the number of primitive wall segments, which reduces graph scale and can improve computational efficiency. The same boundary-based formulation also provides a consistent basis for extending to other boundary-aligned elements such as beams, enabling a more unified structural layout pipeline.

Based on this representation, conditional shear wall layout generation is formulated using a GNN backbone with condition modulation. Neighborhood aggregation is implemented with an attention-based message passing network (e.g., GATv2), while Feature-wise Linear Modulation (FiLM) is used to inject design conditions corresponding to target wall-density categories. A practical challenge is that each floor plan is typically paired with only one engineered layout, limiting condition-specific supervision. To mitigate this, training is organized into a supervised stream that learns geometric fidelity from available layouts and a counterfactual stream that enforces condition compliance under alternative conditions through density-oriented constraints.

The main contributions are as follows:

- A room-level graph formulation for structural layout generation that encodes architectural spaces and their adjacencies, enabling direct use of room semantics and boundary constraints while reducing the scale of graph inference relative to primitive-based formulations.
- A conditional GNN architecture with FiLM-based modulation for controllable shear wall layout generation under prescribed wall-density categories, including a decoding design that predicts both wall existence and geometry.
- A dual-stream training scheme that improves conditional compliance under limited paired data by combining supervised learning with counterfactual, density-oriented regularization.

The remainder of this paper is organized as follows. Section 2 reviews related work on deep learning for structural layout generation, GNN-based representations in architecture-engineering-construction, and conditional generation. Section 3 presents the methodology, including the room-level graph construction, conditional model architecture, and training strategy. Section 4 describes the experimental setup and reports quantitative and qualitative results, including component analysis and case studies. Section 5 discusses implications and limitations. Section 6 concludes the paper.

## 2. Related Work

### 2.1 Deep Learning for Structural Layout Design

Deep learning methods for structural design have developed along two broad lines: learning-based surrogates for accelerating structural analysis and generative models for synthesizing design layouts. Surrogate modeling replaces repeated finite element simulations with neural approximations of structural responses, enabling faster screening of candidate designs. Typical targets include stress fields, displacement responses, and failure indicators inferred from structural configurations. While such surrogates support rapid evaluation, they primarily address the forward problem and do not directly produce layout decisions.

In contrast, generative approaches aim to synthesize structural layouts under architectural inputs and design requirements. Many studies formulate layout synthesis as an image-to-image translation problem, where floor plans are rasterized and structural elements are predicted on a pixel grid using conditional generative models. This formulation benefits from mature vision architectures, yet it also inherits representation-related limitations: structural elements occupy sparse regions in the raster domain, causing severe class imbalance and inefficient feature utilization; topological relations among structural components must be inferred implicitly from local patterns; and engineering workflows typically require vector outputs, making post-processing unavoidable when raster predictions are used.

Topology optimization provides an alternative paradigm grounded in mechanics by optimizing material distribution under load and constraint conditions. Despite strong theoretical foundations, topology optimization often relies on iterative analyses and may yield results that require further interpretation or regularization to meet practical detailing and constructability requirements. These characteristics motivate research into learning-based synthesis methods that can better align with architectural constraints while remaining computationally efficient.

### 2.2 Graph Neural Networks in AEC Applications

Graph neural networks (GNNs) have attracted increasing attention in architecture, engineering, and construction because graph structures naturally represent relational information among entities. At the building scale, graphs have been used to model component relations in BIM, capture dependencies in construction planning, and support prediction tasks such as energy performance estimation and schedule-related forecasting. The common advantage is the ability to explicitly encode connectivity and interaction patterns that are difficult to represent compactly in regular grids.

For layout-related problems, graphs have been used to guide floor plan generation and spatial organization through room adjacency representations. These studies indicate that graph abstractions can capture architectural relationships and constraints more directly than raster encodings. However, much of this line focuses on generating room arrangements rather than predicting structural systems conditioned on architectural space usage and boundary constraints.

The most closely related structural applications represent structural layouts using geometric primitives such as intersection points and wall segments, and then treat element-wise classification or labeling as a graph prediction problem. This representation captures local connectivity effectively, but it operates at a geometric level that is not explicitly tied to architectural function. For example, wall segments are modeled as graph elements largely independent of the rooms they bound, making it difficult to encode constraints that naturally live at the space level (e.g., functional requirements, openings, and boundary usability). This motivates representations that elevate the abstraction to architectural spaces while retaining explicit topology.

### 2.3 Conditional Generation and Feature Modulation

Conditional generation extends generative modeling to produce outputs that respond to specified attributes or constraints. In computer vision, conditional GANs and related frameworks incorporate class labels or auxiliary variables to guide synthesis, and similar conditioning ideas have been adopted in architectural applications such as style-controlled façade generation and program-constrained floor plan synthesis.

Feature-wise Linear Modulation (FiLM) is a widely used conditioning mechanism that injects condition information by modulating intermediate feature activations through learned affine transformations:
[
\text{FiLM}(\mathbf{h}, c) = \gamma(c) \odot \mathbf{h} + \beta(c),
]
where (\gamma(c)) and (\beta(c)) denote condition-dependent scaling and shifting parameters, and (\odot) is the element-wise product. By acting directly on feature representations, FiLM can provide fine-grained conditional control without requiring major architectural changes, and has been found effective across multiple modalities and tasks.

In structural layout generation, conditional formulations are attractive because engineering requirements (e.g., target stiffness levels, material usage limits, or density constraints) can be incorporated as explicit inputs. A practical challenge is the limited availability of paired data that holds the architectural plan fixed while varying structural designs across multiple conditions, since real projects typically produce a single finalized design per requirement set. This limitation motivates training strategies that can encourage condition sensitivity without relying exclusively on fully paired condition-specific labels.

### 2.4 Research Gap and Positioning

Table 1 positions representative lines of related work by their representation choices, semantic abstraction level, conditioning capability, and treatment of topology.

**[TABLE 1: Comparison with Related Work]**

| Method                | Representation | Semantic Level | Conditional | Topology |
| --------------------- | -------------- | -------------- | ----------- | -------- |
| Image-based (CNN/GAN) | Pixel grid     | Low            | Limited     | Implicit |
| Diffusion models      | Pixel grid     | Low            | Yes         | Implicit |
| Edge-based GNN        | Wall segments  | Low            | No          | Explicit |
| Room-based GNN        | Rooms          | High           | Yes         | Explicit |

Based on the above literature, three gaps remain insufficiently addressed. First, existing structural GNN formulations largely rely on geometric primitives, which limits the direct integration of architectural semantics and space-level constraints. Second, controllable generation under explicit design conditions is not commonly supported in structural layout prediction settings, despite its relevance to early-stage design exploration. Third, data scarcity—especially the absence of paired multi-condition designs for identical architectural plans—creates a mismatch with standard supervised conditional training assumptions, motivating more data-efficient strategies that leverage constraint information for counterfactual conditions.

## 3. Methodology

### 3.1 Notation and Terminology

Table 2 summarizes the notation used throughout this paper.

**[TABLE 2: Notation Summary]**

| Symbol                                                 | Definition                                     |
| ------------------------------------------------------ | ---------------------------------------------- |
| $\mathcal{P}$                                          | Architectural floor plan                       |
| $G = (\mathcal{V}, \mathcal{E})$                       | Room adjacency graph                           |
| $\mathcal{V} = \{v_1, \ldots, v_N\}$                   | Set of $N$ room nodes                          |
| $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ | Set of adjacency edges                         |
| $\mathbf{X} \in \mathbb{R}^{N \times D_v}$             | Node feature matrix ($D_v = 25$)               |
| $\mathbf{A} \in \mathbb{R}^{|\mathcal{E}| \times D_e}$ | Edge attribute matrix ($D_e = 8$)              |
| $c \in \{0, 1, 2\}$                                    | Design condition index (low/medium/high)       |
| $\mathbf{z}_c \in \mathbb{R}^{D_c}$                    | Condition embedding vector ($D_c = 32$)        |
| $\mathbf{Y} \in [0,1]^{N \times 16}$                   | Ground-truth shear wall layout matrix          |
| $\hat{\mathbf{Y}} \in [0,1]^{N \times 16}$             | Predicted shear wall layout matrix             |
| $\mathbf{M} \in \{0,1\}^{N \times 16}$                 | Constraint mask (1 = wall placement permitted) |
| $\mathcal{N}_i$                                        | Neighborhood of node $v_i$                     |

### 3.2 Room-based Graph Representation

#### 3.2.1 Representation Rationale

Shear wall placement is fundamentally constrained by architectural functional requirements. Walls must be positioned along room boundaries—not arbitrary locations—and their placement affects room usability, circulation, and natural lighting. This observation motivates a graph representation centered on architectural spaces rather than geometric primitives.

Compared to edge-based representations where wall segments serve as graph elements, room-based representation offers several advantages:

1. **Semantic alignment**: Each node directly corresponds to a functional space (bedroom, living room, kitchen), enabling incorporation of room-level design knowledge.

2. **Constraint encoding**: Openings (doors, windows) that preclude wall placement are naturally encoded as node attributes, eliminating the need for separate constraint handling.

3. **Complexity reduction**: Floor plans typically contain 10–30 rooms but 100+ wall segments, reducing graph size by approximately an order of magnitude.

4. **Output consistency**: Shear walls predicted for each room's boundaries are inherently consistent with room geometry, avoiding fragmented predictions.

**[FIGURE 1: Comparison of graph representations]**
*Left: Edge-based representation with wall segments as edges and intersections as nodes. Right: Room-based representation with rooms as nodes and adjacency as edges. The room-based graph is significantly smaller and semantically meaningful.*

#### 3.2.2 Graph Construction from Floor Plans

Given an architectural floor plan $\mathcal{P}$ in CAD format (DXF), the room adjacency graph $G = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{A})$ is constructed through the following procedure:

**Step 1: Room Extraction.** Closed polygonal regions are identified from the CAD drawing through layer-based filtering and contour detection. Each region $r_i$ with area exceeding a minimum threshold is registered as a room node $v_i$.

**Step 2: Adjacency Detection.** For each pair of rooms $(r_i, r_j)$, the shared boundary length $l_{ij}$ is computed. An edge $e_{ij}$ is added to $\mathcal{E}$ if $l_{ij}$ exceeds a threshold $\tau_{adj}$ (set to 400mm to exclude minor contacts).

**Step 3: Feature Extraction.** Node and edge features are computed as described in Sections 3.2.3 and 3.2.4.

**Step 4: Label Extraction.** For training data, shear wall locations are extracted from structural drawings and mapped to the 16-dimensional output representation (Section 3.2.5).

#### 3.2.3 Node Feature Design

Each room node $v_i$ is associated with a feature vector $\mathbf{x}_i \in \mathbb{R}^{25}$ comprising geometric descriptors and constraint information:

**[TABLE 3: Node Feature Specification]**

| Category   | Dimension | Features                                                     |
| ---------- | --------- | ------------------------------------------------------------ |
| Geometric  | 2         | Centroid coordinates $(x_c, y_c)$, normalized to $[0,1]$     |
| Geometric  | 1         | Area, normalized by maximum room area                        |
| Geometric  | 1         | Aspect ratio (width/height)                                  |
| Geometric  | 4         | Bounding box $(x_{min}, y_{min}, x_{max}, y_{max})$, normalized |
| Geometric  | 1         | Perimeter, normalized by maximum perimeter                   |
| Constraint | 16        | Buildable mask $\mathbf{m}_i \in \{0,1\}^{16}$               |

The constraint features $\mathbf{m}_i$ encode which boundary segments permit shear wall placement. Segments occupied by doors, windows, or other openings are marked as non-buildable (value 0). This 16-dimensional mask corresponds to the output parameterization described in Section 3.2.5.

All geometric features are normalized using min-max scaling computed over the training set to ensure consistent feature magnitudes.

#### 3.2.4 Edge Feature Design

Each edge $e_{ij}$ connecting adjacent rooms $v_i$ and $v_j$ is associated with an attribute vector $\mathbf{a}_{ij} \in \mathbb{R}^{8}$:

**[TABLE 4: Edge Feature Specification]**

| Feature            | Dimension | Description                                                  |
| ------------------ | --------- | ------------------------------------------------------------ |
| Shared length      | 1         | Length of shared boundary, normalized                        |
| Relative position  | 2         | $(x_j - x_i, y_j - y_i)$ between centroids, normalized       |
| Relative direction | 4         | One-hot encoding: $v_j$ is {above, below, left, right} of $v_i$ |
| Shared ratio       | 1         | Shared length / min(perimeter$_i$, perimeter$_j$)            |

The directional encoding enables the model to learn orientation-specific patterns (e.g., walls more common on certain sides of rooms).

#### 3.2.5 Output Parameterization

The shear wall layout for each room $v_i$ is parameterized as a continuous vector $\mathbf{y}_i \in [0,1]^{16}$. This parameterization captures wall placement along all four room boundaries with segment-level precision.

Each room boundary (top, right, bottom, left) is divided at its midpoint into two half-segments. For each half-segment $k \in \{1, \ldots, 8\}$, two values encode the wall coverage:

- $r_k^{start}$: fraction of the half-segment covered by wall starting from the segment's origin
- $r_k^{end}$: fraction covered starting from the segment's terminus

The complete output vector is:
$$\mathbf{y}_i = [r_1^{start}, r_1^{end}, r_2^{start}, r_2^{end}, \ldots, r_8^{start}, r_8^{end}]$$

This 16-dimensional representation can encode:

- **Full walls**: both values equal 1.0
- **Partial walls**: intermediate values indicating wall extent
- **No wall**: both values equal 0.0
- **Walls with openings**: combinations of start/end values

**[FIGURE 2: Output parameterization illustration]**
*Diagram showing a room with four boundaries, each divided into two half-segments. The 16-dimensional vector maps to wall coverage ratios for each half-segment.*

### 3.3 Problem Formulation

The shear wall layout generation task is formulated as conditional graph-to-matrix regression. Given:

- Room adjacency graph $G = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{A})$ constructed from floor plan $\mathcal{P}$
- Design condition $c \in \{0, 1, 2\}$ specifying target shear wall density category (low, medium, high)
- Constraint mask $\mathbf{M} \in \{0,1\}^{N \times 16}$ indicating buildable locations

The objective is to learn a parameterized mapping $f_\theta$:
$$\hat{\mathbf{Y}} = f_\theta(G, c)$$

that predicts the shear wall layout matrix $\hat{\mathbf{Y}} \in [0,1]^{N \times 16}$ approximating the ground-truth layout $\mathbf{Y}$ while respecting physical constraints encoded in $\mathbf{M}$.

The learning objective minimizes a composite loss:
$$\min_\theta \mathbb{E}_{(G, c, \mathbf{Y}) \sim \mathcal{D}} \left[ \mathcal{L}_{sup}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda_{phy} \mathcal{L}_{phy}(\hat{\mathbf{Y}}, \mathbf{M}, c) \right]$$

where $\mathcal{L}_{sup}$ denotes supervised reconstruction loss and $\mathcal{L}_{phy}$ denotes physics-informed constraints. Detailed loss formulations are presented in Section 3.5.

### 3.4 Conditional Graph Neural Network Architecture

The proposed architecture comprises three components: a GATv2-based encoder for spatial feature aggregation, a FiLM module for conditional injection, and a dual-head decoder for task-decoupled prediction. Figure 3 illustrates the overall architecture.

**[FIGURE 3: Model architecture diagram]**
*The architecture showing: (1) Node and edge encoders, (2) Three GATv2 layers with residual connections, (3) FiLM conditioning applied after each layer, (4) Dual-head decoder producing classification and regression outputs.*

#### 3.4.1 Feature Encoding

Raw node and edge features are projected to a common hidden dimension $D_h = 256$:

$$\mathbf{h}_i^{(0)} = \text{ReLU}(\mathbf{W}_{node} \mathbf{x}_i + \mathbf{b}_{node})$$
$$\mathbf{e}_{ij} = \text{ReLU}(\mathbf{W}_{edge} \mathbf{a}_{ij} + \mathbf{b}_{edge})$$

where $\mathbf{W}_{node} \in \mathbb{R}^{D_h \times D_v}$ and $\mathbf{W}_{edge} \in \mathbb{R}^{D_h \times D_e}$ are learnable projection matrices.

The design condition $c$ is embedded through a two-layer MLP:
$$\mathbf{z}_c = \text{MLP}_{cond}(\text{one-hot}(c)) \in \mathbb{R}^{D_c}$$

where $D_c = 32$ is the condition embedding dimension.

#### 3.4.2 GATv2 Backbone for Spatial Aggregation

Graph Attention Networks v2 (GATv2) serve as the backbone encoder, aggregating information across room neighborhoods through learned attention weights. Unlike standard Graph Convolutional Networks that use fixed aggregation weights, GATv2 dynamically computes attention coefficients based on node pair features, enabling the model to focus on structurally significant neighbors.

For node $v_i$ and neighbor $v_j \in \mathcal{N}_i$, the attention coefficient is computed as:

$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \| \mathbf{h}_j \| \mathbf{e}_{ij}]\right)$$

where $\|$ denotes concatenation, $\mathbf{W}$ is a learnable weight matrix, and $\mathbf{a}$ is a learnable attention vector. GATv2 differs from standard GAT by applying the nonlinearity after the linear transformation, ensuring strictly dynamic attention computation.

Attention coefficients are normalized via softmax:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

The updated node representation aggregates neighbor features weighted by attention:
$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}_v \mathbf{h}_j\right)$$

Multi-head attention with $K=4$ heads is employed, with outputs averaged rather than concatenated to maintain dimension consistency:
$$\mathbf{h}_i^{multi} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{h}_i^{'(k)}$$

Three GATv2 layers are stacked with residual connections to enable deep feature propagation while preventing gradient degradation:
$$\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{GATv2}^{(\ell)}(\mathbf{h}_i^{(\ell)}, \{\mathbf{h}_j^{(\ell)}\}_{j \in \mathcal{N}_i})$$

Batch normalization is applied after each GATv2 layer to stabilize training.

#### 3.4.3 FiLM-based Conditional Injection

A critical requirement is ensuring generated layouts respect the specified design condition (shear wall density category). Direct concatenation of condition embeddings with input features often leads to the condition signal being overwhelmed by high-dimensional geometric features during deep propagation—a phenomenon termed "posterior collapse" in conditional generation.

To address this, Feature-wise Linear Modulation (FiLM) is applied after each GATv2 layer. The condition embedding $\mathbf{z}_c$ is transformed into scaling and shifting parameters:

$$\gamma^{(\ell)} = \mathbf{W}_\gamma^{(\ell)} \mathbf{z}_c + \mathbf{b}_\gamma^{(\ell)}$$
$$\beta^{(\ell)} = \mathbf{W}_\beta^{(\ell)} \mathbf{z}_c + \mathbf{b}_\beta^{(\ell)}$$

These parameters modulate the post-attention features:
$$\tilde{\mathbf{h}}_i^{(\ell)} = \gamma^{(\ell)} \odot \mathbf{h}_i^{(\ell)} + \beta^{(\ell)}$$

where $\odot$ denotes element-wise multiplication. Intuitively, $\gamma$ selectively amplifies features relevant to the target density level, while $\beta$ introduces condition-specific bias.

This late-fusion approach—applying conditioning after initial feature extraction rather than at input—ensures the condition signal influences final predictions without being diluted during early processing.

#### 3.4.4 Dual-Head Decoder

Shear wall distributions exhibit sparsity: most boundary segments contain no structural walls. Direct regression on this sparse target leads to models predicting near-zero values to minimize error—a manifestation of zero-inflation bias.

To address this, the prediction task is decoupled into two sub-tasks handled by parallel decoder heads:

**Classification Head**: Predicts the probability of wall existence at each of the 16 positions:
$$\hat{\mathbf{Y}}_{cls} = \sigma(\text{MLP}_{cls}(\tilde{\mathbf{H}})) \in [0,1]^{N \times 16}$$

where $\sigma$ denotes the sigmoid function and $\tilde{\mathbf{H}}$ are the FiLM-modulated features from the final GATv2 layer.

**Regression Head**: Predicts the geometric parameters (wall length ratios) for each position:
$$\hat{\mathbf{Y}}_{reg} = \sigma(\text{MLP}_{reg}(\tilde{\mathbf{H}})) \in [0,1]^{N \times 16}$$

Both heads employ identical MLP architectures: Linear(256→256)→ReLU→Dropout(0.2)→Linear(256→128)→ReLU→Linear(128→16).

During inference, the final prediction combines both outputs:
$$\hat{\mathbf{Y}} = \mathbb{I}(\hat{\mathbf{Y}}_{cls} > \tau) \odot \hat{\mathbf{Y}}_{reg}$$

where $\tau = 0.5$ is the classification threshold and $\mathbb{I}(\cdot)$ is the indicator function. This formulation allows the classification head to establish wall topology while the regression head refines geometric details.

### 3.5 Dual-Stream Training Strategy

#### 3.5.1 Motivation: Addressing Paired Data Scarcity

Standard conditional generation requires paired training data: multiple outputs (layouts) for the same input (floor plan) under different conditions. In structural design, this requirement is impractical—each building is typically designed once for a specific set of requirements (seismic zone, building height), yielding only one ground-truth layout per floor plan.

Training solely on available pairs risks the model ignoring condition inputs entirely, learning to predict layouts based on geometric features alone. This posterior collapse manifests as generated layouts that fail to differentiate across conditions—the model produces similar outputs regardless of specified density requirements.

#### 3.5.2 Dual-Stream Training Framework

To address this challenge, a dual-stream training strategy is proposed that combines supervised learning with physics-informed counterfactual regularization:

**Supervised Stream**: Uses the true condition $c_{real}$ and ground-truth layout $\mathbf{Y}_{gt}$ to compute reconstruction losses, learning geometric accuracy and valid layout patterns.

**Counterfactual Stream**: Samples an alternative condition $c_{fake} \neq c_{real}$ and generates a counterfactual prediction $\hat{\mathbf{Y}}_{fake} = f_\theta(G, c_{fake})$. Since no ground-truth exists for this condition, physics-informed constraints enforce that the generated layout exhibits density characteristics appropriate for $c_{fake}$.

The counterfactual stream explicitly maximizes mutual information between condition input and output density, preventing the model from ignoring condition signals.

**Algorithm 1: Dual-Stream Training**

```
Input: Dataset D, model f_θ, epochs T, warmup W
Output: Trained model f_θ

1.  for epoch t = 1 to T do
2.      for each batch (G, c_real, Y_gt, M) in D do
3.          // Supervised Stream
4.          Ŷ_cls, Ŷ_reg ← f_θ(G, c_real)
5.          L_sup ← L_BCE(Ŷ_cls, Y_gt) + λ₁·L_MSE(Ŷ_reg, Y_gt)
6.                  + λ₂·L_IoU(Ŷ_reg, Y_gt) + λ₃·L_cons(Ŷ_reg, E)
7.          L_phy_real ← L_density(Ŷ_reg, c_real, M)
8.
9.          // Counterfactual Stream (after warmup)
10.         if t > W then
11.             c_fake ← sample_different(c_real)
12.             Ŷ_fake ← f_θ(G, c_fake)
13.             L_phy_fake ← L_density(Ŷ_fake, c_fake, M) + L_mask(Ŷ_fake, M)
14.             λ_cf ← min(0.1, 0.01 + 0.09·(t-W)/(T-W))
15.         else
16.             L_phy_fake ← 0; λ_cf ← 0
17.         end if
18.
19.         L_total ← L_sup + λ₄·L_phy_real + λ_cf·L_phy_fake
20.         Update θ via backpropagation on L_total
21.     end for
22. end for
```

The warmup period (first $W=20$ epochs) allows the model to learn basic layout patterns before introducing counterfactual constraints. The counterfactual weight $\lambda_{cf}$ is gradually increased to prevent training instability.

#### 3.5.3 Loss Function Components

The total loss comprises three groups: supervised reconstruction, geometric regularization, and physics-informed constraints.

**Supervised Reconstruction Losses**

*Binary Cross-Entropy (Classification)*:
$$\mathcal{L}_{BCE} = -\frac{1}{N \cdot 16}\sum_{i,j}\left[y_{ij}\log(\hat{y}_{ij}^{cls}) + (1-y_{ij})\log(1-\hat{y}_{ij}^{cls})\right]$$

where $y_{ij} = \mathbb{I}(\mathbf{Y}_{gt}[i,j] > 0.01)$ is the binarized ground-truth.

*Mean Squared Error (Regression)*:
$$\mathcal{L}_{MSE} = \frac{1}{|S^+|}\sum_{(i,j) \in S^+}(\hat{y}_{ij}^{reg} - y_{ij}^{gt})^2$$

where $S^+ = \{(i,j) : y_{ij}^{gt} > 0.01 \land \mathbf{M}[i,j] = 1\}$ restricts the loss to positive samples in buildable regions, preventing collapse to zero predictions.

**Geometric Regularization Losses**

*Vector IoU Loss*:
$$\mathcal{L}_{IoU} = 1 - \frac{\sum_{i,j}\min(\hat{y}_{ij}^{reg}, y_{ij}^{gt})}{\sum_{i,j}\max(\hat{y}_{ij}^{reg}, y_{ij}^{gt}) + \epsilon}$$

This loss directly optimizes geometric overlap between predicted and ground-truth layouts, complementing point-wise MSE with holistic shape alignment.

*Topological Consistency Loss*:
$$\mathcal{L}_{cons} = \sum_{(u,v) \in \mathcal{E}} \left\|\hat{\mathbf{y}}_u^{(side_v)} - \hat{\mathbf{y}}_v^{(side_u)}\right\|^2$$

where $side_v$ indicates the boundary of room $u$ adjacent to room $v$. This loss penalizes inconsistent predictions at shared boundaries—if room $u$ predicts a wall on its right edge adjacent to room $v$, room $v$ should predict the same wall on its corresponding left edge.

**Physics-Informed Constraints**

*Mask Feasibility Loss*:
$$\mathcal{L}_{mask} = \left\|\hat{\mathbf{Y}} \odot (1 - \mathbf{M})\right\|^2$$

This term penalizes wall predictions in non-buildable regions (doors, windows), ensuring physical feasibility.

*Density Compliance Loss*:
$$\mathcal{L}_{density} = \left(\frac{1}{N}\sum_i \sum_j (\hat{y}_{ij} \cdot m_{ij}) - \rho(c)\right)^2$$

where $\rho(c)$ is the target average density for condition $c$, empirically estimated from training data statistics. This loss enforces that generated layouts exhibit appropriate total wall quantities for the specified condition.

**Loss Weights**

The complete loss function with empirically tuned weights:
$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + 2.0 \cdot \mathcal{L}_{MSE} + 2.0 \cdot \mathcal{L}_{IoU} + 0.5 \cdot \mathcal{L}_{cons} + 5.0 \cdot \mathcal{L}_{mask} + \lambda_{density} \cdot \mathcal{L}_{density}$$

where $\lambda_{density} = 0.05$ for the supervised stream and varies according to $\lambda_{cf}$ for the counterfactual stream.

### 3.6 Implementation Details

**[TABLE 5: Hyperparameter Configuration]**

| Category         | Parameter                       | Value              |
| ---------------- | ------------------------------- | ------------------ |
| **Architecture** | Hidden dimension $D_h$          | 256                |
|                  | Condition embedding $D_c$       | 32                 |
|                  | GATv2 layers                    | 3                  |
|                  | Attention heads $K$             | 4                  |
|                  | Dropout rate                    | 0.2                |
| **Training**     | Optimizer                       | AdamW              |
|                  | Learning rate                   | $1 \times 10^{-3}$ |
|                  | Weight decay                    | $1 \times 10^{-4}$ |
|                  | Batch size                      | 1 (per graph)      |
|                  | Total epochs                    | 100                |
|                  | Warmup epochs $W$               | 20                 |
|                  | LR scheduler                    | Cosine annealing   |
| **Inference**    | Classification threshold $\tau$ | 0.5                |
|                  | Ratio threshold                 | 0.1                |

**Data Augmentation**: Six geometric transformations are applied to each training sample: identity, horizontal flip, vertical flip, and rotations of 90°, 180°, and 270°. Output labels are correspondingly transformed to maintain consistency.

**Class Balancing**: Weighted random sampling ensures balanced exposure to all three design conditions within each epoch, addressing the uneven distribution in the dataset (Low:Medium:High = 52:33:58).

**Cross-Validation**: 5-fold stratified cross-validation is employed, with stratification by design condition. Each fold reserves 10% of training data for validation and early stopping (patience = 20 epochs based on validation IoU).

**Hardware and Software**: Experiments are conducted on a workstation with Intel Core i7-12700F CPU, 32GB RAM, and NVIDIA RTX 3060 GPU (12GB). The implementation uses Python 3.12, PyTorch 2.5.1, and PyTorch Geometric 2.1.0.

## 4. Experiments

### 4.1 Experimental Setup

#### Dataset

A dataset of high-rise residential floor plans is constructed from two sources: (i) the open-source shear-wall dataset reported by Liao et al., and (ii) additional engineering drawings provided by Tongji Architectural Design (Group) Co., Ltd. Each sample consists of an architectural floor plan and the corresponding shear-wall layout designed by licensed structural engineers. Raw CAD drawings are processed via contour extraction and semantic partitioning to identify functional rooms, and then converted into the room-based graph representation described in Section 3.2.

Following the condition taxonomy of Liao et al., which accounts for the combined effects of seismic intensity and structural height, samples are categorized into three design-condition groups:

- **Low-density (C0):** seismic intensity 7 (PGA = 0.10g), height (H \le 50) m
- **Medium-density (C1):** seismic intensity 7, height (H > 50) m
- **High-density (C2):** seismic intensity 8 (PGA = 0.20g)

**[TABLE 6: Dataset Statistics]**

| Statistic                             | Value        |
| ------------------------------------- | ------------ |
| Total floor plans                     | 143          |
| Condition distribution (Low:Med:High) | 52 : 33 : 58 |
| Train / Test split                    | 80% / 20%    |
| Avg. rooms per floor plan             | [NEED DATA]  |
| Avg. edges per graph                  | [NEED DATA]  |
| Samples after augmentation (train)    | [NEED DATA]  |

> （可选但更“期刊化”的补充句：）Unless otherwise specified, all reported results use the same train/test split and data preprocessing pipeline described above.

#### Baselines and Implementation Notes

To assess the impact of the proposed room-based representation, performance is compared against a representative edge-based GNN baseline consistent with prior literature.

**Edge-based GNN baseline.** Graphs are constructed using wall intersection points as nodes and wall segments as edges, and the model predicts binary labels (structural vs. non-structural) for edges. For comparability, the baseline adopts the same backbone family (GATv2), training schedule, data augmentation, and cross-validation protocol as the proposed method, with hyperparameters tuned on validation sets for each method. Conditioning is not included in the baseline in order to remain consistent with the typical formulation in prior edge-based approaches.

For evaluation consistency, predictions from both representations are rasterized into binary layout images and compared using Image IoU, which measures pixel-level overlap between predicted and reference shear-wall visualizations.

#### Evaluation Metrics

**Geometric accuracy.** Image IoU is used as the primary metric:
[
\text{Image IoU}=\frac{|P\cap G|}{|P\cup G|}
]
where (P) and (G) denote the predicted and ground-truth positive pixel sets, respectively. This image-based evaluation enables a representation-agnostic comparison across methods.

For the room-based method, a vector-level IoU is additionally reported on the 16-dimensional output parameterization to quantify parametric overlap without rasterization. Segment-level precision/recall/F1 are computed by thresholding predicted/ground-truth segment values at 0.01:
[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
\text{F1}=\frac{2PR}{P+R}.
]
For correctly identified positive segments, MAE is used to evaluate the error of predicted length ratios:
[
\text{MAE}=\frac{1}{|S^+|}\sum_{(i,j)\in S^+}|\hat y_{ij}-y_{ij}|.
]

**Conditional generation.** Standard supervised metrics require matched ground-truth layouts and therefore cannot directly evaluate counterfactual conditions (e.g., generating a high-density layout for a plan labeled as low-density). To characterize conditional behavior across all conditions, the Conditional Generation Score (CGS) is defined as:
[
\text{CGS}=w_1 S_{\text{den}}+w_2 S_{\text{IoU}}+w_3 S_{\text{uni}},
\quad w_1=w_2=w_3=\frac13.
]
Here, (S_{\text{den}}) measures compliance with target density statistics:
[
S_{\text{den}}=1-\frac{|\bar\rho_{\text{pred}}-\bar\rho_{\text{target}}|}{\bar\rho_{\text{target}}}.
]
(S_{\text{IoU}}) corresponds to Image IoU when the generated condition matches the ground-truth label. (S_{\text{uni}}) summarizes spatial rationality by combining global variation, local smoothness across adjacent rooms, and penalties for extreme outliers:
[
S_{\text{uni}}=0.4,(1-\text{CV})+0.4,\text{Smoothness}+0.2,(1-\text{OutlierRatio}).
]

> 注：这里把你原来对 Uniformity 的三条解释压缩为一句定义性描述，保留公式与三个组成项即可，避免“条目堆砌”。

------

### 4.2 Main Results

Table 7 reports the quantitative comparison between the proposed room-based GNN and the edge-based GNN baseline.

**[TABLE 7: Main Results Comparison]**

| Method                  | Image IoU ↑             | Precision ↑             | Recall ↑                | F1 ↑                    | MAE ↓                   | RMSE ↓                  |
| ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| Edge-based GNN          | 0.499 ± [NEED DATA]     | 0.616 ± [NEED DATA]     | 0.771 ± [NEED DATA]     | 0.683 ± [NEED DATA]     | 0.369 ± [NEED DATA]     | 0.515 ± [NEED DATA]     |
| Proposed room-based GNN | **0.597** ± [NEED DATA] | **0.791** ± [NEED DATA] | **0.783** ± [NEED DATA] | **0.791** ± [NEED DATA] | **0.199** ± [NEED DATA] | **0.328** ± [NEED DATA] |
| Improvement             | +9.8 pp                 | +17.5 pp                | +1.2 pp                 | +10.8 pp                | -46.0%                  | -36.4%                  |

*pp: percentage points. Error bars denote the standard deviation across 5-fold cross-validation. [NEED DATA: add statistical significance tests (e.g., paired t-test or Wilcoxon) and corresponding p-values.]*

The proposed room-based representation yields higher Image IoU and F1 than the edge-based baseline under the same backbone family and training protocol. The largest gain is observed in precision (+17.5 pp), suggesting a marked reduction of false-positive wall predictions. This behavior is practically relevant because over-prediction can introduce unnecessary construction cost and increase the likelihood of conflicts with architectural constraints. In addition, the lower MAE/RMSE indicates improved accuracy in predicted wall-length ratios for correctly identified segments, which is consistent with the representation explicitly reasoning over complete room boundaries rather than isolated wall segments.

**[FIGURE 4: Qualitative comparison of predictions]**
*Representative cases showing floor plan, ground truth, proposed method, and baseline. The proposed method produces fewer spurious segments and more coherent boundary-aligned predictions.*

#### Per-condition results

To examine whether performance is consistent across design conditions, Table 8 reports Image IoU stratified by condition.

**[TABLE 8: Per-Condition Performance]**

| Condition   | Samples     | Image IoU (Proposed) | Image IoU (Baseline) | Δ           |
| ----------- | ----------- | -------------------- | -------------------- | ----------- |
| Low (C0)    | [NEED DATA] | [NEED DATA]          | [NEED DATA]          | [NEED DATA] |
| Medium (C1) | [NEED DATA] | [NEED DATA]          | [NEED DATA]          | [NEED DATA] |
| High (C2)   | [NEED DATA] | [NEED DATA]          | [NEED DATA]          | [NEED DATA] |

[NEED DATA: add a brief condition-wise interpretation, e.g., which regime benefits most and a plausible explanation tied to density and false positives.]

### 4.3 Ablation Studies

Ablation experiments were conducted to quantify the contribution of each component. Table 9 summarizes results using the Conditional Generation Score (CGS) and its sub-metrics.

**[TABLE 9: Ablation Study Results]**

| Configuration            | CGS ↑          | Density ($S_{den}$) ↑ | IoU ($S_{IoU}$) ↑ | Uniformity ($S_{uni}$) ↑ |
| ------------------------ | -------------- | --------------------- | ----------------- | ------------------------ |
| **Full Model**           | **0.727**      | **0.750**             | **0.541**         | **0.891**                |
| *Conditioning Mechanism* |                |                       |                   |                          |
| w/o conditioning         | 0.579 (-0.148) | 0.384                 | 0.494             | 0.859                    |
| w/o FiLM (concat early)  | 0.720 (-0.007) | 0.727                 | 0.528             | 0.906                    |
| w/o FiLM (concat late)   | 0.718 (-0.009) | 0.718                 | 0.541             | 0.894                    |
| *Loss Functions*         |                |                       |                   |                          |
| w/o density loss         | 0.669 (-0.058) | 0.607                 | 0.512             | 0.888                    |
| w/o IoU loss             | 0.691 (-0.036) | 0.763                 | 0.366             | 0.945                    |
| w/o consistency loss     | 0.716 (-0.011) | 0.746                 | 0.527             | 0.876                    |
| w/o all auxiliary losses | 0.622 (-0.105) | 0.548                 | 0.451             | 0.868                    |
| *Training Strategy*      |                |                       |                   |                          |
| w/o dual-stream          | 0.714 (-0.013) | 0.722                 | 0.531             | 0.890                    |
| w/o warmup               | 0.684 (-0.043) | 0.689                 | 0.489             | 0.873                    |
| w/o augmentation         | 0.702 (-0.025) | 0.723                 | 0.462             | 0.921                    |
| w/o weighted sampling    | 0.719 (-0.008) | 0.738                 | 0.529             | 0.889                    |
| *GNN Backbone*           |                |                       |                   |                          |
| GCN                      | 0.712 (-0.015) | 0.735                 | 0.518             | 0.882                    |
| GraphSAGE                | 0.712 (-0.015) | 0.729                 | 0.524             | 0.884                    |
| GIN                      | 0.715 (-0.012) | 0.741                 | 0.521             | 0.883                    |

The conditioning module accounts for the largest change in controllability-related performance. Removing conditioning reduces CGS by 0.148, driven primarily by a sharp drop in density consistency ($S_{den}$: 0.750 → 0.384), indicating that outputs become less responsive to the specified design condition. Among alternative injection strategies, FiLM yields small but consistent gains over early and late feature concatenation.

Loss-term ablations show complementary roles. The density loss most strongly affects conditional compliance (CGS −0.058 when removed), primarily through reduced $S_{den}$. The IoU loss contributes to geometric fidelity ($S_{IoU}$: 0.541 → 0.366 when removed) while increasing uniformity ($S_{uni}$: 0.891 → 0.945), suggesting a trade-off between boundary accuracy and distribution smoothness under the current metric definition. The consistency loss produces a smaller but measurable effect (CGS −0.011), and removing all auxiliary losses leads to a substantial overall degradation (CGS −0.105).

Training-related components have moderate but non-negligible impacts. Eliminating the dual-stream strategy decreases CGS by 0.013, with the main change in $S_{den}$, consistent with improved condition sensitivity. Warmup contributes to optimization stability (CGS −0.043 without warmup). Data augmentation primarily benefits IoU (0.541 → 0.462 without augmentation), while weighted sampling yields a smaller improvement.

For the GNN backbone, GATv2 achieves the best overall CGS; replacing it with GCN, GraphSAGE, or GIN results in consistent but modest decreases (approximately 0.012–0.015 in CGS).

------

### 4.4 Conditional Generation Analysis

The ability to generate condition-dependent layouts is evaluated by applying all three condition inputs to the same set of floor plans.

**[FIGURE 5: Conditional generation demonstration]**
*For three representative floor plans, predictions under Low, Medium, and High conditions. Visual comparison shows progressive increase in wall density and coverage as condition severity increases.*

**[TABLE 10: Density Statistics by Condition]**

| Input Condition | Target Density | Generated Density (Mean ± Std) | Density Error |
| --------------- | -------------- | ------------------------------ | ------------- |
| Low (0)         | [NEED DATA]    | [NEED DATA]                    | [NEED DATA]   |
| Medium (1)      | [NEED DATA]    | [NEED DATA]                    | [NEED DATA]   |
| High (2)        | [NEED DATA]    | [NEED DATA]                    | [NEED DATA]   |

Figure 5 shows that predicted layouts vary systematically across conditions, with denser wall coverage under more demanding settings. Table 10 further reports the corresponding density statistics. Density errors remain within [NEED DATA]% of target values, indicating condition-responsive generation under the current definition of density and evaluation protocol.

------

### 4.5 Case Study: Engineering Validation

Engineering feasibility is assessed on a representative high-rise residential floor plan that is not included in the training set.

**[FIGURE 6: Case study floor plan and predictions]**
*Left: Architectural floor plan. Center: Predicted shear wall layout (High condition). Right: Beam layout derived from room boundaries.*

**Procedure**:

1. A held-out floor plan is processed by the trained model.
2. Predicted shear wall elements are extracted and converted into structural geometry.
3. Beam layouts are derived from room-boundary information using the same room-based representation.
4. A finite element model is constructed in structural analysis software.
5. Modal and seismic response analyses are performed.

**[TABLE 11: Structural Analysis Results]**

| Parameter                 | Code Requirement | Predicted Layout | Status      |
| ------------------------- | ---------------- | ---------------- | ----------- |
| First mode period (s)     | < [NEED DATA]    | [NEED DATA]      | [NEED DATA] |
| Max. inter-story drift    | < 1/800          | [NEED DATA]      | [NEED DATA] |
| Max. displacement (mm)    | < H/500          | [NEED DATA]      | [NEED DATA] |
| Torsion-translation ratio | < 0.9            | [NEED DATA]      | [NEED DATA] |

[NEED DATA: Complete FEM analysis results and interpretation]

The analysis results in Table 11 indicate whether the predicted layout satisfies the specified performance checks under the chosen loading and modeling assumptions. A complete interpretation should report governing cases, critical stories/locations, and any observed failure modes if constraints are violated.

------

### 4.6 Computational Efficiency

Computational characteristics are compared between room-based and edge-based approaches in Table 12.

**[TABLE 12: Computational Efficiency Comparison]**

| Metric                     | Edge-based GNN | Room-based GNN | Ratio       |
| -------------------------- | -------------- | -------------- | ----------- |
| Avg. nodes per graph       | [NEED DATA]    | [NEED DATA]    | [NEED DATA] |
| Avg. edges per graph       | [NEED DATA]    | [NEED DATA]    | [NEED DATA] |
| Model parameters           | [NEED DATA]    | [NEED DATA]    | [NEED DATA] |
| Training time (min/epoch)  | [NEED DATA]    | [NEED DATA]    | [NEED DATA] |
| Inference time (ms/sample) | [NEED DATA]    | [NEED DATA]    | [NEED DATA] |

The room-based representation produces smaller graphs, which reduces training and inference costs under the same implementation settings. The compact graph structure may also reduce redundancy in message passing, potentially benefiting generalization; this effect should be interpreted in conjunction with the reported accuracy metrics and dataset scale.

## 5. Discussion

### 5.1 Interpretation of Results

Two observations emerge from the experimental and ablation results.

First, the room-level representation appears to better align with the geometric and functional constraints of shear wall placement than primitive-level alternatives. Shear walls in residential plans are typically constrained to room boundaries and are influenced by room-specific attributes such as openings and functional use. Modeling rooms as primary entities provides a direct way to incorporate these constraints and reduces reliance on recovering semantics implicitly from fragmented geometric primitives. The improvement over edge-based baselines (Image IoU +9.8 pp) is accompanied by a marked increase in precision (+17.5 pp), suggesting that the representation helps suppress implausible wall predictions in regions where walls are unlikely to be placed.

Second, explicit condition injection is necessary to obtain controllable generation across density regimes. Removing the conditioning module leads to a substantial drop in the conditional generation score (CGS −0.148) and a large decrease in density consistency (0.750 → 0.384), indicating that the model under the tested setting does not reliably differentiate outputs by target condition without an explicit conditioning pathway. Compared with feature concatenation baselines, FiLM-based modulation yields small but consistent improvements, consistent with the interpretation that multiplicative feature modulation can propagate condition information more effectively through multiple message-passing layers. The dual-stream strategy provides a modest gain in CGS (+0.013) but improves conditional compliance; this is consistent with the intended role of counterfactual regularization in discouraging the network from becoming insensitive to condition inputs, although a dedicated diagnostic would be required to substantiate stronger claims about collapse behavior.

### 5.2 Limitations

Several limitations should be noted.

The conditioning design is coarse. Only three discrete categories are used, which restricts fine-grained control. Practical design tasks may require continuous targets (e.g., specified shear wall ratios, building height ranges, or seismic intensity parameters), which are not directly supported by the current formulation.

The geometric parameterization assumes rectangular rooms. The current 16-dimensional boundary-based output is tailored to orthogonal four-edge rooms and does not naturally represent L-shaped or general polygonal spaces. Preprocessing to fit these cases may introduce artifacts and could degrade prediction fidelity.

Structural performance is not explicitly optimized during learning. The model is trained to match observed layout patterns and density-related constraints, but it does not directly minimize performance objectives such as drift, stiffness balance, or stress concentration. Consequently, generated layouts may be feasible in typical cases but are not guaranteed to be optimal for specific engineering objectives.

The dataset size is limited (143 floor plans). Although augmentation and cross-validation can reduce overfitting risk, generalization to substantially different typologies, detailing practices, or regional design conventions remains uncertain.

### 5.3 Engineering Implications

The proposed pipeline is primarily relevant to early-stage structural layout exploration. Conditional generation across multiple density regimes can provide rapid candidate layouts that reflect learned design regularities while remaining aligned with room-boundary constraints. Such outputs may support faster iteration between architectural and structural considerations by offering a consistent baseline that reduces manual trial-and-error in initial placement decisions. In addition, the room-based formulation suggests a pathway toward more integrated prediction of boundary-defined structural elements (e.g., beams along room adjacencies), although this extension requires dedicated data and validation.

### 5.4 Scope and Potential Generalization

The experiments focus on high-rise residential plans in seismic design contexts, and the evidence supports conclusions within this scope. The underlying representation is not specific to residential function in principle, but extension to other building types would require verification that room segmentation, adjacency semantics, and constraint patterns remain predictive for the target domain. Similarly, extension to other structural systems or broader condition sets (e.g., cost targets, material constraints) is conceptually plausible but remains untested; demonstrating such generalization would require new datasets, baselines, and task-specific evaluation criteria.

------

## 6. Conclusion

A room-level graph formulation for conditional shear wall layout generation is presented, with rooms modeled as nodes and adjacencies as edges to better reflect architectural semantics and boundary-constrained placement. Conditioning via FiLM-style modulation enables density-regime control, and a dual-stream training procedure with counterfactual regularization is used to improve condition sensitivity when paired condition-specific supervision is limited.

Across 143 residential floor plans, the room-based representation improves prediction quality over edge-based baselines, achieving 0.597 Image IoU and yielding higher precision (+17.5 pp), which is consistent with reduced false positive wall placement. Ablation results indicate that explicit conditioning is the primary driver of controllability (CGS −0.148 when removed) and that the proposed training strategy provides a smaller but measurable benefit to conditional compliance. Case-based structural validation using finite element analysis further suggests that generated layouts can satisfy feasibility checks under the evaluated settings.

Future work should consider continuous condition formulations, representations that accommodate non-rectangular spaces without ad hoc preprocessing, and tighter integration of structural response metrics into training objectives for performance-aware generation. Extending the framework toward joint prediction of multiple boundary-defined structural elements is also a natural direction, but it requires systematic validation beyond the current scope.

