# 论文配图设计指南与生成提示词

## Figure 1: Overall Framework (核心框架图)

### 设计理念

这张图是论文的"门面"，需要在一张图中清晰传达：
1. **输入**：建筑平面图 + 设计条件
2. **核心创新**：房间级图表示（而非几何图元）
3. **模型**：条件化GNN架构
4. **输出**：剪力墙布置预测

### 布局结构设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │   Input     │      │ Graph           │      │  Conditional    │         │
│  │             │  →   │ Construction    │  →   │  GNN Model      │  →  Output
│  │ Floor Plan  │      │                 │      │                 │         │
│  │ + Condition │      │ Room-based      │      │ GATv2 + FiLM    │         │
│  └─────────────┘      └─────────────────┘      └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 各模块视觉元素

**Module A: Input (左侧)**
- 建筑平面图的简化示意（房间轮廓 + 门窗位置）
- 设计条件的可视化（三个等级的图标或色块）

**Module B: Graph Construction (中左)**
- 房间变成节点（彩色圆形）
- 邻接关系变成边（连接线）
- 强调"Room as Node"的概念
- 可选：与几何图元方法的对比小图

**Module C: Conditional GNN (中右)**
- 网络架构的简化示意
- GATv2层的堆叠
- FiLM调制的可视化（条件向量影响特征）
- 双头输出（分类 + 回归）

**Module D: Output (右侧)**
- 预测的剪力墙布置结果
- 与输入平面图对应的可视化

### 配色方案

```
主色调：学术蓝 + 辅助灰 + 点缀色

- 背景：纯白 #FFFFFF
- 模块边框：深灰 #4A4A4A
- 模块填充：浅蓝 #E3F2FD / 浅灰 #F5F5F5
- 箭头/流程线：深蓝 #1976D2
- 节点颜色：
  - 房间节点：柔和蓝 #64B5F6
  - 边连接：灰色 #9E9E9E
- 条件标识：
  - Low: 浅绿 #A5D6A7
  - Medium: 浅橙 #FFCC80
  - High: 浅红 #EF9A9A
- 剪力墙预测：深蓝 #1565C0
- 强调/创新点：橙色 #FF7043
```

---

## 英文生成提示词 (AI Image Generation Prompts)

### Prompt Version 1: 完整框架图 (Recommended for DALL-E / Midjourney)

```
Scientific illustration of a deep learning framework for structural engineering, horizontal flowchart layout, white background, academic paper style.

Left section "Input": simplified architectural floor plan showing room boundaries with colored rectangles, door/window openings marked, accompanied by a small condition indicator showing three levels (low/medium/high) as colored bars.

Center-left section "Graph Construction": transformation visualization showing rooms becoming circular nodes (soft blue color) connected by gray edges, with label "Room-based Graph G=(V,E)", node features listed as "25-dim" and edge features as "8-dim".

Center section "Conditional GNN Model" (vertical layout): neural network architecture diagram showing three stacked GATv2 layers with residual connections, FiLM conditioning modules receiving condition input from above, dual output heads labeled "Classification" and "Regression".

Right section "Output": predicted shear wall layout overlaid on floor plan, walls shown as thick blue lines along room boundaries, with "16-dim prediction vector" annotation.

Flow arrows connecting sections, clean minimalist design, professional color palette with blue (#1976D2) as primary color, light gray backgrounds for modules, no gradients, vector graphics style, suitable for academic publication, 300 DPI print quality.
```

### Prompt Version 2: 简洁版 (Suitable for Illustrator recreation)

```
Technical diagram of a graph neural network pipeline for building design, academic illustration style, horizontal left-to-right flow.

Four main stages connected by blue arrows:
1. "Input" box containing a simple floor plan sketch with room outlines and a 3-level condition selector
2. "Graph Construction" box showing transformation from rectangular rooms to a node-edge graph structure, nodes as circles, edges as lines
3. "Model" box depicting a neural network with attention layers and conditioning mechanism, labeled "GATv2 + FiLM"
4. "Output" box showing the same floor plan with predicted wall placements highlighted

Clean white background, subtle drop shadows on boxes, consistent blue color theme (#1976D2, #64B5F6, #E3F2FD), sans-serif labels, IEEE/Elsevier journal figure style, high resolution vector format.
```

### Prompt Version 3: 强调创新点版

```
Research paper figure illustrating a novel room-based graph neural network for shear wall prediction, infographic style with academic rigor.

Prominent comparison callout showing "Our Approach: Room-based" versus "Previous: Geometric Primitives" with visual distinction.

Main pipeline:
- Architectural floor plan input with seismic design condition (3 categories shown as icons)
- Graph abstraction step highlighting rooms as nodes (not wall segments), with annotation "Nodes = Architectural Spaces"
- Conditional GNN block with FiLM modulation arrows from condition input
- Shear wall prediction output with overlay visualization

Color coding: input elements in warm gray, graph nodes in sky blue, neural network in navy blue, output predictions in teal. Annotation labels in dark gray, white background, no decorative elements, publication-ready scientific illustration.
```

---

## Figure 2: Graph Construction Detail (图构建细节图)

### 设计要点

展示从建筑平面图到图结构的转换过程，强调房间级表示的优势。

### 英文提示词

```
Scientific diagram comparing two graph construction approaches for building structural design, side-by-side comparison layout, academic paper illustration.

Left panel "Geometric Primitive Approach (Previous Work)":
- Floor plan with many small nodes placed on wall segments
- Dense graph with numerous edges
- Label: "Nodes = Wall Segments, |V| ~ 100+"
- Visual impression: complex, cluttered

Right panel "Room-based Approach (Ours)":
- Same floor plan with fewer, larger nodes at room centers
- Sparse graph with clear edge connections
- Label: "Nodes = Rooms, |V| ~ 10-30"
- Visual impression: clean, semantic

Bottom section showing feature extraction:
- Node features (25-dim): geometric properties + boundary constraints
- Edge features (8-dim): shared length, relative position, direction
- Output vector (16-dim): 4 edges × 2 segments × 2 endpoints

Blue and gray color scheme, clean vector graphics, suitable for two-column journal layout, annotations in English, no 3D effects.
```

---

## Figure 3: Model Architecture (模型架构图)

### 设计要点

详细展示GATv2 + FiLM条件化机制的网络结构。

### 英文提示词

```
Neural network architecture diagram for conditional graph attention network, vertical flow with horizontal layer details, technical illustration style.

Top: Input section showing node features X_v (N×25), edge features X_e (E×8), and condition vector c (1×3)

Middle: Encoder section
- Node Encoder: Linear layer transforming 25-dim to 256-dim
- Edge Encoder: Linear layer transforming 8-dim to 256-dim
- Condition Encoder: MLP transforming 3-dim to 32-dim embedding

Core: Three identical GATv2 blocks stacked vertically, each containing:
- GATv2 Convolution layer (4 attention heads)
- Batch Normalization
- FiLM Layer receiving condition embedding (shown as side input with γ and β symbols)
- ReLU activation
- Residual connection (skip arrow)

Bottom: Decoder section
- MLP decoder (256→128→32)
- Split into two heads: Classification (sigmoid, 16-dim) and Regression (sigmoid, 16-dim)
- Final output: element-wise multiplication of two heads

Color scheme: layers in light blue boxes, activations in green, condition flow in orange arrows, residual connections as dashed lines, white background, clean technical style suitable for machine learning publications.
```

---

## Figure 4: Dual-Stream Training Strategy (双流训练策略图)

### 设计要点

清晰展示监督流和约束流的训练机制。

### 英文提示词

```
Training strategy diagram showing dual-stream learning approach, flowchart style with two parallel paths, academic illustration.

Input at top: Training batch with floor plan graph and real condition label

Two streams diverging:

LEFT STREAM "Supervised Stream":
- Arrow labeled "Real Condition"
- Model forward pass
- Prediction output
- Loss computation box containing: BCE Loss + MSE Loss + IoU Loss + Consistency Loss
- Arrow to "Supervised Loss L_sup"

RIGHT STREAM "Constraint Stream":
- Arrow labeled "Fake Condition (Random)"
- Same model (shown with dashed connection to indicate shared weights)
- Different prediction output
- Loss computation: Global Density Loss
- Arrow to "Density Loss L_density"
- Note: "Activated after warmup (epoch > 20)"

Bottom: Gradient aggregation
- Both streams combine
- Total Loss equation: L = L_sup + λ₁·L_density_real + λ₂·L_density_fake
- Backpropagation arrow

Visual distinction: supervised stream in blue tones, constraint stream in orange tones, shared model in purple, loss functions in green boxes, white background, clean diagram suitable for methods section.
```

---

## Figure 5: Qualitative Results (定性结果对比图)

### 设计要点

展示预测结果与Ground Truth、Baseline的对比。

### 英文提示词

```
Qualitative comparison figure for shear wall prediction results, grid layout with 3 rows × 4 columns, scientific visualization style.

Each row represents one building case (Case A, B, C)

Column 1 "Floor Plan":
- Architectural floor plan showing room boundaries, doors, windows
- Clean line drawing style, rooms labeled or color-coded

Column 2 "Ground Truth":
- Same floor plan with actual shear walls highlighted
- Walls shown as thick green lines along room boundaries
- Label showing IoU = 1.00

Column 3 "Ours":
- Predicted shear walls shown as thick blue lines
- IoU score displayed (e.g., IoU = 0.72)
- Good alignment with ground truth

Column 4 "Baseline":
- Baseline method prediction shown as thick red lines
- Lower IoU score displayed (e.g., IoU = 0.58)
- Visible discrepancies from ground truth

Consistent scale across all subfigures, clear legend at bottom, case labels on left, method labels on top, white background, thin black borders around each subfigure, suitable for full-page width in journal.
```

---

## Figure 6: Conditional Generation Demonstration (条件化生成展示图)

### 设计要点

展示同一建筑在不同设计条件下的预测差异。

### 英文提示词

```
Conditional generation results showing same building under different seismic design conditions, horizontal layout with one floor plan and three prediction variants.

Left: Base floor plan
- Architectural layout clearly shown
- Rooms, doors, windows visible
- Neutral gray color scheme

Three prediction panels to the right:

Panel 1 "Low Seismic Intensity":
- Sparse shear wall distribution
- Light blue walls, fewer in number
- Density indicator bar showing ~30%
- Annotation: "Category 0"

Panel 2 "Medium Seismic Intensity":
- Moderate wall distribution
- Medium blue walls
- Density indicator bar showing ~50%
- Annotation: "Category 1"

Panel 3 "High Seismic Intensity":
- Dense shear wall distribution
- Dark blue walls, comprehensive coverage
- Density indicator bar showing ~70%
- Annotation: "Category 2"

Gradient arrow below showing "Low → High" condition progression
Clear visual difference in wall density across conditions
Professional color gradient from light to dark blue
White background, clean layout suitable for demonstrating controllable generation
```

---

## 配图制作工具推荐

### 矢量图工具（推荐用于最终出图）
1. **Adobe Illustrator** - 专业级，期刊投稿首选
2. **Inkscape** - 免费开源替代
3. **Figma** - 在线协作，易上手
4. **draw.io** - 免费流程图工具

### AI辅助生成（用于初稿/灵感）
1. **DALL-E 3** - 理解复杂描述能力强
2. **Midjourney** - 美学质量高，需调整专业性
3. **Stable Diffusion** - 可本地部署，可控性强

### 科研配图专用
1. **BioRender** - 生物医学风格，可改造用于工程
2. **Matplotlib/Seaborn** - 数据可视化
3. **NetworkX + Matplotlib** - 图结构可视化

---

## 配图规范检查清单

### 技术要求
- [ ] 分辨率：300 DPI以上（用于印刷）
- [ ] 格式：PDF/EPS（矢量）或 TIFF/PNG（位图）
- [ ] 尺寸：符合期刊栏宽要求（单栏~8.5cm，双栏~17.5cm）
- [ ] 字体：Arial/Helvetica，最小8pt
- [ ] 线宽：最小0.5pt

### 学术规范
- [ ] 所有文字使用英文
- [ ] 标注清晰、无歧义
- [ ] 颜色对比度足够（考虑灰度打印）
- [ ] 图例完整
- [ ] 子图标签(a)(b)(c)统一格式

### 期刊特定要求
- [ ] 查看目标期刊的Figure Guidelines
- [ ] Engineering Structures: 单栏优先
- [ ] Advanced Engineering Informatics: 支持彩色图

---

## 快速生成建议

如果时间紧迫，建议的制作流程：

1. **Figure 1 (Overall Framework)**: 使用 Figma/draw.io 手绘，这是最重要的图，值得花时间
2. **Figure 3 (Model Architecture)**: 使用 draw.io 的神经网络模板
3. **Figure 5, 6 (Results)**: 直接从代码生成（matplotlib），后期在Illustrator中美化

对于Figure 1，我建议**不使用AI生成**，而是手动绘制，因为：
- 需要精确控制每个元素的位置和标注
- AI生成的图往往需要大量后期修改
- 手绘可以确保与论文内容完全一致
