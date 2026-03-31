# RC-GNN

面向剪力墙结构智能设计研究的代码库。项目核心目标围绕住宅建筑平面布局，完成从数据准备、房间级图建模、剪力墙布局预测，到后续结构建模分析的一整套研究流程。


## 研究问题

本项目聚焦于：

- 输入建筑布局信息，预测哪些墙段位置应布置剪力墙
- 当前仅关注剪力墙位置预测，不涉及构件截面尺寸设计
- 当前方法以数据驱动为主，模型主体为房间级条件图神经网络，可称为 `RC-GNN (Room-level Conditional GNN)`

与很多以墙段为基本图元的做法不同，本项目以“房间”作为图节点，希望尽量保留住宅平面中的空间语义与分区信息。

## 当前主流程

整个项目可以概括为 4 条主线。

### 1. PNG 数据集整理与标注准备

原始开源数据集多为彩色构件图像。项目先通过图像处理流程，将图中墙体、门、窗等构件提取出来并转换为 DXF，便于后续人工标注房间。

对应目录：

- `pngtool/`

典型内容：

- `image_clustering.py`: 图像去重
- `png2dxf_converter.py`: 从图像提取构件轮廓并写入 DXF

### 2. 标注 DXF 到训练数据

在 DXF 中完成房间标注后，项目会提取墙体、门窗、房间等几何对象，并将其转换为房间级图结构，用于模型训练。

对应目录：

- `preprocess/`

典型内容：

- `dxf_extractor.py`: 从 DXF 提取构件
- `room_calibrator.py`: 校准相邻房间边界，减少绘制误差
- `room_analyzer.py`: 计算房间各边上的剪力墙分布标签
- `layout_graph.py`: 构建房间级图

### 3. 剪力墙布局预测

训练与评估代码位于 `shearwall_pred/`，这是当前研究主线中最稳定、最核心的模型模块之一。

对应目录：

- `shearwall_pred/`

典型内容：

- 数据集定义与缓存
- 数据增强
- 模型结构
- 损失函数
- 训练、测试、可视化
- 交叉验证与集成

### 4. CAD 推理与结构建模分析

实际应用时，输入不再是数据集图像，而是原始 CAD 导出的构件数据。项目通过几何处理将双线墙体提取为中轴线，再分解出矩形房间，转换成与训练阶段一致的图输入。随后可将预测结果继续解析为 FEM 建模数据，并接入 ETABS、盈建科或 OpenSees 分析流程。

对应目录：

- `axis_engine/`
- `experiments/case_study/`
- `shearwall_modeling/`

典型内容：

- `axis_engine/wall_centerline.py`: 墙体中轴线提取
- `axis_engine/line_network_calibrator.py`: 线网校准
- `axis_engine/rect_decomposer.py`: 矩形分解
- `preprocess/cad_inference.py`: 将 CAD 前处理结果桥接为推理输入
- `experiments/case_study/run_case_study.py`: 案例级预测与构件解析
- `shearwall_modeling/`: OpenSees 建模与规范校核相关代码

## 顶层目录说明

以下是当前项目中值得长期维护的主干目录。

```text
Png2Dxf/
├── axis_engine/          # CAD 几何解析与中轴线/矩形分解
├── pngtool/              # PNG 数据清洗与 DXF 转换
├── preprocess/           # 标注 DXF 到训练图数据
├── shearwall_pred/       # 房间级剪力墙预测模型
├── shearwall_modeling/   # OpenSees 建模、分析与校核
├── experiments/          # 论文实验、评估、案例分析
├── data/dxf/                  # 原始或处理中 DXF / CAD 数据
├── outputs/result/               # 模型、图表、案例输出结果
├── data/cache/           # 训练缓存数据
├── misc/                 # 通用辅助工具
├── baseline_edge_gnn/    # 既有墙段级方法复现
└── beam_pred/            # 历史探索模块，当前不是主线
```

其他说明：

- `experiments/` 中很多脚本是论文期临时代码，不适合继续无限扩张
- `experiments/case_study/` 已经承载了部分“准正式流程”，后续更适合独立抽离
- `experiments/pipelines/` 是当前整理阶段建立的流程命名空间骨架
- `beam_pred/` 当前不是项目主线，可视为保留分支
- `%TEMP%/`、`paper/` 等目录更多是辅助性质，不建议继续承载核心逻辑

结构整理约束可参考：

- [`docs/PROJECT_STRUCTURE.md`](/E:/Common/Desktop/Research/deepLearning/codes/Png2Dxf/docs/PROJECT_STRUCTURE.md)

## 环境

项目当前使用本地虚拟环境 `.venv`。

Python 版本要求见 [`pyproject.toml`](/E:/Common/Desktop/Research/deepLearning/codes/Png2Dxf/pyproject.toml)。

推荐使用 `uv` 或现有 `.venv` 环境安装依赖。

```bash
.venv\Scripts\activate
uv sync
```

如果不使用 `uv`，也可以按需参考 `pyproject.toml` 或 `requirements.txt` 手动安装。

## 主要入口


### 训练剪力墙预测模型

```bash
.venv\Scripts\activate
python -m shearwall_pred.trainer --mode train
```

测试：

```bash
python -m shearwall_pred.trainer --mode test --ckpt outputs\result\...\best_model.pth
```

可视化：

```bash
python -m shearwall_pred.trainer --mode visualize --ckpt outputs\result\...\final_model.pth
```

### CAD JSON 转 DXF / 推理图输入

```bash
python -m preprocess.cad_inference --cad-json input.json --dxf-out output.dxf
```

### 案例级推理与 FEM 构件解析

```bash
python -m experiments.pipelines.case_study.run_case_study ^
  --dxf_path path\to\layout.dxf ^
  --model_dir outputs\result\shearwall_pred\...\ ^
  --output_dir outputs\result\case_study
```

兼容说明：

- 旧路径 `experiments.case_study.run_case_study` 仍可使用
- 新代码应优先使用 `experiments.pipelines.case_study.*`

### OpenSees 建模分析

可参考：

- [`shearwall_analyzer.py`](/E:/Common/Desktop/Research/deepLearning/codes/Png2Dxf/shearwall_analyzer.py)

## 当前代码组织上的问题

从长期维护角度看，目前最明显的问题不是“代码太多”，而是“研究脚本、稳定模块、临时分析、案例流程”混在同一层级中继续生长。这样发展下去，问题主要有：

- 顶层目录语义不够稳定，新功能容易继续随手堆到 `experiments/` 或根目录
- “研究原型”与“长期保留模块”边界不清，后续难以判断哪些代码应该重构、测试、文档化
- 从训练到推理到建模的链路已经形成，但入口仍然分散，命名风格也不统一
- 输出目录、数据目录、缓存目录和论文图表目录混杂，容易让结果管理失控

## 对项目结构的整理建议

不建议一次性大搬家。更稳妥的做法是先建立“新的组织原则”，再逐步迁移。

### 建议 1：先明确 3 层代码边界

建议把代码按下面三层理解：

- `core`: 长期维护的核心能力，如几何处理、图构建、模型、建模
- `pipelines`: 端到端流程入口，如训练、推理、案例分析、批处理
- `research`: 论文实验、消融、指标统计、临时可视化

即便暂时不重命名目录，也建议后续新代码按这个原则落位，避免继续往 `experiments/` 里堆正式流程。

### 建议 2：把 `experiments/case_study` 抽成正式流程模块

这是目前最值得优先处理的一点。

`case_study` 已经不只是论文案例脚本，而是承载了：

- 模型推理
- 预测结果后处理
- FEM 拓扑构建
- ETABS / 盈建科流程桥接

这部分更像正式应用流水线，建议后续迁移为类似：

```text
pipelines/
├── inference/
├── fem/
└── external_tools/
```

或者至少先改成顶层独立目录，例如 `inference_pipeline/`、`structural_pipeline/`，避免继续被归入“实验代码”。

### 建议 3：给顶层目录做“主干 / 档案 / 数据”分区

当前顶层可以逐步收敛为：

- 主干代码目录：`axis_engine`、`preprocess`、`shearwall_pred`、`shearwall_modeling`
- 研究或历史目录：`experiments`、`baseline_edge_gnn`、`beam_pred`
- 数据与结果目录：`dxf`、`data/cache`、`result`

也就是说，今后应该避免再新增语义模糊的顶层目录。

### 建议 4：根目录不要继续放业务脚本

目前根目录已有：

- `main.py`
- `shearwall_analyzer.py`
- `test.py`

这类脚本可以逐步迁移到统一入口目录，例如：

```text
scripts/
├── train_shearwall.py
├── run_case_study.py
├── analyze_shearwall_models.py
└── debug_*.py
```

这样根目录只保留项目说明、依赖配置和少量元文件。

### 建议 5：统一“输入/中间结果/输出”目录约定

建议尽快统一以下约定，否则后面最容易乱：

- 原始数据：`data/raw/`
- 人工标注或中间产物：`data/interim/`
- 训练缓存：`data/cache/`
- 结果输出：`outputs/`
- 论文图表：`outputs/paper/`

你现在的 `data/dxf/`、`data/cache/`、`outputs/result/` 都分别承担了这些角色，但边界不够严格。可以不立刻重命名，先在 README 和新代码里把约定写清楚。

### 建议 6：把“可复用算法”和“项目特定脚本”分开

例如：

- `axis_engine/`、`preprocess/` 中很多代码是可复用模块
- `experiments/analysis/`、`experiments/plots/` 中很多是一次性研究脚本

以后新增功能时，先问一句：

- 这是一个可复用能力，还是一次性实验脚本？

如果是前者，就不要放进 `experiments/`。

### 建议 7：优先补的是模块级 README，而不是立刻全面重构

当前代码量已经很大，直接做大规模目录迁移，收益未必高，反而容易打断研究节奏。更现实的路径是：

1. 先把顶层 README 重写清楚
2. 给核心目录补短 README
3. 给正式入口脚本统一命名
4. 再按使用频率逐步抽离 `experiments/case_study`

这比一次性大改结构更稳。

## 一个更适合长期演进的目标结构

如果后续你准备做一次中等规模整理，我建议目标可以类似下面这样：

```text
Png2Dxf/
├── src/
│   ├── data_engine/          # pngtool + preprocess 的稳定部分
│   ├── geometry_engine/      # axis_engine
│   ├── shearwall_pred/       # 预测模型
│   ├── structural_modeling/  # shearwall_modeling
│   └── common/               # misc
├── pipelines/
│   ├── train/
│   ├── inference/
│   ├── fem/
│   └── external_tools/
├── research/
│   ├── ablation/
│   ├── metrics/
│   ├── analysis/
│   └── plots/
├── scripts/
├── data/
├── outputs/
└── README.md
```

这里的关键不是目录名字本身，而是把“稳定模块”和“研究脚本”强行分层。

## 现阶段最值得优先做的整理动作

如果只做少量高收益整理，我建议优先顺序如下：

1. 重写顶层 README，明确项目目标、主流程、目录语义
2. 将 `experiments/case_study` 视为待抽离正式流程，停止继续向其中堆新能力
3. 新增统一 `scripts/` 目录，逐步迁移根目录脚本
4. 给 `axis_engine/`、`preprocess/`、`shearwall_pred/`、`shearwall_modeling/` 各补一个短 README
5. 逐步建立统一的数据与输出目录约定

## 说明

这份 README 基于当前项目梳理与少量入口代码整理而成，目的是帮助后续维护者快速理解项目主线，而不是枚举全部实现细节。对大型研究型代码库来说，先把“主干边界”和“长期维护原则”讲清楚，比把每个脚本都写进 README 更重要。
