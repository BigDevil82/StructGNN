# RC-GNN

面向住宅剪力墙结构智能设计研究的代码库。项目主线是将建筑平面输入逐步转换为房间级图表示，完成剪力墙布局预测，并继续生成结构分析所需的标准化输入。

## 当前能力

- 从 DXF / CAD 相关数据构建房间级图输入
- 训练与评估房间级剪力墙预测模型
- 运行案例级推理、对称后处理与 FEM 拓扑构建
- 导出共享结构输入，供 ETABS、OpenSees、YJK 等结构后端使用
- 使用 OpenSees 进行结构建模与规范校核

## 目录结构

```text
Png2Dxf/
├── src/
│   ├── axis_engine/          # CAD / 几何处理能力
│   ├── data_engine/          # 数据预处理、DXF 提取、PNG 工具
│   ├── misc/                 # 通用工具
│   ├── shearwall_modeling/   # OpenSees 建模与校核
│   └── shearwall_pred/       # 剪力墙预测模型
├── pipelines/
│   ├── case_study/           # DXF -> 推理 -> FEM 拓扑 -> 结构输入
│   ├── structural/etabs/     # ETABS 建模后端
│   └── yjk_pipeline/         # 盈建科流程
├── scripts/                  # 项目级入口脚本
├── experiments/              # 研究、论文实验、分析与绘图
├── data/                     # 输入数据与缓存
├── outputs/                  # 结果输出
├── docs/                     # 项目文档与说明
└── README.md
```

## 主要流程

### 1. 数据与图构建

`src/data_engine/` 负责从 DXF / CAD / PNG 相关数据中提取墙体、门窗、房间等几何信息，并整理为训练和推理可用的图表示。

### 2. 剪力墙预测

`src/shearwall_pred/` 包含数据集、增强、模型、损失、训练、测试、交叉验证与集成推理相关实现。

### 3. 案例级推理

`pipelines/case_study/` 负责正式案例流程：

- 读取 DXF
- 生成推理图输入
- 运行剪力墙预测
- 执行对称后处理
- 构建 FEM 拓扑
- 导出 `*_structural_input.json`

### 4. 结构建模与分析

- `pipelines/structural/etabs/` 负责 ETABS 后端建模
- `src/shearwall_modeling/` 负责 OpenSees 建模与规范校核
- `pipelines/yjk_pipeline/` 负责独立的盈建科流程

## 环境

项目当前使用本地虚拟环境 `.venv`。

推荐：

```bash
.venv\Scripts\activate
uv sync
```

也可以按 `pyproject.toml` 或 `requirements.txt` 手动安装依赖。

## 常用入口

### 案例级推理

```bash
.venv\Scripts\python.exe pipelines\case_study\run_case_study.py ^
  --dxf_path .\data\dxf\cad_json_data\archi_comp.dxf ^
  --category 2 ^
  --symmetry_mode union
```

输出位于 `outputs/result/case_study/`，其中包含可供结构后端复用的 `*_structural_input.json`。

### OpenSees 结构分析

```bash
.venv\Scripts\python.exe -m scripts.shearwall_analyzer_main
```

### ETABS 后端

如果已经有共享结构输入 JSON，可直接使用：

```bash
.venv\Scripts\python.exe pipelines\structural\etabs\create_etabs_model.py ^
  --structural_input_path outputs\result\case_study\archi_comp_structural_input.json
```

### 训练 / 测试剪力墙预测模型

```bash
.venv\Scripts\python.exe -m src.shearwall_pred.trainer --mode train
```

```bash
.venv\Scripts\python.exe -m src.shearwall_pred.trainer --mode test --ckpt outputs\result\...\best_model.pth
```

## 当前约定

- 正式流程放在 `pipelines/`
- 核心能力放在 `src/`
- 研究与论文代码放在 `experiments/`
- 新的业务入口放在 `scripts/`
- 结构后端优先消费共享结构输入，而不是直接依赖案例流程内部细节

## 相关文档

- [`docs/PROJECT_STRUCTURE.md`](/docs/PROJECT_STRUCTURE.md)
- [`docs/notes/项目梳理.md`](/docs/notes/项目梳理.md)
