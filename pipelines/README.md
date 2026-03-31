# Pipelines

本目录承载项目中的正式流程与结构后端。

当前已包含：

- `case_study/`: DXF -> 推理 -> 后处理 -> FEM 拓扑 -> 结构输入导出
- `structural/etabs/`: ETABS 结构建模后端
- `yjk_pipeline/`: 独立的盈建科流程

放置原则：

- 端到端流程放在 `pipelines/`
- 具体结构软件后端放在 `pipelines/structural/`
- 不再将正式流程放回 `experiments/`
