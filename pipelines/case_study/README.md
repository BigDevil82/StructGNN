# Case Study Pipeline

本目录负责案例级正式流程：

- DXF 布局读取
- 剪力墙推理
- 对称后处理
- FEM 拓扑构建
- 可视化
- 导出共享结构输入 `*_structural_input.json`

主要入口：

- `run_case_study.py`
- `pipeline_core.predict_structural_result`

边界约定：

- `pipelines.case_study` 负责生成结构结果与共享输入
- `pipelines.structural.*` 负责消费共享输入并调用具体结构建模后端
