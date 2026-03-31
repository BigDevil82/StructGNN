# ETABS Backend

本目录负责 ETABS 结构建模后端，不负责 DXF 推理流程。

推荐输入方式：

- 直接传入 `FEMInput`
- 或读取 `case_study` 导出的 `*_structural_input.json`

兼容输入方式：

- 继续通过旧的 `fem_result` 字典
- 或由本模块内部临时执行 DXF -> 预测 -> FEM 拓扑流程

推荐逐步迁移到：

- `pipelines.case_study` 负责生成标准结构输入
- `pipelines.structural.etabs` 只消费标准结构输入并建模
