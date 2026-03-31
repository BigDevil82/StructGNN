# preprocess

本目录负责从标注好的 DXF 或推理前处理中间结果，构建模型所需的房间级图数据。

## 主要职责

- DXF 构件提取
- 房间校准
- 剪力墙标签分析
- 房间级图构建
- CAD 推理结果桥接到训练同构输入

## 主要文件

- `dxf_extractor.py`: 提取墙、门、窗、房间等几何对象
- `room_calibrator.py`: 校准房间边界
- `room_analyzer.py`: 生成房间边剪力墙标签
- `layout_graph.py`: 构建房间级图
- `cad_inference.py`: 将 CAD JSON 结果转成推理所需图输入

## 边界建议

- 这里应专注“数据表示转换”
- 不建议加入模型训练逻辑
- 与 `axis_engine/` 的关系应保持清晰：
  - `axis_engine/` 负责几何基础能力
  - `preprocess/` 负责训练/推理输入构建
