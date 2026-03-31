# shearwall_modeling

本目录负责将结构方案转换为可分析的结构模型，并进行 OpenSees 相关分析与校核。

## 主要职责

- 结构域对象定义
- 参数配置
- 几何与模态分析基础能力
- DXF 到结构建模数据转换
- OpenSees 建模、分析与规范校核

## 主要文件

- `config.py`: 建模配置
- `domain.py`: 结构域对象
- `geometry.py`: 几何处理
- `evaluation.py`: 结果评估与校核
- `convert_dxf.py`: DXF 到 FEM 拓扑转换
- `builders/`: 具体建模器实现

## 边界建议

- 这里应聚焦“结构分析建模”
- 不建议继续堆积论文期临时分析脚本
- 与 `experiments/case_study` 或后续 `pipelines` 的关系应保持为：
  - 流程层调用本目录
  - 本目录尽量不依赖上层实验脚本
