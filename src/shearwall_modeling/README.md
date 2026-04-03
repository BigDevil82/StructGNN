# shearwall_modeling

本目录负责将结构方案转换为可分析的结构模型，并进行 OpenSees 相关分析与校核。

## 目录结构

```text
shearwall_modeling/
├── core/        # 配置、常量、领域模型
├── geometry/    # 几何原语、缩放、质量估算
├── analysis/    # 模态组合、反应谱分析、规范校核
├── builders/    # OpenSees 建模器
├── io/          # DXF/JSON 等输入输出适配
├── parametric/  # 参数化模型生成
└── *.py         # 顶层兼容转发模块，逐步废弃
```

## 使用建议

- 新代码优先从子包导入，例如 `shearwall_modeling.core.domain`
- 结构分析建模逻辑放在本目录，流程编排留给上层 `pipelines`
