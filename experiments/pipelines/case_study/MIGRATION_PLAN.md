# Case Study Migration Plan

本文件用于约束 `experiments.case_study` 的低风险迁移顺序。

当前原则：

- 不做一次性整体搬迁
- 每次只迁移一小层能力
- 旧路径保留兼容层
- 每一步迁移后都做最小验证并单独提交

## 当前依赖结论

`case_study` 目前可以拆成 3 层：

### A. 纯几何 / FEM 构件生成层

相对低风险，依赖较少：

- `fem_builder.py`
- `symmetry_postprocess.py`
- `unit.py`

特点：

- 主要依赖 `preprocess/`、`shapely`、`matplotlib`
- 不依赖 ETABS COM
- 可被其他模块复用

### B. 推理流程入口层

中等风险：

- `run_case_study.py`

特点：

- 依赖 `shearwall_pred`
- 依赖 A 层模块
- 是当前案例流程的主入口

### C. ETABS / 外部软件桥接层

高风险，最后处理：

- `create_etabs_model.py`
- `etabs_util.py`
- `yjk_pipeline/`

特点：

- 依赖 COM / 外部软件环境
- 依赖单位系统
- 运行环境要求高
- 迁移时最容易引入隐藏问题

## 已确认的外部引用

当前仓库中明确引用 `experiments.case_study` 的位置：

- `shearwall_modeling/convert_dxf.py` -> `fem_builder`
- `experiments/case_study/run_case_study.py` -> `fem_builder`, `symmetry_postprocess`
- `experiments/case_study/create_etabs_model.py` -> `etabs_util`, `symmetry_postprocess`, `unit`, `fem_builder`

这说明：

- `fem_builder.py` 是最核心、最值得优先包装的模块
- `create_etabs_model.py` 是耦合最深的模块，不适合先动

## 推荐迁移顺序

### 第 1 步：先迁纯工具模块的包装层

优先级最高：

1. `fem_builder.py`
2. `symmetry_postprocess.py`
3. `unit.py`

做法：

- 在 `experiments/pipelines/case_study/` 下创建同名包装模块
- 新路径先转发到旧实现
- 暂不改旧实现内部 import

收益：

- 新流程命名空间开始完整化
- 风险极低
- 可以让后续新代码优先使用新路径

### 第 2 步：再处理主入口

在第 1 步稳定后：

4. `run_case_study.py`

做法：

- 继续保留旧入口
- 新入口逐步改为只依赖 `experiments.pipelines.case_study.*`
- 当依赖全部切到新路径后，再考虑移动真实实现

### 第 3 步：最后处理 ETABS 链路

最后再动：

5. `etabs_util.py`
6. `create_etabs_model.py`
7. `yjk_pipeline/`

做法：

- 先包装，再迁移
- 迁移前必须确认外部软件环境和最小验证方式

## 不建议立即处理的问题

### `etabs_util.py` 的导入写法

当前文件里存在：

```python
from unit import *
```

这是一个高风险点，因为它依赖运行目录与 `sys.path` 环境，不适合在当前阶段顺手修。

正确策略是：

- 等 ETABS 链路单独处理时一起修
- 不要和当前结构整理混在同一批提交中

### `run_case_study.py` 中的 `sys.path` 注入

当前为了脚本直跑，文件里有项目根路径注入逻辑。

这不是当前最该先动的点。更稳妥的做法是：

- 先建立新命名空间
- 再逐步减少脚本内路径修补代码

## 近期可执行动作

按照低风险原则，下一批建议是：

1. 为 `fem_builder.py` 建立新路径兼容包装
2. 为 `symmetry_postprocess.py` 建立新路径兼容包装
3. 为 `unit.py` 建立新路径兼容包装
4. 更新 `run_case_study` 包装层使其开始依赖新命名空间

这 4 步完成后，`case_study` 的新旧路径关系就会清晰很多，但仍然不会破坏旧入口。
