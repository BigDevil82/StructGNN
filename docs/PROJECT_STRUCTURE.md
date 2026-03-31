# Project Structure Guide

本文件用于约束项目后续的目录演进，避免顶层结构继续发散。

## 顶层目录分层

当前项目建议按 4 类理解顶层目录。

### 1. 核心能力模块

这类目录应作为长期维护主干：

- `src/axis_engine/`
- `src/data_engine/`
- `src/shearwall_pred/`
- `src/shearwall_modeling/`
- `src/misc/`

规则：

- 只放可复用模块
- 不放论文期一次性脚本
- 不放特定实验结论导出代码

### 2. 正式流程与入口

这类目录用于承载端到端流程入口与结构后端：

- `scripts/`
- `pipelines/`

规则：

- 允许保留少量必要入口包装
- 应作为稳定入口命名空间
- 新的流程型代码应优先考虑落在这里，而不是根目录

### 3. 研究与分析

这类目录主要服务于论文与实验分析：

- `experiments/ablation/`
- `experiments/analysis/`
- `experiments/metrics/`
- `experiments/plots/`
- `experiments/research/`

规则：

- 允许一定临时性
- 不应反向依赖流程层
- 不应成为正式功能的长期落点

### 4. 数据与输出

这类目录不承载核心逻辑：

- `data/`
- `outputs/`
- `docs/`

规则：

- 只放输入、缓存、结果、论文材料
- 不放新的可执行业务逻辑

## 根目录约束

根目录应尽量只保留：

- 项目说明
- 依赖配置
- 少量入口脚本
- 协作说明文件

原则：

- 不再新增新的业务脚本到根目录
- 新入口优先放 `scripts/`
- 新流程优先放 `pipelines/`

## 后续推荐方向

当前建议维持如下理解：

```text
src/          # 核心能力模块
pipelines/    # 正式流程
experiments/  # 研究与分析
data/         # 原始/中间数据
outputs/      # 结果输出
docs/         # 文档与说明
```
