# Project Structure Guide

本文件用于约束项目后续的目录演进，避免顶层结构继续发散。

## 顶层目录分层

当前项目建议按 4 类理解顶层目录。

### 1. 核心能力模块

这类目录应作为长期维护主干：

- `axis_engine/`
- `preprocess/`
- `shearwall_pred/`
- `shearwall_modeling/`
- `misc/`

规则：

- 只放可复用模块
- 不放论文期一次性脚本
- 不放特定实验结论导出代码

### 2. 流程与兼容层

这类目录用于承载端到端流程入口与迁移兼容层：

- `scripts/`
- `experiments/pipelines/`

规则：

- 允许保留包装层
- 允许作为稳定入口命名空间
- 新的流程型代码应优先考虑落在这里，而不是根目录

### 3. 研究与分析

这类目录主要服务于论文与实验分析：

- `experiments/ablation/`
- `experiments/analysis/`
- `experiments/metrics/`
- `experiments/plots/`
- `baseline_edge_gnn/`
- `beam_pred/`

规则：

- 允许一定临时性
- 不应反向依赖流程层
- 不应成为正式功能的长期落点

### 4. 数据与输出

这类目录不承载核心逻辑：

- `dxf/`
- `data_cache/`
- `result/`
- `paper/`

规则：

- 只放输入、缓存、结果、论文材料
- 不放新的可执行业务逻辑

## 根目录约束

根目录应尽量只保留：

- 项目说明
- 依赖配置
- 少量兼容入口
- 协作说明文件

原则：

- 不再新增新的业务脚本到根目录
- 新入口优先放 `scripts/`
- 新流程优先放 `experiments/pipelines/` 或未来独立 `pipelines/`

## case_study 的特殊约束

`experiments/case_study/` 目前仍保留旧实现，但它已经不是单纯实验脚本。

规则：

- 不再继续向旧 `case_study/` 深度堆积新能力
- 新增流程优先放到 `experiments/pipelines/case_study/`
- 真正迁移实现时必须保留兼容层

## 后续推荐方向

当前阶段不强制重命名目录，但后续可逐步收敛到：

```text
src/          # 核心能力模块
pipelines/    # 正式流程
research/     # 研究与分析
data/         # 原始/中间数据
outputs/      # 结果输出
```

在没有完成兼容层之前，不建议直接整体搬迁。
