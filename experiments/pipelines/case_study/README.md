# Case Study Pipeline Wrapper

本目录是 `experiments.case_study` 的目标迁移位置。

当前阶段仍然采用兼容包装策略：

- 旧实现继续保留在 `experiments/case_study/`
- 新位置先提供统一入口与说明
- 外部若希望开始使用新路径，可逐步切换到：
  - `experiments.pipelines.case_study`
  - `experiments.pipelines.case_study.run_case_study`

这样做的目的：

- 不立即修改大面积 import
- 先建立正式流程命名空间
- 为后续真正迁移实现文件提供稳定落点

## 当前状态

当前仅包装以下入口：

- `run_case_study`
- `main`

后续如继续整理，可逐步把 `fem_builder.py`、`symmetry_postprocess.py`、`create_etabs_model.py` 等实现迁入此目录，并保留旧路径兼容层。
