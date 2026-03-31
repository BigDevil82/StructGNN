# Research Skeleton

本目录是 `experiments/` 中“研究脚本”部分的目标落点。

当前已有的研究脚本仍保留在旧目录：

- `ablation/`
- `analysis/`
- `metrics/`
- `plots/`

本阶段不立即移动这些目录，仅先建立目标骨架，降低重构风险。

后续如果继续整理，可逐步迁移为：

```text
research/
├── ablation/
├── analysis/
├── metrics/
└── plots/
```

原则：

- 一次性实验、论文分析、统计绘图，归入 `research`
- 可重复执行的业务流程，不再归入 `research`
