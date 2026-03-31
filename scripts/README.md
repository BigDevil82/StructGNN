# Scripts

本目录用于放置项目级入口脚本。

当前阶段的整理原则是：

- 将根目录中的可执行脚本逐步迁移到 `scripts/`
- 根目录保留薄包装文件，避免旧命令立即失效
- 暂不移动核心业务包，优先降低重构风险

当前已迁移的入口：

- `scripts.visualize_dataset_main`
- `scripts.shearwall_analyzer_main`
- `scripts.misc_test_main`
