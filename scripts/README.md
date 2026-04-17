# Scripts

本目录用于放置项目级入口脚本。

放置原则：

- 项目级可执行入口统一放在 `scripts/`
- 根目录不再新增业务脚本
- 入口脚本只做参数解析与调用，不承载核心实现
- 每个脚步入口处需加上常规示例调用命令

当前入口：

- `scripts.visualize_dataset_main`
- `scripts.shearwall_analyzer_main`
- `scripts.misc_test_main`
- `scripts.parametric_dataset`
- `scripts.inspect_member_graph`
