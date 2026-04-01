# YJK Pipeline

该目录提供统一接口，完成以下流程：

1. 读取智能设计导出的 JSON（支持 shearwalls/beams 或 members 结构）
2. 调用 YJK API 进行建模
3. 执行结构分析命令
4. 提取层间位移角等关键结果并导出报告

## 对外接口

- Python API: `run_pipeline_from_json(json_path, config)`
- YJK 入口函数: `pyyjks()` in `run_case_study_pipeline.py`

## 主要模块

- `config.py`: 参数配置
- `json_loader.py`: JSON 解析与坐标预处理
- `model_builder.py`: JSON 驱动建模
- `analysis_runner.py`: 分析命令执行
- `result_extractor.py`: 后处理结果提取
- `pipeline.py`: 统一编排
