# 综合评估指标模块

本模块为剪力墙预测模型提供测试集综合评估功能，包含以下指标：

## 评估指标

### 1. 几何精度
- **Vector IoU**: 预测与GT的交并比

### 2. 分类性能（墙体存在性）
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数
- **Accuracy**: 准确率

### 3. 回归性能（墙体长度）
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差

## 使用方法

### 基础评估
```bash
python experiments/metrics/run_comprehensive_eval.py --result_dir result/shearwall_pred/0126_cond_kfold
```

### 详细评估（包含per-edge指标）
```bash
python experiments/metrics/run_comprehensive_eval.py --result_dir result/shearwall_pred/0126_cond_kfold --detailed
```

### 自定义batch size
```bash
python experiments/metrics/run_comprehensive_eval.py --result_dir result/shearwall_pred/0126_cond_kfold --batch_size 8
```

## 输出结果

运行后会在指定目录下生成：
```
result/shearwall_pred/0126_cond_kfold/comprehensive_metrics/
├── metrics_basic.json      # 基础评估结果
└── metrics_detailed.json   # 详细评估结果（如果使用--detailed）
```

### 结果示例

```json
{
  "iou": 0.6534,
  "precision": 0.8721,
  "recall": 0.8245,
  "f1": 0.8476,
  "accuracy": 0.9156,
  "mae": 0.1234,
  "rmse": 0.1876,
  "confusion_matrix": {
    "tp": 1245,
    "fp": 178,
    "fn": 265,
    "tn": 5432
  }
}
```

## 代码结构

```
experiments/metrics/
├── __init__.py                   # 模块导出
├── classification_metrics.py    # P/R/F1计算
├── regression_metrics.py        # MAE/RMSE计算
├── test_evaluator.py            # 综合评估器
├── run_comprehensive_eval.py    # 运行脚本
└── README.md                    # 本文档
```

## 在Python代码中使用

```python
from experiments.metrics import ComprehensiveTestEvaluator
from shearwall_pred.cross_validate import EnsembleShearWallGNN

# 加载模型
ensemble_model = EnsembleShearWallGNN(model_paths, model_config)

# 创建评估器
evaluator = ComprehensiveTestEvaluator(ensemble_model, test_loader)

# 运行评估
results = evaluator.evaluate()

# 打印结果
print(f"IoU: {results['iou']:.4f}")
print(f"F1-Score: {results['f1']:.4f}")
print(f"MAE: {results['mae']:.4f}")
```

## 与论文对应关系

| 论文指标 | 代码实现 |
|---------|---------|
| Vector IoU | `results['iou']` |
| Precision | `results['precision']` |
| Recall | `results['recall']` |
| F1-Score | `results['f1']` |
| MAE | `results['mae']` |
| Density Consistency | 见 `experiments/ablation/analyze_results.py` |
