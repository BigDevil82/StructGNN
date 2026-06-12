"""
综合指标评估脚本

用于对训练好的K-Fold Ensemble模型在测试集上进行完整评估，
计算IoU、Precision/Recall/F1、MAE/RMSE等指标。

使用方法:
    python experiments/metrics/run_comprehensive_eval.py --result_dir outputs/result/shearwall_pred/0126_cond_kfold
    python experiments/metrics/run_comprehensive_eval.py --result_dir outputs/result/shearwall_pred/0126_cond_kfold --detailed
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docs.paper.experiments.metrics.test_evaluator import ComprehensiveTestEvaluator
from src.shearwall_pred.config import data_config, model_config, training_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.dataset import ShearWallDataset


def main():
    parser = argparse.ArgumentParser(description="综合指标评估")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="outputs/result/shearwall_pred/0126_cond_kfold",
        help="K-Fold训练结果目录",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="是否进行详细评估（包含per-edge和per-sample指标）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="测试时的batch size（默认使用配置文件中的值）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("综合指标评估 - 测试集")
    print("=" * 70)

    # 1. 加载测试集
    print(f"\n📂 加载测试集: {data_config.DXF_DIR}/test")
    test_set = ShearWallDataset(
        f"{data_config.CACHE_DIR}/test",
        f"{data_config.DXF_DIR}/test",
        is_test=True,
    )

    batch_size = args.batch_size if args.batch_size is not None else training_config.BATCH_SIZE

    test_loader = DataLoader(
        Subset(test_set, list(range(len(test_set)))),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"   测试集样本数: {len(test_set)}")
    print(f"   Batch size: {batch_size}")

    # 2. 加载Ensemble模型
    cv_path = Path(args.result_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        print(f"\n❌ 错误: 在 {cv_path} 下未找到fold_*/best_model.pth")
        print("   请确认result_dir路径正确，且包含K-Fold训练结果")
        sys.exit(1)

    print(f"\n🔧 加载Ensemble模型 ({len(model_paths)} 个fold)")
    for i, p in enumerate(model_paths, 1):
        print(f"   Fold {i}: {p.name}")

    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(training_config.DEVICE)
    ensemble_model.eval()

    print(f"   设备: {training_config.DEVICE}")

    # 3. 运行评估
    print(f"\n🚀 开始评估...")
    evaluator = ComprehensiveTestEvaluator(ensemble_model, test_loader)

    if args.detailed:
        results = evaluator.evaluate_detailed()
        result_type = "detailed"
    else:
        results = evaluator.evaluate(verbose=True)
        result_type = "basic"

    # 4. 保存结果
    output_dir = cv_path / "comprehensive_metrics"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"metrics_{result_type}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. 打印结果
    print("\n" + "=" * 70)
    print("测试集综合评估结果")
    print("=" * 70)

    print(f"\n📊 几何精度:")
    print(f"  Vector IoU:    {results['iou']:.4f}")

    print(f"\n📊 分类性能 (墙体存在性):")
    print(f"  Precision:     {results['precision']:.4f}")
    print(f"  Recall:        {results['recall']:.4f}")
    print(f"  F1-Score:      {results['f1']:.4f}")
    print(f"  Accuracy:      {results['accuracy']:.4f}")

    print(f"\n📊 回归性能 (墙体长度):")
    print(f"  MAE:           {results['mae']:.4f}")
    print(f"  RMSE:          {results['rmse']:.4f}")

    print(f"\n📊 混淆矩阵:")
    cm = results["confusion_matrix"]
    print(f"  TP (True Pos):  {cm['tp']}")
    print(f"  FP (False Pos): {cm['fp']}")
    print(f"  FN (False Neg): {cm['fn']}")
    print(f"  TN (True Neg):  {cm['tn']}")

    if args.detailed:
        print(f"\n📊 每条边的分类性能:")
        for edge_name, metrics in results["per_edge_classification"].items():
            print(
                f"  {edge_name.capitalize():8s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            )

        print(f"\n📊 每条边的回归性能:")
        for edge_name, metrics in results["per_edge_regression"].items():
            print(f"  {edge_name.capitalize():8s}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print(f"\n📁 结果已保存至: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
