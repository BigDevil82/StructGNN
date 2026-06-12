"""
综合测试评估器

集成所有评估指标，对测试集进行完整评估
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from docs.paper.experiments.metrics.classification_metrics import compute_classification_metrics
from docs.paper.experiments.metrics.regression_metrics import compute_regression_metrics
from src.shearwall_pred.utils import calculate_vector_iou


class ComprehensiveTestEvaluator:
    """
    测试集综合评估器

    整合Vector IoU、Precision/Recall/F1、MAE/RMSE等多项指标，
    对训练好的模型在测试集上进行全面评估。

    示例:
        >>> from shearwall_pred.cross_validate import EnsembleShearWallGNN
        >>> ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
        >>> evaluator = ComprehensiveTestEvaluator(ensemble_model, test_loader)
        >>> results = evaluator.evaluate()
        >>> print(f"IoU: {results['iou']:.4f}, F1: {results['f1']:.4f}")
    """

    def __init__(self, model: torch.nn.Module, test_loader: DataLoader):
        """
        Args:
            model: 训练好的模型（可以是Ensemble模型）
            test_loader: 测试集DataLoader
        """
        self.model = model
        self.test_loader = test_loader
        self.device = next(model.parameters()).device

    def evaluate(self, verbose: bool = True) -> dict:
        """
        在测试集上运行所有指标

        Args:
            verbose: 是否打印详细信息

        Returns:
            dict: 包含以下指标
                - iou: Vector IoU
                - precision: 精确率
                - recall: 召回率
                - f1: F1分数
                - accuracy: 准确率
                - mae: 平均绝对误差
                - rmse: 均方根误差
                - confusion_matrix: 混淆矩阵 (tp, fp, fn, tn)
        """
        self.model.eval()

        all_pred_prob = []
        all_pred_ratio = []
        all_gt = []

        if verbose:
            print(f"正在评估测试集（{len(self.test_loader)} 批次）...")

        # 1. 收集所有预测和GT
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                batch = batch.to(self.device)
                pred_prob, pred_ratio = self.model(batch)

                all_pred_prob.append(pred_prob.cpu().numpy())
                all_pred_ratio.append(pred_ratio.cpu().numpy())
                all_gt.append(batch.y.cpu().numpy())

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  处理进度: {batch_idx + 1}/{len(self.test_loader)}")

        # 2. 拼接所有批次
        pred_prob_all = np.concatenate(all_pred_prob, axis=0)
        pred_ratio_all = np.concatenate(all_pred_ratio, axis=0)
        gt_all = np.concatenate(all_gt, axis=0)

        if verbose:
            print(f"\n总样本数: {pred_ratio_all.shape[0]}")
            print("正在计算指标...")

        # 3. 计算所有指标
        # 3.1 Vector IoU（几何精度）
        iou = calculate_vector_iou(pred_ratio_all, gt_all)

        # 3.2 分类指标（墙体存在性）
        clf_metrics = compute_classification_metrics(pred_prob_all, pred_ratio_all, gt_all)

        # 3.3 回归指标（墙体长度）
        reg_metrics = compute_regression_metrics(pred_ratio_all, gt_all)

        # 4. 汇总结果
        results = {
            # 几何精度
            "iou": iou,
            # 分类性能
            "precision": clf_metrics["precision"],
            "recall": clf_metrics["recall"],
            "f1": clf_metrics["f1"],
            "accuracy": clf_metrics["accuracy"],
            # 回归性能
            "mae": reg_metrics["mae"],
            "rmse": reg_metrics["rmse"],
            # 混淆矩阵
            "confusion_matrix": {
                "tp": clf_metrics["tp"],
                "fp": clf_metrics["fp"],
                "fn": clf_metrics["fn"],
                "tn": clf_metrics["tn"],
            },
            # 元信息
            "num_samples": pred_ratio_all.shape[0],
            "valid_wall_ratio": reg_metrics["valid_ratio"],
        }

        if verbose:
            print("\n✅ 评估完成!")

        return results

    def evaluate_detailed(self) -> dict:
        """
        详细评估，包含per-edge和per-sample的细粒度指标

        Returns:
            dict: 包含基础指标 + per_edge_metrics + per_sample_mae
        """
        from docs.paper.experiments.metrics.classification_metrics import (
            compute_per_edge_classification_metrics,
        )
        from docs.paper.experiments.metrics.regression_metrics import compute_per_edge_regression_metrics

        # 1. 基础评估
        results = self.evaluate(verbose=False)

        # 2. 收集数据
        self.model.eval()
        all_pred_prob = []
        all_pred_ratio = []
        all_gt = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                pred_prob, pred_ratio = self.model(batch)
                all_pred_prob.append(pred_prob.cpu().numpy())
                all_pred_ratio.append(pred_ratio.cpu().numpy())
                all_gt.append(batch.y.cpu().numpy())

        pred_prob_all = np.concatenate(all_pred_prob, axis=0)
        pred_ratio_all = np.concatenate(all_pred_ratio, axis=0)
        gt_all = np.concatenate(all_gt, axis=0)

        # 3. 每条边的指标
        per_edge_clf = compute_per_edge_classification_metrics(pred_prob_all, pred_ratio_all, gt_all)
        per_edge_reg = compute_per_edge_regression_metrics(pred_ratio_all, gt_all)

        # 4. 每个样本的MAE
        reg_metrics = compute_regression_metrics(pred_ratio_all, gt_all)
        per_sample_mae = reg_metrics["mae_per_sample"]

        # 5. 合并结果
        results["per_edge_classification"] = per_edge_clf
        results["per_edge_regression"] = per_edge_reg
        results["per_sample_mae"] = per_sample_mae.tolist()  # 转为list方便JSON序列化

        return results
