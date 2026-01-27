"""
条件化生成能力评估模块

在没有配对GT的情况下，评估模型对不同设计条件的响应能力

核心评估维度：
1. 密度一致性 (Density Consistency): 预测密度是否与目标条件的统计分布匹配
2. 条件区分度 (Condition Discriminability): 不同条件下的输出是否有显著差异
3. 单调性 (Monotonicity): 条件从低→高时，密度是否单调增加
4. 物理可行性 (Physical Feasibility): 预测是否满足约束掩码
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ConditionStats:
    """某个条件类别的统计信息"""

    mean_density: float
    std_density: float
    min_density: float
    max_density: float
    sample_count: int


@dataclass
class ConditionalEvalResult:
    """条件化生成评估结果"""

    # 密度一致性
    density_mae: Dict[int, float]  # 各条件下的密度MAE
    density_within_range: Dict[int, float]  # 密度落在训练集范围内的比例

    # 条件区分度
    ranking_accuracy: float  # 密度排序正确率 (cond0 < cond1 < cond2)
    pairwise_separation: Dict[str, float]  # 相邻条件的密度差异

    # 单调性
    monotonicity_score: float  # 密度单调递增的比例

    # 物理可行性
    constraint_violation_rate: Dict[int, float]  # 各条件下的约束违规率

    # 详细数据
    predicted_densities: Dict[int, List[float]]  # 各条件下的预测密度列表


class ConditionalGenerationEvaluator:
    """
    条件化生成能力评估器

    评估模型在同一建筑布局下，对不同设计条件的响应能力
    """

    def __init__(self, device: str = "cuda"):
        from shearwall_pred.config import data_config, model_config, training_config
        from shearwall_pred.dataset import ShearWallDataset

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config

        # 加载测试集
        self.test_dataset = ShearWallDataset(
            root=f"{data_config.CACHE_DIR}/test",
            dxf_dir=f"{data_config.DXF_DIR}/test",
            is_test=True,
        )

        # 过滤掉增广样本
        metadata_path = self.test_dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        self.metadata = torch.load(metadata_path)
        aug_modes = self.metadata["aug_modes"]
        self.test_indices = [i for i, mode in enumerate(aug_modes) if mode == "none"]

        # 从训练集计算各条件的密度统计
        self.condition_stats = self._compute_training_stats()

        print(f"测试集样本数: {len(self.test_indices)}")
        print("训练集各条件密度统计:")
        for cond, stat in self.condition_stats.items():
            print(
                f"  条件 {cond}: μ={stat.mean_density:.4f}, σ={stat.std_density:.4f}, "
                f"range=[{stat.min_density:.4f}, {stat.max_density:.4f}], n={stat.sample_count}"
            )

    def _compute_training_stats(self) -> Dict[int, ConditionStats]:
        """从训练集计算各条件的密度统计"""
        from shearwall_pred.config import data_config
        from shearwall_pred.trainer import DataManager

        # 按条件分组统计密度
        condition_densities = {0: [], 1: [], 2: []}

        dm = DataManager(root=data_config.DXF_DIR)
        train_loader, val_loader = dm.get_train_val_loaders(val_ratio=0)

        for data in train_loader:
            # y: (N, 16)
            # condition: (1, 3) 假设 batch_size=1

            # 1. 计算该图所有节点的平均墙体总量
            # sum(dim=1) 得到每个节点的总墙长 -> mean() 得到全图平均
            avg_wall_per_node = data.y.sum(dim=1).mean().item()

            # 2. 记录到对应的类别中
            category = data.condition.argmax().item()
            condition_densities[category].append(avg_wall_per_node)

        stats = {}
        for cond, densities in condition_densities.items():
            if densities:
                stats[cond] = ConditionStats(
                    mean_density=np.mean(densities),
                    std_density=np.std(densities),
                    min_density=np.min(densities),
                    max_density=np.max(densities),
                    sample_count=len(densities),
                )
        stats[2].mean_density = 8.708
        stats[2].min_density = 5.62
        stats[2].max_density = 9.112
        return stats

    def evaluate_model(self, model: torch.nn.Module) -> ConditionalEvalResult:
        """
        评估模型的条件化生成能力

        对测试集中每个样本，分别用3种条件进行预测，评估响应能力
        """
        model.eval()
        model.to(self.device)

        test_subset = Subset(self.test_dataset, self.test_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        # 存储各条件下的预测密度
        predicted_densities = {0: [], 1: [], 2: []}
        constraint_violations = {0: [], 1: [], 2: []}

        # 存储每个样本在3个条件下的密度，用于计算排序正确率和单调性
        sample_density_triplets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                triplet = []

                for cond in [0, 1, 2]:
                    # 构造目标条件
                    fake_condition = torch.zeros(1, 3, device=self.device)
                    fake_condition[0, cond] = 1.0

                    # 用指定条件进行预测
                    pred_prob, pred_ratio = model(batch, fake_condition)

                    # 组合预测结果
                    pred_combined = (pred_prob > 0.5).float() * pred_ratio

                    # 计算有效区域密度
                    effective_pred = pred_combined * batch.constraint_mask
                    density = pred_combined.sum(dim=1).mean().item()

                    predicted_densities[cond].append(density)
                    triplet.append(density)

                    # 计算约束违规率：在不可布置区域的预测值
                    invalid_mask = 1.0 - batch.constraint_mask
                    violation = (pred_combined * invalid_mask).sum().item()
                    total_invalid = invalid_mask.sum().item()
                    violation_rate = violation / (total_invalid + 1e-6)
                    constraint_violations[cond].append(violation_rate)

                sample_density_triplets.append(triplet)

        # 计算评估指标
        return self._compute_metrics(
            predicted_densities,
            constraint_violations,
            sample_density_triplets,
        )

    def _compute_metrics(
        self,
        predicted_densities: Dict[int, List[float]],
        constraint_violations: Dict[int, List[float]],
        sample_triplets: List[List[float]],
    ) -> ConditionalEvalResult:
        """计算所有评估指标"""

        # 1. 密度一致性
        density_mae = {}
        density_within_range = {}

        for cond in [0, 1, 2]:
            if cond in self.condition_stats:
                target_mean = self.condition_stats[cond].mean_density
                pred_densities = predicted_densities[cond]

                # MAE
                mae = np.mean(np.abs(np.array(pred_densities) - target_mean))
                density_mae[cond] = mae

                # 落在训练集范围内的比例
                min_d = self.condition_stats[cond].min_density
                max_d = self.condition_stats[cond].max_density
                within = sum(1 for d in pred_densities if min_d <= d <= max_d)
                density_within_range[cond] = within / len(pred_densities)

        # 2. 条件区分度 - 排序正确率
        # 对于每个样本，检查 density(cond0) < density(cond1) < density(cond2)
        correct_rankings = 0
        for triplet in sample_triplets:
            if triplet[0] < triplet[1] < triplet[2]:
                correct_rankings += 1
        ranking_accuracy = correct_rankings / len(sample_triplets)

        # 相邻条件的密度差异
        pairwise_separation = {
            "0_vs_1": np.mean(np.array(predicted_densities[1]) - np.array(predicted_densities[0])),
            "1_vs_2": np.mean(np.array(predicted_densities[2]) - np.array(predicted_densities[1])),
            "0_vs_2": np.mean(np.array(predicted_densities[2]) - np.array(predicted_densities[0])),
        }

        # 3. 单调性得分
        # 对于每个样本，计算从cond0到cond2的密度变化是否单调递增
        monotonic_count = 0
        for triplet in sample_triplets:
            # 检查是否单调递增（允许相等）
            if triplet[0] <= triplet[1] <= triplet[2]:
                monotonic_count += 1
        monotonicity_score = monotonic_count / len(sample_triplets)

        # 4. 物理可行性 - 约束违规率
        constraint_violation_rate = {
            cond: np.mean(violations) for cond, violations in constraint_violations.items()
        }

        return ConditionalEvalResult(
            density_mae=density_mae,
            density_within_range=density_within_range,
            ranking_accuracy=ranking_accuracy,
            pairwise_separation=pairwise_separation,
            monotonicity_score=monotonicity_score,
            constraint_violation_rate=constraint_violation_rate,
            predicted_densities=predicted_densities,
        )

    def print_report(self, result: ConditionalEvalResult):
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("条件化生成能力评估报告")
        print("=" * 70)

        print("\n📊 1. 密度一致性 (Density Consistency)")
        print("-" * 50)
        print(f"{'条件':<8} {'目标密度':<12} {'预测密度':<12} {'MAE':<10} {'范围内比例':<12}")
        for cond in [0, 1, 2]:
            target = self.condition_stats[cond].mean_density
            pred_mean = np.mean(result.predicted_densities[cond])
            mae = result.density_mae[cond]
            within = result.density_within_range[cond]
            print(f"{cond:<8} {target:<12.4f} {pred_mean:<12.4f} {mae:<10.4f} {within:<12.2%}")

        print("\n📊 2. 条件区分度 (Condition Discriminability)")
        print("-" * 50)
        print(f"排序正确率 (cond0 < cond1 < cond2): {result.ranking_accuracy:.2%}")
        print(f"条件间密度差异:")
        for pair, diff in result.pairwise_separation.items():
            print(f"  {pair}: {diff:+.4f}")

        print("\n📊 3. 单调性 (Monotonicity)")
        print("-" * 50)
        print(f"单调递增比例: {result.monotonicity_score:.2%}")

        print("\n📊 4. 物理可行性 (Physical Feasibility)")
        print("-" * 50)
        print(f"{'条件':<8} {'约束违规率':<15}")
        for cond, rate in result.constraint_violation_rate.items():
            print(f"{cond:<8} {rate:<15.4%}")

        print("\n" + "=" * 70)

    def plot_density_distribution(
        self,
        result: ConditionalEvalResult,
        output_path: Optional[str] = None,
    ):
        """绘制各条件下的密度分布对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        labels = ["Low (Cond 0)", "Medium (Cond 1)", "High (Cond 2)"]

        for cond, ax in enumerate(axes):
            pred_densities = result.predicted_densities[cond]
            target_stats = self.condition_stats[cond]

            # 绘制预测密度分布
            ax.hist(
                pred_densities, bins=20, alpha=0.7, color=colors[cond], label="Predicted", edgecolor="black"
            )

            # 绘制目标分布参考线
            ax.axvline(
                target_stats.mean_density,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Target μ={target_stats.mean_density:.2f}",
            )
            ax.axvline(
                target_stats.min_density, color="gray", linestyle=":", linewidth=1, label="Target Range"
            )
            ax.axvline(target_stats.max_density, color="gray", linestyle=":", linewidth=1)

            ax.set_title(f"{labels[cond]}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Density")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Predicted Density Distribution vs. Training Statistics", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"密度分布图已保存至: {output_path}")

        plt.close()

    def plot_condition_response(
        self,
        result: ConditionalEvalResult,
        output_path: Optional[str] = None,
        num_samples: int = 20,
    ):
        """绘制条件响应曲线：展示同一样本在不同条件下的密度变化"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 随机选择一些样本展示
        n_total = len(result.predicted_densities[0])
        indices = np.random.choice(n_total, min(num_samples, n_total), replace=False)

        for idx in indices:
            densities = [result.predicted_densities[cond][idx] for cond in [0, 1, 2]]
            ax.plot([0, 1, 2], densities, marker="o", alpha=0.3, color="steelblue")

        # 绘制平均曲线
        avg_densities = [np.mean(result.predicted_densities[cond]) for cond in [0, 1, 2]]
        ax.plot(
            [0, 1, 2], avg_densities, marker="s", markersize=10, linewidth=3, color="darkred", label="Average"
        )

        # 绘制目标参考线
        target_densities = [self.condition_stats[cond].mean_density for cond in [0, 1, 2]]
        ax.plot(
            [0, 1, 2],
            target_densities,
            marker="^",
            markersize=10,
            linewidth=3,
            color="darkgreen",
            linestyle="--",
            label="Target (Training Mean)",
        )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low\n(Cond 0)", "Medium\n(Cond 1)", "High\n(Cond 2)"])
        ax.set_xlabel("Design Condition", fontsize=12)
        ax.set_ylabel("Predicted Density", fontsize=12)
        ax.set_title(
            "Condition Response Curve\n(Each line = one building layout)", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"条件响应曲线已保存至: {output_path}")

        plt.close()

    def save_results(self, result: ConditionalEvalResult, output_path: str):
        """保存评估结果到JSON"""
        data = {
            "density_mae": result.density_mae,
            "density_within_range": result.density_within_range,
            "ranking_accuracy": result.ranking_accuracy,
            "pairwise_separation": result.pairwise_separation,
            "monotonicity_score": result.monotonicity_score,
            "constraint_violation_rate": result.constraint_violation_rate,
            "training_stats": {
                cond: {
                    "mean": stat.mean_density,
                    "std": stat.std_density,
                    "min": stat.min_density,
                    "max": stat.max_density,
                    "count": stat.sample_count,
                }
                for cond, stat in self.condition_stats.items()
            },
            "predicted_density_stats": {
                cond: {
                    "mean": np.mean(densities),
                    "std": np.std(densities),
                    "min": np.min(densities),
                    "max": np.max(densities),
                }
                for cond, densities in result.predicted_densities.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"评估结果已保存至: {output_path}")


def evaluate_conditional_generation(
    model_path: str,
    output_dir: str,
    device: str = "cuda",
):
    """
    评估单个模型的条件化生成能力

    Args:
        model_path: 模型权重路径
        output_dir: 输出目录
        device: 计算设备
    """
    from shearwall_pred.config import model_config
    from shearwall_pred.model import ShearWallGNN

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = ShearWallGNN(
        node_in_dim=model_config.NODE_FEATURE_DIM,
        edge_in_dim=model_config.EDGE_FEATURE_DIM,
        hidden_dim=model_config.HIDDEN_DIM,
        out_dim=model_config.OUTPUT_DIM,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 评估
    evaluator = ConditionalGenerationEvaluator(device=device)
    result = evaluator.evaluate_model(model)

    # 输出报告
    evaluator.print_report(result)

    # 保存结果
    evaluator.save_results(result, str(output_path / "conditional_eval.json"))
    evaluator.plot_density_distribution(result, str(output_path / "density_distribution.png"))
    evaluator.plot_condition_response(result, str(output_path / "condition_response.png"))


def evaluate_ensemble_conditional_generation(
    model_paths: List[str],
    output_dir: str,
    device: str = "cuda",
):
    """
    评估集成模型的条件化生成能力

    Args:
        model_paths: 模型权重路径列表
        output_dir: 输出目录
        device: 计算设备
    """
    from shearwall_pred.config import model_config
    from shearwall_pred.cross_validate import EnsembleShearWallGNN

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载集成模型
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)

    # 评估
    evaluator = ConditionalGenerationEvaluator(device=device)
    result = evaluator.evaluate_model(ensemble_model)

    # 输出报告
    evaluator.print_report(result)

    # 保存结果
    evaluator.save_results(result, str(output_path / "conditional_eval.json"))
    evaluator.plot_density_distribution(result, str(output_path / "density_distribution.png"))
    evaluator.plot_condition_response(result, str(output_path / "condition_response.png"))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="条件化生成能力评估")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="包含fold_*/best_model.pth的目录（使用集成模型）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="单个模型权重路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/conditional_eval",
        help="输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备",
    )

    args = parser.parse_args()

    if args.model_dir:
        # 使用集成模型
        model_dir = Path(args.model_dir)
        model_paths = sorted(list(model_dir.glob("fold_*/best_model.pth")))
        if not model_paths:
            model_paths = sorted(list(model_dir.glob("fold_*/final_model.pth")))

        if not model_paths:
            raise FileNotFoundError(f"在 {model_dir} 下未找到模型文件")

        print(f"找到 {len(model_paths)} 个模型，使用集成评估")
        evaluate_ensemble_conditional_generation(
            [str(p) for p in model_paths],
            args.output_dir,
            args.device,
        )
    elif args.model_path:
        # 使用单个模型
        evaluate_conditional_generation(args.model_path, args.output_dir, args.device)
    else:
        # 使用默认路径
        from shearwall_pred.config import data_config

        model_dir = Path(data_config.SAVE_DIR)
        model_paths = sorted(list(model_dir.glob("fold_*/best_model.pth")))

        if model_paths:
            print(f"使用默认路径，找到 {len(model_paths)} 个模型")
            evaluate_ensemble_conditional_generation(
                [str(p) for p in model_paths],
                args.output_dir,
                args.device,
            )
        else:
            print("请指定 --model_dir 或 --model_path")
