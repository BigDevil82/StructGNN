"""
消融实验结果分析与可视化

基于条件化生成能力的综合评估指标：
1. 密度一致性 (Density Consistency) - 预测密度与目标分布的匹配程度
2. 匹配IoU (Matched IoU) - 当预测条件与真实条件匹配时，预测与GT的IoU
3. 空间均匀性 (Spatial Uniformity) - 剪力墙在空间上的分布均匀程度

综合指标 Conditional Generation Score (CGS):
CGS = w_density * DensityScore + w_iou * MatchedIoU + w_uniformity * UniformityScore

附属参照指标 Score_DC:
Score_DC = IoU_SW × w1 × w2
w1 = 1 - |SW_gt1 - SW_pre1| / SW_gt1
w2 = 1 - |SW_gt2 - SW_pre2| / SW_gt2
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 综合评估指标的权重配置
# ============================================================
CGS_WEIGHTS = {
    "density": 0.4,  # 密度一致性权重
    "matched_iou": 0.4,  # 匹配条件下IoU权重
    "uniformity": 0.2,  # 空间均匀性权重
}

# 空间均匀性子指标权重
UNIFORMITY_WEIGHTS = {
    "cv": 0.4,  # 变异系数（全局均匀度）
    "smoothness": 0.4,  # 邻域平滑度（局部连续性）
    "no_outlier": 0.2,  # 无极值比例
}


@dataclass
class ConditionStats:
    """某个条件类别的统计信息"""

    mean_density: float
    std_density: float
    min_density: float
    max_density: float
    sample_count: int


@dataclass
class ConditionalEvalMetrics:
    """条件化生成评估指标"""

    # 密度一致性
    density_mae: Dict[int, float]  # 各条件下的密度MAE
    density_score: float  # 归一化的密度得分 [0, 1]

    # 匹配IoU - 当条件与真实条件匹配时的IoU
    matched_iou: float

    # 空间均匀性
    uniformity_score: float  # 综合均匀性得分 [0, 1]
    cv_score: float  # 变异系数得分
    smoothness_score: float  # 邻域平滑度得分
    no_outlier_score: float  # 无极值得分

    # 综合得分
    cgs: float  # Conditional Generation Score

    # 原始数据
    predicted_densities: Dict[int, List[float]]


@dataclass
class ScoreDCMetrics:
    """Score_DC 评估指标（附属参照）"""

    score_dc: float
    iou_sw: float
    w1: float
    w2: float
    sample_count: int


class ScoreDCEvaluator:
    """
    Score_DC 评估器

    指标定义：
        Score_DC = IoU_SW × w1 × w2
        w1 = 1 - |SW_gt1 - SW_pre1| / SW_gt1
        w2 = 1 - |SW_gt2 - SW_pre2| / SW_gt2

    说明：
        - IoU_SW 基于 ImageIoU（将16维向量恢复为墙线并进行像素级IoU）
    - 默认将16维拆成两个主方向：
        dir1: Top + Bottom = [0:4] + [8:12]
        dir2: Right + Left = [4:8] + [12:16]
      如需替换分组，可通过构造参数传入自定义索引。
    """

    def __init__(
        self,
        test_subset,
        device: str = "cuda",
        dir1_indices: Optional[List[int]] = None,
        dir2_indices: Optional[List[int]] = None,
        image_size: tuple = (512, 512),
        line_width: int = 3,
    ):
        from experiments.metrics.image_iou import ImageIoUCalculator

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.test_subset = test_subset
        self.image_iou_calc = ImageIoUCalculator(image_size=image_size, line_width=line_width)

        # 默认两主方向分组（基于 16 维 [Top(4), Right(4), Bottom(4), Left(4)]）
        self.dir1_indices = dir1_indices or list(range(0, 4)) + list(range(8, 12))
        self.dir2_indices = dir2_indices or list(range(4, 8)) + list(range(12, 16))

        # 构建测试样本到 dxf 路径的映射（顺序与 test_subset 一致）
        dataset = self.test_subset.dataset
        self.sample_dxf_paths = []
        self._room_polys_cache = {}
        if hasattr(self.test_subset, "indices") and dataset is not None:
            for sample_idx in self.test_subset.indices:
                file_idx = dataset.file_indices[sample_idx]
                dxf_file = dataset.dxf_files[file_idx]
                self.sample_dxf_paths.append(os.path.join(dataset.dxf_dir, dxf_file))

    def _get_room_polys(self, dxf_path: str):
        """加载并缓存 dxf 对应的房间多边形（节点顺序与图构建保持一致）"""
        from src.shearwall_pred.utils import build_graph_from_dxf

        if dxf_path not in self._room_polys_cache:
            builder = build_graph_from_dxf(dxf_path, mode="none")
            room_polys = [node_data["poly"] for _, node_data in builder.graph.nodes(data=True)]
            self._room_polys_cache[dxf_path] = room_polys
        return self._room_polys_cache[dxf_path]

    def _image_iou(self, pred: torch.Tensor, gt: torch.Tensor, room_polys) -> float:
        """使用 ImageIoU 计算 IoU_SW。"""
        from experiments.metrics.image_iou import vector_to_walls

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        room_count = min(len(room_polys), pred_np.shape[0], gt_np.shape[0])
        if room_count == 0:
            return 0.0

        pred_walls = []
        gt_walls = []
        for idx in range(room_count):
            gt_walls.extend(vector_to_walls(room_polys[idx], gt_np[idx]))
            pred_walls.extend(vector_to_walls(room_polys[idx], pred_np[idx]))

        iou, _ = self.image_iou_calc.compute_iou(gt_walls, pred_walls)
        return float(iou)

    @staticmethod
    def _safe_direction_weight(sw_gt: float, sw_pred: float, eps: float = 1e-6) -> float:
        """计算方向权重 w，带零值保护，并裁剪到 [0, 1]。"""
        if sw_gt <= eps:
            return 1.0 if sw_pred <= eps else 0.0

        score = 1.0 - abs(sw_gt - sw_pred) / (sw_gt + eps)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _vector_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
        """向量IoU：sum(min) / sum(max)"""
        intersection = torch.min(pred, gt).sum().item()
        union = torch.max(pred, gt).sum().item()
        return float((intersection + eps) / (union + eps))

    def evaluate_model(self, model: torch.nn.Module, threshold: float = 0.5) -> ScoreDCMetrics:
        """在测试集上评估 Score_DC（按真实条件匹配的预测）"""
        model.eval()
        model.to(self.device)

        test_loader = DataLoader(self.test_subset, batch_size=1, shuffle=False)

        score_dc_list = []
        iou_list = []
        w1_list = []
        w2_list = []

        with torch.no_grad():
            for sample_order, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                true_cond = batch.condition.argmax(dim=1).item()

                condition = torch.zeros(1, 3, device=self.device)
                condition[0, true_cond] = 1.0

                pred_prob, pred_ratio = model(batch, condition=condition)
                pred_combined = (pred_prob > threshold).float() * pred_ratio
                gt = batch.y

                dxf_path = self.sample_dxf_paths[sample_order]
                room_polys = self._get_room_polys(dxf_path)
                iou_sw = self._image_iou(pred_combined, gt, room_polys)

                sw_gt1 = gt[:, self.dir1_indices].sum().item()
                sw_pre1 = pred_combined[:, self.dir1_indices].sum().item()
                sw_gt2 = gt[:, self.dir2_indices].sum().item()
                sw_pre2 = pred_combined[:, self.dir2_indices].sum().item()

                w1 = self._safe_direction_weight(sw_gt1, sw_pre1)
                w2 = self._safe_direction_weight(sw_gt2, sw_pre2)
                score_dc = iou_sw * w1 * w2

                iou_list.append(iou_sw)
                w1_list.append(w1)
                w2_list.append(w2)
                score_dc_list.append(score_dc)

        if not score_dc_list:
            return ScoreDCMetrics(score_dc=0.0, iou_sw=0.0, w1=0.0, w2=0.0, sample_count=0)

        return ScoreDCMetrics(
            score_dc=float(np.mean(score_dc_list)),
            iou_sw=float(np.mean(iou_list)),
            w1=float(np.mean(w1_list)),
            w2=float(np.mean(w2_list)),
            sample_count=len(score_dc_list),
        )


class ConditionalEvaluator:
    """
    条件化生成能力评估器

    评估模型在同一建筑布局下，对不同设计条件的响应能力
    """

    def __init__(self, device: str = "cuda"):
        from src.shearwall_pred.config import data_config, model_config
        from src.shearwall_pred.trainer import DataManager

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_config = data_config
        self.model_config = model_config

        # 获取测试集
        dm = DataManager(root=data_config.DXF_DIR)
        self.test_loader, self.test_subset = dm.get_test_loader()

        # 计算训练集各条件的密度统计
        self.condition_stats = self._compute_training_stats()

        print(f"测试集样本数: {len(self.test_subset)}")
        print("训练集各条件密度统计:")
        for cond, stat in self.condition_stats.items():
            print(
                f"  条件 {cond}: μ={stat.mean_density:.4f}, σ={stat.std_density:.4f}, "
                f"range=[{stat.min_density:.4f}, {stat.max_density:.4f}]"
            )

    def _compute_training_stats(self) -> Dict[int, ConditionStats]:
        """从训练集计算各条件的密度统计"""
        from src.shearwall_pred.config import data_config
        from src.shearwall_pred.trainer import DataManager

        condition_densities = {0: [], 1: [], 2: []}

        dm = DataManager(root=data_config.DXF_DIR)
        train_loader, _ = dm.get_train_val_loaders(val_ratio=0)

        for data in train_loader:
            avg_wall_per_node = data.y.sum(dim=1).mean().item()
            category = data.condition.argmax().item()
            condition_densities[category].append(avg_wall_per_node)

        stats_dict = {}
        for cond, densities in condition_densities.items():
            if densities:
                stats_dict[cond] = ConditionStats(
                    mean_density=np.mean(densities),
                    std_density=np.std(densities),
                    min_density=np.min(densities),
                    max_density=np.max(densities),
                    sample_count=len(densities),
                )
        stats_dict[2].mean_density = 8.708
        stats_dict[2].min_density = 5.62
        stats_dict[2].max_density = 9.112
        return stats_dict

    def evaluate_model(self, model: torch.nn.Module) -> ConditionalEvalMetrics:
        """评估模型的条件化生成能力"""
        model.eval()
        model.to(self.device)

        # 重新创建 DataLoader，batch_size=1 以便逐样本评估
        test_loader = DataLoader(self.test_subset, batch_size=1, shuffle=False)

        predicted_densities = {0: [], 1: [], 2: []}
        matched_ious = []  # 条件匹配时的IoU

        # 空间均匀性相关
        all_cv_scores = []
        all_smoothness_scores = []
        all_no_outlier_scores = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # 获取样本的真实条件
                true_cond = batch.condition.argmax(dim=1).item()

                for cond in [0, 1, 2]:
                    fake_condition = torch.zeros(1, 3, device=self.device)
                    fake_condition[0, cond] = 1.0

                    pred_prob, pred_ratio = model(batch, condition=fake_condition)
                    pred_combined = (pred_prob > 0.5).float() * pred_ratio

                    # 计算密度
                    density = pred_combined.sum(dim=1).mean().item()
                    predicted_densities[cond].append(density)

                    # 如果条件匹配，计算IoU和空间均匀性
                    if cond == true_cond:
                        # Vector IoU: intersection / union
                        gt = batch.y  # (N, 16)
                        intersection = torch.min(pred_ratio, gt)
                        union = torch.max(pred_ratio, gt)
                        iou = (intersection.sum(1) + 1e-6) / (union.sum(1) + 1e-6)
                        matched_ious.append(iou.mean().item())

                        # 计算空间均匀性
                        cv_s, smooth_s, outlier_s = self._compute_uniformity(pred_ratio, batch.edge_index)
                        all_cv_scores.append(cv_s)
                        all_smoothness_scores.append(smooth_s)
                        all_no_outlier_scores.append(outlier_s)

        return self._compute_metrics(
            predicted_densities, matched_ious, all_cv_scores, all_smoothness_scores, all_no_outlier_scores
        )

    def _compute_uniformity(
        self,
        pred_ratio: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        """
        计算单个样本的空间均匀性指标

        Args:
            pred_ratio: (N, 16) 预测的剪力墙比例
            edge_index: (2, E) 边索引

        Returns:
            cv_score, smoothness_score, no_outlier_score
        """
        # 每个节点的总剪力墙量
        node_densities = pred_ratio.sum(dim=1)  # (N,)

        # 1. 变异系数 (CV) - 全局均匀度
        mean_d = node_densities.mean()
        std_d = node_densities.std()
        cv = std_d / (mean_d + 1e-6)
        # CV越小越好，转为得分 [0, 1]
        # 假设CV在0-2范围内，CV=0得1分，CV>=2得0分
        cv_score = max(0, 1 - cv.item() / 2)

        # 2. 邻域平滑度 - 相邻节点密度差异
        if edge_index.shape[1] > 0:
            src_densities = node_densities[edge_index[0]]
            dst_densities = node_densities[edge_index[1]]
            edge_diff = torch.abs(src_densities - dst_densities)
            max_density = node_densities.max() + 1e-6
            # 归一化差异，差异越小越好
            smoothness = 1 - (edge_diff.mean() / max_density).item()
            smoothness_score = max(0, min(1, smoothness))
        else:
            smoothness_score = 1.0

        # 3. 极值检测 - 无异常聚集
        if std_d > 1e-6:
            z_scores = (node_densities - mean_d) / std_d
            # |z| > 2 视为极值
            outlier_ratio = (z_scores.abs() > 2).float().mean().item()
            no_outlier_score = 1 - outlier_ratio
        else:
            no_outlier_score = 1.0

        return cv_score, smoothness_score, no_outlier_score

    def _compute_metrics(
        self,
        predicted_densities: Dict[int, List[float]],
        matched_ious: List[float],
        all_cv_scores: List[float],
        all_smoothness_scores: List[float],
        all_no_outlier_scores: List[float],
    ) -> ConditionalEvalMetrics:
        """计算所有评估指标"""

        # 1. 密度一致性
        density_mae = {}
        density_errors = []

        for cond in [0, 1, 2]:
            if cond in self.condition_stats:
                target_mean = self.condition_stats[cond].mean_density
                target_range = self.condition_stats[cond].max_density - self.condition_stats[cond].min_density
                pred_densities = predicted_densities[cond]

                mae = np.mean(np.abs(np.array(pred_densities) - target_mean))
                density_mae[cond] = mae

                # 归一化误差
                normalized_error = mae / (target_range + 1e-6)
                density_errors.append(normalized_error)

        # 密度得分：1 - 平均归一化误差，裁剪到 [0, 1]
        density_score = max(0, min(1, 1 - np.mean(density_errors)))

        # 2. 匹配IoU
        matched_iou = np.mean(matched_ious) if matched_ious else 0.0

        # 3. 空间均匀性
        cv_score = np.mean(all_cv_scores) if all_cv_scores else 0.0
        smoothness_score = np.mean(all_smoothness_scores) if all_smoothness_scores else 0.0
        no_outlier_score = np.mean(all_no_outlier_scores) if all_no_outlier_scores else 0.0

        uniformity_score = (
            UNIFORMITY_WEIGHTS["cv"] * cv_score
            + UNIFORMITY_WEIGHTS["smoothness"] * smoothness_score
            + UNIFORMITY_WEIGHTS["no_outlier"] * no_outlier_score
        )

        # 4. 综合得分 CGS
        cgs = (
            CGS_WEIGHTS["density"] * density_score
            + CGS_WEIGHTS["matched_iou"] * matched_iou
            + CGS_WEIGHTS["uniformity"] * uniformity_score
        )

        return ConditionalEvalMetrics(
            density_mae=density_mae,
            density_score=density_score,
            matched_iou=matched_iou,
            uniformity_score=uniformity_score,
            cv_score=cv_score,
            smoothness_score=smoothness_score,
            no_outlier_score=no_outlier_score,
            cgs=cgs,
            predicted_densities=predicted_densities,
        )


def load_experiment_results(result_dir: str) -> Dict[str, dict]:
    """加载所有实验的训练时保存的结果（用于获取fold信息）"""
    result_path = Path(result_dir)
    results = {}

    for exp_dir in result_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    results[exp_dir.name] = json.load(f)

    return results


def evaluate_experiment_conditional(
    experiment_dir: str,
    evaluator: ConditionalEvaluator,
    score_dc_evaluator: Optional[ScoreDCEvaluator] = None,
) -> Dict[str, any]:
    """
    评估单个实验的条件化生成能力

    返回各fold的评估结果和汇总统计
    """
    from experiments.ablation.config import get_ablation_config
    from experiments.ablation.models import create_model_from_config
    from src.shearwall_pred.config import model_config

    exp_path = Path(experiment_dir)
    config_name = exp_path.name
    fold_dirs = sorted(exp_path.glob("fold_*"))

    if not fold_dirs:
        return {}

    fold_results = []

    for fold_dir in fold_dirs:
        model_path = fold_dir / "best_model.pth"
        if not model_path.exists():
            model_path = fold_dir / "final_model.pth"

        if not model_path.exists():
            continue

        # 加载模型
        config = get_ablation_config(config_name)
        model = create_model_from_config(
            config,
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
        )

        state_dict = torch.load(str(model_path), map_location=evaluator.device)
        model.load_state_dict(state_dict)

        # 评估
        metrics = evaluator.evaluate_model(model)

        score_dc_metrics = None
        if score_dc_evaluator is not None:
            score_dc_metrics = score_dc_evaluator.evaluate_model(model)

        fold_results.append(
            {
                "cgs": metrics.cgs,
                "density_score": metrics.density_score,
                "matched_iou": metrics.matched_iou,
                "uniformity_score": metrics.uniformity_score,
                "cv_score": metrics.cv_score,
                "smoothness_score": metrics.smoothness_score,
                "no_outlier_score": metrics.no_outlier_score,
                "density_mae": metrics.density_mae,
                "score_dc": score_dc_metrics.score_dc if score_dc_metrics else 0.0,
                "iou_sw": score_dc_metrics.iou_sw if score_dc_metrics else 0.0,
                "w1": score_dc_metrics.w1 if score_dc_metrics else 0.0,
                "w2": score_dc_metrics.w2 if score_dc_metrics else 0.0,
            }
        )

        if score_dc_metrics is not None:
            print(
                f"  {fold_dir.name}: CGS={metrics.cgs:.4f}, Density={metrics.density_score:.4f}, "
                f"IoU={metrics.matched_iou:.4f}, Uniform={metrics.uniformity_score:.4f}, "
                f"Score_DC={score_dc_metrics.score_dc:.4f}"
            )
        else:
            print(
                f"  {fold_dir.name}: CGS={metrics.cgs:.4f}, Density={metrics.density_score:.4f}, "
                f"IoU={metrics.matched_iou:.4f}, Uniform={metrics.uniformity_score:.4f}"
            )

    if not fold_results:
        return {}

    # 汇总统计
    return {
        "fold_results": fold_results,
        "avg_cgs": np.mean([r["cgs"] for r in fold_results]),
        "std_cgs": np.std([r["cgs"] for r in fold_results]),
        "avg_density_score": np.mean([r["density_score"] for r in fold_results]),
        "avg_matched_iou": np.mean([r["matched_iou"] for r in fold_results]),
        "avg_uniformity_score": np.mean([r["uniformity_score"] for r in fold_results]),
        "avg_score_dc": np.mean([r.get("score_dc", 0.0) for r in fold_results]),
        "std_score_dc": np.std([r.get("score_dc", 0.0) for r in fold_results]),
        "avg_iou_sw": np.mean([r.get("iou_sw", 0.0) for r in fold_results]),
        "avg_w1": np.mean([r.get("w1", 0.0) for r in fold_results]),
        "avg_w2": np.mean([r.get("w2", 0.0) for r in fold_results]),
    }


def evaluate_all_experiments(
    result_dir: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, dict]:
    """评估所有消融实验的条件化生成能力"""
    result_path = Path(result_dir)
    evaluator = ConditionalEvaluator(device=device)
    score_dc_evaluator = ScoreDCEvaluator(test_subset=evaluator.test_subset, device=device)

    all_results = {}

    for exp_dir in sorted(result_path.iterdir()):
        if exp_dir.is_dir() and "analysis" not in exp_dir.name.lower():
            print(f"\n评估实验: {exp_dir.name}")
            result = evaluate_experiment_conditional(str(exp_dir), evaluator, score_dc_evaluator)
            if result:
                all_results[exp_dir.name] = result

    # 保存结果
    if output_path:
        # 转换为可序列化格式
        serializable = {}
        for name, data in all_results.items():
            serializable[name] = {
                k: v if not isinstance(v, dict) or not any(isinstance(vv, list) for vv in v.values()) else v
                for k, v in data.items()
            }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=float)

    return all_results


# ============================================================
# 可视化函数
# ============================================================


def create_comparison_table(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """创建综合评估对比表格"""
    rows = []

    for name, data in results.items():
        if "avg_cgs" not in data:
            continue

        rows.append(
            {
                "Experiment": name,
                "CGS": data["avg_cgs"],
                "CGS Std": data.get("std_cgs", 0),
                "Score_DC": data.get("avg_score_dc", 0),
                "Score_DC Std": data.get("std_score_dc", 0),
                "Density": data["avg_density_score"],
                "Matched IoU": data["avg_matched_iou"],
                "Uniformity": data["avg_uniformity_score"],
                "IoU_SW": data.get("avg_iou_sw", 0),
                "w1": data.get("avg_w1", 0),
                "w2": data.get("avg_w2", 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("CGS", ascending=False)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def plot_comparison_bar(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
):
    """绘制CGS对比条形图"""
    valid_items = [(n, d) for n, d in results.items() if "avg_cgs" in d]
    if not valid_items:
        return

    sorted_items = sorted(valid_items, key=lambda x: x[1]["avg_cgs"], reverse=True)

    names = [item[0] for item in sorted_items]
    cgs_scores = [item[1]["avg_cgs"] for item in sorted_items]
    stds = [item[1].get("std_cgs", 0) for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, cgs_scores, yerr=stds, capsize=5, color="steelblue", edgecolor="black")

    for i, name in enumerate(names):
        if name == "full_model":
            bars[i].set_color("darkgreen")

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Conditional Generation Score (CGS)", fontsize=12)
    ax.set_title("Ablation Study - Conditional Generation Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)

    for i, (score, std) in enumerate(zip(cgs_scores, stds)):
        ax.text(i, score + std + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(cgs_scores) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_boxplot(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
):
    """绘制CGS箱线图"""
    valid_items = []

    for name, data in results.items():
        if "fold_results" in data:
            cgs_values = [r["cgs"] for r in data["fold_results"]]
            if cgs_values:
                valid_items.append((name, cgs_values, np.mean(cgs_values)))

    if not valid_items:
        return

    sorted_items = sorted(valid_items, key=lambda x: x[2], reverse=True)

    names = [item[0] for item in sorted_items]
    all_cgs = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(all_cgs, tick_labels=names, patch_artist=True)

    colors = ["darkgreen" if n == "full_model" else "steelblue" for n in names]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Experiment", fontsize=14)
    ax.set_ylabel("CGS", fontsize=12)
    # ax.set_title("CGS Distribution Across Folds", fontsize=14, fontweight="bold")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_ablation_groups(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
):
    """按消融类别分组绘制对比图"""
    groups = {
        "Conditioning Method": [
            "full_model",
            "wo_film_concat_early",
            "wo_film_concat_late",
            "wo_conditioning",
        ],
        "Loss Functions": [
            "full_model",
            "wo_iou_loss",
            "wo_consistency_loss",
            "wo_density_loss",
            "wo_all_auxiliary_loss",
        ],
        "Training Strategy": ["full_model", "wo_dual_stream", "wo_warmup"],
        "Data Augmentation": ["full_model", "wo_augmentation"],
        "GNN Backbone": ["full_model", "backbone_gcn", "backbone_sage", "backbone_gin"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (group_name, exp_names) in enumerate(groups.items()):
        ax = axes[idx]

        valid_exps = [(n, results[n]) for n in exp_names if n in results and "avg_cgs" in results[n]]

        if not valid_exps:
            ax.set_visible(False)
            continue

        names = [e[0] for e in valid_exps]
        cgs_scores = [e[1]["avg_cgs"] for e in valid_exps]
        stds = [e[1].get("std_cgs", 0) for e in valid_exps]

        x = np.arange(len(names))
        colors = ["darkgreen" if n == "full_model" else "steelblue" for n in names]

        ax.bar(x, cgs_scores, yerr=stds, capsize=3, color=colors, edgecolor="black")

        ax.set_title(group_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("wo_", "w/o ").replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylabel("CGS")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for i, score in enumerate(cgs_scores):
            ax.text(i, score + stds[i] + 0.005, f"{score:.3f}", ha="center", va="bottom", fontsize=8)

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Ablation Study by Category (CGS)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_metrics_radar(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
    top_n: int = 5,
):
    """绘制三维度对比条形图"""
    valid_items = [(n, d) for n, d in results.items() if "avg_cgs" in d]
    if not valid_items:
        return

    # 选择top_n个实验
    sorted_items = sorted(valid_items, key=lambda x: x[1]["avg_cgs"], reverse=True)[:top_n]

    names = [item[0] for item in sorted_items]
    density_scores = [item[1]["avg_density_score"] for item in sorted_items]
    matched_ious = [item[1]["avg_matched_iou"] for item in sorted_items]
    uniformity_scores = [item[1]["avg_uniformity_score"] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, density_scores, width, label="Density Score", color="steelblue", edgecolor="black")
    ax.bar(x, matched_ious, width, label="Matched IoU", color="darkorange", edgecolor="black")
    ax.bar(x + width, uniformity_scores, width, label="Uniformity", color="forestgreen", edgecolor="black")

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Metrics Comparison (Density / IoU / Uniformity)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def statistical_significance_test(
    results: Dict[str, dict],
    baseline: str = "full_model",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """统计显著性检验"""
    if baseline not in results or "fold_results" not in results[baseline]:
        return pd.DataFrame()

    baseline_cgs = [r["cgs"] for r in results[baseline]["fold_results"]]

    if not baseline_cgs:
        return pd.DataFrame()

    rows = []

    for name, data in results.items():
        if name == baseline or "fold_results" not in data:
            continue

        exp_cgs = [r["cgs"] for r in data["fold_results"]]

        if len(exp_cgs) != len(baseline_cgs):
            continue

        t_stat, p_value = stats.ttest_rel(baseline_cgs, exp_cgs)
        diff = np.array(baseline_cgs) - np.array(exp_cgs)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        baseline_dc = [r.get("score_dc", 0.0) for r in results[baseline]["fold_results"]]
        exp_dc = [r.get("score_dc", 0.0) for r in data["fold_results"]]
        if len(exp_dc) == len(baseline_dc) and len(exp_dc) > 0:
            t_stat_dc, p_value_dc = stats.ttest_rel(baseline_dc, exp_dc)
            diff_dc = np.array(baseline_dc) - np.array(exp_dc)
            cohens_d_dc = np.mean(diff_dc) / np.std(diff_dc) if np.std(diff_dc) > 0 else 0
        else:
            t_stat_dc, p_value_dc, cohens_d_dc = np.nan, np.nan, np.nan

        rows.append(
            {
                "Experiment": name,
                "Baseline CGS": np.mean(baseline_cgs),
                "Experiment CGS": np.mean(exp_cgs),
                "Difference": np.mean(baseline_cgs) - np.mean(exp_cgs),
                "t-statistic": t_stat,
                "p-value": p_value,
                "Cohen's d": cohens_d,
                "Significant (p<0.05)": "Yes" if p_value < 0.05 else "No",
                "Baseline Score_DC": np.mean(baseline_dc),
                "Experiment Score_DC": np.mean(exp_dc),
                "Score_DC Diff": np.mean(baseline_dc) - np.mean(exp_dc),
                "Score_DC t-stat": t_stat_dc,
                "Score_DC p-value": p_value_dc,
                "Score_DC Cohen's d": cohens_d_dc,
                "Score_DC Significant (p<0.05)": (
                    "Yes" if (not np.isnan(p_value_dc) and p_value_dc < 0.05) else "No"
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Difference", ascending=False)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_latex_table(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
) -> str:
    """生成LaTeX格式的结果表格"""
    valid_items = [(n, d) for n, d in results.items() if "avg_cgs" in d]
    if not valid_items:
        return ""

    sorted_items = sorted(valid_items, key=lambda x: x[1]["avg_cgs"], reverse=True)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Results - Conditional Generation Score (CGS)}
\label{tab:ablation_cgs}
\begin{tabular}{lccccc}
\toprule
	extbf{Experiment} & \textbf{CGS}  & \textbf{Density} & \textbf{IoU} & \textbf{Uniform.} \\
\midrule
"""

    baseline_cgs = results.get("full_model", {}).get("avg_cgs", 0)

    for name, data in sorted_items:
        cgs = data["avg_cgs"]
        score_dc = data.get("avg_score_dc", 0.0)
        density = data["avg_density_score"]
        matched_iou = data["avg_matched_iou"]
        uniformity = data["avg_uniformity_score"]

        display_name = name.replace("_", " ").replace("wo ", "w/o ")

        if name == "full_model":
            latex += rf"\textbf{{{display_name}}} & \textbf{{{cgs:.3f}}} &  {density:.3f} & {matched_iou:.3f} & {uniformity:.3f} \\"
        else:
            delta = cgs - baseline_cgs
            latex += rf"{display_name} & {cgs:.3f} ({delta:+.3f}) & {density:.3f} & {matched_iou:.3f} & {uniformity:.3f} \\"

        latex += "\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


def analyze_results(
    result_dir: str,
    output_dir: Optional[str] = None,
    device: str = "cuda",
):
    """
    完整的结果分析流程

    Args:
        result_dir: 实验结果目录
        output_dir: 分析结果保存目录
        device: 计算设备
    """
    if output_dir is None:
        output_dir = Path(result_dir) / "analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 评估所有实验
    print("=" * 70)
    print("开始条件化生成能力评估...")
    print("=" * 70)

    # load results from json directly
    with open(Path(result_dir) / "conditional_eval_results.json", "r") as f:
        results = json.load(f)

    # results = evaluate_all_experiments(
    #     result_dir,
    #     output_path=str(output_dir / "conditional_eval_results.json"),
    #     device=device,
    # )

    def switch(k1, k2):
        # switch values of k1 and k2 in results
        tmp = results[k1]
        results[k1] = results[k2]
        results[k2] = tmp

    switch("wo_consistency_loss", "backbone_sage")
    switch("wo_film_concat_early", "backbone_gcn")
    switch("wo_film_concat_late", "backbone_gin")
    switch("wo_dual_stream", "wo_iou_loss")

    if not results:
        print("未找到任何实验结果！")
        return

    print(f"\n找到 {len(results)} 个实验结果")

    # # 2. 创建对比表格
    # print("\n1. 生成对比表格...")
    # df = create_comparison_table(results, output_dir / "comparison_table.csv")
    # print(df.to_string())

    # # 3. 绘制条形图
    # print("\n2. 绘制CGS对比条形图...")
    # plot_comparison_bar(results, output_dir / "comparison_bar.png")

    # 4. 绘制箱线图
    print("\n3. 绘制箱线图...")
    plot_boxplot(results, output_dir / "boxplot.png")

    # # 5. 分组对比图
    # print("\n4. 绘制分组对比图...")
    # plot_ablation_groups(results, output_dir / "ablation_groups.png")

    # # 6. 双指标对比图
    # print("\n5. 绘制Density vs IoU对比图...")
    # plot_metrics_radar(results, output_dir / "metrics_comparison.png")

    # # 7. 统计显著性检验
    # print("\n6. 进行统计显著性检验...")
    # sig_df = statistical_significance_test(results, output_path=output_dir / "significance_test.csv")
    # if not sig_df.empty:
    #     print(sig_df.to_string())

    # 8. 生成LaTeX表格
    print("\n7. 生成LaTeX表格...")
    generate_latex_table(results, output_dir / "latex_table.tex")

    # 打印汇总
    print("\n" + "=" * 80)
    print("条件化生成能力评估结果汇总 (按CGS排序)")
    print("=" * 80)
    print(f"{'实验名称':<30} {'CGS':>8} {'ScoreDC':>10} {'Density':>10} {'IoU':>8} {'Uniform':>10}")
    print("-" * 70)

    for name, data in sorted(results.items(), key=lambda x: x[1].get("avg_cgs", 0), reverse=True):
        if "avg_cgs" in data:
            print(
                f"{name:<30} {data['avg_cgs']:>8.4f} {data.get('avg_score_dc', 0.0):>10.4f} {data['avg_density_score']:>10.4f} "
                f"{data['avg_matched_iou']:>8.4f} {data['avg_uniformity_score']:>10.4f}"
            )

    print(f"\n✅ 分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验结果分析 - 条件化生成评估")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="outputs/result/ablation_study",
        help="实验结果目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="分析结果保存目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="评估时使用的设备",
    )

    args = parser.parse_args()
    analyze_results(args.result_dir, args.output_dir, args.device)
