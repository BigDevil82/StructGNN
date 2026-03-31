"""
消融实验运行脚本

功能：
1. 根据配置运行单个或多个消融实验
2. 支持K-Fold交叉验证
3. 自动保存结果和检查点
4. 支持断点续训
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablation.config import (
    ABLATION_CONFIGS,
    AblationConfig,
    ConditioningMethod,
    get_ablation_config,
    list_ablation_configs,
)
from experiments.ablation.losses import AblationHybridLoss
from experiments.ablation.models import create_model_from_config
from shearwall_pred.config import data_config, model_config, training_config
from shearwall_pred.dataset import ShearWallDataset
from shearwall_pred.utils import get_file_category


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AblationDataManager:
    """消融实验数据管理器"""

    def __init__(
        self,
        config: AblationConfig,
        train_dxf_dir: str,
        test_dxf_dir: str,
        cache_dir: str,
    ):
        self.config = config
        self.train_dxf_dir = train_dxf_dir
        self.test_dxf_dir = test_dxf_dir
        self.cache_dir = cache_dir

    def get_kfold_splits(self, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """生成K-Fold划分（基于文件级别）"""
        dataset = ShearWallDataset(
            root=f"{self.cache_dir}/train",
            dxf_dir=self.train_dxf_dir,
        )

        metadata_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(metadata_path)

        file_indices = metadata["file_indices"]
        dxf_files = metadata["data/dxf_files"]
        aug_modes = metadata["aug_modes"]

        # 获取唯一文件及其类别
        unique_files = sorted(set(file_indices))
        file_categories = [get_file_category(dxf_files[f]) for f in unique_files]

        # 分层K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for train_idx, val_idx in skf.split(unique_files, file_categories):
            train_files = [unique_files[i] for i in train_idx]
            val_files = [unique_files[i] for i in val_idx]
            splits.append((train_files, val_files))

        return splits, dataset, metadata

    def get_fold_loaders(
        self,
        fold_idx: int,
        splits: List,
        dataset: ShearWallDataset,
        metadata: dict,
    ) -> Tuple[DataLoader, DataLoader]:
        """获取指定fold的训练和验证DataLoader"""
        train_files, val_files = splits[fold_idx]

        file_indices = metadata["file_indices"]
        aug_modes = metadata["aug_modes"]

        # 建立文件到样本的映射
        file_to_samples = defaultdict(list)
        for sample_idx, f_idx in enumerate(file_indices):
            file_to_samples[f_idx].append(sample_idx)

        # 展开训练样本索引（包含增广）
        train_indices = []
        for fid in train_files:
            if self.config.use_augmentation:
                train_indices.extend(file_to_samples[fid])
            else:
                # 仅保留无增广样本
                train_indices.extend(
                    [s for s in file_to_samples[fid] if aug_modes[s] == "none"]
                )

        # 验证集：不包含增广
        val_indices = []
        for fid in val_files:
            val_indices.extend(
                [s for s in file_to_samples[fid] if aug_modes[s] == "none"]
            )

        # 加权采样
        if self.config.use_weighted_sampling:
            train_cats = []
            dxf_files = metadata["data/dxf_files"]
            for idx in train_indices:
                f_idx = file_indices[idx]
                train_cats.append(get_file_category(dxf_files[f_idx]))

            cat_counts = Counter(train_cats)
            total = len(train_cats)
            weights_map = {c: total / cnt for c, cnt in cat_counts.items()}
            sample_weights = [weights_map[c] for c in train_cats]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
        )

        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def get_test_loader(self) -> DataLoader:
        """获取测试集DataLoader"""
        dataset = ShearWallDataset(
            root=f"{self.cache_dir}/test",
            dxf_dir=self.test_dxf_dir,
        )

        metadata_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(metadata_path)
        aug_modes = metadata["aug_modes"]

        # 仅保留无增广样本
        test_indices = [i for i, mode in enumerate(aug_modes) if mode == "none"]

        return DataLoader(
            Subset(dataset, test_indices),
            batch_size=self.config.batch_size,
            shuffle=False,
        )


class AblationTrainer:
    """消融实验训练器"""

    def __init__(
        self,
        config: AblationConfig,
        model: torch.nn.Module,
        criterion: AblationHybridLoss,
        device: torch.device,
        save_dir: Path,
    ):
        self.config = config
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.history = {
            "train_loss": [],
            "val_iou": [],
            "loss_components": [],
        }
        self.best_val_iou = -float("inf")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> dict:
        """执行训练"""
        print(f"  训练样本: {len(train_loader.dataset)} | 验证样本: {len(val_loader.dataset)}")

        for epoch in range(epochs):
            # 计算密度损失权重（预热策略）
            if self.config.use_warmup and epoch < self.config.warmup_epochs:
                density_weight = 0.0
            elif self.config.use_warmup:
                progress = (epoch - self.config.warmup_epochs) / max(1, epochs - self.config.warmup_epochs)
                density_weight = min(self.config.density_weight_fake_max, progress * self.config.density_weight_fake_max)
            else:
                density_weight = self.config.density_weight_fake_max

            # 训练一个epoch
            train_loss, loss_components = self._train_epoch(train_loader, density_weight)
            self.history["train_loss"].append(train_loss)
            self.history["loss_components"].append(loss_components)

            # 验证
            val_iou = self._validate_epoch(val_loader)
            self.history["val_iou"].append(val_iou)

            # 保存最佳模型
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self._save_checkpoint("best_model.pth")

            # 日志输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"    Epoch [{epoch+1:3d}/{epochs}] "
                    f"Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}"
                )

        # 保存最终模型
        self._save_checkpoint("final_model.pth")
        self._save_history()

        return {
            "best_val_iou": self.best_val_iou,
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_iou": self.history["val_iou"][-1],
        }

    def _train_epoch(
        self, loader: DataLoader, density_weight: float
    ) -> Tuple[float, dict]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_loss_components = defaultdict(float)

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # 前向传播
            pred_prob, pred_ratio = self.model(batch)

            # 计算监督损失
            loss, loss_dict = self.criterion(pred_prob, pred_ratio, batch)

            # 真实条件的密度损失
            if self.config.use_density_loss:
                density_loss_real = self.criterion.compute_density_loss(
                    pred_ratio, batch.batch, batch.condition, batch.constraint_mask
                )
                loss = loss + self.config.density_weight_real * density_loss_real
                loss_dict["density_real"] = density_loss_real.item()

            # 双流训练：伪造条件
            if self.config.use_dual_stream and density_weight > 0:
                fake_indices = torch.randint(0, 3, (batch.num_graphs,), device=self.device)
                fake_condition = torch.nn.functional.one_hot(fake_indices, num_classes=3).float()

                _, pred_ratio_fake = self.model(batch, fake_condition)

                density_loss_fake = self.criterion.compute_density_loss(
                    pred_ratio_fake, batch.batch, fake_condition, batch.constraint_mask
                )
                loss = loss + density_weight * density_loss_fake
                loss_dict["density_fake"] = density_loss_fake.item()

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                all_loss_components[k] += v

        n_batches = len(loader)
        avg_components = {k: v / n_batches for k, v in all_loss_components.items()}

        return total_loss / n_batches, avg_components

    def _validate_epoch(self, loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_iou = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred_prob, pred_ratio = self.model(batch)

                # 计算Vector IoU
                intersection = torch.min(pred_ratio, batch.y)
                union = torch.max(pred_ratio, batch.y)
                iou = (intersection.sum(1) + 1e-6) / (union.sum(1) + 1e-6)

                total_iou += iou.mean().item()
                count += 1

        return total_iou / count

    def _save_checkpoint(self, name: str):
        """保存模型检查点"""
        torch.save(self.model.state_dict(), self.save_dir / name)

    def _save_history(self):
        """保存训练历史"""
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def run_single_ablation(
    config: AblationConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"实验: {config.name}")
    print(f"{'='*60}")

    # 数据管理器
    dm = AblationDataManager(
        config=config,
        train_dxf_dir=f"{data_config.DXF_DIR}/train",
        test_dxf_dir=f"{data_config.DXF_DIR}/test",
        cache_dir=data_config.CACHE_DIR,
    )

    # K-Fold交叉验证
    splits, dataset, metadata = dm.get_kfold_splits(n_splits=config.n_folds)

    fold_results = []
    experiment_dir = output_dir / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(config.n_folds):
        print(f"\n📂 Fold {fold + 1}/{config.n_folds}")

        # 获取数据
        train_loader, val_loader = dm.get_fold_loaders(fold, splits, dataset, metadata)

        # 创建模型
        model = create_model_from_config(
            config,
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
        )

        # 创建损失函数
        criterion = AblationHybridLoss(config)

        # 训练
        fold_dir = experiment_dir / f"fold_{fold}"
        trainer = AblationTrainer(config, model, criterion, device, fold_dir)
        result = trainer.train(train_loader, val_loader, config.epochs)

        fold_results.append(result)

    # 汇总结果
    avg_iou = np.mean([r["best_val_iou"] for r in fold_results])
    std_iou = np.std([r["best_val_iou"] for r in fold_results])

    summary = {
        "config_name": config.name,
        "n_folds": config.n_folds,
        "avg_val_iou": float(avg_iou),
        "std_val_iou": float(std_iou),
        "fold_results": fold_results,
        "config": {
            "backbone": config.backbone.value,
            "conditioning": config.conditioning.value,
            "use_dual_stream": config.use_dual_stream,
            "use_iou_loss": config.use_iou_loss,
            "use_consistency_loss": config.use_consistency_loss,
            "use_density_loss": config.use_density_loss,
            "use_augmentation": config.use_augmentation,
        },
    }

    # 保存汇总
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 实验 {config.name} 完成")
    print(f"   平均验证IoU: {avg_iou:.4f} ± {std_iou:.4f}")

    return summary


def run_ablation_study(
    experiments: List[str],
    output_dir: str,
    device: str = "cuda",
):
    """运行一组消融实验"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_name in experiments:
        set_seed(42)  # 每个实验重置种子
        config = get_ablation_config(exp_name)
        result = run_single_ablation(config, output_path, device)
        all_results[exp_name] = result

    # 保存所有结果
    with open(output_path / f"all_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(f"{'实验名称':<30} {'平均IoU':>12} {'标准差':>12}")
    print("-" * 70)

    for name, result in all_results.items():
        print(f"{name:<30} {result['avg_val_iou']:>12.4f} {result['std_val_iou']:>12.4f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="消融实验运行脚本")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["full_model"],
        help=f"要运行的实验名称列表，可选: {list_ablation_configs()}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有预定义的消融实验",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/result/ablation_study",
        help="结果保存目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备",
    )

    args = parser.parse_args()

    if args.all:
        experiments = list_ablation_configs()
    else:
        experiments = args.experiments

    print(f"将运行以下实验: {experiments}")
    run_ablation_study(experiments, args.output_dir, args.device)


if __name__ == "__main__":
    main()
