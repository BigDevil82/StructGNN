import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train.config import data_config, model_config, training_config
from train.dataset import ShearWallDataset
from train.losses import ConsistencyLoss, HybridLoss
from train.model import ShearWallGNN
from train.utils import get_file_category
from train.visualize_test import visualize_single_case

# 全局设置
DEVICE = torch.device(training_config.DEVICE if torch.cuda.is_available() else "cpu")
torch.manual_seed(training_config.RANDOM_SEED)
random.seed(training_config.RANDOM_SEED)
np.random.seed(training_config.RANDOM_SEED)


class DataManager:
    """数据管理类：负责数据集加载、分层划分与加权采样Loader构建"""

    def __init__(self, root_dir: str, dxf_dir: str):
        self.dataset = ShearWallDataset(root=root_dir, dxf_dir=dxf_dir)
        # 加载元数据以便查阅
        metadata_path = self.dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        self.metadata = torch.load(metadata_path)

    def split_and_get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
        """
        核心逻辑升级：
        1. 分层划分 (Stratified Split)：确保 Train/Val/Test 中各类别比例一致。
        2. 加权采样 (Weighted Sampling)：在训练时对少数类进行过采样。
        """
        file_indices = self.metadata["file_indices"]
        aug_modes = self.metadata["aug_modes"]
        dxf_files = self.metadata["dxf_files"]

        # 1. 建立映射并识别类别
        file_to_samples = defaultdict(list)
        file_to_category = {}  # file_idx -> category_id

        for sample_idx, file_idx in enumerate(file_indices):
            file_to_samples[file_idx].append(sample_idx)
            # 只在第一次遇到该文件时解析类别
            if file_idx not in file_to_category:
                fname = dxf_files[file_idx]
                file_to_category[file_idx] = get_file_category(fname)

        # 2. 按类别分组文件
        files_by_category = defaultdict(list)
        for f_idx, cat in file_to_category.items():
            files_by_category[cat].append(f_idx)

        # 3. 分层划分 (Stratified Split)
        train_files, val_files, test_files = [], [], []

        print(f"📊 数据集类别分布 (文件数):")
        for cat, f_list in files_by_category.items():
            random.shuffle(f_list)
            n = len(f_list)
            n_train = int(n * training_config.TRAIN_RATIO)
            n_val = int(n * training_config.VAL_RATIO)

            # 确保每个集至少有文件 (避免除零错误)
            train_subset = f_list[:n_train]
            val_subset = f_list[n_train : n_train + n_val]
            test_subset = f_list[n_train + n_val :]

            train_files.extend(train_subset)
            val_files.extend(val_subset)
            test_files.extend(test_subset)

            print(
                f"  - Class {cat}: Total {n} -> Train {len(train_subset)} / Val {len(val_subset)} / Test {len(test_subset)}"
            )

        # 4. 展开为样本索引
        train_indices = self._expand_indices(train_files, file_to_samples)
        val_indices = self._expand_indices(val_files, file_to_samples)
        # 测试集：只取 aug_mode == 'none'
        test_indices = self._expand_indices(
            test_files, file_to_samples, filter_no_aug=True, aug_modes=aug_modes
        )

        print(f"📊 最终样本统计:")
        print(f"  - 训练集: {len(train_indices)} 样本 (含增广)")
        print(f"  - 验证集: {len(val_indices)} 样本 (含增广)")
        print(f"  - 测试集: {len(test_indices)} 样本 (仅原始)")

        # 5. 构建加权采样器 (Weighted Random Sampler) 用于训练集
        # 计算训练集中每个样本的权重
        train_sample_categories = []
        for idx in train_indices:
            # 找到该样本对应的文件，再找到类别
            f_idx = file_indices[idx]
            train_sample_categories.append(file_to_category[f_idx])

        # 计算类别权重: 1 / frequency
        cat_counts = Counter(train_sample_categories)
        total_samples = len(train_sample_categories)
        class_weights = {cat: total_samples / count for cat, count in cat_counts.items()}

        # 为每个样本分配权重
        sample_weights = [class_weights[cat] for cat in train_sample_categories]

        # 创建采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # 6. 构建 Loader
        # 注意：使用 sampler 时，shuffle 必须为 False
        train_loader = DataLoader(
            Subset(self.dataset, train_indices),
            batch_size=training_config.BATCH_SIZE,
            sampler=sampler,  # 应用加权采样
            shuffle=False,  # 互斥
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_indices), batch_size=training_config.BATCH_SIZE, shuffle=False
        )
        test_set = Subset(self.dataset, test_indices)
        test_loader = DataLoader(test_set, batch_size=training_config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader, test_set

    def _expand_indices(self, file_ids, mapping, filter_no_aug=False, aug_modes=None):
        indices = []
        for fid in file_ids:
            samples = mapping[fid]
            if filter_no_aug:
                samples = [s for s in samples if aug_modes[s] == "none"]
            indices.extend(samples)
        return indices


class Trainer:
    """训练管理类：负责模型训练、验证与保存"""

    def __init__(
        self, model: ShearWallGNN, optimizer: torch.optim.Optimizer, hybrid_loss, consist_loss, save_dir
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.hybrid_loss = hybrid_loss
        self.consist_loss = consist_loss
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def run(self, train_loader, val_loader, epochs):
        print(f"\n🚀 开始训练 (Epochs: {epochs})")

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # 打印与保存
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model("best_model.pth")

        self._save_model("final_model.pth")
        self._plot_curves()
        print(f"✅ 训练完成，最佳验证 Loss: {self.best_val_loss:.4f}")

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            self.optimizer.zero_grad()
            pred_prob, pred_ratio = self.model(batch)
            loss = self.hybrid_loss(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
            loss_consist = self.consist_loss(pred_ratio, batch.edge_index, batch.edge_attr)
            loss += loss_consist
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                pred_prob, pred_ratio = self.model(batch)
                loss = self.hybrid_loss(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _save_model(self, name):
        torch.save(self.model.state_dict(), self.save_dir / name)

    def _plot_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train")
        plt.plot(self.history["val_loss"], label="Val")
        plt.title("Loss Curve")
        plt.legend()
        plt.savefig(self.save_dir / "loss_curve.png")
        plt.close()


class Evaluator:
    """评估管理类：负责测试集评估与可视化"""

    def __init__(self, model_path: str, dxf_dir: str):
        self.model = self._load_model(model_path)
        self.dxf_dir = dxf_dir

    def _load_model(self, path):
        print(f"📥 加载模型: {path}")
        model = ShearWallGNN(
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            out_dim=model_config.OUTPUT_DIM,
        ).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model

    def run_test(self, test_set, output_dir=None):
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        dataset: ShearWallDataset = test_set.dataset if isinstance(test_set, Subset) else test_set
        # 加载必要元数据来找到原始DXF路径
        meta_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(meta_path)

        print(f"\n🧪 开始测试集评估 (样本数: {len(test_set)})")
        iou_scores = []

        # 统计每个类别的IoU
        category_ious = defaultdict(list)

        for sample_idx in test_set.indices:
            # 解析元数据
            file_idx = metadata["file_indices"][sample_idx]
            mode = metadata["aug_modes"][sample_idx]
            dxf_file = metadata["dxf_files"][file_idx]
            dxf_path = os.path.join(self.dxf_dir, dxf_file)

            # 获取类别用于统计
            # 注意：这里需要实例化 DataManager 或复用其静态逻辑，这里简化处理
            # 建议 Evaluator 接收一个 category_getter 函数
            # 此处仅打印文件名供人工核对

            save_path = None
            if output_dir:
                suffix = f"_{mode}" if mode != "none" else ""
                save_path = str(output_dir / f"{Path(dxf_file).stem}{suffix}.png")

            try:
                # 调用 visualize_test.py 中的核心函数
                iou = visualize_single_case(dxf_path, self.model, save_path=save_path, mode=mode)
                iou_scores.append(iou)
                print(f"  Process: {dxf_file} | IoU: {iou:.4f}")
            except Exception as e:
                print(f"  Error: {dxf_file} - {e}")

        self._save_results(iou_scores, output_dir)

    def _save_results(self, scores, output_dir):
        if not scores:
            print("⚠️ 无有效测试结果")
            return

        stats = {
            "avg_iou": float(np.mean(scores)),
            "max_iou": float(np.max(scores)),
            "min_iou": float(np.min(scores)),
            "count": len(scores),
        }

        print("\n📈 测试结果统计:")
        print(f"  平均 IoU: {stats['avg_iou']:.4f}")
        print(f"  最大 IoU: {stats['max_iou']:.4f}")
        print(f"  最小 IoU: {stats['min_iou']:.4f}")

        if output_dir:
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(stats, f, indent=4)


# ================= 主入口 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "cv_test"], help="运行模式"
    )
    parser.add_argument("--ckpt", type=str, help="测试模式下的模型路径")
    args = parser.parse_args()

    # 1. 准备数据
    dm = DataManager(data_config.CACHE_DIR, data_config.DXF_DIR)
    train_loader, val_loader, test_loader, test_set = dm.split_and_get_loaders()

    if args.mode == "train":
        # 2. 初始化组件
        # 确保你的 Config 里 NODE_FEATURE_DIM 已经改为 28
        model = ShearWallGNN(
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            out_dim=model_config.OUTPUT_DIM,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY
        )
        hybrid_loss = HybridLoss(cls_weight=training_config.CLS_WEIGHT, reg_weight=training_config.REG_WEIGHT)
        consist_loss = ConsistencyLoss(weight=0.3)

        # 3. 开始训练
        trainer = Trainer(model, optimizer, hybrid_loss, consist_loss, data_config.SAVE_DIR)
        trainer.run(train_loader, val_loader, training_config.EPOCHS)

        # 4. 训练后自动测试
        evaluator = Evaluator(trainer.save_dir / "final_model.pth", data_config.DXF_DIR)
        evaluator.run_test(test_set, output_dir=trainer.save_dir / "test_set_results")

    elif args.mode == "test":
        if not args.ckpt:
            raise ValueError("测试模式必须提供 --ckpt 参数")

        evaluator = Evaluator(args.ckpt, data_config.DXF_DIR)
        # 默认保存到模型同级目录下的 test_results
        out_dir = Path(args.ckpt).parent / "test_set_results"
        evaluator.run_test(test_set, output_dir=out_dir)


if __name__ == "__main__":
    main()
