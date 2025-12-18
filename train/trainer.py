import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train.config import data_config, model_config, training_config
from train.dataset import ShearWallDataset
from train.losses import HybridLoss
from train.model import ShearWallGNN
from train.visualize_test import visualize_single_case

# 全局设置
DEVICE = torch.device(training_config.DEVICE if torch.cuda.is_available() else "cpu")
torch.manual_seed(training_config.RANDOM_SEED)
random.seed(training_config.RANDOM_SEED)
np.random.seed(training_config.RANDOM_SEED)


class DataManager:
    """数据管理类：负责数据集加载、划分与Loader构建"""

    def __init__(self, root_dir: str, dxf_dir: str):
        self.dataset = ShearWallDataset(root=root_dir, dxf_dir=dxf_dir)
        # 加载元数据以便查阅
        metadata_path = self.dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        self.metadata = torch.load(metadata_path)

    def split_and_get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
        """
        核心逻辑：
        1. 按文件ID划分，杜绝数据泄露。
        2. 训练/验证集：包含所有增广数据。
        3. 测试集：只保留 aug_mode == 'none' 的数据。
        """
        file_indices = self.metadata["file_indices"]
        aug_modes = self.metadata["aug_modes"]

        # 1. 建立映射: file_idx -> [sample_idx1, sample_idx2, ...]
        file_to_samples = defaultdict(list)
        for sample_idx, file_idx in enumerate(file_indices):
            file_to_samples[file_idx].append(sample_idx)

        # 2. 划分文件ID
        unique_files = list(file_to_samples.keys())
        random.shuffle(unique_files)

        n_files = len(unique_files)
        train_n = int(n_files * training_config.TRAIN_RATIO)
        val_n = int(n_files * training_config.VAL_RATIO)

        train_files = unique_files[:train_n]
        val_files = unique_files[train_n : train_n + val_n]
        test_files = unique_files[train_n + val_n :]

        # 3. 展开为样本索引（核心过滤逻辑在这里）
        train_indices = self._expand_indices(train_files, file_to_samples)
        val_indices = self._expand_indices(val_files, file_to_samples)

        # 【修改点】测试集只取 aug_mode == 'none'
        test_indices = self._expand_indices(
            test_files, file_to_samples, filter_no_aug=True, aug_modes=aug_modes
        )

        print(f"📊 数据划分统计:")
        print(f"  - 训练集: {len(train_files)} 文件 -> {len(train_indices)} 样本 (含增广)")
        print(f"  - 验证集: {len(val_files)} 文件 -> {len(val_indices)} 样本 (含增广)")
        print(f"  - 测试集: {len(test_files)} 文件 -> {len(test_indices)} 样本 (仅原始)")

        # 4. 构建 Loader
        train_loader = DataLoader(
            Subset(self.dataset, train_indices), batch_size=training_config.BATCH_SIZE, shuffle=True
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
                # 只保留 mode 为 none 的样本
                samples = [s for s in samples if aug_modes[s] == "none"]
            indices.extend(samples)
        return indices


class Trainer:
    """训练管理类：负责模型训练、验证与保存"""

    def __init__(self, model: ShearWallGNN, optimizer: torch.optim.Optimizer, criterion, save_dir):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
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

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            self.optimizer.zero_grad()
            pred_prob, pred_ratio = self.model(batch)
            loss = self.criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
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
                loss = self.criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
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

    def __init__(self, model_path: str, data_config_obj):
        self.model = self._load_model(model_path)
        self.data_config = data_config_obj

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

        dataset = test_set.dataset
        # 加载必要元数据来找到原始DXF路径
        meta_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(meta_path)

        print(f"\n🧪 开始测试集评估 (样本数: {len(test_set)})")
        iou_scores = []

        for sample_idx in test_set.indices:
            # 解析元数据
            file_idx = metadata["file_indices"][sample_idx]
            mode = metadata["aug_modes"][sample_idx]
            dxf_file = metadata["dxf_files"][file_idx]
            dxf_path = os.path.join(self.data_config.DXF_DIR, dxf_file)

            # 理论上这里 mode 应该全是 none，因为我们在 DataManager 里过滤了
            # 但为了安全起见，依然处理后缀
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

        if output_dir:
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(stats, f, indent=4)


# ================= 主入口 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--ckpt", type=str, help="测试模式下的模型路径")
    args = parser.parse_args()

    # 1. 准备数据
    dm = DataManager(data_config.CACHE_DIR, data_config.DXF_DIR)
    train_loader, val_loader, test_loader, test_set = dm.split_and_get_loaders()

    if args.mode == "train":
        # 2. 初始化组件
        model = ShearWallGNN(
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            out_dim=model_config.OUTPUT_DIM,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY
        )
        criterion = HybridLoss(cls_weight=training_config.CLS_WEIGHT, reg_weight=training_config.REG_WEIGHT)

        # 3. 开始训练
        trainer = Trainer(model, optimizer, criterion, data_config.SAVE_DIR)
        trainer.run(train_loader, val_loader, training_config.EPOCHS)

        # 4. 训练后自动测试
        evaluator = Evaluator(trainer.save_dir / "final_model.pth", data_config)
        evaluator.run_test(test_set, output_dir=trainer.save_dir / "visualizations")

    elif args.mode == "test":
        if not args.ckpt:
            raise ValueError("测试模式必须提供 --ckpt 参数")

        evaluator = Evaluator(args.ckpt, data_config)
        # 默认保存到模型同级目录下的 test_results
        out_dir = Path(args.ckpt).parent / "test_results_clean"
        evaluator.run_test(test_set, output_dir=out_dir)


if __name__ == "__main__":
    main()
