import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from train.config import data_config, model_config, training_config
from train.dataset import ShearWallDataset
from train.losses import ConsistencyLoss, HybridLoss
from train.model import ShearWallGNN
from train.utils import calculate_vector_iou, get_file_category
from train.visualize_test import predict_shear_walls, prepare_graph_data_for_inference, visualize_single_case

# 全局设置
DEVICE = torch.device(training_config.DEVICE if torch.cuda.is_available() else "cpu")
torch.manual_seed(training_config.RANDOM_SEED)
random.seed(training_config.RANDOM_SEED)
np.random.seed(training_config.RANDOM_SEED)


class DataManager:
    """简化版数据管理：直接从分好的文件夹加载数据"""

    def __init__(self, root: str):

        self.train_dxf_dir = str(Path(root) / "train")
        self.test_dxf_dir = str(Path(root) / "test")

        # 缓存路径建议分开，以免元数据冲突
        # 例如: data_cache/train 和 data_cache/test
        self.train_cache_root = str(Path(data_config.CACHE_DIR) / "train")
        self.test_cache_root = str(Path(data_config.CACHE_DIR) / "test")

    def get_train_loader(self) -> DataLoader:
        """加载训练集并应用加权采样"""
        print(f"📂 加载训练集 from: {self.train_dxf_dir}")
        dataset = ShearWallDataset(root=self.train_cache_root, dxf_dir=self.train_dxf_dir)

        # 加载元数据用于计算权重
        metadata_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(metadata_path)

        file_indices = metadata["file_indices"]
        dxf_files = metadata["dxf_files"]

        # --- 计算样本权重 (Weighted Sampling) ---
        sample_categories = []
        for idx in range(len(dataset)):
            # 找到该样本对应的文件
            f_idx = file_indices[idx]
            fname = dxf_files[f_idx]
            sample_categories.append(get_file_category(fname))

        # 计算类别权重: 1 / frequency
        cat_counts = Counter(sample_categories)
        total_samples = len(sample_categories)
        class_weights = {cat: total_samples / count for cat, count in cat_counts.items()}

        # 分配每个样本的权重
        sample_weights = [class_weights[cat] for cat in sample_categories]

        print(f"📊 训练集统计: 总样本 {total_samples} (含增广)")
        for cat, count in cat_counts.items():
            print(f"  - Class {cat}: {count} samples")

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=training_config.BATCH_SIZE,
            sampler=sampler,
            shuffle=False,  # 使用 sampler 时必须为 False
        )

    def get_test_loader(self) -> Tuple[DataLoader, Subset]:
        """加载测试集并过滤掉自动生成的增广数据"""
        print(f"📂 加载测试集 from: {self.test_dxf_dir}")
        dataset = ShearWallDataset(root=self.test_cache_root, dxf_dir=self.test_dxf_dir)

        # 加载元数据用于过滤
        metadata_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(metadata_path)
        aug_modes = metadata["aug_modes"]

        # --- 过滤逻辑: 只保留 aug_mode == 'none' ---
        test_indices = [i for i, mode in enumerate(aug_modes) if mode == "none"]

        test_subset = Subset(dataset, test_indices)

        print(f"📊 测试集统计: 原始DXF {len(test_subset)} 张 (过滤掉了增广版本)")

        return DataLoader(test_subset, batch_size=training_config.BATCH_SIZE, shuffle=False), test_subset


class Trainer:
    """简化版训练器：无验证集，只保存最终模型"""

    def __init__(
        self,
        model: ShearWallGNN,
        optimizer: torch.optim.Optimizer,
        criterion: HybridLoss,
        consistency_criterion: ConsistencyLoss,
        save_dir,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.hybrid_loss = criterion
        self.consist_loss = consistency_criterion
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": []}
        self.best_val_iou = float("-inf")

    def run(self, train_loader, val_loader, epochs):
        print(f"\n🚀 开始全量训练 (Epochs: {epochs}, 无验证集)")

        self.model.train()  # 始终保持训练模式

        for epoch in range(epochs):
            avg_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(avg_loss)

            avg_iou = self._validate_epoch(val_loader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Avg IoU: {avg_iou:.4f}"
                )
            if avg_iou > self.best_val_iou:
                self.best_val_iou = avg_iou
                self._save_model("best_model.pth")

        # 训练结束，保存最终模型
        self._save_model("final_model.pth")
        self._plot_curves()
        print(f"✅ 训练完成，模型已保存至 {self.save_dir}")

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            self.optimizer.zero_grad()
            pred_prob, pred_ratio = self.model(batch)
            loss = self.hybrid_loss(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
            loss += self.consist_loss(pred_ratio, batch.edge_index, batch.edge_attr)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _validate_epoch(self, loader):
        self.model.eval()
        total_iou = 0  # 改用 IoU 累计
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                pred_prob, pred_ratio = self.model(batch)

                # 使用之前实现的 VectorIoULoss 原理计算 IoU，或者直接算 metric
                # 这里简单用 vector 计算方式作为指标
                intersection = torch.min(pred_ratio, batch.y)
                union = torch.max(pred_ratio, batch.y)
                iou = (intersection.sum(1) + 1e-6) / (union.sum(1) + 1e-6)

                total_iou += iou.mean().item()
                count += 1

        avg_iou = total_iou / count
        return avg_iou  # 返回 IoU 而不是 Loss

    def _save_model(self, name):
        torch.save(self.model.state_dict(), self.save_dir / name)

    def _plot_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / "loss_curve.png")
        plt.close()


class Evaluator:
    """评估器：逻辑不变"""

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

    def run_test(self, test_set: Subset, output_dir=None):
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        dataset: ShearWallDataset = test_set.dataset
        meta_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(meta_path)

        print(f"\n🧪 开始测试集评估 (样本数: {len(test_set)})")
        iou_scores = []

        # 由于 test_set 是 Subset，我们需要通过 indices 访问
        for sample_idx in test_set.indices:
            file_idx = metadata["file_indices"][sample_idx]
            mode = metadata["aug_modes"][sample_idx]
            dxf_file = metadata["dxf_files"][file_idx]
            dxf_path = os.path.join(dataset.dxf_dir, dxf_file)

            # 1. 准备数据
            data_batch, builder = prepare_graph_data_for_inference(dxf_path, mode=mode)

            # 2. 模型预测
            predictions = predict_shear_walls(self.model, data_batch)
            gt_vectors = np.array(
                [node_data["sw_vector"] for node_id, node_data in builder.graph.nodes(data=True)]
            )

            # 3. 计算 IoU
            iou = calculate_vector_iou(predictions, gt_vectors)
            iou_scores.append(iou)

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

    def visulize_testset(self, test_set: Subset, output_dir: str):
        output_dir: Path = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset: ShearWallDataset = test_set.dataset
        meta_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(meta_path)

        print(f"\n🎨 开始测试集可视化 (样本数: {len(test_set)})")

        iou_scores = []
        for sample_idx in test_set.indices:
            file_idx = metadata["file_indices"][sample_idx]
            mode = metadata["aug_modes"][sample_idx]
            dxf_file = metadata["dxf_files"][file_idx]
            dxf_path = os.path.join(dataset.dxf_dir, dxf_file)

            avg_iou = visualize_single_case(
                dxf_path,
                self.model,
                save_path=output_dir / f"{Path(dxf_file).stem}_vis.png",
                mode=mode,
            )
            iou_scores.append(avg_iou)
            print(f"  - {dxf_file}: Avg IoU = {avg_iou:.4f}")
        self._save_results(iou_scores, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "visualize"])
    parser.add_argument("--ckpt", type=str, help="测试模式下的模型路径")

    args = parser.parse_args()

    # 1. 数据管理
    dm = DataManager(root=data_config.DXF_DIR)

    if args.mode == "train":
        # 获取训练 Loader
        train_loader = dm.get_train_loader()

        # 初始化模型
        model = ShearWallGNN(
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            out_dim=model_config.OUTPUT_DIM,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY
        )
        # 你的混合 Loss
        criterion = HybridLoss(cls_weight=training_config.CLS_WEIGHT, reg_weight=training_config.REG_WEIGHT)
        consistency_criterion = ConsistencyLoss(weight=0.5)

        # 训练
        trainer = Trainer(model, optimizer, criterion, consistency_criterion, data_config.SAVE_DIR)
        trainer.run(train_loader, epochs=training_config.EPOCHS)

        # 训练完顺便测一下
        print("\n🔎 正在对测试集进行最终评估...")
        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(trainer.save_dir / "final_model.pth", data_config)
        evaluator.run_test(test_set, output_dir=trainer.save_dir / "test_set_results")

    elif args.mode == "test":
        if not args.ckpt:
            raise ValueError("请提供 --ckpt")

        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(args.ckpt, data_config)
        out_dir = Path(args.ckpt).parent / "test_set_results"
        evaluator.run_test(test_set, output_dir=out_dir)

    elif args.mode == "visualize":
        if not args.ckpt:
            raise ValueError("请提供 --ckpt")

        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(args.ckpt, data_config)
        out_dir = Path(args.ckpt).parent / "test_set_results"
        evaluator.visulize_testset(test_set, output_dir=out_dir)


if __name__ == "__main__":
    main()
