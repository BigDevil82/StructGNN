import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from train.config import data_config, model_config, training_config
from train.dataset import ShearWallDataset
from train.losses import GlobalDensityLoss, HybridLoss
from train.model import ShearWallGNN
from train.utils import calculate_vector_iou, get_file_category
from train.visualize_test import predict_shear_walls, prepare_graph_data_for_inference, visualize_single_case

# 全局设置
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

    def get_train_val_loaders(
        self, val_ratio: float = 0.0
    ) -> Tuple[DataLoader, Optional[DataLoader], Subset, Optional[Subset]]:
        """从训练集随机切分出一部分作为验证集。

        参数:
        - val_ratio: 验证集比例，范围 [0, 1]。<= 0 时不切分。

        返回:
        - train_loader, val_loader(可能为 None), train_subset, val_subset(可能为 None)
        """
        print(f"📂 加载训练集 from: {self.train_dxf_dir}")
        dataset = ShearWallDataset(root=self.train_cache_root, dxf_dir=self.train_dxf_dir)

        # 加载元数据用于计算权重
        metadata_path = dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        metadata = torch.load(metadata_path)

        file_indices = metadata["file_indices"]
        dxf_files = metadata["dxf_files"]

        # 计算样本类别及权重
        sample_categories = []
        for idx in range(len(dataset)):
            f_idx = file_indices[idx]
            fname = dxf_files[f_idx]
            sample_categories.append(get_file_category(fname))

        cat_counts = Counter(sample_categories)
        total_samples = len(sample_categories)
        class_weights = {cat: total_samples / count for cat, count in cat_counts.items()}
        sample_weights = [class_weights[cat] for cat in sample_categories]

        print(f"📊 训练集统计: 总样本 {total_samples} (含增广)")
        for cat, count in cat_counts.items():
            print(f"  - Class {cat}: {count} samples")

        # 不切分 -> 返回全量训练 loader
        if val_ratio <= 0.0:
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            train_loader = DataLoader(
                dataset,
                batch_size=training_config.BATCH_SIZE,
                sampler=sampler,
                shuffle=False,
            )
            return train_loader, None

        # 切分训练/验证索引（随机但可复现）
        rng = random.Random(training_config.RANDOM_SEED)
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)
        val_size = max(1, int(len(dataset) * val_ratio))
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # 仅针对训练子集构建加权采样器
        train_weights = [sample_weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights), replacement=True
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=training_config.BATCH_SIZE,
            sampler=train_sampler,
            shuffle=False,
        )

        val_loader = DataLoader(val_subset, batch_size=training_config.BATCH_SIZE, shuffle=False)

        print(f"🔀 已划分验证集: 训练 {len(train_subset)} | 验证 {len(val_subset)} (ratio={val_ratio:.2f})")

        return train_loader, val_loader

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
        save_dir,
    ):
        self.model = model.to(training_config.DEVICE)
        self.optimizer = optimizer
        self.hybrid_loss = criterion
        self.density_loss = GlobalDensityLoss()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": []}
        self.best_val_iou = float("-inf")

    def run(self, train_loader, val_loader: Optional[DataLoader], epochs):
        has_val = val_loader is not None
        print(f"\n🚀 开始训练 (Epochs: {epochs}, 验证集: {'启用' if has_val else '未启用'})")

        self.model.train()  # 始终保持训练模式

        for epoch in range(epochs):
            avg_loss = self._train_epoch(train_loader, epoch)
            self.history["train_loss"].append(avg_loss)

            avg_iou = self._validate_epoch(val_loader) if has_val else None

            if (epoch + 1) % 10 == 0 or epoch == 0:
                if has_val:
                    print(
                        f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Avg IoU: {avg_iou:.4f}"
                    )
                else:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_loss:.4f}")
            if has_val and avg_iou is not None and avg_iou > self.best_val_iou:
                self.best_val_iou = avg_iou
                self._save_model("best_model.pth")

        # 训练结束，保存最终模型
        self._save_model("final_model.pth")
        self._plot_curves()
        print(f"✅ 训练完成，模型已保存至 {self.save_dir}")

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0

        if epoch < 20:
            density_weight = 0.0
        else:
            density_weight = min(0.1, 0.01 * (epoch - 20))

        for batch in loader:
            batch = batch.to(training_config.DEVICE)
            self.optimizer.zero_grad()
            pred_prob, pred_ratio = self.model(batch)
            loss_supervised = self.hybrid_loss(pred_prob, pred_ratio, batch)
            loss_density_real = self.density_loss(
                pred_ratio, batch.batch, batch.condition, batch.constraint_mask
            )

            loss_density_fake = torch.tensor(0.0).to(training_config.DEVICE)

            if density_weight > 0:
                # 生成假条件
                fake_indices = torch.randint(0, 3, (batch.num_graphs,)).to(training_config.DEVICE)
                fake_condition = torch.nn.functional.one_hot(fake_indices, num_classes=3).float()

                # --- 技巧：防止 BN 统计量被假数据污染 ---
                # 某些情况下，可以临时切换到 eval 模式跑 forward，但如果不方便
                # 只要权重够小，通常影响可控。
                # 更严谨的做法是：
                # self.model.eval()
                # _, pred_ratio_fake = self.model(batch, fake_condition)
                # self.model.train()

                _, pred_ratio_fake = self.model(batch, fake_condition)

                # 计算 Fake 的密度 Loss (同样要传入 mask!)
                loss_density_fake = self.density_loss(
                    pred_ratio_fake, batch.batch, fake_condition, batch.constraint_mask
                )

            # ==========================
            # 3. 总 Loss 与 反向传播
            # ==========================
            loss = loss_supervised + 0.05 * loss_density_real + density_weight * loss_density_fake

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
                batch = batch.to(training_config.DEVICE)
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
        ).to(training_config.DEVICE)
        model.load_state_dict(torch.load(path, map_location=training_config.DEVICE))
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
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="从训练集切分出的验证集比例 (0 表示不切分)",
    )

    args = parser.parse_args()

    # 1. 数据管理
    dm = DataManager(root=data_config.DXF_DIR)

    if args.mode == "train":
        # 获取训练/验证 Loader（按需切分）
        train_loader, val_loader = dm.get_train_val_loaders(val_ratio=0.1)

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
        criterion = HybridLoss(
            cls_weight=training_config.CLS_WEIGHT,
            reg_weight=training_config.REG_WEIGHT,
            consistency_weight=training_config.CONSISTENCY_WEIGHT,
        )

        # 训练
        trainer = Trainer(model, optimizer, criterion, data_config.SAVE_DIR)
        trainer.run(train_loader, val_loader, epochs=training_config.EPOCHS)

        # 训练完顺便测一下
        print("\n🔎 正在对测试集进行最终评估...")
        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(trainer.save_dir / "final_model.pth", data_config)
        evaluator.run_test(test_set, output_dir=trainer.save_dir / "test_set_results")

    elif args.mode == "test":
        if not args.ckpt:
            ckpt = data_config.SAVE_DIR + "/best_model.pth"
        else:
            ckpt = args.ckpt

        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(ckpt, data_config)
        out_dir = Path(ckpt).parent / "test_set_results"
        evaluator.run_test(test_set, output_dir=out_dir)

    elif args.mode == "visualize":
        if not args.ckpt:
            ckpt = data_config.SAVE_DIR + "/final_model.pth"
        else:
            ckpt = args.ckpt

        test_loader, test_set = dm.get_test_loader()
        evaluator = Evaluator(ckpt, data_config)
        out_dir = Path(ckpt).parent / "test_set_results"
        evaluator.visulize_testset(test_set, output_dir=out_dir)


if __name__ == "__main__":
    main()
