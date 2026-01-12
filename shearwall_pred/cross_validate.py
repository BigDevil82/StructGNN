import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

# 复用你现有项目中的组件
from shearwall_pred.config import ModelConfig, data_config, model_config, training_config
from shearwall_pred.dataset import ShearWallDataset
from shearwall_pred.losses import ConsistencyLoss, HybridLoss
from shearwall_pred.model import ShearWallGNN
from shearwall_pred.trainer import DataManager, Evaluator, Trainer
from shearwall_pred.utils import get_file_category  # 假设你已经把 Trainer 类封装好了


class KFoldDataManager:
    """K-Fold 数据管理：负责生成交叉验证的 Loaders"""

    def __init__(self, root_dir: str, dxf_dir: str, n_splits: int = 5, seed: int = 42):
        self.dataset = ShearWallDataset(root=root_dir, dxf_dir=dxf_dir)
        self.n_splits = n_splits
        self.seed = seed

        # 加载元数据
        metadata_path = self.dataset.processed_paths[0].replace(".pt", "_metadata.pt")
        self.metadata = torch.load(metadata_path)

        # 预处理：分组文件并生成 Folds
        self.splits = self._prepare_splits()

    def _prepare_splits(self):
        """
        执行分层 K-Fold 划分 (基于文件 ID)
        """
        file_indices = self.metadata["file_indices"]
        dxf_files = self.metadata["dxf_files"]

        # 1. 整理去重的文件列表和对应的类别
        # 我们只关心 unique files，不关心样本数量（样本包含增广）
        unique_file_indices = sorted(list(set(file_indices)))
        file_categories = []

        for f_idx in unique_file_indices:
            fname = dxf_files[f_idx]
            file_categories.append(get_file_category(fname))

        # 2. 使用 sklearn 进行分层划分
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # splits 存储格式: List[ (train_file_indices, val_file_indices) ]
        splits = []
        X = np.array(unique_file_indices)
        y = np.array(file_categories)  # 用于分层的标签

        for train_idx_idx, val_idx_idx in skf.split(X, y):
            # skf 返回的是数组的索引，需要映射回 file_indices
            train_files = X[train_idx_idx].tolist()
            val_files = X[val_idx_idx].tolist()
            splits.append((train_files, val_files))

        return splits

    def get_fold_loaders(self, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
        """
        获取指定 Fold 的训练集和验证集 Loader
        """
        train_files, val_files = self.splits[fold_idx]

        # 建立 file_idx -> sample_indices 的映射
        file_to_samples = defaultdict(list)
        for sample_idx, f_idx in enumerate(self.metadata["file_indices"]):
            file_to_samples[f_idx].append(sample_idx)

        # 1. 展开样本索引
        # 训练集：包含增广数据
        train_indices = []
        for fid in train_files:
            train_indices.extend(file_to_samples[fid])

        # 验证集：【关键】严格过滤，不包含增广数据，只保留 aug_mode='none'
        # 这样验证集的表现才是真实的泛化能力
        val_indices = []
        aug_modes = self.metadata["aug_modes"]
        for fid in val_files:
            samples = file_to_samples[fid]
            val_indices.extend([s for s in samples if aug_modes[s] == "none"])

        # 2. 计算训练集加权采样权重 (Weighted Sampling)
        # 获取训练样本的类别
        train_cats = []
        for idx in train_indices:
            f_idx = self.metadata["file_indices"][idx]
            fname = self.metadata["dxf_files"][f_idx]
            train_cats.append(get_file_category(fname))

        cat_counts = Counter(train_cats)
        total_samples = len(train_cats)
        # 权重 = 总数 / 频次 (稀有类别权重高)
        weights_map = {c: total_samples / cnt for c, cnt in cat_counts.items()}
        sample_weights = [weights_map[c] for c in train_cats]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # 3. 构建 Loader
        train_loader = DataLoader(
            Subset(self.dataset, train_indices),
            batch_size=training_config.BATCH_SIZE,
            sampler=sampler,  # 使用采样器
            shuffle=False,
        )

        val_loader = DataLoader(
            Subset(self.dataset, val_indices), batch_size=training_config.BATCH_SIZE, shuffle=False
        )

        return train_loader, val_loader


class EnsembleShearWallGNN(torch.nn.Module):
    """
    集成模型包装器：内部管理多个模型，前向传播时输出平均值
    """

    def __init__(self, model_paths: List[str], model_config: ModelConfig):
        super().__init__()
        self.models = torch.nn.ModuleList()

        print(f"🔄 正在初始化集成模型，共 {len(model_paths)} 个子模型...")

        for path in model_paths:
            # 1. 初始化子模型结构
            model = ShearWallGNN(
                node_in_dim=model_config.NODE_FEATURE_DIM,
                edge_in_dim=model_config.EDGE_FEATURE_DIM,
                hidden_dim=model_config.HIDDEN_DIM,
                out_dim=model_config.OUTPUT_DIM,
            )

            # 2. 加载权重
            # map_location 确保在 CPU/GPU 间正确加载
            state_dict = torch.load(path, map_location=training_config.DEVICE)
            model.load_state_dict(state_dict)

            # 3. 设置为评估模式 (非常重要，关闭 Dropout 等)
            model.eval()
            model.to(training_config.DEVICE)

            self.models.append(model)

        print("✅ 集成模型初始化完成")

    def forward(self, data):
        """
        前向传播：运行所有子模型并取平均
        """
        # data 需要移动到设备上 (虽然 DataLoader 通常做了，但为了保险)
        # 注意：这里假设输入的 data 已经在正确的 device 上，或者子模型能处理

        prob_sum = 0
        ratio_sum = 0
        n_models = len(self.models)

        with torch.no_grad():  # 确保不计算梯度
            for model in self.models:
                # 获取子模型输出 (假设输出已经是 Sigmoid 后的 [0,1])
                prob, ratio = model(data)

                prob_sum += prob
                ratio_sum += ratio

        # 取平均值
        avg_prob = prob_sum / n_models
        avg_ratio = ratio_sum / n_models

        return avg_prob, avg_ratio


def cross_validate_train(n_folds=5, epochs=100):
    """主执行函数"""

    # 初始化数据管理器
    kfold_dm = KFoldDataManager(
        f"{data_config.CACHE_DIR}/train", f"{data_config.DXF_DIR}/train", n_splits=n_folds
    )

    # 记录每一折的最佳结果
    fold_results = []
    base_save_dir = Path(f"{data_config.SAVE_DIR}")
    base_save_dir.mkdir(exist_ok=True)

    print(f"🔥 开始 {n_folds}-Fold 交叉验证")
    print(f"========================================")

    for fold in range(n_folds):
        print(f"\n📂 FOLD {fold+1}/{n_folds}")

        # 1. 获取数据
        train_loader, val_loader = kfold_dm.get_fold_loaders(fold)
        print(f"   Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

        # 2. 初始化全新的模型 (每一折都要重置参数)
        model = ShearWallGNN(
            node_in_dim=model_config.NODE_FEATURE_DIM,
            edge_in_dim=model_config.EDGE_FEATURE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            out_dim=model_config.OUTPUT_DIM,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY
        )

        # 使用你现有的混合 Loss
        criterion = HybridLoss(cls_weight=training_config.CLS_WEIGHT, reg_weight=training_config.REG_WEIGHT)

        # 3. 运行训练
        # 为每一折创建一个单独的保存目录
        fold_save_dir = base_save_dir / f"fold_{fold}"

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            save_dir=fold_save_dir,
        )

        # 运行训练循环 (假设 trainer.run 返回 None，我们自己读取 best_loss)
        trainer.run(train_loader, val_loader, epochs=epochs)

        # 记录结果
        fold_results.append(
            {
                "fold": fold,
                "model_path": str(fold_save_dir / "final_model.pth"),
            }
        )

    # 保存结果到 JSON
    with open(base_save_dir / "cv_summary.json", "w") as f:
        json.dump(fold_results, f, indent=4)


def cross_validate_test():
    """将训练好的模型在测试集上运行，将结果保存到各fold文件夹下"""
    # 1. 准备数据
    test_set = ShearWallDataset(f"{data_config.CACHE_DIR}/test", f"{data_config.DXF_DIR}/test", is_test=True)
    # 包装为Subset兼容
    test_set = Subset(test_set, list(range(len(test_set))))

    # 1. 寻找所有 Fold 的模型路径
    cv_path = Path(data_config.SAVE_DIR)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        raise FileNotFoundError(f"在 {cv_path} 下未找到任何 fold_*/final_model.pth")
    print(f"📂 找到 {len(model_paths)} 个模型检查点:")
    for p in model_paths:
        print(f"  - {p}")

    # 2. 构建集成模型
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(training_config.DEVICE)

    # 复用 Evaluator 的 save 逻辑，但覆盖其 model
    evaluator = Evaluator(str(model_paths[0]), data_config)  # 这里的路径只是为了初始化，马上会被覆盖
    evaluator.model = ensemble_model  # 【关键】替换为集成模型

    # 设置输出目录
    output_dir = cv_path / "ensemble_test_results"

    print(f"\n🚀 开始集成模型评估...")
    evaluator.visulize_testset(test_set, output_dir=str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="运行模式")
    parser.add_argument("--folds", type=int, default=5, help="Fold 数量")
    args = parser.parse_args()

    if args.mode == "train":
        cross_validate_train(n_folds=args.folds, epochs=training_config.EPOCHS)
    else:
        cross_validate_test()
