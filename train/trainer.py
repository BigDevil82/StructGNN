"""
训练脚本

功能：
1. 加载数据集并划分训练/验证/测试集
2. 初始化模型和优化器
3. 训练循环（带验证）
4. 保存模型和训练曲线
5. 测试集可视化评估
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train.config import data_config, model_config, training_config
from train.dataset import ShearWallDataset
from train.losses import HybridLoss
from train.model import ShearWallGNN
from train.visualize_test import visualize_single_case

# 设置随机种子以确保可复现性
torch.manual_seed(training_config.RANDOM_SEED)
random.seed(training_config.RANDOM_SEED)

# 配置设备
DEVICE = torch.device(training_config.DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(data_config.SAVE_DIR, exist_ok=True)


# ================= 辅助函数 =================
def prepare_data() -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
    """
    准备训练、验证和测试数据加载器

    Returns:
        (train_loader, val_loader, test_loader, test_set)
    """
    print("正在加载数据集...")
    dataset = ShearWallDataset(root=data_config.CACHE_DIR, dxf_dir=data_config.DXF_DIR)

    print(f"数据集大小: {len(dataset)} 个样本")

    # 划分数据集
    train_size = int(len(dataset) * training_config.TRAIN_RATIO)
    val_size = int(len(dataset) * training_config.VAL_RATIO)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=training_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=training_config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=training_config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_set


def train_epoch(
    model: ShearWallGNN, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: HybridLoss
) -> float:
    """
    训练一个epoch

    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数

    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch: Data = batch.to(DEVICE)
        optimizer.zero_grad()

        # 前向传播
        pred_prob, pred_ratio = model(batch)

        # 计算损失
        loss: torch.Tensor = criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model: ShearWallGNN, loader: DataLoader, criterion: HybridLoss) -> float:
    """
    在验证/测试集上评估模型

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数

    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch: Data = batch.to(DEVICE)
            pred_prob, pred_ratio = model(batch)
            loss: torch.Tensor = criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
            total_loss += loss.item()

    return total_loss / len(loader)


def save_training_curve(loss_history: List[float], save_path: str):
    """
    保存训练损失曲线

    Args:
        loss_history: 损失历史记录
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, linewidth=2)
    plt.title("Training Loss Curve", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def visualize_test_set(model, test_set: Subset, save_dir: str):
    """
    在测试集上运行可视化并保存结果

    Args:
        model: 训练好的模型
        test_set: 测试集子集
        save_dir: 可视化结果保存目录
    """
    # 创建保存目录
    save_dir: Path = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试集的原始索引
    test_indices = test_set.indices

    dxf_files = sorted([f for f in os.listdir(data_config.DXF_DIR) if f.endswith(".dxf")])

    print(f"测试集样本数: {len(test_indices)}")
    print("=" * 60)

    # 遍历测试集并可视化
    iou_scores = []
    for idx in test_indices:
        if idx >= len(dxf_files):
            print(f"⚠️  警告：索引 {idx} 超出范围，跳过")
            continue

        dxf_file = dxf_files[idx]
        dxf_path = os.path.join(data_config.DXF_DIR, dxf_file)

        # 生成保存路径
        save_path = save_dir / f"{Path(dxf_file).stem}.png"

        try:
            iou = visualize_single_case(dxf_path, model, save_path=str(save_path))
            iou_scores.append(iou)
        except Exception as e:
            print(f"❌ 可视化 {dxf_file} 时出错: {e}")
            continue

    # 统计结果
    print("=" * 60)
    print(f"可视化完成！结果保存在: {save_dir}")
    if iou_scores:
        avg_iou = sum(iou_scores) / len(iou_scores)
        max_iou = max(iou_scores)
        min_iou = min(iou_scores)
        print(f"\nIoU 统计:")
        print(f"  - 平均值: {avg_iou:.4f}")
        print(f"  - 最大值: {max_iou:.4f}")
        print(f"  - 最小值: {min_iou:.4f}")
    else:
        print("⚠️  警告：没有成功生成任何可视化结果")


def evaluate_model_on_test_set(
    model_path: str,
    data_dir: str = None,
    save_visualizations: bool = True,
    output_dir: str = None,
) -> dict:
    """
    加载指定模型并在测试集上运行评估

    Args:
        model_path: 模型权重文件路径（.pth文件）
        data_dir: DXF数据目录路径（可选，默认使用配置文件中的路径）
        save_visualizations: 是否保存可视化结果
        output_dir: 可视化结果保存目录（可选，默认使用模型路径同级目录）
    """
    print("=" * 60)
    print("开始测试集评估")
    print("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"模型路径: {model_path}")

    # 设置数据目录
    if data_dir is not None:
        original_dxf_dir = data_config.DXF_DIR
        data_config.DXF_DIR = data_dir
        print(f"数据目录: {data_dir}")
    else:
        print(f"数据目录: {data_config.DXF_DIR}")

    # 设置输出目录
    if output_dir is None:
        model_dir = Path(model_path).parent
        output_dir = model_dir / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据集
    print("正在加载数据集...")
    dataset = ShearWallDataset(root=data_config.CACHE_DIR, dxf_dir=data_config.DXF_DIR)

    # 划分数据集（使用相同的随机种子确保一致性）
    train_size = int(len(dataset) * training_config.TRAIN_RATIO)
    val_size = int(len(dataset) * training_config.VAL_RATIO)
    test_size = len(dataset) - train_size - val_size

    _, _, test_set = random_split(dataset, [train_size, val_size, test_size])

    print(f"测试集样本数: {len(test_set)}")

    # 2. 初始化模型
    print("\n正在初始化模型...")
    model = ShearWallGNN(
        node_in_dim=model_config.NODE_FEATURE_DIM,
        edge_in_dim=model_config.EDGE_FEATURE_DIM,
        hidden_dim=model_config.HIDDEN_DIM,
        out_dim=model_config.OUTPUT_DIM,
    ).to(DEVICE)

    # 3. 加载模型权重
    print("正在加载模型权重...")
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功")
    except Exception as e:
        raise RuntimeError(f"加载模型权重失败: {e}")

    # 4. 可视化结果
    iou_scores = []
    if save_visualizations:
        print("\n" + "=" * 60)
        print("生成可视化结果...")
        print("=" * 60)

        # 获取测试集索引和DXF文件
        test_indices = test_set.indices
        dxf_files = sorted([f for f in os.listdir(data_config.DXF_DIR) if f.endswith(".dxf")])

        print(f"正在处理 {len(test_indices)} 个测试样本...")

        for i, idx in enumerate(test_indices, 1):
            if idx >= len(dxf_files):
                print(f"  [{i}/{len(test_indices)}] ⚠️  索引 {idx} 超出范围，跳过")
                continue

            dxf_file = dxf_files[idx]
            dxf_path = os.path.join(data_config.DXF_DIR, dxf_file)
            save_path = output_dir / f"{Path(dxf_file).stem}.png"

            try:
                iou = visualize_single_case(dxf_path, model, save_path=str(save_path))
                iou_scores.append(iou)
                print(f"  [{i}/{len(test_indices)}] ✅ {dxf_file}: IoU={iou:.4f}")
            except Exception as e:
                print(f"  [{i}/{len(test_indices)}] ❌ {dxf_file}: {e}")
                continue

    # 5. 统计结果
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    results = {
        "model_path": model_path,
        "num_samples": len(test_set),
        "iou_scores": iou_scores,
    }

    if iou_scores:
        avg_iou = sum(iou_scores) / len(iou_scores)
        max_iou = max(iou_scores)
        min_iou = min(iou_scores)

        results.update(
            {
                "avg_iou": avg_iou,
                "max_iou": max_iou,
                "min_iou": min_iou,
            }
        )

        print(f"\n📊 评估结果:")
        print(f"  - 平均 IoU: {avg_iou:.4f}")
        print(f"  - 最大 IoU: {max_iou:.4f}")
        print(f"  - 最小 IoU: {min_iou:.4f}")
        print(f"  - 成功评估样本数: {len(iou_scores)}/{len(test_set)}")

        if save_visualizations:
            print(f"\n💾 可视化结果已保存到: {output_dir}")
    else:
        print("⚠️  警告：未能计算IoU分数")

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        # 转换为可序列化的格式
        json_results = {
            "model_path": str(results["model_path"]),
            "num_samples": int(results["num_samples"]),
            "avg_iou": float(results.get("avg_iou", 0)),
            "max_iou": float(results.get("max_iou", 0)),
            "min_iou": float(results.get("min_iou", 0)),
            "num_evaluated": len(iou_scores),
        }
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"📄 评估结果已保存到: {results_file}")

    # 恢复原始数据目录配置
    if data_dir is not None:
        data_config.DXF_DIR = original_dxf_dir

    return results


# ================= 主训练流程 =================
def train():
    """
    主训练流程

    步骤：
    1. 准备数据
    2. 初始化模型、优化器和损失函数
    3. 训练循环
    4. 保存模型和训练曲线
    5. 测试集评估和可视化
    """
    # 1. 准备数据
    train_loader, val_loader, test_loader, test_set = prepare_data()

    # 2. 初始化模型
    print("\n初始化模型...")
    model = ShearWallGNN(
        node_in_dim=model_config.NODE_FEATURE_DIM,
        edge_in_dim=model_config.EDGE_FEATURE_DIM,
        hidden_dim=model_config.HIDDEN_DIM,
        out_dim=model_config.OUTPUT_DIM,
    ).to(DEVICE)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY
    )

    # 损失函数
    criterion = HybridLoss(cls_weight=training_config.CLS_WEIGHT, reg_weight=training_config.REG_WEIGHT)

    # 3. 训练循环
    print(f"\n开始训练 (共{training_config.EPOCHS}个epoch)...")
    print("=" * 60)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")

    for epoch in range(training_config.EPOCHS):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_loss_history.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion)
        val_loss_history.append(val_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{training_config.EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(data_config.SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

    print("=" * 60)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")

    # 4. 保存最终模型和训练曲线
    final_model_path = os.path.join(data_config.SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

    loss_curve_path = os.path.join(data_config.SAVE_DIR, "training_loss.png")
    save_training_curve(train_loss_history, loss_curve_path)

    # 5. 测试集评估
    print("\n在测试集上评估...")
    test_loss = validate(model, test_loader, criterion)
    print(f"测试集损失: {test_loss:.4f}")

    # 6. 测试集可视化
    print("\n生成测试集可视化结果...")
    viz_save_dir = os.path.join(data_config.SAVE_DIR, "visualizations")
    visualize_test_set(model, test_set, viz_save_dir)

    print("\n所有任务完成！")


if __name__ == "__main__":
    # 默认执行训练
    # train()

    evaluate_model_on_test_set(model_path="result/ckpt_1217/final_model.pth", save_visualizations=True)
