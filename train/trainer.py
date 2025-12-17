import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

from train.dataset import ShearWallDataset
from train.model import ShearWallGNN
from train.visualize_test import visualize_single_case

torch.manual_seed(42)
random.seed(42)
# ================= 配置参数 =================
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100
DXF_PATH = r"dxf/to_process/room_finished"  # 你的DXF文件夹路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "result/ckpt_1217"
os.makedirs(SAVE_DIR, exist_ok=True)


# ================= 自定义 Loss =================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, mask_penalty_weight=2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = mask_penalty_weight

    def forward(self, pred, target, constraint_mask):
        """
        pred: 模型预测 (N, 16)
        target: 真实标签 (N, 16)
        constraint_mask: 可布置区域掩码 (N, 16). 1=可布置, 0=不可布置
        """
        # 1. 基础回归 Loss (MSE)
        basic_loss = self.mse(pred, target)

        # 2. 物理约束 Loss (Mask Penalty)
        # 如果 constraint_mask 为 0 (不能布置)，但 pred > 0，则惩罚
        # logic: 惩罚项 = pred * (1 - mask)
        # 如果 mask=1 (可布置), (1-mask)=0, 惩罚为0
        # 如果 mask=0 (不可布置), (1-mask)=1, 惩罚为 pred 的值
        violation = pred * (1 - constraint_mask)
        penalty_loss = torch.mean(violation**2)

        return basic_loss + self.penalty_weight * penalty_loss


class HybridLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=2.0):
        super().__init__()
        self.bce = nn.BCELoss()  # 二分类交叉熵
        self.mse = nn.MSELoss(reduction="none")  # 回归 Loss
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, pred_prob, pred_ratio, target_ratio, mask):
        # 1. 生成分类标签 (Target Classification)
        # 如果真实长度 > 0，则分类标签为 1，否则为 0
        target_cls = (target_ratio > 0.01).float()

        # 2. 分类 Loss (BCE)
        # 即使 mask=0 (不可布置)，如果 GT 没有墙，模型也该预测 0，所以这里是否应用 mask 看情况
        # 建议：仅在 mask=1 的区域计算 Loss，或者全局计算
        cls_loss = self.bce(pred_prob, target_cls)

        # 3. 回归 Loss (只在有墙的地方计算！)
        # 关键点：如果 GT 没墙，不要惩罚回归头预测了什么，防止干扰
        reg_loss_all = self.mse(pred_ratio, target_ratio)

        # 掩码1：只计算 ground truth 中有墙的部分 (正样本)
        pos_mask = target_cls
        # 掩码2：同时也得是物理可布置区域 (双重保险)
        valid_mask = pos_mask * mask

        # 避免除以 0
        num_pos = valid_mask.sum() + 1e-6
        reg_loss = (reg_loss_all * valid_mask).sum() / num_pos

        return self.cls_weight * cls_loss + self.reg_weight * reg_loss


# ================= 训练流程 =================
def main():
    # 1. 准备数据
    print("正在准备数据集...")
    dataset = ShearWallDataset(root="data_cache", dxf_dir=DXF_PATH)

    # 划分训练/验证集 (80/20)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = ShearWallGNN(node_in_dim=25).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # criterion = PhysicsInformedLoss(mask_penalty_weight=5.0)  # 强惩罚
    criterion = HybridLoss(cls_weight=1.0, reg_weight=2.0)

    # 3. 训练循环
    print("开始训练...")
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            # 前向传播
            pred_prob, pred_ratio = model(batch)

            # 计算 Loss
            # 注意：dataset中需要把 constraint_mask 传递过来
            loss = criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
            validate(model, val_loader, criterion)

    # 4. 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "shear_wall_predictor.pth"))
    print("模型已保存！")

    # 绘制 Loss 曲线
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"))
    plt.show()
    # save loss

    # test the model
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    validate(model, test_loader, criterion)

    # 运行并保存测试集上的可视化结果
    print("\n正在可视化测试集结果...")
    visualize_test_set(model, dataset, test_set)


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred_prob, pred_ratio = out
            loss = criterion(pred_prob, pred_ratio, batch.y, batch.constraint_mask)
            val_loss += loss.item()
    print(f"   >>> Val Loss: {val_loss / len(loader):.4f}")
    return out


def visualize_test_set(model, dataset: ShearWallDataset, test_set: Subset, save_dir):
    """
    在测试集上运行可视化并保存结果
    """
    import os
    from pathlib import Path

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试集的原始索引
    test_indices = test_set.indices

    # 获取原始 DXF 文件列表
    dxf_files = [f for f in os.listdir(dataset.dxf_dir) if f.endswith(".dxf")]

    print(f"测试集共 {len(test_indices)} 个样本")

    # 遍历测试集
    IoU_scores = []
    for idx in test_indices:
        if idx >= len(dxf_files):
            print(f"警告：索引 {idx} 超出范围")
            continue

        dxf_file = dxf_files[idx]
        dxf_path = os.path.join(dataset.dxf_dir, dxf_file)

        # 生成保存路径
        save_path = save_dir / f"{Path(dxf_file).stem}.png"

        # try:
        IoU = visualize_single_case(dxf_path, model, save_path=str(save_path))
        IoU_scores.append(IoU)
        # except Exception as e:
        #     print(f"可视化 {dxf_file} 时出错: {e}")
        #     continue

    print(f"\n所有可视化结果已保存到: {save_dir}")
    if IoU_scores:
        avg_IoU = sum(IoU_scores) / len(IoU_scores)
        print(f"平均 IoU 分数: {avg_IoU:.4f}")


if __name__ == "__main__":
    main()
