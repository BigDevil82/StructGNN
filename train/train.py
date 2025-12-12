import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from train.dataset import ShearWallDataset
from train.model import ShearWallGNN

torch.manual_seed(42)
random.seed(42)
# ================= 配置参数 =================
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 300
DXF_PATH = r"dxf/to_process/room_finished"  # 你的DXF文件夹路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ================= 训练流程 =================
def main():
    # 1. 准备数据
    print("正在准备数据集...")
    dataset = ShearWallDataset(root="data_cache", dxf_dir=DXF_PATH)

    # 划分训练/验证集 (80/20)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = ShearWallGNN(node_in_dim=25, out_dim=16).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = PhysicsInformedLoss(mask_penalty_weight=5.0)  # 强惩罚

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
            out = model(batch)

            # 计算 Loss
            # 注意：dataset中需要把 constraint_mask 传递过来
            loss = criterion(out, batch.y, batch.constraint_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
            validate(model, val_loader, criterion)

    # 4. 保存模型
    torch.save(model.state_dict(), "shear_wall_predictor.pth")
    print("模型已保存！")

    # 绘制 Loss 曲线
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")
    plt.show()
    # save loss

    # test the model
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    validate(model, test_loader, criterion)


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y, batch.constraint_mask)
            val_loss += loss.item()
    print(f"   >>> Val Loss: {val_loss / len(loader):.4f}")


if __name__ == "__main__":
    main()
