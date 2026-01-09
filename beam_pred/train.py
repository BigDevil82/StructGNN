import glob
import os

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from beam_pred.beam_dataset import BeamDataset
from beam_pred.model import BeamPredictorGNN


def train_pipeline(dxf_files):
    # 1. 准备数据
    dataset = BeamDataset(root="data_cache/beam_dataset/train", dxf_files=dxf_files)

    # 简单的切分
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 2. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeamPredictorGNN(node_in_dim=6, edge_in_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    # --- 记录训练历史 ---
    history = {"train_loss": [], "val_precision": [], "val_recall": [], "val_f1": []}

    # 3. 训练循环
    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.target_edge_index)
            loss = criterion(pred_logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # 每个 epoch 都评估一次，以便画出平滑曲线
        metrics = evaluate(model, val_loader, device, print_log=((epoch + 1) % 10 == 0))
        history["val_precision"].append(metrics["p"])
        history["val_recall"].append(metrics["r"])
        history["val_f1"].append(metrics["f1"])

    save_dir = "beam_pred/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "beam_predictor.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存至: {save_path}")

    return model, history


def evaluate(model, loader, device, print_log=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.target_edge_index)
            preds = torch.sigmoid(logits) > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    if print_log:
        print(f"Validation -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    return {"p": p, "r": r, "f1": f1}


if __name__ == "__main__":
    # Windows 下 multiprocessing 必须放在 if __name__ == "__main__": 下
    input_dir = r"dxf/to_process/beam_finish_modified_with_rooms"
    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))

    if not dxf_files:
        print(f"Warning: No DXF files found in {input_dir}")

    # 不再需要预先构建 builders，直接把文件路径给 dataset 即可
    train_pipeline(dxf_files)
