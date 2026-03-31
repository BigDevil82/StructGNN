import os
import random

import torch
from torch.utils.data import random_split

from src.shearwall_pred.dataset import ShearWallDataset
from src.shearwall_pred.model import ShearWallGNN
from src.shearwall_pred.trainer import visualize_test_set


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)

    dxf_path = r"data/dxf/to_process/room_finished"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "outputs/result/ckpt_1217"
    model_path = f"{save_dir}/shear_wall_predictor.pth"

    print("正在准备数据集...")
    dataset = ShearWallDataset(root="data/cache", dxf_dir=dxf_path)

    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size])

    model = ShearWallGNN().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重已加载")
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    visualize_test_set(model, dataset, test_set, f"{save_dir}/visualization")


if __name__ == "__main__":
    main()
