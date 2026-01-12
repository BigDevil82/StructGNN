# ================= 配置 =================
import os
import random

import torch
from torch.utils.data import Subset, random_split

from shearwall_pred.dataset import ShearWallDataset
from shearwall_pred.model import ShearWallGNN
from shearwall_pred.trainer import visualize_test_set
from shearwall_pred.visualize_test import visualize_single_case

torch.manual_seed(42)
random.seed(42)

DXF_PATH = r"dxf/to_process/room_finished"  # 你的DXF文件夹路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "result/ckpt_1217"

MODEL_PATH = f"{SAVE_DIR}/shear_wall_predictor.pth"  # 训练好的模型路径
DXF_FILE_PATH = r"dxf/to_process/room_finished/L27_231.dxf"  # 替换为你想要测试的具体DXF文件路径

print("正在准备数据集...")
dataset = ShearWallDataset(root="data_cache", dxf_dir=DXF_PATH)

# 划分训练/验证集 (80/20)
train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# 1. 加载模型结构
# 注意：这里参数要和你训练时的一致 (node_in_dim=25)
model = ShearWallGNN().to(DEVICE)

# 2. 加载权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("模型权重已加载")
else:
    print("未找到模型文件，请检查路径")
    exit()

# 3. 运行可视化
visualize_test_set(model, dataset, test_set, f"{SAVE_DIR}/visualization")

# 也可以遍历文件夹进行批量测试
# test_dir = r"dxf/to_process/room_finished"
# for f in os.listdir(test_dir)[:5]: # 只测前5个
#     if f.endswith(".dxf"):
#         visualize_single_case(os.path.join(test_dir, f), model)
