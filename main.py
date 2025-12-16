# ================= 配置 =================
import os

import torch

from train.model import ShearWallGNN
from train.visualize_test import visualize_single_case

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "result/ckpt/shear_wall_predictor.pth"  # 训练好的模型路径
DXF_FILE_PATH = r"dxf/to_process/room_finished/L27_231.dxf"  # 替换为你想要测试的具体DXF文件路径

# 1. 加载模型结构
# 注意：这里参数要和你训练时的一致 (node_in_dim=25)
model = ShearWallGNN(node_in_dim=25, out_dim=16).to(DEVICE)

# 2. 加载权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("模型权重已加载")
else:
    print("未找到模型文件，请检查路径")
    exit()

# 3. 运行可视化
# 可以是单个文件
visualize_single_case(DXF_FILE_PATH, model)

# 也可以遍历文件夹进行批量测试
# test_dir = r"dxf/to_process/room_finished"
# for f in os.listdir(test_dir)[:5]: # 只测前5个
#     if f.endswith(".dxf"):
#         visualize_single_case(os.path.join(test_dir, f), model)
