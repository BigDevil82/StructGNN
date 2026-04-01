from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from src.shearwall_pred.config import model_config, viz_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.utils import build_graph_from_dxf


def load_ensemble_model(cv_dir: str, device: str = "cuda") -> EnsembleShearWallGNN:
    """加载K-Fold Ensemble模型"""
    cv_path = Path(cv_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        raise FileNotFoundError(f"在 {cv_path} 下未找到模型")

    print(f"加载 {len(model_paths)} 个fold模型...")
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(device)
    ensemble_model.eval()
    return ensemble_model


def predict_shear_walls(model, data_batch) -> np.ndarray:
    """使用模型预测剪力墙分布"""
    model.eval()
    with torch.no_grad():
        pred_prob, pred_ratio = model(data_batch)
        pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
        predictions = pred_combined.cpu().numpy()
        predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)
    return predictions


def prepare_case_graph(dxf_path: str, category: int | None, device: str):
    """构建案例研究所需图数据及房间几何信息。"""
    builder_graph = build_graph_from_dxf(dxf_path, mode="none")
    data = builder_graph.to_pyg_data()

    if category is not None:
        cate_one_hot = np.zeros((1, 3))
        cate_one_hot[:, category] = 1.0
        data.condition = torch.tensor(cate_one_hot, dtype=torch.float)
        print(f"  设置建筑类别: {category}")

    data_batch = Batch.from_data_list([data]).to(device)

    node_ids = list(builder_graph.graph.nodes())
    room_polys = [builder_graph.graph.nodes[node_id]["poly"] for node_id in node_ids]
    masks_list = [builder_graph.graph.nodes[node_id].get("masks", []) for node_id in node_ids]

    return builder_graph, data_batch, room_polys, masks_list
