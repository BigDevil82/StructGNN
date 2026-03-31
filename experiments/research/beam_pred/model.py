import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool


class BeamPredictorGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super(BeamPredictorGNN, self).__init__()

        # 1. Graph Encoder (GNN)
        # GINEConv 适合处理带有边特征的图
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_in_dim,
        )
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_in_dim,
        )
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_in_dim,
        )

        # 2. Link Predictor (Decoder)
        # 输入: Concat(Node_u, Node_v, Geometric_Feats)
        # Geometric_Feats: 两个节点之间的相对距离信息，帮助判断长梁/短梁
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, 64),  # +2 是因为要把 dx, dy 作为显式特征输入
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 输出 Logits
        )

    def forward(self, x, edge_index, edge_attr, target_edge_index):
        # x: [Num_Nodes, 2] (Coords)
        # edge_attr: [Num_Edges, 6] (Type + Length)

        # --- Encoder Phase ---
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv3(h, edge_index, edge_attr)  # [Num_Nodes, Hidden_Dim]

        # --- Decoder Phase ---
        # 提取需要预测的边的节点特征
        row, col = target_edge_index
        h_src = h[row]
        h_dst = h[col]

        # 计算相对几何特征 (Explicit Geometric Features)
        # 原始坐标 x 还在显存里，可以直接用
        pos_src = x[row][:, :2]
        pos_dst = x[col][:, :2]
        # 计算 abs(dx), abs(dy)
        geo_feat = torch.abs(pos_src - pos_dst)

        # 拼接特征
        link_feat = torch.cat([h_src, h_dst, geo_feat], dim=1)

        # 预测
        return self.predictor(link_feat).squeeze()
