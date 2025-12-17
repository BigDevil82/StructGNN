import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATv2Conv


class ShearWallGNN(nn.Module):
    def __init__(self, node_in_dim=25, edge_in_dim=8, hidden_dim=128, out_dim=32):
        super(ShearWallGNN, self).__init__()

        # 1. Input Encoder
        self.node_enc = nn.Linear(node_in_dim, hidden_dim)
        self.edge_enc = nn.Linear(edge_in_dim, hidden_dim)

        # 2. Graph Layers (Message Passing)
        # 使用 GATv2，因为它能更好地处理具有不同重要性的邻居
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # 3. Output Decoder (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            # nn.Sigmoid(),  # 输出归一化到 [0, 1]，表示剪力墙比例
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embedding
        x = F.relu(self.node_enc(x))
        edge_attr = F.relu(self.edge_enc(edge_attr))

        # Layer 1
        x_in = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x) + x_in  # Residual

        # Layer 2
        x_in = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x) + x_in

        # Layer 3
        x_in = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x) + x_in

        # Decode
        out = self.decoder(x)
        # 分割输出
        prob_logits = out[:, :16]  # 前16维：分类
        ratio_logits = out[:, 16:]  # 后16维：回归

        return torch.sigmoid(prob_logits), torch.sigmoid(ratio_logits)  # 输出概率和比例
