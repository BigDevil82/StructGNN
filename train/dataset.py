"""
剪力墙预测数据集模块

功能：
1. 从DXF文件批量构建图数据
2. 提取节点特征（几何 + 约束）和标签（剪力墙向量）
3. 构建边特征和邻接关系
4. 缓存处理后的PyTorch Geometric数据
"""

import os
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from train.config import model_config
from train.utils import build_graph_from_dxf, mask_to_constraint_vector


class ShearWallDataset(InMemoryDataset):
    """剪力墙分布预测数据集"""

    def __init__(self, root: str, dxf_dir: str, transform=None, pre_transform=None):
        """
        Args:
            root: 缓存目录路径
            dxf_dir: DXF文件目录路径
            transform: PyG数据转换函数
            pre_transform: PyG数据预转换函数
        """
        self.dxf_dir = dxf_dir
        super(ShearWallDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """返回所有原始DXF文件名"""
        return [f for f in os.listdir(self.dxf_dir) if f.endswith(".dxf")]

    @property
    def processed_file_names(self) -> List[str]:
        """返回处理后的缓存文件名"""
        return ["data.pt"]

    def process(self):
        """
        处理所有DXF文件，构建PyG数据对象

        流程：
        1. 遍历所有DXF文件
        2. 提取几何数据 → 校准 → 分析 → 构图
        3. 从图中提取节点/边特征
        4. 构建PyG Data对象
        5. 保存到缓存
        """
        data_list = []
        dxf_files = self.raw_file_names

        for dxf_file in tqdm(dxf_files, desc="Processing DXF files"):
            dxf_path = os.path.join(self.dxf_dir, dxf_file)

            try:
                # 使用工具函数构建图（封装了提取、校准、分析、构图全流程）
                graph_builder, calibrated_rooms, analysis_results = build_graph_from_dxf(dxf_path)
                G = graph_builder.graph

                # 提取节点特征和标签
                x_list = []  # 节点特征：[几何(9) + 约束(16)]
                y_list = []  # 标签：剪力墙向量(16)
                mask_list = []  # 约束掩码(16)

                node_mapping = {node: i for i, node in enumerate(G.nodes())}

                for node in G.nodes():
                    node_data = G.nodes[node]

                    # 1. 几何特征 (9维)
                    geo_feature = node_data["geo_feature"]

                    # 2. 标签 (16维)
                    sw_vector = node_data.get("sw_vector")
                    if sw_vector is None:
                        # 如果没有标签，填充零向量（异常情况）
                        sw_vector = np.zeros(model_config.CONSTRAINT_DIM, dtype=np.float32)

                    # 3. 约束特征 (16维) - 从masks转换
                    masks = node_data.get("masks", [])
                    constraint_vector = mask_to_constraint_vector(masks)

                    # 拼接输入特征: [Geo(9) + Constraint(16)] = 25维
                    x_feat = np.concatenate([geo_feature, constraint_vector])

                    x_list.append(x_feat)
                    y_list.append(sw_vector)
                    mask_list.append(constraint_vector)

                # 构建边索引和边特征
                edge_index = []
                edge_attr = []

                for u, v, edge_data in G.edges(data=True):
                    edge_index.append([node_mapping[u], node_mapping[v]])
                    edge_attr.append(edge_data["feature"])

                # 转换为Tensor
                x = torch.tensor(np.array(x_list), dtype=torch.float)
                y = torch.tensor(np.array(y_list), dtype=torch.float)
                constraint_mask = torch.tensor(np.array(mask_list), dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

                # 创建PyG Data对象
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    constraint_mask=constraint_mask,  # 保存约束掩码用于Loss计算
                )

                data_list.append(data)

            except Exception as e:
                print(f"Error processing {dxf_file}: {e}")
                continue

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
