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

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from train.utils import build_graph_from_dxf


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

        # 加载元数据（文件索引映射）
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            self.file_indices = metadata["file_indices"]
            self.aug_modes = metadata["aug_modes"]
            self.dxf_files = metadata["dxf_files"]
        else:
            # 兼容旧版本缓存
            self.file_indices = None
            self.aug_modes = None
            self.dxf_files = None

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
        file_indices_list = []  # 记录每个样本对应的原始文件索引
        aug_modes_list = []  # 记录每个样本的增广模式
        dxf_files = self.raw_file_names
        aug_modes = ["none", "flip_x", "flip_y"]

        for file_idx, dxf_file in enumerate(tqdm(dxf_files, desc="Processing DXF files")):
            dxf_path = os.path.join(self.dxf_dir, dxf_file)

            for mode in aug_modes:
                try:
                    # 1. 获取 Builder
                    builder = build_graph_from_dxf(dxf_path, mode=mode)

                    # 2. 一键获取 PyG Data (逻辑已经在 Builder 内部闭环)
                    data = builder.to_pyg_data()

                    # 记录元数据
                    data_list.append(data)
                    file_indices_list.append(file_idx)
                    aug_modes_list.append(mode)

                except Exception as e:
                    print(f"Error: {e}")
                    continue

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # 保存元数据
        metadata = {"file_indices": file_indices_list, "aug_modes": aug_modes_list, "dxf_files": dxf_files}
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        torch.save(metadata, metadata_path)
        print(f"元数据已保存: {len(file_indices_list)} 个样本来自 {len(dxf_files)} 个文件")
