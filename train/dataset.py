import os
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from dxf_extractor import DXFExtractor
from preprocess.layout_graph import LayoutGraphBuilder
from preprocess.room_analyzer import RoomAnalyzer
from preprocess.room_calibrator import calibrate_rooms


def compute_anchor_ratios(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    """计算边两端的剪力墙比例"""
    if not intervals:
        return 0.0, 0.0
    if len(intervals) == 1:
        if intervals[0][0] < 1e-3:
            return intervals[0][1], 0.0
        else:
            return 0.0, 1.0 - intervals[0][0]
    start_ratio = intervals[0][1] if intervals[0][0] < 1e-3 else 0.0
    end_ratio = 1.0 - intervals[-1][0]
    return start_ratio, end_ratio


class ShearWallDataset(InMemoryDataset):
    def __init__(self, root, dxf_dir, transform=None, pre_transform=None):
        self.dxf_dir = dxf_dir
        super(ShearWallDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.dxf_dir) if f.endswith(".dxf")]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def mask_to_vector(self, buildable_masks_list):
        """
        将 list 形式的 masks [(start_ratio, end_ratio), ...] 转换为 16维的 0/1 向量
        1.0 表示可布置区域，0.0 表示不可布置（门窗）
        """
        # 对应 4条边 x 2半边 x 2端点 (Start/End)
        # 这里简化处理：如果端点所在的半边大部分被 mask 覆盖，则标记为 0
        # 实际逻辑需根据你的 RoomAnalyzer 对齐，这里提供一个特征占位
        vector = np.ones(16, dtype=np.float32)

        buildable_ratios = []
        for intervals in buildable_masks_list:
            start_ratio, end_ratio = compute_anchor_ratios(intervals)
            buildable_ratios.extend([start_ratio, end_ratio])

        for i, ratio in enumerate(buildable_ratios):
            if ratio < 0.4:
                vector[i] = 0.0

        return vector

    def process(self):
        data_list = []
        dxf_files = self.raw_file_names

        for dxf_file in tqdm(dxf_files, desc="Processing DXFs"):
            dxf_path = os.path.join(self.dxf_dir, dxf_file)

            try:
                # ================= 你的原始 Pipeline =================
                extractor = DXFExtractor()
                extractor.extract_from_file(dxf_path)

                # 转换几何体 & 校准
                from shapely.geometry import Polygon

                raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
                sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
                infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

                if not raw_rooms:
                    continue
                calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

                # Ground Truth 分析
                analyzer = RoomAnalyzer(sw_polys, infill_polys)
                analysis_results = []
                for i, room in enumerate(raw_rooms):  # 注意：Analyzer用原始room匹配墙体
                    if room.is_valid and room.area > 1:
                        sw, ms = analyzer.process_room(room)
                        analysis_results.append({"room_index": i, "sw_vector": sw, "masks": ms})

                # 建图
                gb = LayoutGraphBuilder(calibrated_rooms)
                gb.add_analysis_results(analysis_results)
                G = gb.graph
                # =====================================================

                # NetworkX -> PyG Data
                x_list = []  # 节点特征
                y_list = []  # 标签 (剪力墙分布)
                mask_feat_list = []  # 物理约束特征

                # 建立节点索引映射
                node_mapping = {node: i for i, node in enumerate(G.nodes())}

                for node in G.nodes():
                    node_data = G.nodes[node]

                    # 1. 几何特征 (9维)
                    geo = node_data["geo_feature"]

                    # 2. 标签 (16维)
                    if node_data["sw_vector"] is None:
                        # 异常处理：如果没有标签（可能是非房间区域），填0
                        sw_vec = np.zeros(16, dtype=np.float32)
                    else:
                        sw_vec = node_data["sw_vector"]

                    # 3. 约束特征 (16维) - 非常重要！
                    # 你需要把 masks 转化为向量，告诉模型哪里是窗户
                    # 这里的逻辑需要你根据 mask 数据结构具体实现
                    # 假设我们将其处理为：1=可布置，0=有门窗不可布置
                    constraint_vec = self.mask_to_vector(node_data["masks"])

                    # 拼接输入特征: [Geo(9) + Constraint(16)] = 25维
                    x_feat = np.concatenate([geo, constraint_vec])

                    x_list.append(x_feat)
                    y_list.append(sw_vec)
                    mask_feat_list.append(constraint_vec)

                # 构建边索引和边特征
                edge_index = []
                edge_attr = []
                for u, v, edge_data in G.edges(data=True):
                    edge_index.append([node_mapping[u], node_mapping[v]])
                    edge_attr.append(edge_data["feature"])  # 7维边特征

                x = torch.tensor(np.array(x_list), dtype=torch.float)
                y = torch.tensor(np.array(y_list), dtype=torch.float)
                mask_feat = torch.tensor(np.array(mask_feat_list), dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                # 将 mask_feat 保存到 data 中，方便在 Loss 中调用
                data.constraint_mask = mask_feat

                data_list.append(data)

            except Exception as e:
                print(f"Error processing {dxf_file}: {e}")
                continue

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
