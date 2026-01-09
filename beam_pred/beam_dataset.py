import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from beam_pred.data_aug import GeometryAugmentor
from preprocess.beam_ir_builder import Node, StructuralGraphBuilder
from preprocess.dxf_extractor import LineSegment, Room


def get_node_type(node: Node) -> int:
    # 根据 node.connected_types (在 beam_ir_builder 中已定义) 判断
    types = node.connected_types
    if "shear_wall" in types:
        return 0  # 最高优先级
    if "window" in types or "door" in types:
        return 1
    if "infill_wall" in types:
        return 2
    return 3  # 其他


def process_single_dxf(dxf_path: str, augment: bool) -> List[Data]:
    """
    单个 DXF 文件的处理函数（支持数据增广）
    返回: List[Data]
    """
    data_list = []
    # 定义增广模式
    aug_modes = ["none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"] if augment else ["none"]

    try:
        # 1. 读取原始数据 (只读一次作为基准)
        base_builder = StructuralGraphBuilder(dxf_path)
        extractor = base_builder.extractor

        # 2. 将基准数据转换为 Shapely 对象 (用于增广输入)
        raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
        raw_sw = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
        raw_infill = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]
        raw_doors = [Polygon([(p.x, p.y) for p in w]) for w in extractor.doors]
        raw_windows = [Polygon([(p.x, p.y) for p in w]) for w in extractor.windows]
        raw_beams = [
            LineString([(b.start_point.x, b.start_point.y), (b.end_point.x, b.end_point.y)])
            for b in extractor.beams
        ]

        # 3. 遍历增广模式
        for mode in aug_modes:
            try:
                # A. 应用几何变换
                (aug_rooms, aug_sw, aug_infill, aug_doors, aug_windows, aug_beams) = (
                    GeometryAugmentor.apply_augmentation(
                        raw_rooms, raw_sw, raw_infill, raw_doors, raw_windows, raw_beams, mode
                    )
                )

                # B. 创建新的 Builder 实例
                # 注意：这里重新实例化会再次读取文件，虽然略有浪费但能保证状态纯净。
                # 如果追求极致性能，可以修改 StructuralGraphBuilder 支持空初始化。
                builder = StructuralGraphBuilder(dxf_path)

                # C. 【关键】将增广后的 Shapely 对象还原为 Builder 需要的自定义对象
                # 覆盖 builder.extractor 中的数据

                # 还原房间
                builder.extractor.rooms = []
                for poly in aug_rooms:
                    # Shapely 多边形首尾相接，切片 [:-1] 去掉重复点
                    pts = [Point(x, y) for x, y in poly.exterior.coords[:-1]]
                    builder.extractor.rooms.append(Room(pts))

                # 辅助函数：还原多边形构件列表
                def polys_to_points_list(polys):
                    return [[Point(x, y) for x, y in p.exterior.coords[:-1]] for p in polys]

                builder.extractor.shear_walls = polys_to_points_list(aug_sw)
                builder.extractor.infill_walls = polys_to_points_list(aug_infill)
                builder.extractor.doors = polys_to_points_list(aug_doors)
                builder.extractor.windows = polys_to_points_list(aug_windows)

                # 还原梁
                builder.extractor.beams = []
                for line in aug_beams:
                    p1 = Point(line.coords[0][0], line.coords[0][1])
                    p2 = Point(line.coords[1][0], line.coords[1][1])
                    builder.extractor.beams.append(LineSegment(p1, p2))

                # D. 执行核心处理逻辑 (映射、构建图)
                builder.process()

                # E. 提取特征转换为 PyG Data (逻辑同之前，提取为独立函数或内联)
                data = builder_to_pyg_data(builder)  # 假设你将之前的转换逻辑封装成了这个函数
                if data:
                    data_list.append(data)

            except Exception as e_aug:
                print(f"Warning: Augmentation {mode} failed for {dxf_path}: {e_aug}")
                continue

        return data_list

    except Exception as e:
        print(f"Error processing {dxf_path}: {e}")
        return []


def builder_to_pyg_data(builder: StructuralGraphBuilder) -> Optional[Data]:
    type_map = {"shear_wall": 0, "infill_wall": 1, "window": 2, "door": 3, "empty": 4}

    # --- 节点特征 ---
    nodes = builder.node_manager.nodes
    if not nodes:
        return None

    coords = np.array([[n.x, n.y] for n in nodes], dtype=np.float32)
    # 归一化坐标
    centroid = np.mean(coords, axis=0)
    std = np.std(coords, axis=0) + 1e-6
    norm_coords = (coords - centroid) / std

    node_type_onehot = []
    for n in nodes:
        t = get_node_type(n)
        one_hot = [0] * 4
        one_hot[t] = 1
        node_type_onehot.append(one_hot)

    x = torch.cat([torch.tensor(norm_coords), torch.tensor(node_type_onehot)], dim=1)

    # --- 边特征 (输入图) ---
    edge_indices = []
    edge_attrs = []

    for room in builder.rooms:
        for edge_list in room.edges:
            for seg in edge_list:
                # 添加双向边
                edge_indices.append([seg.u, seg.v])
                edge_indices.append([seg.v, seg.u])

                # 边特征: [Type_OneHot, Normalized_Length]
                attr = [0] * len(type_map)
                if seg.type in type_map:
                    attr[type_map[seg.type]] = 1

                len_feat = seg.length / 1000.0
                feat_vec = attr + [len_feat]

                edge_attrs.append(feat_vec)
                edge_attrs.append(feat_vec)

    if not edge_indices:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # --- 监督信号 (目标边) ---
    # A. 正样本
    pos_edges = set()
    for beam in builder.beams:
        u, v = sorted((beam.u, beam.v))
        pos_edges.add((u, v))

    # B. 候选边生成
    candidate_pairs = set()
    for room in builder.rooms:
        room_node_ids = set()
        for edge_list in room.edges:
            for seg in edge_list:
                room_node_ids.add(seg.u)
                room_node_ids.add(seg.v)

        room_nodes = list(room_node_ids)
        for i in range(len(room_nodes)):
            for j in range(i + 1, len(room_nodes)):
                u, v = room_nodes[i], room_nodes[j]
                if u == v:
                    continue

                n_u = builder.node_manager.get_node(u)
                n_v = builder.node_manager.get_node(v)

                dx = abs(n_u.x - n_v.x)
                dy = abs(n_u.y - n_v.y)
                ALIGN_TOL = 50.0  # mm

                if (dx < ALIGN_TOL) or (dy < ALIGN_TOL):
                    candidate_pairs.add(tuple(sorted((u, v))))

    # 构建 Label
    labels = []
    target_indices = []
    for u, v in candidate_pairs:
        target_indices.append([u, v])
        labels.append(1.0 if (u, v) in pos_edges else 0.0)

    if not target_indices:
        return None

    y = torch.tensor(labels, dtype=torch.float)
    target_edge_index = torch.tensor(target_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, target_edge_index=target_edge_index, y=y)


# =============================================================================
# Dataset 类
# =============================================================================
class BeamDataset(InMemoryDataset):
    def __init__(
        self, root, dxf_files: Optional[List[str]] = None, transform=None, pre_transform=None, augment=True
    ):
        """
        Args:
            root: 数据缓存目录
            dxf_files: DXF 文件路径列表。
                       注意：如果 processed 文件已存在，该参数将被忽略。
                       只有在首次生成数据时需要提供。
        """
        self.dxf_files = dxf_files
        self.augment = augment
        super().__init__(root, transform, pre_transform)
        # 加载已处理的数据（会自动检查 processed_file_names 是否存在）
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["beam_data.pt"]

    def process(self):
        if not self.dxf_files:
            raise RuntimeError("Processed data not found, and 'dxf_files' was not provided.")

        print(
            f"开始并行处理 {len(self.dxf_files)} 个 DXF 文件 {'(含 6x 数据增广)' if self.augment else ''}..."
        )

        args = [(dxf_path, self.augment) for dxf_path in self.dxf_files]

        with Pool(processes=cpu_count()) as pool:
            # map 返回的是 List[List[Data]]
            nested_results = list(
                tqdm(
                    pool.starmap(process_single_dxf, args),
                    total=len(self.dxf_files),
                    desc="Processing & Augmenting" if self.augment else "Processing",
                )
            )

        # 扁平化结果: 将 [[Data1, Data2], [Data3, Data4]] 展平为 [Data1, Data2, Data3, Data4]
        data_list = []
        for sublist in nested_results:
            if sublist:  # 确保不为 None 或空列表
                data_list.extend(sublist)

        if not data_list:
            raise RuntimeError("没有成功生成任何数据，请检查 DXF 文件或处理逻辑。")

        print(f"成功处理并增广得到 {len(data_list)} 个样本，正在保存缓存...")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"数据已保存至: {self.processed_paths[0]}")
