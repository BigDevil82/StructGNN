"""
房间坐标校准模块

主要功能：
1. 将房间校准为标准矩形（使用 bounding box）
2. 对齐共享边：通过坐标聚类消除绘图误差
"""

from typing import List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN

from preprocess.dxf_extractor import DXFExtractor


class RoomCalibrator:
    """房间坐标校准器"""

    def __init__(self, alignment_threshold: float = 1.0):
        """
        Args:
            alignment_threshold: 坐标对齐的阈值（同一簇内的坐标差异小于此值）
        """
        self.alignment_threshold = alignment_threshold

    def _extract_all_coordinates(self, room_polys: List[Polygon]) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取所有房间的 x 和 y 坐标

        Args:
            room_polys: 房间多边形列表

        Returns:
            x_coords: 所有 x 坐标的数组
            y_coords: 所有 y 坐标的数组
        """
        x_coords = []
        y_coords = []

        for room_poly in room_polys:
            minx, miny, maxx, maxy = room_poly.bounds
            x_coords.extend([minx, maxx])
            y_coords.extend([miny, maxy])

        return np.array(x_coords), np.array(y_coords)

    def _cluster_coordinates(self, coords: np.ndarray) -> dict:
        """
        对坐标进行聚类，返回每个原始坐标到校准坐标的映射

        Args:
            coords: 一维坐标数组

        Returns:
            mapping: {原始坐标: 校准坐标} 的字典
        """
        if len(coords) == 0:
            return {}

        # 去重并排序
        unique_coords = np.unique(coords)

        # 使用 DBSCAN 进行聚类
        # eps: 同一簇内点的最大距离
        # min_samples: 簇的最小样本数（设为1表示单个点也可以成簇）
        coords_2d = unique_coords.reshape(-1, 1)
        clustering = DBSCAN(eps=self.alignment_threshold, min_samples=1).fit(coords_2d)

        # 计算每个簇的平均值
        mapping = {}
        for label in np.unique(clustering.labels_):
            cluster_mask = clustering.labels_ == label
            cluster_coords = unique_coords[cluster_mask]
            aligned_coord = np.mean(cluster_coords)

            # 将簇内所有坐标映射到平均值
            for coord in cluster_coords:
                mapping[coord] = aligned_coord

        return mapping

    def _align_coordinate(self, coord: float, mapping: dict) -> float:
        """
        根据映射表对齐单个坐标

        Args:
            coord: 原始坐标
            mapping: 坐标映射表

        Returns:
            校准后的坐标
        """
        # 找到最接近的映射键
        if coord in mapping:
            return mapping[coord]

        # 如果不在映射表中，找最近的键
        keys = np.array(list(mapping.keys()))
        closest_key = keys[np.argmin(np.abs(keys - coord))]
        return mapping[closest_key]

    def calibrate_rooms(self, room_polys: List[Polygon]) -> List[Polygon]:
        """
        校准房间列表

        步骤：
        1. 将每个房间转换为其 bounding box（标准矩形）
        2. 对所有 x、y 坐标分别聚类
        3. 用聚类中心替换原坐标，实现共享边对齐

        Args:
            room_polys: 原始房间多边形列表

        Returns:
            校准后的房间多边形列表
        """
        if not room_polys:
            return []

        # 步骤1: 转换为 bounding box
        bbox_rooms = []
        for room_poly in room_polys:
            minx, miny, maxx, maxy = room_poly.bounds
            bbox = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            bbox_rooms.append(bbox)

        # 步骤2: 提取所有坐标
        x_coords, y_coords = self._extract_all_coordinates(bbox_rooms)

        # 步骤3: 对 x 和 y 坐标分别聚类
        x_mapping = self._cluster_coordinates(x_coords)
        y_mapping = self._cluster_coordinates(y_coords)

        # 步骤4: 应用映射，生成校准后的房间
        calibrated_rooms = []
        for bbox in bbox_rooms:
            minx, miny, maxx, maxy = bbox.bounds

            # 校准坐标
            minx_aligned = self._align_coordinate(minx, x_mapping)
            maxx_aligned = self._align_coordinate(maxx, x_mapping)
            miny_aligned = self._align_coordinate(miny, y_mapping)
            maxy_aligned = self._align_coordinate(maxy, y_mapping)

            # 确保 min < max（防止聚类后反转）
            if minx_aligned > maxx_aligned:
                minx_aligned, maxx_aligned = maxx_aligned, minx_aligned
            if miny_aligned > maxy_aligned:
                miny_aligned, maxy_aligned = maxy_aligned, miny_aligned

            # 创建校准后的矩形
            calibrated_bbox = Polygon(
                [
                    (minx_aligned, miny_aligned),
                    (maxx_aligned, miny_aligned),
                    (maxx_aligned, maxy_aligned),
                    (minx_aligned, maxy_aligned),
                ]
            )

            calibrated_rooms.append(calibrated_bbox)

        return calibrated_rooms


def calibrate_rooms(room_polys: List[Polygon], alignment_threshold: float = 1.0) -> List[Polygon]:
    """
    便捷函数：校准房间列表

    Args:
        room_polys: 原始房间多边形列表
        alignment_threshold: 坐标对齐阈值

    Returns:
        校准后的房间多边形列表
    """
    calibrator = RoomCalibrator(alignment_threshold=alignment_threshold)
    return calibrator.calibrate_rooms(room_polys)


# ==========================================
# 示例和测试
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon

    # 创建测试数据：3个有微小偏差的房间，共享边
    print("=" * 60)
    print("测试：房间坐标校准")
    print("=" * 60)

    # 提取DXF数据
    dxf_path = r"dxf/to_process/room_finished/L1L28_232.dxf"
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    room_polys = [Polygon([(p.x, p.y) for p in room.polygon]) for room in extractor.rooms]
    calibrated_rooms = calibrate_rooms(room_polys, 200)
    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 原始房间
    ax1.set_title("原始房间（有误差）", fontsize=14, fontweight="bold")
    colors = ["red", "green", "blue"]
    for i, room in enumerate(room_polys):
        x, y = room.exterior.xy
        ax1.plot(x, y, color=colors[i % 3], linewidth=1, label=f"房间{i+1}")
        # ax1.fill(x, y, color=colors[i%3], alpha=0.2)

    # 校准后房间
    ax2.set_title("校准后房间（对齐共享边）", fontsize=14, fontweight="bold")
    for i, room in enumerate(calibrated_rooms):
        x, y = room.exterior.xy
        ax2.plot(x, y, color=colors[i % 3], linewidth=1, label=f"房间{i+1}")
        # ax2.fill(x, y, color=colors[i%3], alpha=0.2)

    # 设置相同的坐标轴范围
    for ax in [ax1, ax2]:
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")

    plt.tight_layout()
    plt.show()
