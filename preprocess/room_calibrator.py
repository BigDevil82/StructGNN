"""
房间坐标校准模块

主要功能：
1. 将房间校准为标准矩形（使用 bounding box）
2. 对齐共享边：通过坐标聚类消除绘图误差
"""

from typing import List, Tuple

from shapely.geometry import Polygon

from preprocess.dxf_extractor import DXFExtractor


class RoomCalibrator:
    """房间坐标校准器"""

    def __init__(self, alignment_threshold: float = 200.0):
        """
        Args:
            alignment_threshold: 坐标对齐的阈值（小于此值的坐标将被合并）
        """
        self.alignment_threshold = alignment_threshold
        self.x_grid: List[float] = []
        self.y_grid: List[float] = []

    def _cluster_1d(self, values: List[float]) -> List[float]:
        """
        对一维坐标进行快速聚类
        逻辑：排序 -> 线性扫描 -> 平均值
        复杂度: O(N log N)
        """
        if not values:
            return []

        # 1. 排序
        v = sorted(values)

        # 2. 线性扫描聚类
        clusters = [[v[0]]]
        for x in v[1:]:
            if x - clusters[-1][-1] <= self.alignment_threshold:
                clusters[-1].append(x)
            else:
                clusters.append([x])

        # 3. 计算中心 (均值)
        centers = [sum(c) / len(c) for c in clusters]
        return centers

    def _snap_to_grid(self, val: float, grid: List[float]) -> float:
        """
        将数值吸附到最近的网格线
        """
        if not grid:
            return val

        # 简单的线性查找 (因为 Grid 数量很少，通常 < 100，二分查找都可以省了)
        # 如果追求极致性能，这里可以用 bisect，但 Python 循环在这个量级下极快
        nearest = min(grid, key=lambda g: abs(g - val))

        if abs(nearest - val) <= self.alignment_threshold:
            return nearest
        return val

    def calibrate_rooms(self, room_polys: List[Polygon]) -> List[Polygon]:
        """
        校准房间列表
        """
        if not room_polys:
            return []

        # 1. 提取所有边界坐标
        xs = []
        ys = []
        bounds_list = []  # 缓存 bounds 避免重复计算

        for room in room_polys:
            minx, miny, maxx, maxy = room.bounds
            bounds_list.append((minx, miny, maxx, maxy))
            xs.extend([minx, maxx])
            ys.extend([miny, maxy])

        # 2. 生成 X 和 Y 轴的对齐网格
        self.x_grid = self._cluster_1d(xs)
        self.y_grid = self._cluster_1d(ys)

        # 3. 将每个房间吸附到网格
        calibrated_rooms = []
        for minx, miny, maxx, maxy in bounds_list:

            # 吸附
            nx1 = self._snap_to_grid(minx, self.x_grid)
            nx2 = self._snap_to_grid(maxx, self.x_grid)
            ny1 = self._snap_to_grid(miny, self.y_grid)
            ny2 = self._snap_to_grid(maxy, self.y_grid)

            # 防御性编程：防止吸附后房间反转或塌陷
            if nx2 <= nx1:
                nx2 = nx1 + 1.0
            if ny2 <= ny1:
                ny2 = ny1 + 1.0

            # 构造新的矩形多边形
            # 顺序: 左下 -> 右下 -> 右上 -> 左上 -> 左下 (Shapely 不需要闭合点，这里给4个角即可)
            new_poly = Polygon([(nx1, ny1), (nx2, ny1), (nx2, ny2), (nx1, ny2)])
            calibrated_rooms.append(new_poly)

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
    dxf_path = r"data/dxf/to_process/room_finished/L1L28_232.dxf"
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
