#!/usr/bin/env python3
"""
从DXF文件中提取剪力墙和梁的构件信息
"""

import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import ezdxf
import matplotlib.pyplot as plt
import numpy as np

Polygon = List["Point"]


class Point:
    """点坐标"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_dict(self):
        return {"X": self.x, "Y": self.y}


class Wall:
    """墙体（多边形）"""

    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def to_dict(self):
        return [point.to_dict() for point in self.polygon]


class LineSegment:
    """线段"""

    def __init__(self, start_point: Point, end_point: Point):
        self.start_point = start_point
        self.end_point = end_point

    def to_dict(self):
        return {"StartPoint": self.start_point.to_dict(), "EndPoint": self.end_point.to_dict()}


class Room:
    """房间分区"""

    def __init__(self, polygon: List[Point]):
        self.polygon = polygon

    def to_dict(self):
        return {
            "Polygon": [point.to_dict() for point in self.polygon],
        }


class DXFExtractor:
    """DXF构件信息提取器"""

    def __init__(self):
        self.shear_walls: List[Polygon] = []
        self.beams: List[LineSegment] = []
        self.doors: List[Polygon] = []
        self.windows: List[Polygon] = []
        self.infill_walls: List[Polygon] = []
        self.rooms: List[Room] = []

    def extract_from_file(self, dxf_path: str) -> Dict:
        """
        从DXF文件提取构件信息

        Args:
            dxf_path: DXF文件路径

        Returns:
            包含walls和beams的字典
        """
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            self.shear_walls: List[Polygon] = []
            self.beams: List[LineSegment] = []
            self.doors: List[Polygon] = []
            self.windows: List[Polygon] = []
            self.infill_walls: List[Polygon] = []
            self.rooms: List[Room] = []

            # 遍历所有实体
            for entity in msp:
                self._process_entity(entity)

            return {
                "shear_walls": [wall for wall in self.shear_walls],
                "beams": [beam.to_dict() for beam in self.beams],
                "doors": [door for door in self.doors],
                "windows": [window for window in self.windows],
                "infill_walls": [infill_wall for infill_wall in self.infill_walls],
                "rooms": [room.to_dict() for room in self.rooms],
            }

        except Exception as e:
            print(f"读取DXF文件失败 {dxf_path}: {e}")
            return {
                "shear_walls": [],
                "beams": [],
                "doors": [],
                "windows": [],
                "infill_walls": [],
                "rooms": [],
            }

    def _process_entity(self, entity):
        """处理单个实体"""
        entity_type = entity.dxftype()
        layer_name = entity.dxf.layer

        # 图层到构件列表的映射
        layer_mapping = {
            "SHEAR_WALLS": self.shear_walls,
            "DOORS": self.doors,
            "WINDOWS": self.windows,
            "INFILL_WALLS": self.infill_walls,
        }

        # 处理多边形类型的构件（剪力墙、门、窗、填充墙）
        if layer_name in layer_mapping:
            polygon = self._extract_geometry(entity)
            if polygon:
                layer_mapping[layer_name].append(polygon)

        # 处理梁 - 拆分为线段
        elif layer_name == "BEAMS":
            segments = self._extract_line_segments(entity)
            if segments:
                self.beams.extend(segments)

        # 处理房间
        elif layer_name == "ROOM":
            polygon = self._extract_geometry(entity)
            if polygon:
                self.rooms.append(Room(polygon))

    def _extract_points_from_entity(self, entity) -> List[Tuple[float, float]]:
        """从任意实体（LINE, LWPOLYLINE, POLYLINE）提取点坐标列表

        Args:
            entity: DXF实体

        Returns:
            点坐标列表 [(x1, y1), (x2, y2), ...]
        """
        entity_type = entity.dxftype()
        points: List[Tuple[float, float]] = []

        try:
            if entity_type == "LINE":
                # 直线：起点和终点
                start = entity.dxf.start
                end = entity.dxf.end
                points = [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]

            elif entity_type == "LWPOLYLINE":
                # 轻量级多段线
                for p in entity.get_points():
                    x, y = p[0], p[1]
                    points.append((float(x), float(y)))

            elif entity_type == "POLYLINE":
                # 经典多段线
                for v in entity.vertices:
                    loc = getattr(v.dxf, "location", None)
                    if loc is not None:
                        points.append((float(loc.x), float(loc.y)))
                    else:
                        x = getattr(v.dxf, "x", None)
                        y = getattr(v.dxf, "y", None)
                        if x is not None and y is not None:
                            points.append((float(x), float(y)))

        except Exception as e:
            print(f"提取点坐标失败 ({entity_type}): {e}")
            return []

        return points

    def _extract_geometry(self, entity) -> Polygon:
        """提取几何图形为多边形（点列表）

        支持 LINE, LWPOLYLINE, POLYLINE

        Args:
            entity: DXF实体

        Returns:
            多边形点列表
        """
        points = self._extract_points_from_entity(entity)
        if not points:
            return []

        return [Point(x, y) for x, y in points]

    def _extract_line_segments(self, entity) -> List[LineSegment]:
        """将实体拆分为线段列表

        - LINE: 返回单个线段
        - LWPOLYLINE/POLYLINE: 将顶点按顺序拆分为多条线段

        Args:
            entity: DXF实体

        Returns:
            线段列表
        """
        points = self._extract_points_from_entity(entity)
        if len(points) < 2:
            return []

        segments: List[LineSegment] = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            segments.append(LineSegment(Point(x1, y1), Point(x2, y2)))

        return segments

    def _calculate_distance(self, p1: Point, p2: Point) -> float:
        """计算两点距离"""
        return np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    def extract_batch(self, dxf_dir: str, output_dir: str = "extracted_data"):
        """
        批量提取DXF文件

        Args:
            dxf_dir: DXF文件目录
            output_dir: 输出JSON文件目录
        """
        dxf_dir = Path(dxf_dir)
        output_dir: Path = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        dxf_files = list(dxf_dir.glob("*.dxf"))

        if not dxf_files:
            print(f"在目录 {dxf_dir} 中未找到DXF文件")
            return

        print(f"🔍 找到 {len(dxf_files)} 个DXF文件")

        results = {}

        for dxf_file in dxf_files:
            print(f"处理: {dxf_file.name}")

            # 提取构件信息
            data = self.extract_from_file(str(dxf_file))

            # 保存为JSON
            json_file = output_dir / f"{dxf_file.stem}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                # Re-serializing Point objects for JSON output
                json_data = {
                    "shear_walls": [[p.to_dict() for p in wall] for wall in data["shear_walls"]],
                    "beams": data["beams"],
                    "doors": [[p.to_dict() for p in door] for door in data["doors"]],
                    "windows": [[p.to_dict() for p in window] for window in data["windows"]],
                    "infill_walls": [[p.to_dict() for p in wall] for wall in data["infill_walls"]],
                    "rooms": data["rooms"],
                }
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            results[dxf_file.name] = {
                "shear_walls_count": len(data["shear_walls"]),
                "beams_count": len(data["beams"]),
                "json_file": str(json_file),
            }

        # 输出统计信息
        print(f"\n📊 提取结果统计:")
        total_walls = sum(r["shear_walls_count"] for r in results.values())
        total_beams = sum(r["beams_count"] for r in results.values())

        print(f"   总剪力墙: {total_walls}")
        print(f"   总梁线段: {total_beams}")
        print(f"   输出目录: {output_dir}")

        for filename, stats in results.items():
            print(f"   {filename}: {stats['shear_walls_count']} 剪力墙, {stats['beams_count']} 梁")

        return results

    def plot_structure(self, data: Dict = None, save_path: str = None):
        """
        绘制提取的结构线段

        Args:
            json_path: JSON文件路径
            data: 直接传入的数据字典
            save_path: 保存图片路径
        """
        color = {
            "shear_walls": "gray",
            # "beams": "orange",
            # "rooms": "pink",
            "doors": "blue",
            "windows": "green",
            "infill_walls": "gray",
        }

        # 创建绘图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 根据数据结构分别绘制
        for name, items in data.items():
            if name not in color:
                continue
            c = color.get(name, "k")

            if name == "beams":
                # beams 为字典列表，包含 StartPoint/EndPoint
                for i, seg in enumerate(items):
                    start = seg["StartPoint"]
                    end = seg["EndPoint"]
                    ax.plot(
                        [start["X"], end["X"]],
                        [start["Y"], end["Y"]],
                        color=c,
                        linewidth=5,
                        label=name if i == 0 else "",
                    )
            elif name == "rooms":
                for room in items:
                    polygon = room["Polygon"]
                    x_coords = [point["X"] for point in polygon] + [polygon[0]["X"]]
                    y_coords = [point["Y"] for point in polygon] + [polygon[0]["Y"]]
                    ax.plot(
                        x_coords,
                        y_coords,
                        color=c,
                        linewidth=3,
                        alpha=0.8,
                    )
            else:
                # 其它（shear_walls/doors/windows/infill_walls）为多边形列表
                # 在 extract_from_file 中，这些是 Point 对象列表；
                # 在批量导出的 JSON 中，这些会是点字典列表。
                for i, polygon in enumerate(items):
                    # 兼容 Point 对象或字典
                    def get_xy(pt):
                        if isinstance(pt, dict):
                            return pt.get("X", 0), pt.get("Y", 0)
                        else:
                            return pt.x, pt.y

                    if not polygon:
                        continue

                    xs = []
                    ys = []
                    for pt in polygon:
                        x, y = get_xy(pt)
                        xs.append(x)
                        ys.append(y)
                    # 闭合多边形
                    xs.append(xs[0])
                    ys.append(ys[0])

                    ax.plot(
                        xs,
                        ys,
                        color=c,
                        linewidth=1,
                        alpha=0.8,
                        label=name if i == 0 else "",
                    )
                    # fill 多边形
                    ax.fill(xs, ys, color=c)

        # 设置图形属性
        ax.set_aspect("equal")
        ax.grid(False)
        ax.axis("off")
        # ax.set_title("结构平面图")

        # 调整布局
        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 结构图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_rooms(self, rooms: List[Room], save_path: str = None):
        """
        绘制房间分区

        Args:
            rooms: 房间列表
            save_path: 保存图片路径
        """
        # 创建绘图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 绘制房间分区 - 黄色多边形
        for room in rooms:
            polygon = room.polygon
            x_coords = [point.x for point in polygon] + [polygon[0].x]
            y_coords = [point.y for point in polygon] + [polygon[0].y]
            ax.plot(
                x_coords,
                y_coords,
                color="orange",
                linewidth=1,
                alpha=0.8,
            )

        # 设置图形属性
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"房间分区图")

        # 调整布局
        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 房间分区图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()


def process_dxf(dxf_file: str):
    """处理单个DXF文件（顶层函数，便于多进程 pickling）"""
    extractor = DXFExtractor()
    data = extractor.extract_from_file(dxf_file)
    save_path = os.path.join("dxf/plots/raw", Path(dxf_file).stem + ".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    extractor.plot_structure(data, save_path=str(save_path))
    return Path(dxf_file).stem


def main():
    """主函数"""
    # import shutil

    # # 复制所有文件到新文件夹
    # src_dir = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\beam_finished"
    # target_dir = (
    #     r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\beam_finished_extracted"
    # )
    # for root, dirs, files in os.walk(src_dir):
    #     for fname in files:
    #         fpath = os.path.join(root, fname)
    #         if os.path.isfile(fpath) and fname.lower().endswith(".dxf"):
    #             shutil.copy(fpath, os.path.join(target_dir, fname.replace("-XG", "")))

    extractor = DXFExtractor()

    # # 示例：提取单个DXF文件
    # f_path = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\beam_finished_extracted\L1L28_232.dxf"
    # data = extractor.extract_from_file(f_path)
    # extractor.plot_structure(data)

    # 批量处理DXF文件，使用多进程加速

    dxf_files = []
    for root, dirs, files in os.walk(
        r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\room_finished\final"
    ):
        dxf_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(".dxf")])

    if dxf_files:
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_dxf, dxf_files)
        print(f"✅ 完成处理 {len(results)} 个DXF文件")


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support

        freeze_support()
    except Exception:
        pass
    main()
