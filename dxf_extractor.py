#!/usr/bin/env python3
"""
从DXF文件中提取剪力墙和梁的构件信息
"""

import json
import os
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

        # 处理剪力墙 - 来自SHEAR_WALLS图层的多段线
        if layer_name == "SHEAR_WALLS" and entity_type == "LWPOLYLINE":
            polygon = self._extract_polygon(entity)
            if polygon:
                self.shear_walls.append(polygon)
        # 处理梁 - 来自BEAMS图层的直线
        elif layer_name == "BEAMS" and entity_type == "LINE":
            line_segment = self._extract_line(entity)
            if line_segment:
                self.beams.append(line_segment)
        elif layer_name == "DOORS" and entity_type == "LWPOLYLINE":
            polygon = self._extract_polygon(entity)
            if polygon:
                self.doors.append(polygon)
        elif layer_name == "WINDOWS" and entity_type == "LWPOLYLINE":
            polygon = self._extract_polygon(entity)
            if polygon:
                self.windows.append(polygon)
        elif layer_name == "INFILL_WALLS" and entity_type == "LWPOLYLINE":
            polygon = self._extract_polygon(entity)
            if polygon:
                self.infill_walls.append(polygon)
        elif layer_name == "ROOM" and entity_type == "LWPOLYLINE":
            self.rooms.append(self._extract_rect(entity))

    def _extract_polygon(self, entity) -> Polygon:
        """从LWPOLYLINE实体中提取多边形顶点"""
        try:
            # ezdxf `get_points` for closed polylines does not repeat the start point
            points = list(entity.get_points())
            return [Point(p[0], p[1]) for p in points]
        except Exception as e:
            print(f"提取多边形失败: {e}")
            return []

    def _extract_line(self, line):
        """从直线实体提取梁"""
        try:
            start = line.dxf.start
            end = line.dxf.end

            start_point = Point(start[0], start[1])
            end_point = Point(end[0], end[1])

            return LineSegment(start_point, end_point)

        except Exception as e:
            print(f"提取梁直线失败: {e}")

    def _extract_rect(self, entity):
        """从矩形实体提取线段"""
        try:
            points = list(entity.get_points())

            room = Room([Point(p[0], p[1]) for p in points])

        except Exception as e:
            print(f"提取矩形线段失败: {e}")

        return room

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
            "shear_walls": "red",
            # "beams": "cyan",
            "doors": "blue",
            "windows": "green",
            "infill_walls": "gray",
        }

        # 创建绘图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 绘制剪力墙 - 红色粗线
        for name, lines in data.items():
            if name not in color:
                continue
            c = color.get(name, "k")
            for line in lines:
                start = line["StartPoint"]
                end = line["EndPoint"]
                ax.plot(
                    [start["X"], end["X"]],
                    [start["Y"], end["Y"]],
                    color=c,
                    linewidth=1,
                    alpha=0.8,
                    label=name if line == lines[0] else "",
                )

        # 设置图形属性
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"结构平面图")

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


def main():
    """主函数"""
    extractor = DXFExtractor()

    # 示例：提取单个DXF文件
    f_path = (
        r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\room_finished\L1L28_232.dxf"
    )
    data = extractor.extract_from_file(f_path)

    extractor.plot_rooms(extractor.rooms)

    # # 批量提取
    # dxf_directory = "output"  # DXF文件目录

    # if os.path.exists(dxf_directory):
    #     results = extractor.extract_batch(dxf_directory)

    #     # 验证第一个结果
    #     if results:
    #         first_result = next(iter(results.values()))
    #         extractor.validate_extraction(first_result["json_file"])
    # else:
    #     print(f"❌ 目录不存在: {dxf_directory}")
    #     print("请先运行PNG到DXF转换，生成DXF文件")


if __name__ == "__main__":
    main()
