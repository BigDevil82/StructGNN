import os
from typing import List, Tuple, Union

import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import LineString, Point, Polygon

from preprocess.dxf_extractor import DXFExtractor


class GeometryAugmentor:
    """
    几何增广器：对房间、墙体、门窗及梁进行几何变换
    支持 Polygon (面) 和 LineString (线)
    """

    @staticmethod
    def flip_horizontal(geoms: List[Union[Polygon, LineString]]) -> List[Union[Polygon, LineString]]:
        """水平翻转 (左右镜像)"""
        # 以 (0,0) 为轴翻转
        return [scale(p, xfact=-1.0, yfact=1.0, origin=(0, 0)) for p in geoms]

    @staticmethod
    def flip_vertical(geoms: List[Union[Polygon, LineString]]) -> List[Union[Polygon, LineString]]:
        """垂直翻转 (上下镜像)"""
        return [scale(p, xfact=1.0, yfact=-1.0, origin=(0, 0)) for p in geoms]

    @staticmethod
    def rotate_90(geoms: List[Union[Polygon, LineString]]) -> List[Union[Polygon, LineString]]:
        """逆时针旋转 90 度"""
        return [rotate(p, angle=90, origin=(0, 0)) for p in geoms]

    @staticmethod
    def rotate_180(geoms: List[Union[Polygon, LineString]]) -> List[Union[Polygon, LineString]]:
        """旋转 180 度"""
        return [rotate(p, angle=180, origin=(0, 0)) for p in geoms]

    @staticmethod
    def rotate_270(geoms: List[Union[Polygon, LineString]]) -> List[Union[Polygon, LineString]]:
        """逆时针旋转 270 度"""
        return [rotate(p, angle=270, origin=(0, 0)) for p in geoms]

    @staticmethod
    def apply_augmentation(
        rooms: List[Polygon],
        shear_walls: List[Polygon],
        infill_walls: List[Polygon],
        doors: List[Polygon],
        windows: List[Polygon],
        beams: List[LineString],
        mode: str,
    ) -> Tuple[List[Polygon], List[Polygon], List[Polygon], List[Polygon], List[Polygon], List[LineString]]:
        """
        统一应用变换
        mode: 'none', 'flip_x', 'flip_y', 'rot_90', 'rot_180', 'rot_270'
        """
        if mode == "none":
            return rooms, shear_walls, infill_walls, doors, windows, beams

        ops = {
            "none": None,
            "flip_x": GeometryAugmentor.flip_horizontal,
            "flip_y": GeometryAugmentor.flip_vertical,
            "rot_90": GeometryAugmentor.rotate_90,
            "rot_180": GeometryAugmentor.rotate_180,
            "rot_270": GeometryAugmentor.rotate_270,
        }

        func = ops.get(mode)
        if not func:
            return rooms, shear_walls, infill_walls, doors, windows, beams

        # 对所有类型的几何体应用相同的变换
        new_rooms = func(rooms)
        new_sw = func(shear_walls)
        new_infill = func(infill_walls)
        new_doors = func(doors)
        new_windows = func(windows)
        new_beams = func(beams)

        return new_rooms, new_sw, new_infill, new_doors, new_windows, new_beams


# ==========================================
# 可视化验证模块
# ==========================================
def visualize_augmentations(dxf_path: str, save_path: str = None):
    """
    可视化数据增广效果 (包含梁、门、窗)
    """
    import sys

    import matplotlib.pyplot as plt

    # 确保能找到 preprocess 模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocess.room_calibrator import calibrate_rooms

    print(f"正在加载 DXF 文件: {dxf_path}")

    # 1. 提取DXF数据
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换为Shapely几何体
    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]
    door_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.doors]
    window_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.windows]

    # 梁是 LineSegment 对象，需转换为 Shapely LineString
    beam_lines = [
        LineString([(b.start_point.x, b.start_point.y), (b.end_point.x, b.end_point.y)])
        for b in extractor.beams
    ]

    print(f"  - 房间: {len(raw_rooms)}, 剪力墙: {len(sw_polys)}, 填充墙: {len(infill_polys)}")
    print(f"  - 门: {len(door_polys)}, 窗: {len(window_polys)}, 梁: {len(beam_lines)}")

    # 2. 校准房间
    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

    # 3. 定义增广模式
    augmentation_modes = [
        ("none", "原始图"),
        ("flip_x", "水平翻转"),
        ("flip_y", "垂直翻转"),
        ("rot_90", "旋转90°"),
        ("rot_180", "旋转180°"),
        ("rot_270", "旋转270°"),
    ]

    # 4. 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (mode, title) in enumerate(augmentation_modes):
        ax = axes[idx]

        # 应用增广
        res = GeometryAugmentor.apply_augmentation(
            calibrated_rooms, sw_polys, infill_polys, door_polys, window_polys, beam_lines, mode
        )
        aug_rooms, aug_sw, aug_infill, aug_doors, aug_windows, aug_beams = res

        # A. 绘制房间轮廓
        for room in aug_rooms:
            x, y = room.exterior.xy
            ax.plot(x, y, "b-", linewidth=1, alpha=0.3)
            ax.fill(x, y, color="lightblue", alpha=0.1)

        # B. 绘制构件
        # 剪力墙 (红)
        for poly in aug_sw:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="red", alpha=0.8, label="剪力墙" if poly == aug_sw[0] else "")
        # 填充墙 (灰)
        for poly in aug_infill:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="lightgray", alpha=0.5, label="填充墙" if poly == aug_infill[0] else "")
        # 门 (棕)
        for poly in aug_doors:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="brown", alpha=0.6, label="门" if poly == aug_doors[0] else "")
        # 窗 (青)
        for poly in aug_windows:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="cyan", alpha=0.6, label="窗" if poly == aug_windows[0] else "")

        # C. 绘制梁 (橙色粗线)
        for beam in aug_beams:
            x, y = beam.xy
            ax.plot(x, y, color="orange", linewidth=3, alpha=0.9, label="梁" if beam == aug_beams[0] else "")

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(False)
        ax.axis("off")

        # 只在第一张图显示图例
        if idx == 0:
            ax.legend(loc="upper right", fontsize="small")

    plt.suptitle(f"梁预测数据增广效果\n文件: {os.path.basename(dxf_path)}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ 可视化结果已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def verify_augmentation_consistency(dxf_path: str):
    """验证增广的一致性 (包括梁的长度守恒)"""
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocess.room_calibrator import calibrate_rooms

    print("=" * 60)
    print("增广一致性验证 (含梁数据)")
    print("=" * 60)

    # 提取
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换
    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]
    door_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.doors]
    window_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.windows]
    beam_lines = [
        LineString([(b.start_point.x, b.start_point.y), (b.end_point.x, b.end_point.y)])
        for b in extractor.beams
    ]

    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

    # 统计原始指标
    orig_counts = {"sw": len(sw_polys), "beam": len(beam_lines), "door": len(door_polys)}
    orig_total_area = sum(room.area for room in calibrated_rooms)
    orig_total_beam_len = sum(beam.length for beam in beam_lines)

    print(
        f"原始统计: 剪力墙{orig_counts['sw']}个, 梁{orig_counts['beam']}根, 总梁长{orig_total_beam_len:.1f}"
    )

    modes = ["flip_x", "flip_y", "rot_90", "rot_180", "rot_270"]
    all_passed = True

    for mode in modes:
        res = GeometryAugmentor.apply_augmentation(
            calibrated_rooms, sw_polys, infill_polys, door_polys, window_polys, beam_lines, mode
        )
        aug_rooms, aug_sw, aug_infill, aug_doors, aug_windows, aug_beams = res

        # 检查数量
        count_ok = (
            (len(aug_sw) == orig_counts["sw"])
            and (len(aug_beams) == orig_counts["beam"])
            and (len(aug_doors) == orig_counts["door"])
        )

        # 检查面积守恒
        aug_area = sum(r.area for r in aug_rooms)
        area_ok = abs(aug_area - orig_total_area) < 1.0

        # 检查梁长度守恒
        aug_beam_len = sum(b.length for b in aug_beams)
        len_ok = abs(aug_beam_len - orig_total_beam_len) < 1.0

        passed = count_ok and area_ok and len_ok
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"  {status} {mode:10s} - 梁数量:{len(aug_beams)} 总长:{aug_beam_len:.1f}")

        if not passed:
            if not len_ok:
                print(f"      ⚠️ 梁长度不一致: {aug_beam_len:.1f} != {orig_total_beam_len:.1f}")

    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=str, required=True, help="DXF文件路径")
    parser.add_argument("--output", type=str, default="aug_viz.png")
    args = parser.parse_args()

    verify_augmentation_consistency(args.dxf)
    visualize_augmentations(args.dxf)
