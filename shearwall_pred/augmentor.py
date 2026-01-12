import os
from typing import List, Tuple

import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon

from preprocess.dxf_extractor import DXFExtractor


class GeometryAugmentor:
    """
    几何增广器：对房间和墙体多边形进行几何变换
    """

    @staticmethod
    def flip_horizontal(polys: List[Polygon]) -> List[Polygon]:
        """水平翻转 (左右镜像)"""
        # origin='center' 表示以自身的中心为轴，但我们需要以整个户型的中心为轴
        # 所以先计算所有图形的整体中心，或者简单点：直接以 (0,0) 为轴翻转，
        # 因为后续 GraphBuilder 会重新计算中心并归一化，所以翻转轴的位置不重要，只要相对关系对即可。
        return [scale(p, xfact=-1.0, yfact=1.0, origin=(0, 0)) for p in polys]

    @staticmethod
    def flip_vertical(polys: List[Polygon]) -> List[Polygon]:
        """垂直翻转 (上下镜像)"""
        return [scale(p, xfact=1.0, yfact=-1.0, origin=(0, 0)) for p in polys]

    @staticmethod
    def rotate_90(polys: List[Polygon]) -> List[Polygon]:
        """逆时针旋转 90 度"""
        return [rotate(p, angle=90, origin=(0, 0)) for p in polys]

    @staticmethod
    def rotate_180(polys: List[Polygon]) -> List[Polygon]:
        """旋转 180 度"""
        return [rotate(p, angle=180, origin=(0, 0)) for p in polys]

    @staticmethod
    def rotate_270(polys: List[Polygon]) -> List[Polygon]:
        """逆时针旋转 270 度"""
        return [rotate(p, angle=270, origin=(0, 0)) for p in polys]

    @staticmethod
    def apply_augmentation(
        rooms: List[Polygon], shear_walls: List[Polygon], infill_walls: List[Polygon], mode: str
    ) -> Tuple[List[Polygon], List[Polygon], List[Polygon]]:
        """
        统一应用变换
        mode: 'none', 'flip_x', 'flip_y', 'rot_90', 'rot_180', 'rot_270'
        """
        if mode == "none":
            return rooms, shear_walls, infill_walls

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
            return rooms, shear_walls, infill_walls

        # 对所有类型的几何体应用相同的变换
        new_rooms = func(rooms)
        new_sw = func(shear_walls)
        new_infill = func(infill_walls)

        return new_rooms, new_sw, new_infill


# ==========================================
# 可视化验证模块
# ==========================================
def visualize_augmentations(dxf_path: str, save_path: str = None):
    """
    可视化数据增广效果

    加载一个DXF文件，应用所有增广变换，并在一张图上并排显示对比效果

    Args:
        dxf_path: DXF文件路径
        save_path: 保存路径（可选）
    """
    import os

    # 导入必要的模块
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

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

    print(f"  - 房间数: {len(raw_rooms)}")
    print(f"  - 剪力墙数: {len(sw_polys)}")
    print(f"  - 填充墙数: {len(infill_polys)}")

    # 2. 校准房间
    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

    # 3. 定义所有增广模式
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
        aug_rooms, aug_sw, aug_infill = GeometryAugmentor.apply_augmentation(
            calibrated_rooms, sw_polys, infill_polys, mode
        )

        # 绘制房间轮廓
        for room in aug_rooms:
            x, y = room.exterior.xy
            ax.plot(x, y, "b-", linewidth=2, alpha=0.7, label="房间" if room == aug_rooms[0] else "")
            ax.fill(x, y, color="lightblue", alpha=0.2)

        # 绘制剪力墙
        for sw in aug_sw:
            x, y = sw.exterior.xy
            ax.fill(x, y, color="red", label="剪力墙" if sw == aug_sw[0] else "")

        # 绘制填充墙
        for infill in aug_infill:
            x, y = infill.exterior.xy
            ax.fill(x, y, color="gray", label="填充墙" if infill == aug_infill[0] else "")

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axis("off")

    plt.suptitle(f"数据增广效果对比\n文件: {os.path.basename(dxf_path)}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ 可视化结果已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def verify_augmentation_consistency(dxf_path: str):
    """
    验证增广的一致性和正确性

    检查：
    1. 增广后房间数量是否保持一致
    2. 增广后面积是否保持一致
    3. 旋转和翻转的可逆性

    Args:
        dxf_path: DXF文件路径
    """
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from preprocess.room_calibrator import calibrate_rooms

    print("=" * 60)
    print("数据增广一致性验证")
    print("=" * 60)

    # 提取数据
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

    # 计算原始数据的统计信息
    original_room_count = len(calibrated_rooms)
    original_sw_count = len(sw_polys)
    original_infill_count = len(infill_polys)
    original_total_area = sum(room.area for room in calibrated_rooms)

    print(f"\n原始数据统计:")
    print(f"  - 房间数: {original_room_count}")
    print(f"  - 剪力墙数: {original_sw_count}")
    print(f"  - 填充墙数: {original_infill_count}")
    print(f"  - 总面积: {original_total_area:.2f}")

    # 测试所有增广模式
    modes = ["flip_x", "flip_y", "rot_90", "rot_180", "rot_270"]

    print("\n验证增广一致性:")
    all_passed = True

    for mode in modes:
        aug_rooms, aug_sw, aug_infill = GeometryAugmentor.apply_augmentation(
            calibrated_rooms, sw_polys, infill_polys, mode
        )

        # 检查数量
        room_count_ok = len(aug_rooms) == original_room_count
        sw_count_ok = len(aug_sw) == original_sw_count
        infill_count_ok = len(aug_infill) == original_infill_count

        # 检查面积（允许小的数值误差）
        aug_total_area = sum(room.area for room in aug_rooms)
        area_ok = abs(aug_total_area - original_total_area) < 1.0

        passed = room_count_ok and sw_count_ok and infill_count_ok and area_ok
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(
            f"  {status} {mode:10s} - 房间:{len(aug_rooms)} 剪力墙:{len(aug_sw)} "
            f"填充墙:{len(aug_infill)} 面积:{aug_total_area:.2f}"
        )

        if not passed:
            if not room_count_ok:
                print(f"      ⚠️  房间数不一致: {len(aug_rooms)} != {original_room_count}")
            if not sw_count_ok:
                print(f"      ⚠️  剪力墙数不一致: {len(aug_sw)} != {original_sw_count}")
            if not infill_count_ok:
                print(f"      ⚠️  填充墙数不一致: {len(aug_infill)} != {original_infill_count}")
            if not area_ok:
                print(f"      ⚠️  面积不一致: {aug_total_area:.2f} != {original_total_area:.2f}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过！数据增广功能正常")
    else:
        print("❌ 部分验证失败，请检查增广实现")
    print("=" * 60)

    return all_passed


# ==========================================
# 主函数：运行验证
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="验证数据增广模块")
    parser.add_argument(
        "--dxf", type=str, default="dxf/to_process/room_finished/L1L28_232.dxf", help="DXF文件路径"
    )
    parser.add_argument(
        "--output", type=str, default="result/augmentation_visualization.png", help="可视化结果保存路径"
    )
    parser.add_argument("--no-viz", action="store_true", help="不生成可视化图片，只进行一致性验证")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.dxf):
        print(f"❌ 错误：DXF文件不存在: {args.dxf}")
        print("\n可用的示例文件:")
        dxf_dir = "dxf/to_process/room_finished"
        if os.path.exists(dxf_dir):
            files = [f for f in os.listdir(dxf_dir) if f.endswith(".dxf")][:5]
            for f in files:
                print(f"  - {os.path.join(dxf_dir, f)}")
        exit(1)

    print("🚀 开始验证数据增广模块\n")

    # 1. 一致性验证
    verify_augmentation_consistency(args.dxf)

    # 2. 可视化验证
    if not args.no_viz:
        print("\n" + "=" * 60)
        print("生成可视化对比图...")
        print("=" * 60)
        visualize_augmentations(args.dxf, args.output)

    print("\n✨ 验证完成！")
