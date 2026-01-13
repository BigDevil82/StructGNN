import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union

from preprocess.wall_centerline import (
    estimate_wall_thickness,
    extract_wall_centerline,
    visualize_wall_extraction,
)


def extract_mixed_thickness_walls(
    wall_geom_input,
    min_thickness_threshold=50.0,  # 最小墙厚阈值，低于此不再提取
    max_iterations=3,  # 最大剥离次数，防止死循环
):
    """
    处理混合厚度墙体的中心线提取（剥离法）。
    """
    if wall_geom_input is None or wall_geom_input.is_empty:
        return []

    # 预处理
    current_geom = wall_geom_input.buffer(0)
    all_centerlines = []

    # 记录已处理的厚度，避免重复死循环
    processed_thicknesses = set()

    print(f"开始混合厚度提取流程...")

    for i in range(max_iterations):
        if current_geom.is_empty or current_geom.area < 1e-3:
            print(f"  [Pass {i+1}] 几何体已空，停止提取。")
            break

        # 1. 估算当前剩余几何体的主导厚度
        current_thickness = estimate_wall_thickness(current_geom)
        current_thickness = int(round(current_thickness / 50) * 50)  # 四舍五入到最近的50mm

        # 终止条件：厚度太小或无法检测
        if current_thickness < min_thickness_threshold:
            print(f"  [Pass {i+1}] 检测厚度 {current_thickness:.2f} < 阈值，停止剥离。")
            break

        print(f"  [Pass {i+1}] 识别主导厚度: {current_thickness:.2f}")
        processed_thicknesses.add(current_thickness)

        # 清洗碎片：去掉面积极小的噪点
        if isinstance(current_geom, MultiPolygon):
            valid_polys = [p for p in current_geom.geoms if p.area > (current_thickness * current_thickness)]
            current_geom = unary_union(valid_polys)
        elif isinstance(current_geom, Polygon):
            if current_geom.area < (current_thickness * current_thickness):
                current_geom = Polygon()

        # 2. 提取当前厚度的中心线
        # 注意：这里我们只提取符合当前厚度的部分，tolerance设紧一点
        lines = extract_wall_centerline(
            current_geom,
            thickness=current_thickness,
            tolerance_ratio=0.2,  # 容差稍微给大一点点，适应施工误差
            merge_lines=False,  # 先不合并，方便后续处理
        )

        if lines.is_empty:
            print(f"  [Pass {i+1}] 未提取到有效线段，跳过。")
            continue

        # 收集线段
        if isinstance(lines, LineString):
            all_centerlines.append((lines, current_thickness))
        elif isinstance(lines, MultiLineString):
            for line in lines.geoms:
                all_centerlines.append((line, current_thickness))

        # 3. 构造遮罩并剥离 (Peeling)
        # 用提取出的线段，按当前厚度生成Buffer，从原图中挖掉
        # 技巧：buffer稍微大一点点(比如+0.1mm)，确保切断连接处，防止残留细丝
        lines = linemerge(lines)
        mask = lines.buffer(current_thickness / 2.0 + 0.1, cap_style=2, join_style=2)
        current_geom = current_geom.difference(mask)
        print(f"  [Pass {i+1}] 剥离后剩余面积: {current_geom.area:.2f}")
        # visualize current geometry for debug
        # visualize_wall_extraction(current_geom, lines, title=f"Current Geometry after {i+1} passes")

    return all_centerlines


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 构造一个混合厚度场景：
    # 一个粗墙 (200mm) 形成的 L 型，中间接一个细墙 (100mm)

    # 1. 200mm 墙
    thick_wall = LineString([(0, 0), (0, 10), (10, 10)]).buffer(1, cap_style=2, join_style=2)
    # 2. 100mm 墙，垂直连接在 (5, 10) 处，向下延伸
    thin_wall = LineString([(5, 10), (5, 5)]).buffer(0.5, cap_style=2, join_style=2)

    # 合并成一个整体多边形
    mixed_wall = unary_union([thick_wall, thin_wall])
    mixed_wall = scale(mixed_wall, 100, 100)  # 放大到实际尺寸 (mm)

    # 运行提取
    result = extract_mixed_thickness_walls(mixed_wall, min_thickness_threshold=0.2, max_iterations=2)

    # 可视化 (假设你有 visualize_wall_extraction)
    visualize_wall_extraction(mixed_wall, result, title="Mixed Thickness Extraction")
