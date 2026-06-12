import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from docs.paper.experiments.research.baseline_edge_gnn.preprocess.analyze_color2 import analyze_line_colors
from docs.paper.experiments.research.baseline_edge_gnn.preprocess.extract_lines_parallel import (
    clean_lines,
    extract_lines,
    find_endpoints,
    find_intersections,
    merge_close_endpoints,
    merge_close_horizontal_lines,
    merge_close_vertical_lines,
    split_lines_at_intersections,
)


def scale_and_pad(img, target_size=(2048, 1024), padding=20):
    """
    将图像缩先放到目标大小（直接缩放），再添加白色边框

    Args:
        img: 输入图像
        target_size: 目标大小 (width, height)
        padding: 边框大小

    Returns:
        处理后的图像
    """
    # 1. 缩放到目标大小
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

    # 2. 添加白色边框
    img_padded = cv2.copyMakeBorder(
        img_resized, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    return img_padded


def extract_components_from_image(image_path):
    """
    从图像中提取所有线段构件
    返回: red_lines, blue_lines, green_lines, gray_lines (每个都是 [(start, end), ...] 格式)
    """
    # 1. 提取骨架线
    image = cv2.imread(image_path)
    # reshape to (2048, 1024) if needed
    if image.shape != (1024, 2048, 3):
        image = scale_and_pad(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 定义膨胀核并处理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 提取水平和竖直线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # 骨架化处理
    skeleton = cv2.ximgproc.thinning(lines)

    # 2. 获取端点坐标
    horizontal_endpoints, vertical_endpoints = find_endpoints(skeleton)

    # 3. 合并接近的线段
    horizontal_endpoints = merge_close_horizontal_lines(horizontal_endpoints)
    vertical_endpoints = merge_close_vertical_lines(vertical_endpoints)

    # 4. 计算交点
    intersections = find_intersections(horizontal_endpoints, vertical_endpoints)

    # 5. 切分线段
    horizontal_endpoints, vertical_endpoints = split_lines_at_intersections(
        horizontal_endpoints, vertical_endpoints, intersections
    )

    # 6. 合并端点
    updated_horizontal, updated_vertical = merge_close_endpoints(
        horizontal_endpoints, vertical_endpoints, intersections
    )

    # 7. 清理
    updated_horizontal, updated_vertical = clean_lines(updated_horizontal, updated_vertical)

    # 8. 分析线段颜色
    red_lines, blue_lines, green_lines, gray_lines = analyze_line_colors(
        updated_horizontal, updated_vertical, image
    )

    return red_lines, blue_lines, green_lines, gray_lines


def calculate_line_overlap(wall_line, shear_wall_lines, tolerance=5):
    """
    计算墙构件与剪力墙的重叠比例

    Args:
        wall_line: 墙构件线段 (start, end)，其中 start 和 end 是 (x, y) 坐标
        shear_wall_lines: 剪力墙线段列表 [((x1, y1), (x2, y2)), ...]
        tolerance: 位置容差，判断是否在同一位置

    Returns:
        (left_ratio, right_ratio): 两端剪力墙占比的元组，取值范围 [0, 1.0]
                                   当满布时两端均为1.0
    """
    wall_start, wall_end = wall_line

    # 判断是水平线还是竖直线
    is_horizontal = abs(wall_end[0] - wall_start[0]) > abs(wall_end[1] - wall_start[1])

    if is_horizontal:
        # 水平线：确保 start 在左侧
        if wall_start[0] > wall_end[0]:
            wall_start, wall_end = wall_end, wall_start

        wall_y = (wall_start[1] + wall_end[1]) / 2
        wall_x_min, wall_x_max = wall_start[0], wall_end[0]
        wall_length = wall_x_max - wall_x_min

        if wall_length == 0:
            return (0.0, 0.0)

        # 找到所有与此墙构件重叠的剪力墙线段
        overlapping_segments = []
        for sw_start, sw_end in shear_wall_lines:
            # 检查是否是同一条水平线（y 坐标接近）
            sw_y = (sw_start[1] + sw_end[1]) / 2
            if abs(wall_y - sw_y) > tolerance:
                continue

            # 检查 x 方向是否有重叠
            sw_x_min = min(sw_start[0], sw_end[0])
            sw_x_max = max(sw_start[0], sw_end[0])

            # 计算重叠区间
            overlap_start = max(wall_x_min, sw_x_min)
            overlap_end = min(wall_x_max, sw_x_max)

            if overlap_start < overlap_end:
                overlapping_segments.append((overlap_start, overlap_end))

        # 合并重叠的区间
        if not overlapping_segments:
            return (0.0, 0.0)

        overlapping_segments.sort()
        merged_segments = [overlapping_segments[0]]
        for current in overlapping_segments[1:]:
            last = merged_segments[-1]
            if current[0] <= last[1]:
                merged_segments[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_segments.append(current)

        # 计算左端和右端的剪力墙占比
        left_overlap = 0
        right_overlap = 0

        for seg_start, seg_end in merged_segments:
            # 左端重叠（从墙起点开始）
            if seg_start <= wall_x_min + tolerance:
                left_overlap = max(left_overlap, seg_end - wall_x_min)

            # 右端重叠（到墙终点结束）
            if seg_end >= wall_x_max - tolerance:
                right_overlap = max(right_overlap, wall_x_max - seg_start)

        left_ratio = left_overlap / wall_length
        right_ratio = right_overlap / wall_length

    else:
        # 竖直线：确保 start 在上方
        if wall_start[1] > wall_end[1]:
            wall_start, wall_end = wall_end, wall_start

        wall_x = (wall_start[0] + wall_end[0]) / 2
        wall_y_min, wall_y_max = wall_start[1], wall_end[1]
        wall_length = wall_y_max - wall_y_min

        if wall_length == 0:
            return (0.0, 0.0)

        # 找到所有与此墙构件重叠的剪力墙线段
        overlapping_segments = []
        for sw_start, sw_end in shear_wall_lines:
            # 检查是否是同一条竖直线（x 坐标接近）
            sw_x = (sw_start[0] + sw_end[0]) / 2
            if abs(wall_x - sw_x) > tolerance:
                continue

            # 检查 y 方向是否有重叠
            sw_y_min = min(sw_start[1], sw_end[1])
            sw_y_max = max(sw_start[1], sw_end[1])

            # 计算重叠区间
            overlap_start = max(wall_y_min, sw_y_min)
            overlap_end = min(wall_y_max, sw_y_max)

            if overlap_start < overlap_end:
                overlapping_segments.append((overlap_start, overlap_end))

        # 合并重叠的区间
        if not overlapping_segments:
            return (0.0, 0.0)

        overlapping_segments.sort()
        merged_segments = [overlapping_segments[0]]
        for current in overlapping_segments[1:]:
            last = merged_segments[-1]
            if current[0] <= last[1]:
                merged_segments[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_segments.append(current)

        # 计算上端和下端的剪力墙占比
        top_overlap = 0
        bottom_overlap = 0

        for seg_start, seg_end in merged_segments:
            # 上端重叠（从墙起点开始）
            if seg_start <= wall_y_min + tolerance:
                top_overlap = max(top_overlap, seg_end - wall_y_min)

            # 下端重叠（到墙终点结束）
            if seg_end >= wall_y_max - tolerance:
                bottom_overlap = max(bottom_overlap, wall_y_max - seg_start)

        left_ratio = top_overlap / wall_length
        right_ratio = bottom_overlap / wall_length

    return (left_ratio, right_ratio)


def build_dataset_from_pair(image_a_path, image_b_path):
    """
    从成对的图像A和B中构建数据集

    Args:
        image_a_path: 建筑布局图路径（A）
        image_b_path: 结构布局图路径（B）

    Returns:
        dataset: 字典，包含墙、门、窗的数据
        {
            'walls': [{start, end, left_shear_ratio, right_shear_ratio}, ...],
            'doors': [{start, end, left_shear_ratio, right_shear_ratio}, ...],
            'windows': [{start, end, left_shear_ratio, right_shear_ratio}, ...]
        }
    """
    # 提取A图中的构件
    # 红色-不使用, 蓝色-梁/门, 绿色-窗, 灰色-墙
    _, blue_components_a, green_components_a, gray_walls_a = extract_components_from_image(image_a_path)

    # 提取B图中的构件（红色线条为剪力墙）
    red_shear_walls_b, _, _, _ = extract_components_from_image(image_b_path)

    walls_data = []
    doors_data = []
    windows_data = []

    # 对A中的每个墙构件，计算其两端剪力墙占比
    for wall_line in gray_walls_a:
        left_ratio, right_ratio = calculate_line_overlap(wall_line, red_shear_walls_b)

        walls_data.append(
            {
                "start": [float(wall_line[0][0]), float(wall_line[0][1])],
                "end": [float(wall_line[1][0]), float(wall_line[1][1])],
                "left_shear_ratio": float(left_ratio),
                "right_shear_ratio": float(right_ratio),
            }
        )

    # 添加门构件（蓝色），剪力墙占比为0
    for door_line in blue_components_a:
        doors_data.append(
            {
                "start": [float(door_line[0][0]), float(door_line[0][1])],
                "end": [float(door_line[1][0]), float(door_line[1][1])],
                "left_shear_ratio": 0.0,
                "right_shear_ratio": 0.0,
            }
        )

    # 添加窗构件（绿色），剪力墙占比为0
    for window_line in green_components_a:
        windows_data.append(
            {
                "start": [float(window_line[0][0]), float(window_line[0][1])],
                "end": [float(window_line[1][0]), float(window_line[1][1])],
                "left_shear_ratio": 0.0,
                "right_shear_ratio": 0.0,
            }
        )

    return {"walls": walls_data, "doors": doors_data, "windows": windows_data}


def process_single_image_pair(args):
    """
    处理单个图像对（用于多进程）

    Args:
        args: (img_a_path, img_b_path, stem) 的元组

    Returns:
        (stem, result_dict) 或 (stem, None) 如果出错
    """
    img_a_path, img_b_path, stem = args

    try:
        # 构建该图像对的数据集
        dataset = build_dataset_from_pair(img_a_path, img_b_path)

        result = {
            "image_a": img_a_path,
            "image_b": img_b_path,
            "walls": dataset["walls"],
            "doors": dataset["doors"],
            "windows": dataset["windows"],
            "num_walls": len(dataset["walls"]),
            "num_doors": len(dataset["doors"]),
            "num_windows": len(dataset["windows"]),
        }

        return (stem, result)

    except Exception as e:
        print(f"处理 {stem} 时出错: {str(e)}")
        return (stem, None)


def build_dataset_from_folders(folder_a, folder_b, output_file, num_workers=None):
    """
    从A和B文件夹中的成对图像构建完整数据集

    Args:
        folder_a: 建筑布局图文件夹路径
        folder_b: 结构布局图文件夹路径
        output_file: 输出JSON文件路径
        num_workers: 进程数，默认为CPU核心数
    """
    folder_a = Path(folder_a)
    folder_b = Path(folder_b)

    if num_workers is None:
        num_workers = cpu_count()

    print(f"使用 {num_workers} 个进程进行处理...")

    # 获取所有图像文件
    images_b = sorted([f for f in folder_b.glob("*.png")])

    # 准备任务列表
    tasks = []
    for img_b_path in images_b:
        img_a_path = folder_a / img_b_path.name

        if not img_a_path.exists():
            print(f"警告: 找不到对应的A图像 {img_a_path}")
            continue

        tasks.append((str(img_a_path), str(img_b_path), img_a_path.stem))

    # 使用多进程处理
    all_dataset = {}

    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_single_image_pair, tasks), total=len(tasks), desc="处理图像对")
        )

    # 收集结果
    for stem, result in results:
        if result is not None:
            all_dataset[stem] = result

    # 保存数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n数据集已保存到: {output_file}")
    print(f"总共处理了 {len(all_dataset)} 对图像")

    # 统计信息
    total_walls = sum(data["num_walls"] for data in all_dataset.values())
    total_doors = sum(data["num_doors"] for data in all_dataset.values())
    total_windows = sum(data["num_windows"] for data in all_dataset.values())
    print(f"总共提取了 {total_walls} 个墙构件")
    print(f"总共提取了 {total_doors} 个门构件")
    print(f"总共提取了 {total_windows} 个窗构件")

    return all_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从成对的建筑平面图中构建剪力墙预测数据集")
    parser.add_argument(
        "--folder_a", type=str, default="0_datasets/L1_7/test_A", help="建筑布局图文件夹路径（A）"
    )
    parser.add_argument(
        "--folder_b", type=str, default="0_datasets/L1_7/test_B", help="结构布局图文件夹路径（B）"
    )
    parser.add_argument(
        "--output", type=str, default="processed/shear_wall_dataset.json", help="输出数据集文件路径"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="进程数（默认为CPU核心数）")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 构建数据集
    dataset = build_dataset_from_folders(args.folder_a, args.folder_b, args.output, args.num_workers)
