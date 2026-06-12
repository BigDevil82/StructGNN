import argparse
import json
import os
from copy import deepcopy

import cv2
import numpy as np


class DataAugmenter:
    """数据增广类，用于对建筑平面图构件进行几何变换"""

    def __init__(self, image_width=2048, image_height=1024, padding=20):
        """
        初始化数据增广器

        Args:
            image_width: 图像宽度，默认2048
            image_height: 图像高度，默认1024
        """
        self.width = image_width + padding * 2
        self.height = image_height + padding * 2

    def transform_point(self, point, mode):
        """
        对单个点进行变换

        Args:
            point: [x, y] 坐标
            mode: 变换模式 ('none', 'flip_x', 'flip_y', 'rot_90', 'rot_180', 'rot_270')

        Returns:
            变换后的坐标 [x', y']
        """
        x, y = point

        if mode == "none":
            return [x, y]

        elif mode == "flip_x":
            # 沿x轴翻转（上下翻转）
            return [x, self.height - y]

        elif mode == "flip_y":
            # 沿y轴翻转（左右翻转）
            return [self.width - x, y]

        elif mode == "rot_90":
            # 顺时针旋转90度
            # (x, y) -> (height - y, x)
            return [self.height - y, x]

        elif mode == "rot_180":
            # 旋转180度
            return [self.width - x, self.height - y]

        elif mode == "rot_270":
            # 顺时针旋转270度（逆时针90度）
            # (x, y) -> (y, width - x)
            return [y, self.width - x]

        else:
            raise ValueError(f"未知的变换模式: {mode}")

    def transform_component(self, component, mode):
        """
        对单个构件进行变换

        Args:
            component: 构件字典 {'start': [x, y], 'end': [x, y], ...}
            mode: 变换模式

        Returns:
            变换后的构件字典
        """
        transformed = deepcopy(component)
        transformed["start"] = self.transform_point(component["start"], mode)
        transformed["end"] = self.transform_point(component["end"], mode)
        return transformed

    def transform_data(self, data, mode):
        """
        对整个数据集进行变换

        Args:
            data: 包含 walls, doors, windows 的字典
            mode: 变换模式

        Returns:
            变换后的数据字典
        """
        transformed_data = {"walls": [], "doors": [], "windows": []}

        for wall in data.get("walls", []):
            transformed_data["walls"].append(self.transform_component(wall, mode))

        for door in data.get("doors", []):
            transformed_data["doors"].append(self.transform_component(door, mode))

        for window in data.get("windows", []):
            transformed_data["windows"].append(self.transform_component(window, mode))

        return transformed_data

    def get_canvas_size(self, mode):
        """
        获取变换后的画布尺寸

        Args:
            mode: 变换模式

        Returns:
            (width, height) 元组
        """
        if mode in ["rot_90", "rot_270"]:
            # 旋转90度或270度时，宽高互换
            return (self.height, self.width)
        else:
            return (self.width, self.height)


def draw_components(canvas, data, offset_x=0, offset_y=0, padding=50):
    """
    在画布上绘制构件

    Args:
        canvas: 画布
        data: 构件数据
        offset_x: x方向偏移
        offset_y: y方向偏移
        padding: 边缘填充
    """
    # 绘制门构件（蓝色）
    for door in data.get("doors", []):
        start = (int(door["start"][0]) + offset_x + padding, int(door["start"][1]) + offset_y + padding)
        end = (int(door["end"][0]) + offset_x + padding, int(door["end"][1]) + offset_y + padding)
        cv2.line(canvas, start, end, (255, 0, 0), 3)  # 蓝色
    # 绘制窗构件（绿色）
    for window in data.get("windows", []):
        start = (int(window["start"][0]) + offset_x + padding, int(window["start"][1]) + offset_y + padding)
        end = (int(window["end"][0]) + offset_x + padding, int(window["end"][1]) + offset_y + padding)
        cv2.line(canvas, start, end, (0, 200, 0), 3)

    # 绘制墙构件（灰色），带剪力墙占比
    for wall in data.get("walls", []):
        start = (int(wall["start"][0]) + offset_x + padding, int(wall["start"][1]) + offset_y + padding)
        end = (int(wall["end"][0]) + offset_x + padding, int(wall["end"][1]) + offset_y + padding)
        left_ratio = wall.get("left_shear_ratio", 0.0)
        right_ratio = wall.get("right_shear_ratio", 0.0)

        # 判断是水平还是竖直线
        is_horizontal = abs(end[0] - start[0]) > abs(end[1] - start[1])

        # 绘制整条墙（灰色）
        cv2.line(canvas, start, end, (120, 120, 120), 2)

        if is_horizontal:
            if start[0] > end[0]:
                start, end = end, start
                left_ratio, right_ratio = right_ratio, left_ratio

            line_length = end[0] - start[0]

            # 左端剪力墙（红色）
            if left_ratio > 0:
                left_end_x = int(start[0] + line_length * left_ratio)
                cv2.line(canvas, start, (left_end_x, start[1]), (0, 0, 255), 4)

            # 右端剪力墙（红色）
            if right_ratio > 0:
                right_start_x = int(end[0] - line_length * right_ratio)
                cv2.line(canvas, (right_start_x, end[1]), end, (0, 0, 255), 4)
        else:
            if start[1] > end[1]:
                start, end = end, start
                left_ratio, right_ratio = right_ratio, left_ratio

            line_length = end[1] - start[1]

            # 上端剪力墙（红色）
            if left_ratio > 0:
                top_end_y = int(start[1] + line_length * left_ratio)
                cv2.line(canvas, start, (start[0], top_end_y), (0, 0, 255), 4)

            # 下端剪力墙（红色）
            if right_ratio > 0:
                bottom_start_y = int(end[1] - line_length * right_ratio)
                cv2.line(canvas, (end[0], bottom_start_y), end, (0, 0, 255), 4)


def visualize_augmentation(original_data, modes, output_path, padding=50):
    """
    可视化数据增广结果，将原始和多种变换并排显示

    Args:
        original_data: 原始数据
        modes: 变换模式列表
        output_path: 输出图像路径
        padding: 边缘填充
    """
    augmenter = DataAugmenter()

    # 计算每个子图的最大尺寸（考虑旋转）
    max_w = max(augmenter.width, augmenter.height)
    max_h = max(augmenter.width, augmenter.height)

    # 计算画布尺寸（多行多列布局）
    cols = min(3, len(modes))  # 每行最多3个
    rows = (len(modes) + cols - 1) // cols

    canvas_width = cols * (max_w + padding * 2) + (cols - 1) * 20
    canvas_height = rows * (max_h + padding * 2 + 60) + (rows - 1) * 20

    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 绘制每种变换
    for idx, mode in enumerate(modes):
        row = idx // cols
        col = idx % cols

        # 计算当前子图的起始位置
        start_x = col * (max_w + padding * 2 + 20)
        start_y = row * (max_h + padding * 2 + 60 + 20) + 50

        # 应用变换
        transformed_data = augmenter.transform_data(original_data, mode)

        # 绘制边框
        rect_x = start_x
        rect_y = start_y - 50
        rect_w = max_w + padding * 2
        rect_h = max_h + padding * 2 + 50
        cv2.rectangle(canvas, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (200, 200, 200), 2)

        # 绘制标题
        title = f"Mode: {mode}"
        cv2.putText(canvas, title, (start_x + 10, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 获取变换后的画布尺寸
        trans_w, trans_h = augmenter.get_canvas_size(mode)

        # 计算居中偏移
        offset_x = start_x + (max_w - trans_w) // 2
        offset_y = start_y + (max_h - trans_h) // 2

        # 绘制构件
        draw_components(canvas, transformed_data, offset_x, offset_y, padding)

        # 添加统计信息
        stats_y = start_y + max_h + padding * 2 + 20
        stats_text = (
            f"W:{len(transformed_data.get('walls', []))} "
            f"D:{len(transformed_data.get('doors', []))} "
            f"Wi:{len(transformed_data.get('windows', []))}"
        )
        cv2.putText(
            canvas, stats_text, (start_x + 10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
        )

    # 保存结果
    cv2.imwrite(output_path, canvas)
    print(f"可视化结果已保存到: {output_path}")
    return canvas


def augment_dataset(input_json, image_key, modes, output_dir):
    """
    对数据集中的某个图像进行多种增广并可视化

    Args:
        input_json: 输入JSON文件路径
        image_key: 要增广的图像键名
        modes: 变换模式列表
        output_dir: 输出目录
    """
    # 读取数据
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if image_key not in dataset:
        print(f"错误: 找不到图像 '{image_key}'")
        available_keys = list(dataset.keys())[:10]
        print(f"可用的图像键（前10个）: {available_keys}")
        return

    data = dataset[image_key]

    # 准备原始数据（仅包含构件信息）
    original_data = {
        "walls": data.get("walls", []),
        "doors": data.get("doors", []),
        "windows": data.get("windows", []),
    }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 可视化所有变换
    output_path = os.path.join(output_dir, f"{image_key}_augmentation.png")
    visualize_augmentation(original_data, modes, output_path)

    # 保存增广后的数据
    augmenter = DataAugmenter()
    augmented_dataset = {}

    for mode in modes:
        transformed_data = augmenter.transform_data(original_data, mode)
        augmented_key = f"{image_key}_{mode}"
        augmented_dataset[augmented_key] = {
            "original_image": image_key,
            "augmentation_mode": mode,
            "walls": transformed_data["walls"],
            "doors": transformed_data["doors"],
            "windows": transformed_data["windows"],
            "num_walls": len(transformed_data["walls"]),
            "num_doors": len(transformed_data["doors"]),
            "num_windows": len(transformed_data["windows"]),
        }

    # 保存增广数据集
    output_json = os.path.join(output_dir, f"{image_key}_augmented.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)

    print(f"增广数据已保存到: {output_json}")
    print(f"生成了 {len(modes)} 个增广样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建筑平面图数据增广工具")
    parser.add_argument(
        "--input", type=str, default="processed/shear_wall_dataset.json", help="输入数据集JSON文件路径"
    )
    parser.add_argument("--image_key", type=str, default="test (1)", help="要增广的图像键名")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"],
        help="变换模式列表",
    )
    parser.add_argument("--output_dir", type=str, default="processed/augmentation", help="输出目录")

    args = parser.parse_args()

    augment_dataset(args.input, args.image_key, args.modes, args.output_dir)
