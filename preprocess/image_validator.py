#!/usr/bin/env python3
"""
从PNG图片中提取剪力墙和梁的像素，计算剪力墙占比，并移动不符合要求的图片
"""

import os
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


class ImageValidator:
    """图片构件比例验证器"""

    def __init__(self, shear_wall_threshold: float = 0.1):
        """
        初始化验证器

        Args:
            shear_wall_threshold: 剪力墙占比阈值，低于此值的图片会被移动
        """
        self.shear_wall_threshold = shear_wall_threshold

        # 定义颜色范围（BGR格式，因为OpenCV使用BGR）
        # matplotlib的红色 (255, 0, 0) -> BGR (0, 0, 255)
        # matplotlib的橙色 (255, 165, 0) -> BGR (0, 165, 255)
        self.red_color_bgr = np.array([0, 0, 255])
        self.orange_color_bgr = np.array([0, 165, 255])

        # 颜色容差（允许一定的颜色偏差）
        self.color_tolerance = 30

    def extract_component_pixels(self, image_path: str) -> Tuple[int, int, int]:
        """
        提取图片中的剪力墙和梁的像素数量

        Args:
            image_path: 图片路径

        Returns:
            (剪力墙像素数, 梁像素数, 总构件像素数)
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 创建红色掩码（剪力墙）
        red_lower = self.red_color_bgr - self.color_tolerance
        red_upper = self.red_color_bgr + self.color_tolerance
        red_mask = cv2.inRange(img, red_lower, red_upper)

        # 创建橙色掩码（梁）
        orange_lower = self.orange_color_bgr - self.color_tolerance
        orange_upper = self.orange_color_bgr + self.color_tolerance
        orange_mask = cv2.inRange(img, orange_lower, orange_upper)

        # 统计像素数量
        shear_wall_pixels = np.count_nonzero(red_mask)
        beam_pixels = np.count_nonzero(orange_mask)
        total_component_pixels = shear_wall_pixels + beam_pixels

        return shear_wall_pixels, beam_pixels, total_component_pixels

    def calculate_shear_wall_ratio(self, image_path: str) -> float:
        """
        计算剪力墙占总构件的比例

        Args:
            image_path: 图片路径

        Returns:
            剪力墙占比（0-1之间）
        """
        shear_wall_pixels, beam_pixels, total_pixels = self.extract_component_pixels(image_path)

        if total_pixels == 0:
            return 0.0

        ratio = shear_wall_pixels / total_pixels
        return ratio

    def validate_and_move(
        self,
        plots_dir: str,
        output_dir: str = None,
        dry_run: bool = False,
    ) -> dict:
        """
        遍历plots文件夹，验证所有图片，并移动不符合要求的图片

        Args:
            plots_dir: plots文件夹路径
            output_dir: 输出文件夹路径（存放不符合要求的图片），默认为plots_dir/../plots_low_wall_ratio
            dry_run: 如果为True，只统计不移动文件

        Returns:
            统计信息字典
        """
        plots_dir = Path(plots_dir)
        if not plots_dir.exists():
            raise ValueError(f"plots目录不存在: {plots_dir}")

        # 设置输出目录
        if output_dir is None:
            output_dir = plots_dir.parent / "plots_low_wall_ratio"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # 获取所有PNG图片
        image_files = list(plots_dir.glob("*.png"))
        if not image_files:
            print(f"在 {plots_dir} 中未找到PNG图片")
            return {}

        print(f"🔍 找到 {len(image_files)} 张图片")
        print(f"📊 剪力墙占比阈值: {self.shear_wall_threshold:.2%}")
        print(f"📁 低占比图片输出目录: {output_dir}")
        print()

        # 统计信息
        stats = {
            "total_images": len(image_files),
            "valid_images": 0,
            "invalid_images": 0,
            "moved_images": [],
            "errors": [],
        }

        # 遍历所有图片
        for image_path in tqdm(image_files, desc="验证图片"):
            try:
                # 计算剪力墙占比
                ratio = self.calculate_shear_wall_ratio(str(image_path))

                # 判断是否符合要求
                if ratio < self.shear_wall_threshold:
                    stats["invalid_images"] += 1
                    stats["moved_images"].append(
                        {
                            "filename": image_path.name,
                            "ratio": ratio,
                        }
                    )

                    # 移动文件
                    if not dry_run:
                        target_path = output_dir / image_path.name
                        shutil.move(str(image_path), str(target_path))
                else:
                    stats["valid_images"] += 1

            except Exception as e:
                stats["errors"].append(
                    {
                        "filename": image_path.name,
                        "error": str(e),
                    }
                )

        # 输出统计结果
        print()
        print("=" * 60)
        print("📊 验证结果统计")
        print("=" * 60)
        print(f"总图片数: {stats['total_images']}")
        print(f"符合要求: {stats['valid_images']} ({stats['valid_images']/stats['total_images']:.2%})")
        print(f"不符合要求: {stats['invalid_images']} ({stats['invalid_images']/stats['total_images']:.2%})")
        print(f"处理错误: {len(stats['errors'])}")
        print()

        if stats["moved_images"]:
            print(
                f"{'移动' if not dry_run else '待移动'}的图片 (剪力墙占比 < {self.shear_wall_threshold:.2%}):"
            )
            for item in sorted(stats["moved_images"], key=lambda x: x["ratio"]):
                print(f"  - {item['filename']}: {item['ratio']:.2%}")
            print()

        if stats["errors"]:
            print("处理错误:")
            for item in stats["errors"]:
                print(f"  - {item['filename']}: {item['error']}")
            print()

        if dry_run:
            print("🔍 DRY RUN 模式 - 未移动任何文件")
        else:
            print(f"✅ 已移动 {stats['invalid_images']} 张图片到: {output_dir}")

        return stats


def main():
    """主函数"""
    # 创建验证器（阈值设置为10%，可根据需要调整）
    validator = ImageValidator(shear_wall_threshold=0.1)

    # 设置plots文件夹路径
    plots_dir = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\data\data\dxf\plots"

    # 先运行dry_run模式查看统计信息
    print("🔍 运行DRY RUN模式，查看统计信息...")
    print()
    stats = validator.validate_and_move(plots_dir, dry_run=True)

    # 询问是否继续
    print()
    user_input = input("是否继续移动文件？(y/n): ")
    if user_input.lower() == "y":
        print()
        print("🚀 开始移动文件...")
        print()
        validator.validate_and_move(plots_dir, dry_run=False)
    else:
        print("❌ 已取消操作")


if __name__ == "__main__":
    main()
