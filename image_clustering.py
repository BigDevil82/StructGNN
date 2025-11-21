#!/usr/bin/env python3
"""
基于自定义特征的图纸聚类功能
使用DBSCAN算法进行聚类分析
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from element_extractor import ElementExtractor


@dataclass
class ImageFeature:
    """图像特征数据类"""

    image_path: str
    feature_vector: np.ndarray


class ImageClusteringAnalyzer:
    """图纸聚类分析器"""

    def __init__(self, eps: float = 0.5, min_samples: int = 1):
        """初始化聚类分析器"""
        self.eps = eps
        self.min_samples = min_samples
        self.element_types = ["sw", "iw", "window", "door"]
        self.scaler = StandardScaler()

        # 聚类结果
        self.features: List[ImageFeature] = []
        self.labels = None
        self.clusters = {}

    def extract_features(self, image_path: str) -> Optional[ImageFeature]:
        """提取单张图像的特征向量"""
        try:
            extractor = ElementExtractor(image_path)
            total_pixels = extractor.image.shape[0] * extractor.image.shape[1]
            feature_vector = []

            for element_type in self.element_types:
                try:
                    mask = extractor.get_mask_by_hsv(element_type)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 特征1: 像素占比
                    element_pixels = np.sum(mask > 0)
                    ratio = element_pixels / total_pixels
                    feature_vector.append(ratio)

                    # 特征2: 轮廓数量
                    contour_count = len(contours)
                    feature_vector.append(contour_count)

                except Exception:
                    feature_vector.extend([0.0, 0.0])

            feature_vector = np.array(feature_vector, dtype=np.float32)
            return ImageFeature(image_path=image_path, feature_vector=feature_vector)

        except Exception:
            return None

    def analyze_images(self, image_paths: List[str]):
        """分析图像列表并提取特征"""
        print(f"🔍 开始提取 {len(image_paths)} 张图像的特征...")

        self.features = []
        for image_path in image_paths:
            feature = self.extract_features(image_path)
            if feature is not None:
                self.features.append(feature)

        print(f"✅ 成功提取 {len(self.features)} 张图像的特征")

    def perform_clustering(self):
        """使用DBSCAN执行聚类分析"""
        if not self.features:
            print("❌ 没有特征数据，无法进行聚类")
            return

        print("🎯 开始DBSCAN聚类分析...")

        # 准备并标准化特征矩阵
        feature_matrix = np.array([f.feature_vector for f in self.features])
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = dbscan.fit_predict(feature_matrix_scaled)

        # 组织聚类结果
        self.clusters: dict[int, list[ImageFeature]] = {}
        for i, label in enumerate(self.labels):
            if label != -1:  # 忽略噪声点
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(self.features[i])

        print(f"✅ 聚类分析完成！")

    def organize_results(self, output_dir: str):
        """将聚类结果组织到文件夹中"""
        if not self.clusters:
            print("❌ 没有聚类结果可供组织")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 统计信息
        single_images = sum(1 for images in self.clusters.values() if len(images) == 1)
        similar_groups = sum(1 for images in self.clusters.values() if len(images) > 1)

        print("📊 统计结果：")
        print(f"   总图像数: {len(self.features)}")
        print(f"   相似组数: {similar_groups}")
        print(f"   单独图像: {single_images}")

        # 组织文件
        for cluster_id, images in self.clusters.items():
            if len(images) == 1:
                # 单张图像直接放在根目录
                src_path = Path(images[0].image_path)
                dst_path = output_path / src_path.name
                shutil.copy2(src_path, dst_path)
            else:
                # 多张图像创建文件夹
                cluster_dir = output_path / f"cluster_{cluster_id:03d}_{len(images)}"
                cluster_dir.mkdir(exist_ok=True)

                for feature in images:
                    src_path = Path(feature.image_path)
                    dst_path = cluster_dir / src_path.name
                    shutil.copy2(src_path, dst_path)


def main():
    """主函数"""
    images_dir = "imgs/raw"
    output_unique_dir = "imgs/unique_imgs"

    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(".png")]

    # 聚类去重
    analyzer = ImageClusteringAnalyzer(eps=0.5, min_samples=1)
    analyzer.analyze_images(image_paths)
    analyzer.perform_clustering()
    print(f"聚类完成: {len(analyzer.clusters)} 个聚类")

    # 取每个聚类的代表图像
    unique_images = [cluster[0].image_path for cluster in analyzer.clusters.values()]

    # copy to a folder
    Path(output_unique_dir).mkdir(exist_ok=True)
    for img_path in unique_images:
        img_name = Path(img_path).name
        shutil.copy(img_path, os.path.join(output_unique_dir, img_name))


if __name__ == "__main__":
    main()
