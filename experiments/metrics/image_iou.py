"""
图像IoU计算模块

将不同模型的剪力墙预测结果渲染为图像，然后计算像素级IoU。
这种方法可以统一比较不同表示方式的模型（Room-based vs Edge-based）。

使用方法:
    from experiments.metrics.image_iou import ImageIoUCalculator, render_walls_to_image
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, Point, Polygon


def get_room_edges(room_poly: Polygon) -> List[LineString]:
    """获取房间的四条边（按顺序：上、右、下、左）"""
    minx, miny, maxx, maxy = room_poly.bounds
    tl, tr, bl, br = Point(minx, maxy), Point(maxx, maxy), Point(minx, miny), Point(maxx, miny)
    return [
        LineString([tl, tr]),  # Top
        LineString([tr, br]),  # Right
        LineString([bl, br]),  # Bottom
        LineString([tl, bl]),  # Left
    ]


def vector_to_walls(room_poly: Polygon, sw_vector: np.ndarray) -> List[LineString]:
    """
    将16维剪力墙向量转换为墙体线段列表（Room-based模型）

    Args:
        room_poly: 房间多边形
        sw_vector: 16维向量 [4边 × 2半边 × 2端点]

    Returns:
        walls: 墙体线段列表
    """
    edges = get_room_edges(room_poly)
    sw_ratios = sw_vector.reshape(4, 2, 2)  # [4条边, 2半边, 2端点]
    result_walls = []

    for i, edge in enumerate(edges):
        # 将边分为两半
        p_mid = edge.interpolate(0.5, normalized=True)
        halves = [
            LineString([edge.coords[0], p_mid]),
            LineString([p_mid, edge.coords[-1]]),
        ]

        for j, half in enumerate(halves):
            start_ratio, end_ratio = sw_ratios[i][j]

            # 生成起点墙体
            if start_ratio > 0.01:
                w_start = half.interpolate(0, normalized=True)
                w_end = half.interpolate(start_ratio, normalized=True)
                result_walls.append(LineString([w_start, w_end]))

            # 生成终点墙体
            if end_ratio > 0.01:
                w_start = half.interpolate(1.0 - end_ratio, normalized=True)
                w_end = half.interpolate(1.0, normalized=True)
                result_walls.append(LineString([w_start, w_end]))

    return result_walls


def edge_pred_to_walls(edge_line: LineString, ratio_start: float, ratio_end: float) -> List[LineString]:
    """
    将边级预测转换为墙体线段（Edge-based模型）

    Args:
        edge_line: 边的几何线段
        ratio_start: 起始端剪力墙比例
        ratio_end: 结束端剪力墙比例

    Returns:
        walls: 墙体线段列表
    """
    walls = []
    coords = list(edge_line.coords)
    total_len = edge_line.length

    # 起始端墙体
    if ratio_start > 0.01:
        sw_len = total_len * ratio_start
        sw_point = edge_line.interpolate(sw_len)
        walls.append(LineString([coords[0], (sw_point.x, sw_point.y)]))

    # 结束端墙体
    if ratio_end > 0.01:
        sw_len = total_len * ratio_end
        sw_point = edge_line.interpolate(total_len - sw_len)
        walls.append(LineString([(sw_point.x, sw_point.y), coords[-1]]))

    return walls


class ImageIoUCalculator:
    """
    图像IoU计算器

    将墙体线段渲染到图像上，然后计算像素级IoU。
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        line_width: int = 3,
        padding: float = 0.1,
    ):
        """
        Args:
            image_size: 渲染图像尺寸 (width, height)
            line_width: 墙体线宽（像素）
            padding: 边界填充比例
        """
        self.image_size = image_size
        self.line_width = line_width
        self.padding = padding

    def _compute_transform(
        self, walls: List[LineString], bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        计算从世界坐标到图像坐标的变换参数

        Returns:
            (scale, offset_x, offset_y)
        """
        if bounds is None:
            # 从墙体计算边界
            all_coords = []
            for wall in walls:
                all_coords.extend(list(wall.coords))

            if not all_coords:
                return 1.0, 0.0, 0.0

            xs = [c[0] for c in all_coords]
            ys = [c[1] for c in all_coords]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
        else:
            minx, miny, maxx, maxy = bounds

        # 添加padding
        w = maxx - minx
        h = maxy - miny
        pad_x = w * self.padding
        pad_y = h * self.padding
        minx -= pad_x
        maxx += pad_x
        miny -= pad_y
        maxy += pad_y

        # 计算缩放比例
        w = max(maxx - minx, 1e-6)
        h = max(maxy - miny, 1e-6)

        scale_x = (self.image_size[0] - 2) / w
        scale_y = (self.image_size[1] - 2) / h
        scale = min(scale_x, scale_y)

        # 计算偏移（居中）
        offset_x = (self.image_size[0] - w * scale) / 2 - minx * scale
        offset_y = (self.image_size[1] - h * scale) / 2 - miny * scale

        return scale, offset_x, offset_y

    def _world_to_image(
        self, x: float, y: float, scale: float, offset_x: float, offset_y: float
    ) -> Tuple[int, int]:
        """世界坐标转图像坐标（y轴翻转）"""
        img_x = int(x * scale + offset_x)
        img_y = int(self.image_size[1] - (y * scale + offset_y))
        return img_x, img_y

    def render_walls(
        self,
        walls: List[LineString],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        transform: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        将墙体线段渲染为二值图像

        Args:
            walls: 墙体线段列表
            bounds: 可选的边界 (minx, miny, maxx, maxy)
            transform: 可选的变换参数 (scale, offset_x, offset_y)

        Returns:
            image: 二值图像 (H, W)，墙体位置为1，其他为0
            transform: 使用的变换参数
        """
        if transform is None:
            transform = self._compute_transform(walls, bounds)

        scale, offset_x, offset_y = transform

        # 创建空白图像
        img = Image.new("L", self.image_size, 0)
        draw = ImageDraw.Draw(img)

        # 绘制墙体
        for wall in walls:
            if wall.is_empty:
                continue
            coords = list(wall.coords)
            img_coords = []
            for x, y in coords:
                img_x, img_y = self._world_to_image(x, y, scale, offset_x, offset_y)
                img_coords.append((img_x, img_y))

            if len(img_coords) >= 2:
                draw.line(img_coords, fill=255, width=self.line_width)

        return (np.array(img) > 0).astype(np.uint8), transform

    def compute_iou(
        self,
        gt_walls: List[LineString],
        pred_walls: List[LineString],
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, Dict]:
        """
        计算两组墙体的像素级IoU

        Args:
            gt_walls: Ground Truth墙体线段
            pred_walls: 预测墙体线段
            bounds: 统一的渲染边界

        Returns:
            iou: IoU值
            details: 详细信息字典
        """
        # 合并所有墙体计算统一边界
        all_walls = gt_walls + pred_walls
        if bounds is None and all_walls:
            all_coords = []
            for wall in all_walls:
                if not wall.is_empty:
                    all_coords.extend(list(wall.coords))
            if all_coords:
                xs = [c[0] for c in all_coords]
                ys = [c[1] for c in all_coords]
                bounds = (min(xs), min(ys), max(xs), max(ys))

        # 使用统一变换渲染
        transform = self._compute_transform(all_walls, bounds)
        gt_img, _ = self.render_walls(gt_walls, bounds, transform)
        pred_img, _ = self.render_walls(pred_walls, bounds, transform)

        # 计算IoU
        intersection = np.logical_and(gt_img, pred_img).sum()
        union = np.logical_or(gt_img, pred_img).sum()

        if union == 0:
            # 两者都为空
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        # 计算其他指标
        gt_pixels = gt_img.sum()
        pred_pixels = pred_img.sum()

        details = {
            "iou": float(iou),
            "intersection": int(intersection),
            "union": int(union),
            "gt_pixels": int(gt_pixels),
            "pred_pixels": int(pred_pixels),
            "precision": float(intersection / pred_pixels) if pred_pixels > 0 else 0.0,
            "recall": float(intersection / gt_pixels) if gt_pixels > 0 else 0.0,
        }

        return iou, details

    def visualize_comparison(
        self,
        gt_walls: List[LineString],
        pred_walls: List[LineString],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        可视化GT和预测的对比图

        Returns:
            RGB图像: 红色=仅GT，绿色=匹配，蓝色=仅预测
        """
        all_walls = gt_walls + pred_walls
        transform = self._compute_transform(all_walls, bounds)

        gt_img, _ = self.render_walls(gt_walls, bounds, transform)
        pred_img, _ = self.render_walls(pred_walls, bounds, transform)

        # 创建RGB图像
        h, w = gt_img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # 红色: 仅GT
        rgb[..., 0] = np.where(gt_img & ~pred_img, 255, 0)
        # 绿色: 匹配
        rgb[..., 1] = np.where(gt_img & pred_img, 255, 0)
        # 蓝色: 仅预测
        rgb[..., 2] = np.where(~gt_img & pred_img, 255, 0)

        if save_path:
            Image.fromarray(rgb).save(save_path)

        return rgb


class UnifiedModelEvaluator:
    """
    统一模型评估器

    使用图像IoU对不同表示方式的模型进行公平对比。
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512), line_width: int = 3):
        self.iou_calc = ImageIoUCalculator(image_size=image_size, line_width=line_width)

    def evaluate_room_based(
        self,
        room_polys: List[Polygon],
        pred_vectors: np.ndarray,
        gt_vectors: np.ndarray,
    ) -> Dict:
        """
        评估Room-based模型的预测结果

        Args:
            room_polys: 房间多边形列表
            pred_vectors: 预测向量 (N, 16)
            gt_vectors: Ground Truth向量 (N, 16)

        Returns:
            评估结果字典
        """
        all_gt_walls = []
        all_pred_walls = []

        for i, room_poly in enumerate(room_polys):
            gt_walls = vector_to_walls(room_poly, gt_vectors[i])
            pred_walls = vector_to_walls(room_poly, pred_vectors[i])
            all_gt_walls.extend(gt_walls)
            all_pred_walls.extend(pred_walls)

        iou, details = self.iou_calc.compute_iou(all_gt_walls, all_pred_walls)
        return details

    def evaluate_edge_based(
        self,
        edge_lines: List[LineString],
        edge_mask: np.ndarray,
        pred_ratios: np.ndarray,
        gt_ratios: np.ndarray,
    ) -> Dict:
        """
        评估Edge-based模型的预测结果

        Args:
            edge_lines: 边的几何线段列表
            edge_mask: 边类型掩码（1=PSW, 0=其他）
            pred_ratios: 预测比例 (E, 2)
            gt_ratios: Ground Truth比例 (E, 2)

        Returns:
            评估结果字典
        """
        all_gt_walls = []
        all_pred_walls = []

        edge_idx = 0
        for i, edge_line in enumerate(edge_lines):
            if edge_idx >= len(pred_ratios):
                break

            # 只处理PSW类型的边
            if edge_mask[edge_idx] > 0.5:
                ratio_start_gt, ratio_end_gt = gt_ratios[edge_idx]
                ratio_start_pred, ratio_end_pred = pred_ratios[edge_idx]

                gt_walls = edge_pred_to_walls(edge_line, ratio_start_gt, ratio_end_gt)
                pred_walls = edge_pred_to_walls(edge_line, ratio_start_pred, ratio_end_pred)

                all_gt_walls.extend(gt_walls)
                all_pred_walls.extend(pred_walls)

            edge_idx += 2  # Bidirectional edges

        iou, details = self.iou_calc.compute_iou(all_gt_walls, all_pred_walls)
        return details


def compute_image_iou_batch(
    gt_walls_batch: List[List[LineString]],
    pred_walls_batch: List[List[LineString]],
    image_size: Tuple[int, int] = (512, 512),
    line_width: int = 3,
) -> Tuple[float, List[float]]:
    """
    批量计算图像IoU

    Args:
        gt_walls_batch: 每个样本的GT墙体列表
        pred_walls_batch: 每个样本的预测墙体列表
        image_size: 渲染图像尺寸
        line_width: 线宽

    Returns:
        mean_iou: 平均IoU
        ious: 每个样本的IoU列表
    """
    calc = ImageIoUCalculator(image_size=image_size, line_width=line_width)
    ious = []

    for gt_walls, pred_walls in zip(gt_walls_batch, pred_walls_batch):
        iou, _ = calc.compute_iou(gt_walls, pred_walls)
        ious.append(iou)

    mean_iou = np.mean(ious) if ious else 0.0
    return float(mean_iou), ious


if __name__ == "__main__":
    # 测试代码
    from shapely.geometry import box

    # 创建测试房间
    room = box(0, 0, 100, 80)

    # 模拟16维预测向量
    # [上边左半起, 上边左半终, 上边右半起, 上边右半终, ...]
    gt_vector = np.array([
        0.3, 0.0, 0.0, 0.2,  # 上边
        0.2, 0.0, 0.0, 0.3,  # 右边
        0.0, 0.0, 0.0, 0.0,  # 下边
        0.4, 0.0, 0.0, 0.4,  # 左边
    ])

    pred_vector = np.array([
        0.25, 0.0, 0.0, 0.25,  # 上边
        0.15, 0.0, 0.0, 0.35,  # 右边
        0.0, 0.0, 0.0, 0.0,    # 下边
        0.35, 0.0, 0.0, 0.45,  # 左边
    ])

    # 转换为墙体
    gt_walls = vector_to_walls(room, gt_vector)
    pred_walls = vector_to_walls(room, pred_vector)

    print(f"GT walls: {len(gt_walls)}")
    print(f"Pred walls: {len(pred_walls)}")

    # 计算图像IoU
    calc = ImageIoUCalculator(image_size=(256, 256), line_width=5)
    iou, details = calc.compute_iou(gt_walls, pred_walls)

    print(f"\nImage IoU: {iou:.4f}")
    print(f"Details: {details}")

    # 保存可视化
    rgb = calc.visualize_comparison(gt_walls, pred_walls, save_path="test_iou_comparison.png")
    print("Saved visualization to test_iou_comparison.png")
