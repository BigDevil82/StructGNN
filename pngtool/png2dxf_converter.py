#!/usr/bin/env python3
"""
PNG语义图像到DXF文件的转换器
"""

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ezdxf
import numpy as np
from ezdxf import appsettings, units, zoom
from ezdxf.enums import TextEntityAlignment

from pngtool.element_extractor import ElementExtractor
from pngtool.img_util import resize_img


@dataclass
class ElementConfig:
    """构件配置"""

    name: str
    color_code: int
    layer_name: str


class PNG2DXFConverter:
    """PNG到DXF转换器"""

    # 构件配置
    ELEMENTS = {
        "sw": ElementConfig("shear_wall", 1, "SHEAR_WALLS"),  # 红色
        "iw": ElementConfig("infill_wall", 7, "INFILL_WALLS"),  # 灰色
        "window": ElementConfig("window", 3, "WINDOWS"),  # 绿色
        "door": ElementConfig("door", 5, "DOORS"),  # 蓝色
    }

    # 图例配置
    LEGEND_CONFIG = {
        "rect_width": 1600,
        "rect_height": 800,
        "text_height": 400,
        "spacing": 1200,
        "margin": 2000,
        "title_height": 600,
    }

    # 元素名称
    ELEMENT_NAMES = {
        "sw": "剪力墙 / Shear Wall",
        "iw": "填充墙 / Infill Wall",
        "window": "窗户 / Window",
        "door": "门 / Door",
    }

    _IMG_INIT_SCALE = 10  # 预先将图片放大10倍，以提升识别精度

    def __init__(self, img_path: str, scale: float = 80.0):
        self.img_path = img_path

        # 缩放图像，以提升识别精度
        raw_img = cv2.imread(img_path)
        w, h = raw_img.shape[1], raw_img.shape[0]
        img = resize_img(raw_img, size=(int(w * self._IMG_INIT_SCALE), int(h * self._IMG_INIT_SCALE)))
        self.extractor = ElementExtractor(img)

        self.scale = scale / self._IMG_INIT_SCALE  # 由于图片被预先放大了10倍，因此这里需要除以10
        # DXF相关
        self.doc = None
        self.msp = None

    def setup_dxf(self):
        """初始化DXF文档和图层"""
        self.doc = ezdxf.new("R2013")
        self.msp = self.doc.modelspace()

        self.doc.header["$INSUNITS"] = units.MM
        self.doc.header["$LWDISPLAY"] = 1

        # 创建中文字体
        self.doc.styles.new(name="SimSun", dxfattribs={"font": "宋体", "width": 0.8})

        # 创建DASHED线型，如果它不存在
        if "DASHED" not in self.doc.linetypes:
            # pattern: A, total pattern length, dash length, gap length (negative)
            # 定义一个划线长度为200，间隔长度为100的虚线
            self.doc.linetypes.add(
                name="DASHED",
                pattern=[300, 200, -100],
                description="Dashed line with 200 on, 100 off",
            )

        # 创建图层
        for config in self.ELEMENTS.values():
            self.doc.layers.new(
                name=config.layer_name, dxfattribs={"color": config.color_code, "lineweight": 30}
            )
            self.doc.layers.get(config.layer_name).lock()

        # 房间分区图层（黄色）
        self.doc.layers.new(name="ROOM", dxfattribs={"color": 2, "lineweight": 50})

        # 图例图层
        self.doc.layers.new(name="LEGEND", dxfattribs={"color": 7, "lineweight": 25})

        # 梁图层（金黄）
        self.doc.layers.new(name="BEAMS", dxfattribs={"color": 2, "lineweight": 50})

        # 轴线图层（浅灰色）
        self.doc.layers.new(name="AXES", dxfattribs={"color": 9, "lineweight": 5})
        # lock this layer
        self.doc.layers.get("AXES").lock()

    def extract_and_convert_elements(self):
        """提取并转换所有元素"""
        for element_type, config in self.ELEMENTS.items():
            try:
                # 获取轮廓
                mask = self.extractor.get_mask_by_hsv(element_type)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 转换每个轮廓
                for contour in contours:
                    self._create_polygon_from_contour(contour, config)

            except Exception as e:
                print(f"处理 {element_type} 时出错: {e}")

    def add_axis_lines(self, axis_info_json: str):
        with open(axis_info_json) as f:
            axis_info = json.load(f)[os.path.basename(self.img_path)]
        hor_axes, ver_axes = axis_info["vertical_axes"], axis_info["horizontal_axes"]
        min_x, max_x = min(ver_axes), max(ver_axes)
        min_y, max_y = min(hor_axes), max(hor_axes)

        for x in ver_axes:
            self._create_line_from_points((x, min_y), (x, max_y), layer="AXES", color=256)
        for y in hor_axes:
            self._create_line_from_points((min_x, y), (max_x, y), layer="AXES", color=256)

    def _create_polygon_from_contour(self, contour: np.ndarray, config: ElementConfig):
        """从轮廓创建DXF多边形"""
        # 简化轮廓
        epsilon = 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            return

        # 转换坐标系
        world_points = self._pixel_to_world(points)

        # 创建多段线
        self.msp.add_lwpolyline(
            world_points,
            close=True,
            dxfattribs={
                "color": config.color_code,
                "layer": config.layer_name,
                "linetype": "CONTINUOUS",
            },
        )

    def _create_line_from_points(
        self, p1: Tuple[float, float], p2: Tuple[float, float], layer: str, color: int
    ):
        """从两点创建DXF虚线直线"""
        p1, p2 = np.array([p1, p2])
        self.msp.add_line(
            p1,
            p2,
            dxfattribs={
                "color": color,
                "layer": layer,
                "linetype": "DASHED",
            },
        )

    def _pixel_to_world(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """像素坐标转世界坐标"""
        img_height = self.extractor.image.shape[0]
        return [(p[0] * self.scale, (img_height - p[1]) * self.scale) for p in points]

    def add_legend(self):
        """添加图例"""
        # 图像边界
        img_height, img_width = self.extractor.image.shape[:2]
        max_x, max_y = img_width * self.scale, img_height * self.scale

        # 图例位置
        x = max_x + self.LEGEND_CONFIG["margin"]
        y = max_y - self.LEGEND_CONFIG["margin"]

        # 标题
        title = self.msp.add_text(
            "构件图例 / LEGEND",
            dxfattribs={"layer": "LEGEND", "height": self.LEGEND_CONFIG["title_height"], "style": "SimSun"},
        )
        title.set_placement((x, y), align=TextEntityAlignment.LEFT)

        # 各元素图例
        current_y = y - self.LEGEND_CONFIG["spacing"] * 1.5

        for element_type, config in self.ELEMENTS.items():
            # 彩色矩形
            rect_points = [
                (x, current_y),
                (x + self.LEGEND_CONFIG["rect_width"], current_y),
                (x + self.LEGEND_CONFIG["rect_width"], current_y - self.LEGEND_CONFIG["rect_height"]),
                (x, current_y - self.LEGEND_CONFIG["rect_height"]),
            ]

            # 填充矩形
            hatch = self.msp.add_hatch(color=config.color_code)
            hatch.dxf.layer = "LEGEND"
            hatch.paths.add_polyline_path(rect_points, is_closed=True)

            # 矩形边框
            self.msp.add_lwpolyline(rect_points, close=True, dxfattribs={"color": 7, "layer": "LEGEND"})

            # 文字标签
            text_x = x + self.LEGEND_CONFIG["rect_width"] + 200
            text_y = current_y - self.LEGEND_CONFIG["rect_height"] / 2 - 150

            text = self.msp.add_text(
                self.ELEMENT_NAMES[element_type],
                dxfattribs={
                    "layer": "LEGEND",
                    "height": self.LEGEND_CONFIG["text_height"],
                    "style": "SimSun",
                },
            )
            text.set_placement((text_x, text_y), align=TextEntityAlignment.LEFT)

            current_y -= self.LEGEND_CONFIG["spacing"]

    def set_viewport(self):
        """设置默认视图"""
        try:
            img_height, img_width = self.extractor.image.shape[:2]
            max_x, max_y = img_width * self.scale, img_height * self.scale

            center_x, center_y = max_x / 4, max_y / 2
            view_height = max(max_x, max_y) * 1.1

            self.doc.set_modelspace_vport(height=view_height, center=(center_x, center_y))

        except Exception as e:
            print(f"设置视图失败: {e}")

    def convert(self, axis_info_json: str, output_path: str):
        """执行转换"""
        self.setup_dxf()
        self.extract_and_convert_elements()
        self.add_axis_lines(axis_info_json)
        self.add_legend()
        self.set_viewport()
        self.doc.saveas(output_path)


def convert_single_image(args):
    """单图转换函数"""
    img_path, scale, axis_info_json, output_dir = args
    try:
        converter = PNG2DXFConverter(img_path, scale)
        output_file = Path(output_dir) / f"{Path(img_path).stem}.dxf"
        converter.convert(axis_info_json, str(output_file))
        return True
    except Exception as e:
        print(f"转换失败 {img_path}: {e}")
        return False


def convert_batch(
    img_paths: List[str],
    scale_map: Dict,
    axis_info_json: str,
    output_dir: str = "output",
    num_processes: Optional[int] = None,
):
    """批量转换"""
    import multiprocessing as mp

    from tqdm import tqdm

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 进程数
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(img_paths))

    print(f"批量转换: {len(img_paths)} 张图像, {num_processes} 个进程")

    # 准备参数
    args_list = [
        (img_path, scale_map[os.path.basename(img_path)]["scale"], axis_info_json, output_dir)
        for img_path in img_paths
    ]
    start_time = time.time()

    # 多进程转换
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    convert_single_image, args_list, chunksize=max(1, len(img_paths) // num_processes // 2)
                ),
                total=len(img_paths),
                desc="转换进度",
                unit="张",
            )
        )

    # 统计结果
    success_count = sum(results)
    elapsed_time = time.time() - start_time

    print(f"✅ 转换完成: {success_count}/{len(img_paths)} 成功")
    print(f"⏱️ 用时: {elapsed_time:.2f}秒, 速度: {len(img_paths)/elapsed_time:.1f} 张/秒")


def turn_on_global_property():
    dxf_dir = "dxf/to_process/room_raw"
    for root, _, files in os.walk(dxf_dir):
        for file in files:
            if file.lower().endswith(".dxf"):
                dxf_path = os.path.join(root, file)
                doc = ezdxf.readfile(dxf_path)
                doc.header["$LWDISPLAY"] = 1
                appsettings.set_current_layer(doc, "ROOM")
                doc.saveas(dxf_path)
                # print(f"已开启线宽显示: {dxf_path}")


def main():
    """主函数"""
    from pngtool.image_clustering import ImageClusteringAnalyzer

    # # 转换单张做测试
    # axis_info_json = r"E:\Common\Desktop\Research\deepLearning\codes\AxisEngine\data\batch_parsed_axes.json"
    # img_path = "imgs/raw/L1L28_30.png"
    # converter = PNG2DXFConverter(img_path, scale=75.0)
    # converter.convert(axis_info_json, "test2.dxf")

    turn_on_global_property()
    return
    # 批量转换
    unique_img_dir = "imgs/unique_imgs"
    image_paths = [
        os.path.join(unique_img_dir, f) for f in os.listdir(unique_img_dir) if f.lower().endswith(".png")
    ]

    with open("imgs/shear_wall_axes_opening.json") as f:
        scale_map = json.load(f)

    convert_batch(
        image_paths, scale_map=scale_map, axis_info_json=axis_info_json, output_dir="dxf/to_process/room_raw"
    )


if __name__ == "__main__":
    main()
