import numpy as np


def bilinear_interpolate(image, x, y, line_type="horizontal"):
    """
    对给定坐标点进行选择性插值
    line_type: 'horizontal' 或 'vertical'，指定线段类型
    """
    h, w = image.shape[:2]

    # 检查是否为整数坐标
    if isinstance(x, int) and isinstance(y, int):
        # 确保坐标在图像范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return image[y, x]

    # 获取周围的整数坐标点
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = x1 + 1, y1 + 1

    # 确保坐标在图像范围内
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if line_type == "horizontal":
        # 水平线段只在y方向插值
        if isinstance(x, int):
            x_int = int(x)
            wy2 = y - y1
            wy1 = 1 - wy2
            # 反转权重
            wy1, wy2 = wy2, wy1
            return (wy1 * image[y1, x_int] + wy2 * image[y2, x_int]).astype(np.uint8)
    else:  # vertical
        # 竖直线段只在x方向插值
        if isinstance(y, int):
            y_int = int(y)
            wx2 = x - x1
            wx1 = 1 - wx2
            # 反转权重
            wx1, wx2 = wx2, wx1
            return (wx1 * image[y_int, x1] + wx2 * image[y_int, x2]).astype(np.uint8)

    # 如果不满足上述条件，返回最近的像素值
    x_nearest = int(round(x))
    y_nearest = int(round(y))
    x_nearest = max(0, min(x_nearest, w - 1))
    y_nearest = max(0, min(y_nearest, h - 1))
    return image[y_nearest, x_nearest]


def analyze_line_colors(horizontal_lines, vertical_lines, image):
    """
    分析近似水平和近似竖直线段的颜色
    """

    def get_color(pixel):
        # 标准颜色值 (BGR格式)
        RED = np.array([0, 0, 255])  # 红色 RGB(255,0,0)
        BLUE = np.array([255, 0, 0])  # 蓝色 RGB(0,0,255)
        GREEN = np.array([0, 255, 0])  # 绿色 RGB(0,255,0)
        GRAY = np.array([152, 152, 152])  # 灰色 RGB(152,152,152)

        # 计算当前像素到四种标准颜色的欧氏距离
        dist_to_red = np.linalg.norm(pixel - RED)
        dist_to_blue = np.linalg.norm(pixel - BLUE)
        dist_to_green = np.linalg.norm(pixel - GREEN)
        dist_to_gray = np.linalg.norm(pixel - GRAY)

        # 找到距离最小的颜色
        distances = {"red": dist_to_red, "blue": dist_to_blue, "green": dist_to_green, "gray": dist_to_gray}

        # 返回距离最近的颜色
        return min(distances, key=distances.get)

    def analyze_horizontal_line(start, end):
        """处理近似水平的线段"""
        x_start, x_end = min(start[0], end[0]), max(start[0], end[0])
        total_x_dist = x_end - x_start
        if total_x_dist == 0:
            return []

        y_slope = (end[1] - start[1]) / total_x_dist
        MIN_SEGMENT_LENGTH = 5

        # 存储整条线段上所有的颜色信息
        color_sequence = []
        color_positions = []

        # 首先收集整条线段的所有颜色信息
        for x in range(int(x_start), int(x_end) + 1):
            y = start[1] + y_slope * (x - x_start)
            pixel = bilinear_interpolate(image, x, y, line_type="horizontal")
            color = get_color(pixel)

            if not color_sequence or color != color_sequence[-1]:
                color_sequence.append(color)
                color_positions.append(x)

        # 如果整条线段只有一种颜色，直接返回
        if len(color_sequence) == 1:
            return [((x_start, start[1]), (x_end, end[1]), color_sequence[0])]

        # 先进行初始切分
        initial_segments = []
        color_positions.append(x_end)  # 添加终点位置

        for i in range(len(color_positions) - 1):
            segment_start = color_positions[i]
            segment_end = color_positions[i + 1]
            initial_segments.append(
                {
                    "start": (segment_start, start[1] + y_slope * (segment_start - x_start)),
                    "end": (segment_end, start[1] + y_slope * (segment_end - x_start)),
                    "color": color_sequence[i],
                    "length": segment_end - segment_start,
                }
            )

        # 处理短线段
        if len(initial_segments) == 2:
            # 情况1：只有两段
            if (
                initial_segments[0]["length"] < MIN_SEGMENT_LENGTH
                or initial_segments[1]["length"] < MIN_SEGMENT_LENGTH
            ):
                # 合并为一段，使用较长段的颜色
                dominant_color = (
                    initial_segments[0]["color"]
                    if initial_segments[0]["length"] > initial_segments[1]["length"]
                    else initial_segments[1]["color"]
                )
                return [((x_start, start[1]), (x_end, end[1]), dominant_color)]
        else:
            # 情况2：三段或以上
            i = 0
            while i < len(initial_segments):
                if initial_segments[i]["length"] < MIN_SEGMENT_LENGTH:
                    if i == 0:
                        # 开头的短线段，合并到下一段
                        initial_segments[i + 1]["start"] = initial_segments[i]["start"]
                        initial_segments[i + 1]["length"] = (
                            initial_segments[i + 1]["end"][0] - initial_segments[i + 1]["start"][0]
                        )
                        initial_segments.pop(i)
                        continue
                    elif i == len(initial_segments) - 1:
                        # 结尾的短线段，合并到上一段
                        initial_segments[i - 1]["end"] = initial_segments[i]["end"]
                        initial_segments[i - 1]["length"] = (
                            initial_segments[i - 1]["end"][0] - initial_segments[i - 1]["start"][0]
                        )
                        initial_segments.pop(i)
                    else:
                        # 中间的短线段，与前后两段合并成一条
                        # 找出三段中最长的一段的颜色
                        lengths = [
                            initial_segments[i - 1]["length"],
                            initial_segments[i]["length"],
                            initial_segments[i + 1]["length"],
                        ]
                        max_length_idx = lengths.index(max(lengths))
                        dominant_color = initial_segments[i - 1 + max_length_idx]["color"]

                        # 合并三段
                        merged_segment = {
                            "start": initial_segments[i - 1]["start"],
                            "end": initial_segments[i + 1]["end"],
                            "color": dominant_color,
                            "length": initial_segments[i - 1]["length"]
                            + initial_segments[i]["length"]
                            + initial_segments[i + 1]["length"],
                        }

                        # 删除原来的三段，插入合并后的新段
                        initial_segments[i - 1 : i + 2] = [merged_segment]
                        continue
                i += 1

        # 转换回原始格式
        final_segments = [
            (segment["start"], segment["end"], segment["color"]) for segment in initial_segments
        ]
        return final_segments

    def analyze_vertical_line(start, end):
        """处理近似竖直的线段"""
        y_start, y_end = min(start[1], end[1]), max(start[1], end[1])
        total_y_dist = y_end - y_start
        if total_y_dist == 0:
            return []

        x_slope = (end[0] - start[0]) / total_y_dist
        MIN_SEGMENT_LENGTH = 5

        # 存储整条线段上所有的颜色信息
        color_sequence = []
        color_positions = []

        # 首先收集整条线段的所有颜色信息
        for y in range(int(y_start), int(y_end) + 1):
            x = start[0] + x_slope * (y - y_start)
            pixel = bilinear_interpolate(image, x, y, line_type="vertical")
            color = get_color(pixel)

            if not color_sequence or color != color_sequence[-1]:
                color_sequence.append(color)
                color_positions.append(y)

        # 如果整条线段只有一种颜色，直接返回
        if len(color_sequence) == 1:
            return [((start[0], y_start), (end[0], y_end), color_sequence[0])]

        # 先进行初始切分
        initial_segments = []
        color_positions.append(y_end)  # 添加终点位置

        for i in range(len(color_positions) - 1):
            segment_start = color_positions[i]
            segment_end = color_positions[i + 1]
            initial_segments.append(
                {
                    "start": (start[0] + x_slope * (segment_start - y_start), segment_start),
                    "end": (start[0] + x_slope * (segment_end - y_start), segment_end),
                    "color": color_sequence[i],
                    "length": segment_end - segment_start,
                }
            )

        # 处理短线段
        if len(initial_segments) == 2:
            # 情况1：只有两段
            if (
                initial_segments[0]["length"] < MIN_SEGMENT_LENGTH
                or initial_segments[1]["length"] < MIN_SEGMENT_LENGTH
            ):
                # 合并为一段，使用较长段的颜色
                dominant_color = (
                    initial_segments[0]["color"]
                    if initial_segments[0]["length"] > initial_segments[1]["length"]
                    else initial_segments[1]["color"]
                )
                return [((start[0], y_start), (end[0], y_end), dominant_color)]
        else:
            # 情况2：三段或以上
            i = 0
            while i < len(initial_segments):
                if initial_segments[i]["length"] < MIN_SEGMENT_LENGTH:
                    if i == 0:
                        # 开头的短线段，合并到下一段
                        initial_segments[i + 1]["start"] = initial_segments[i]["start"]
                        initial_segments[i + 1]["length"] = (
                            initial_segments[i + 1]["end"][1] - initial_segments[i + 1]["start"][1]
                        )
                        initial_segments.pop(i)
                        continue
                    elif i == len(initial_segments) - 1:
                        # 结尾的短线段，合并到上一段
                        initial_segments[i - 1]["end"] = initial_segments[i]["end"]
                        initial_segments[i - 1]["length"] = (
                            initial_segments[i - 1]["end"][1] - initial_segments[i - 1]["start"][1]
                        )
                        initial_segments.pop(i)
                    else:
                        # 中间的短线段，与前后两段合并成一条
                        # 找出三段中最长的一段的颜色
                        lengths = [
                            initial_segments[i - 1]["length"],
                            initial_segments[i]["length"],
                            initial_segments[i + 1]["length"],
                        ]
                        max_length_idx = lengths.index(max(lengths))
                        dominant_color = initial_segments[i - 1 + max_length_idx]["color"]

                        # 合并三段
                        merged_segment = {
                            "start": initial_segments[i - 1]["start"],
                            "end": initial_segments[i + 1]["end"],
                            "color": dominant_color,
                            "length": initial_segments[i - 1]["length"]
                            + initial_segments[i]["length"]
                            + initial_segments[i + 1]["length"],
                        }

                        # 删除原来的三段，插入合并后的新段
                        initial_segments[i - 1 : i + 2] = [merged_segment]
                        continue
                i += 1

        # 转换回原始格式
        final_segments = [
            (segment["start"], segment["end"], segment["color"]) for segment in initial_segments
        ]
        return final_segments

    # 分析所有线段
    horizontal_segments = []
    vertical_segments = []

    # 处理近似水平线段
    for start, end in horizontal_lines:
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dx > dy:  # 确认是近似水平线
            horizontal_segments.extend(analyze_horizontal_line(start, end))
        else:  # 如果实际上更接近竖直线，则用竖直线处理方法
            vertical_segments.extend(analyze_vertical_line(start, end))

    # 处理近似竖直线段
    for start, end in vertical_lines:
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dy > dx:  # 确认是近似竖直线
            vertical_segments.extend(analyze_vertical_line(start, end))
        else:  # 如果实际上更接近水平线，则用水平线处理方法
            horizontal_segments.extend(analyze_horizontal_line(start, end))

    # 将线段按颜色分类
    red_lines = []
    blue_lines = []
    green_lines = []
    gray_lines = []

    for start, end, color in horizontal_segments + vertical_segments:
        if color == "red":
            red_lines.append((start, end))
        elif color == "blue":
            blue_lines.append((start, end))
        elif color == "green":
            green_lines.append((start, end))
        elif color == "gray":
            gray_lines.append((start, end))

    return red_lines, blue_lines, green_lines, gray_lines
