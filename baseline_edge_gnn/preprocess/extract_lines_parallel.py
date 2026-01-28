import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from .analyze_color2 import analyze_line_colors


def extract_lines(image_path, save_path):
    # 读取和预处理图像
    image = cv2.imread(image_path)
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
    # # 保存骨架化图像
    # cv2.imwrite(
    #     os.path.join(save_path, f'{os.path.basename(image_path).replace(".png", "")}_skeleton.png'), skeleton
    # )
    # cv2.imshow("skeleton", skeleton)
    # cv2.waitKey(0)

    return image, skeleton


def classify_lines_by_color(endpoints, original_image):
    red_lines = []
    blue_lines = []

    for start, end in endpoints:
        start_pixel = original_image[start[1], start[0]]
        if start_pixel[2] > 150 and start_pixel[0] < 100:  # 红色
            red_lines.append((start, end))
        elif start_pixel[0] > 150 and start_pixel[2] < 100:  # 蓝色
            blue_lines.append((start, end))

    return red_lines, blue_lines


# ... existing code ...
def find_endpoints(skeleton):
    """
    提取骨架图中水平和竖直线的端点坐标
    返回: horizontal_endpoints, vertical_endpoints
    每个endpoints列表中包含 [(start_point, end_point), ...]
    """
    height, width = skeleton.shape
    horizontal_endpoints = []
    vertical_endpoints = []

    # 遍历图像找到所有线段的起点
    visited = np.zeros_like(skeleton, dtype=bool)

    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 255 and not visited[y, x]:
                # 检查是水平线还是竖直线
                is_horizontal = False
                if y > 0 and y < height - 1:
                    if x < width - 1 and skeleton[y, x] == 255 and skeleton[y, x + 1] == 255:
                        is_horizontal = True

                if is_horizontal:
                    # 追踪水平线
                    start_x = x
                    curr_x = x
                    while curr_x < width and skeleton[y, curr_x] == 255:
                        visited[y, curr_x] = True
                        curr_x += 1
                    end_x = curr_x - 1
                    horizontal_endpoints.append(((start_x, y), (end_x, y)))
                else:
                    # 追踪竖直线
                    start_y = y
                    curr_y = y
                    while curr_y < height and skeleton[curr_y, x] == 255:
                        visited[curr_y, x] = True
                        curr_y += 1
                    end_y = curr_y - 1
                    vertical_endpoints.append(((x, start_y), (x, end_y)))

    return horizontal_endpoints, vertical_endpoints


def find_intersections(horizontal_endpoints, vertical_endpoints):
    """
    计算水平线和竖直线的交点，包括：
    1. 线段相交点
    2. 水平线段端点在竖直线上的情况(包括距离为1的情况)
    3. 竖直线段端点在水平线上的情况(包括距离为1的情况)
    """
    intersections = []

    for h_start, h_end in horizontal_endpoints:
        h_y = h_start[1]  # 水平线的y坐标
        h_x_min = min(h_start[0], h_end[0])
        h_x_max = max(h_start[0], h_end[0])

        for v_start, v_end in vertical_endpoints:
            v_x = v_start[0]  # 竖直线的x坐标
            v_y_min = min(v_start[1], v_end[1])
            v_y_max = max(v_start[1], v_end[1])

            # 检查线段相交
            if (h_x_min <= v_x <= h_x_max) and (v_y_min <= h_y <= v_y_max):
                intersections.append((v_x, h_y))

            # 检查水平线段的端点是否在竖直线上或距离为1
            for point in [h_start, h_end]:
                if v_y_min <= point[1] <= v_y_max:
                    if point[0] == v_x or abs(point[0] - v_x) == 1:
                        intersections.append((v_x, point[1]))

            # 检查竖直线段的端点是否在水平线上或距离为1
            for point in [v_start, v_end]:
                if h_x_min <= point[0] <= h_x_max:
                    if point[1] == h_y or abs(point[1] - h_y) == 1:
                        intersections.append((point[0], h_y))

    # 去除重复的交点
    return list(set(intersections))


def merge_close_horizontal_lines(horizontal_endpoints):
    """
    合并接近的水平线段，满足以下任一条件时合并：
    1. x和y坐标都在1像素以内
    2. x坐标在1像素内且y坐标在2像素内
    3. x坐标在2像素内且y坐标在1像素内
    """
    if not horizontal_endpoints:
        return []

    def lines_are_close(line1, line2):
        """判断两条线段是否接近"""
        # 获取两条线段的端点
        x1_start, x1_end = line1[0][0], line1[1][0]
        x2_start, x2_end = line2[0][0], line2[1][0]
        y1, y2 = line1[0][1], line2[0][1]

        # 计算x方向的最小距离
        x_dist = min(abs(x1_start - x2_end), abs(x1_end - x2_start))
        # 计算y方向的距离
        y_dist = abs(y1 - y2)

        # 三种情况的判断
        if x_dist <= 1 and y_dist <= 1:  # 情况1
            return True
        if x_dist <= 1 and y_dist <= 2:  # 情况2
            return True
        if x_dist <= 2 and y_dist <= 1:  # 情况3
            return True

        return False

    # 将所有需要合并的线段分组
    groups = []
    used = set()

    for i, line1 in enumerate(horizontal_endpoints):
        if i in used:
            continue

        current_group = [line1]
        used.add(i)

        # 检查其他所有线段
        for j, line2 in enumerate(horizontal_endpoints):
            if j in used:
                continue

            # 检查line2是否与当前组中的任何线段接近
            if any(lines_are_close(existing_line, line2) for existing_line in current_group):
                current_group.append(line2)
                used.add(j)

        groups.append(current_group)

    # 合并每个组中的线段
    merged_lines = []
    for group in groups:
        if len(group) == 1:
            merged_lines.append(group[0])
            continue

        # 找出最长的线段的y坐标
        longest_line = max(group, key=lambda x: abs(x[1][0] - x[0][0]))
        y_coord = longest_line[0][1]

        # 获取所有x坐标
        x_coords = []
        for line in group:
            x_coords.extend([line[0][0], line[1][0]])
        min_x = min(x_coords)
        max_x = max(x_coords)

        # 添加合并后的线段
        merged_lines.append(((min_x, y_coord), (max_x, y_coord)))

    return merged_lines


def merge_close_vertical_lines(vertical_endpoints):
    """
    合并接近的竖直线段，满足以下任一条件时合并：
    1. x和y坐标都在1像素以内
    2. x坐标在2像素内且y坐标在1像素内
    3. x坐标在1像素内且y坐标在2像素内
    """
    if not vertical_endpoints:
        return []

    def lines_are_close(line1, line2):
        """判断两条竖直线段是否接近"""
        # 获取两条线段的端点
        y1_start, y1_end = line1[0][1], line1[1][1]
        y2_start, y2_end = line2[0][1], line2[1][1]
        x1, x2 = line1[0][0], line2[0][0]

        # 计算y方向的最小距离
        y_dist = min(abs(y1_start - y2_end), abs(y1_end - y2_start))
        # 计算x方向的距离
        x_dist = abs(x1 - x2)

        # 三种情况的判断
        if x_dist <= 1 and y_dist <= 1:  # 情况1
            return True
        if x_dist <= 2 and y_dist <= 1:  # 情况2
            return True
        if x_dist <= 1 and y_dist <= 2:  # 情况3
            return True

        return False

    # 将所有需要合并的线段分组
    groups = []
    used = set()

    for i, line1 in enumerate(vertical_endpoints):
        if i in used:
            continue

        current_group = [line1]
        used.add(i)

        # 检查其他所有线段
        for j, line2 in enumerate(vertical_endpoints):
            if j in used:
                continue

            # 检查line2是否与当前组中的任何线段接近
            if any(lines_are_close(existing_line, line2) for existing_line in current_group):
                current_group.append(line2)
                used.add(j)

        groups.append(current_group)

    # 合并每个组中的线段
    merged_lines = []
    for group in groups:
        if len(group) == 1:
            merged_lines.append(group[0])
            continue

        # 找出最长的线段的x坐标
        longest_line = max(group, key=lambda x: abs(x[1][1] - x[0][1]))
        x_coord = longest_line[0][0]

        # 获取所有y坐标
        y_coords = []
        for line in group:
            y_coords.extend([line[0][1], line[1][1]])
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 添加合并后的线段
        merged_lines.append(((x_coord, min_y), (x_coord, max_y)))

    return merged_lines


def merge_close_endpoints(horizontal_endpoints, vertical_endpoints, filtered_intersections):
    """
    一次性合并所有接近的端点，优先使用交叉点，其次使用最长线段的端点
    """

    def get_all_points_with_lines(h_endpoints, v_endpoints):
        """获取所有端点及其所属的线段"""
        points_to_lines = {}
        for line in h_endpoints:
            start, end = line
            if start not in points_to_lines:
                points_to_lines[start] = []
            if end not in points_to_lines:
                points_to_lines[end] = []
            points_to_lines[start].append(line)
            points_to_lines[end].append(line)

        for line in v_endpoints:
            start, end = line
            if start not in points_to_lines:
                points_to_lines[start] = []
            if end not in points_to_lines:
                points_to_lines[end] = []
            points_to_lines[start].append(line)
            points_to_lines[end].append(line)

        return points_to_lines

    def points_are_close(p1, p2):
        """判断两点是否接近"""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])

        # 确保不是完全重合的点
        if dx == 0 and dy == 0:
            return False

        # 三种情况:
        # 1. x相距<=1且y相距<=2
        # 2. x相距<=2且y相距<=1
        # 3. x相距<=1且y相距<=1
        return (dx <= 1 and dy <= 2) or (dx <= 2 and dy <= 1) or (dx <= 1 and dy <= 1)

    def get_line_length(line):
        """计算线段长度"""
        start, end = line
        return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

    def find_longest_line_point(points, points_to_lines):
        """找出与给定点相关的最长线段的端点"""
        max_length = -1
        best_point = None

        for point in points:
            for line in points_to_lines[point]:
                length = get_line_length(line)
                if length > max_length:
                    max_length = length
                    best_point = point

        return best_point

    # 将交叉点转换为元组格式
    intersection_points = set((int(x), int(y)) for x, y in filtered_intersections)

    # 获取所有端点及其相关线段
    points_to_lines = get_all_points_with_lines(horizontal_endpoints, vertical_endpoints)
    all_points = list(points_to_lines.keys())

    point_groups = []
    used_points = set()

    # 对每个点进行分组
    for point in all_points:
        if point in used_points:
            continue

        # 找出所有与当前点接近的点
        current_group = {point}
        points_to_check = [point]

        while points_to_check:
            current_point = points_to_check.pop()

            for other_point in all_points:
                if other_point in used_points:
                    continue

                if other_point not in current_group and points_are_close(current_point, other_point):
                    current_group.add(other_point)
                    points_to_check.append(other_point)
                    used_points.add(other_point)

        used_points.add(point)
        if current_group:
            point_groups.append(current_group)

    # 创建点的映射关系
    point_mapping = {}
    for group in point_groups:
        # 检查组内是否有交叉点
        intersection_in_group = None
        for point in group:
            if point in intersection_points:
                intersection_in_group = point
                break

        # 选择新的坐标点：优先使用交叉点，其次使用最长线段的端点
        if intersection_in_group:
            new_point = intersection_in_group
        else:
            new_point = find_longest_line_point(group, points_to_lines)

        # 建立映射关系
        for point in group:
            point_mapping[point] = new_point

    # 更新线段端点
    updated_horizontal = []
    for start, end in horizontal_endpoints:
        new_start = point_mapping.get(start, start)
        new_end = point_mapping.get(end, end)
        updated_horizontal.append((new_start, new_end))

    updated_vertical = []
    for start, end in vertical_endpoints:
        new_start = point_mapping.get(start, start)
        new_end = point_mapping.get(end, end)
        updated_vertical.append((new_start, new_end))

    return updated_horizontal, updated_vertical


def split_lines_at_intersections(horizontal_endpoints, vertical_endpoints, intersections):
    """
    根据交点将线段切分成多个小线段
    """
    new_horizontal = []
    new_vertical = []

    # 将交点转换为整数坐标的集合
    intersection_points = set((int(x), int(y)) for x, y in intersections)

    # 处理水平线段
    for start, end in horizontal_endpoints:
        y = start[1]  # 水平线的y坐标
        x_start, x_end = min(start[0], end[0]), max(start[0], end[0])

        # 收集这条线上的所有交点
        line_intersections = []
        for x, y_int in intersection_points:
            if y_int == y and x_start <= x <= x_end:
                line_intersections.append(x)

        # 添加起点和终点
        line_intersections.append(x_start)
        line_intersections.append(x_end)
        line_intersections = sorted(set(line_intersections))  # 去重并排序

        # 创建子线段
        for i in range(len(line_intersections) - 1):
            x1, x2 = line_intersections[i], line_intersections[i + 1]
            if x1 != x2:  # 避免长度为0的线段
                new_horizontal.append(((x1, y), (x2, y)))

    # 处理竖直线段
    for start, end in vertical_endpoints:
        x = start[0]  # 竖直线的x坐标
        y_start, y_end = min(start[1], end[1]), max(start[1], end[1])

        # 收集这条线上的所有交点
        line_intersections = []
        for x_int, y in intersection_points:
            if x_int == x and y_start <= y <= y_end:
                line_intersections.append(y)

        # 添加起点和终点
        line_intersections.append(y_start)
        line_intersections.append(y_end)
        line_intersections = sorted(set(line_intersections))  # 去重并排序

        # 创建子线段
        for i in range(len(line_intersections) - 1):
            y1, y2 = line_intersections[i], line_intersections[i + 1]
            if y1 != y2:  # 避免长度为0的线段
                new_vertical.append(((x, y1), (x, y2)))

    return new_horizontal, new_vertical


def clean_lines(horizontal_endpoints, vertical_endpoints):
    """
    清理线段：
    1. 删除长度为0或太短的线段
    2. 删除重复的线段
    """

    def remove_duplicates(lines):
        # 将线段标准化（确保起点在左/上）
        normalized = []
        for start, end in lines:
            if isinstance(start, tuple) and isinstance(end, tuple):
                # 水平线
                if start[0] > end[0] or (start[0] == end[0] and start[1] > end[1]):
                    normalized.append((end, start))
                else:
                    normalized.append((start, end))

        # 使用集合去重
        return list(set(normalized))

    def remove_short_lines(lines, min_length=1):
        filtered = []
        for start, end in lines:
            length = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
            if length >= min_length:
                filtered.append((start, end))
        return filtered

    # 清理水平线段
    h_lines = remove_duplicates(horizontal_endpoints)
    h_lines = remove_short_lines(h_lines)

    # 清理竖直线段
    v_lines = remove_duplicates(vertical_endpoints)
    v_lines = remove_short_lines(v_lines)

    return h_lines, v_lines


def merge_close_points(
    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list, threshold=2
):
    """
    合并相距在threshold像素以内的端点，确保所有点坐标都是整数
    """
    # 将点列表转换为numpy数组便于计算
    points = np.array(points_list)
    merge_groups = []
    processed = set()

    # 寻找需要合并的点组
    for i in range(len(points)):
        if i in processed:
            continue

        current_group = [i]
        for j in range(i + 1, len(points)):
            if j in processed:
                continue

            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if dist <= threshold:
                current_group.append(j)

        if len(current_group) > 1:
            merge_groups.append(current_group)
            processed.update(current_group)

    # 创建点的映射关系：旧点 -> 新点
    point_mapping = {}
    for group in merge_groups:
        group_points = points[group]
        # 计算平均点并取整
        mean_x = int(round(np.mean(group_points[:, 0])))
        mean_y = int(round(np.mean(group_points[:, 1])))
        mean_point = [mean_x, mean_y]

        # 记录每个旧点到新点的映射
        for idx in group:
            old_point = tuple(map(int, points[idx]))  # 确保旧点也是整数
            point_mapping[old_point] = mean_point

    # 更新所有点的坐标为整数
    new_points_set = set()
    for point in points_list:
        point_tuple = tuple(map(int, point))
        if point_tuple in point_mapping:
            new_point = point_mapping[point_tuple]
        else:
            new_point = [int(round(point[0])), int(round(point[1]))]
        new_points_set.add(tuple(new_point))

    # 更新线段端点，确保使用相同的整数坐标
    for lines_list in [red_lines_list, blue_lines_list, green_lines_list, gray_lines_list]:
        for line in lines_list:
            for i in range(2):
                point = tuple(map(int, map(round, line[i])))  # 先round再转为整数
                if point in point_mapping:
                    line[i] = point_mapping[point]
                else:
                    line[i] = [int(round(line[i][0])), int(round(line[i][1]))]

    # 将集合转换为列表
    return (
        [list(point) for point in new_points_set],
        red_lines_list,
        blue_lines_list,
        green_lines_list,
        gray_lines_list,
    )


def line_and_points_deduplication(red_lines, blue_lines, green_lines, gray_lines):
    # 对所有颜色线段去重
    red_lines = list(set(red_lines))
    blue_lines = list(set(blue_lines))
    green_lines = list(set(green_lines))
    gray_lines = list(set(gray_lines))

    # 收集所有端点并去重
    all_points = set()
    for start, end in red_lines + blue_lines + green_lines + gray_lines:
        all_points.add(start)
        all_points.add(end)

    return red_lines, blue_lines, green_lines, gray_lines, all_points


def merge_aligned_lines(points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list):
    """
    合并同向的线段，分别处理各种颜色的线段，
    但如果一个点同时连接了不同颜色的线段，则保留该点
    """

    def get_connected_lines(point, lines):
        """获取与指定点相连的所有线段及其索引"""
        connected = []
        point = tuple(point)  # 确保point是元组
        for i, line in enumerate(lines):
            # 将线段端点转换为元组进行比较
            start = tuple(line[0])
            end = tuple(line[1])
            if point == start or point == end:
                connected.append((i, line))
        return connected

    def is_same_direction(line1, line2, max_angle_diff=20):
        """判断两条线段是否同向（允许少许误差）"""
        # 计算两条线段的方向向量
        vec1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])
        vec2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])

        # 归一化向量
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # 计算夹角（弧度）
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        angle_deg = np.degrees(angle)

        return angle_deg <= max_angle_diff or angle_deg >= 180 - max_angle_diff

    def get_points_from_lines(lines):
        """获取所有线段的端点集合"""
        points = set()
        for line in lines:
            points.add(tuple(line[0]))
            points.add(tuple(line[1]))
        return points

    # 获取所有颜色线段的端点
    red_points = get_points_from_lines(red_lines_list)
    blue_points = get_points_from_lines(blue_lines_list)
    green_points = get_points_from_lines(green_lines_list)
    gray_points = get_points_from_lines(gray_lines_list)

    # 找出同时连接不同颜色线段的点（任意两种颜色的交集）
    all_color_points = [red_points, blue_points, green_points, gray_points]
    shared_points = set()
    for i in range(len(all_color_points)):
        for j in range(i + 1, len(all_color_points)):
            shared_points.update(all_color_points[i].intersection(all_color_points[j]))

    def merge_color_lines(lines, other_color_points):
        """合并同色同向线段，但避开与其他颜色相连的点"""
        points_to_remove = set()
        merged_lines = lines.copy()

        # 收集所有点
        all_points = set()
        for line in merged_lines:
            all_points.add(tuple(line[0]))
            all_points.add(tuple(line[1]))

        # 持续合并直到没有可以合并的线段
        while True:
            merged_any = False

            # 检查每个点
            for point in list(all_points):
                # 如果这个点同时连接了两种颜色的线段，跳过它
                if point in shared_points:
                    continue

                # 获取与当前点相连的所有线段
                connected_lines = get_connected_lines(point, merged_lines)
                # 如果只有两条线段连接到这个点
                if len(connected_lines) == 2:
                    line1_idx, line1 = connected_lines[0]
                    line2_idx, line2 = connected_lines[1]

                    # 检查这两条线段是否同向
                    if is_same_direction(line1, line2):
                        # 获取合并后的新线段的端点
                        all_line_points = [p for p in line1 + line2 if tuple(p) != point]
                        new_line = [all_line_points[0], all_line_points[-1]]

                        # 删除原来的两条线段，添加新的线段
                        if line1_idx > line2_idx:
                            del merged_lines[line1_idx]
                            del merged_lines[line2_idx]
                        else:
                            del merged_lines[line2_idx]
                            del merged_lines[line1_idx]
                        merged_lines.append(new_line)

                        # 记录要删除的点并更新点集
                        points_to_remove.add(point)
                        all_points.remove(point)
                        merged_any = True
                        break

            # 如果没有合并任何线段，说明处理完成
            if not merged_any:
                break

        return merged_lines, points_to_remove

    # 分别处理各种颜色的线段
    other_colors_points = blue_points.union(green_points).union(gray_points)
    new_red_lines, red_points_to_remove = merge_color_lines(red_lines_list, other_colors_points)

    other_colors_points = red_points.union(green_points).union(gray_points)
    new_blue_lines, blue_points_to_remove = merge_color_lines(blue_lines_list, other_colors_points)

    other_colors_points = red_points.union(blue_points).union(gray_points)
    new_green_lines, green_points_to_remove = merge_color_lines(green_lines_list, other_colors_points)

    other_colors_points = red_points.union(blue_points).union(green_points)
    new_gray_lines, gray_points_to_remove = merge_color_lines(gray_lines_list, other_colors_points)

    # 更新点列表，移除不需要的点，但保留共享点
    points_to_remove = (
        red_points_to_remove.union(blue_points_to_remove)
        .union(green_points_to_remove)
        .union(gray_points_to_remove)
    ) - shared_points
    new_points = [point for point in points_list if tuple(point) not in points_to_remove]

    return new_points, new_red_lines, new_blue_lines, new_green_lines, new_gray_lines


def remove_isolated_blue_lines(
    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
):
    """
    删除游离的蓝色线段（两个端点都没有与其他线段相连的蓝色线段）
    """
    # 创建一个字典来记录每个点连接的线段数量
    point_connections = {}

    # 初始化所有点的连接数为0
    for point in points_list:
        point_tuple = tuple(point)
        point_connections[point_tuple] = 0

    # 统计每个点连接的线段数量
    for line in red_lines_list + blue_lines_list + green_lines_list + gray_lines_list:
        start_tuple = tuple(line[0])
        end_tuple = tuple(line[1])
        point_connections[start_tuple] = point_connections.get(start_tuple, 0) + 1
        point_connections[end_tuple] = point_connections.get(end_tuple, 0) + 1

    # 找出需要删除的蓝色线段和对应的点
    lines_to_remove = []
    points_to_remove = set()

    for i, line in enumerate(blue_lines_list):
        start_tuple = tuple(line[0])
        end_tuple = tuple(line[1])

        # 如果线段的两个端点都只连接了这一条线
        if point_connections[start_tuple] == 1 and point_connections[end_tuple] == 1:
            lines_to_remove.append(i)
            points_to_remove.add(start_tuple)
            points_to_remove.add(end_tuple)

    # 删除游离的蓝色线段
    for index in reversed(lines_to_remove):
        del blue_lines_list[index]

    # 删除孤立的点
    points_list = [point for point in points_list if tuple(point) not in points_to_remove]

    return points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list


def remove_short_single_connected_lines(
    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list, min_length=3
):
    """
    删除短的单边连接线段（长度小于等于min_length且只有一端与其他线相连）
    """

    def get_line_length(line):
        """计算线段长度"""
        start, end = line
        return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    def get_point_connections(point, all_lines):
        """获取一个点连接的线段数量"""
        count = 0
        point = tuple(point)
        for line in all_lines:
            if tuple(line[0]) == point or tuple(line[1]) == point:
                count += 1
        return count

    # 所有颜色的线段列表
    all_color_lines = [red_lines_list, blue_lines_list, green_lines_list, gray_lines_list]

    # 处理每种颜色的线段
    for color_lines in all_color_lines:
        lines_to_remove = []
        for i, line in enumerate(color_lines):
            if get_line_length(line) <= min_length:
                # 检查两个端点的连接数
                all_lines = red_lines_list + blue_lines_list + green_lines_list + gray_lines_list
                start_connections = get_point_connections(line[0], all_lines)
                end_connections = get_point_connections(line[1], all_lines)

                # 如果其中一个端点只有一个连接（即当前线段），而另一个端点有多个连接
                if (start_connections == 1 and end_connections > 1) or (
                    end_connections == 1 and start_connections > 1
                ):
                    lines_to_remove.append(i)

        # 删除符合条件的线段
        for i in reversed(lines_to_remove):
            del color_lines[i]

    # 更新点列表，删除不再连接任何线段的点
    used_points = set()
    for line in red_lines_list + blue_lines_list + green_lines_list + gray_lines_list:
        used_points.add(tuple(line[0]))
        used_points.add(tuple(line[1]))

    points_list = [point for point in points_list if tuple(point) in used_points]

    return points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list


def straighten_lines_and_points(
    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list, angle_threshold=20
):
    """
    将所有线段调整为完全水平或竖直，同时保持交点关系

    Args:
        points_list: 所有点的列表
        red_lines_list: 红色线段列表
        blue_lines_list: 蓝色线段列表
        green_lines_list: 绿色线段列表
        gray_lines_list: 灰色线段列表
        angle_threshold: 判断线段是水平还是竖直的角度阈值（度）
    """

    def calculate_angle(line):
        """计算线段与水平线的夹角（0-90度）"""
        start, end = line
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return min(angle, 180 - angle) if angle > 90 else angle

    def is_nearly_horizontal(line):
        """判断线段是否接近水平"""
        return calculate_angle(line) < angle_threshold

    def is_nearly_vertical(line):
        """判断线段是否接近竖直"""
        return abs(90 - calculate_angle(line)) < angle_threshold

    def get_connected_points(point, all_lines):
        """获取与指定点相连的所有点"""
        connected = set()
        point_tuple = tuple(point)
        for line in all_lines:
            start_tuple = tuple(line[0])
            end_tuple = tuple(line[1])
            if point_tuple == start_tuple:
                connected.add(end_tuple)
            elif point_tuple == end_tuple:
                connected.add(start_tuple)
        return connected

    # 将所有点转换为numpy数组以便计算
    points_array = np.array(points_list)
    all_lines = red_lines_list + blue_lines_list + green_lines_list + gray_lines_list

    # 创建点的映射字典
    point_mapping = {tuple(point): i for i, point in enumerate(points_list)}

    # 迭代调整直到收敛
    max_iterations = 100
    for iteration in range(max_iterations):
        changed = False

        # 处理每条线段
        for lines in [red_lines_list, blue_lines_list, green_lines_list, gray_lines_list]:
            for line in lines:
                start_idx = point_mapping[tuple(line[0])]
                end_idx = point_mapping[tuple(line[1])]

                if is_nearly_horizontal(line):
                    # 将线段调整为完全水平
                    avg_y = (points_array[start_idx][1] + points_array[end_idx][1]) / 2
                    old_start_y = points_array[start_idx][1]
                    old_end_y = points_array[end_idx][1]

                    points_array[start_idx][1] = avg_y
                    points_array[end_idx][1] = avg_y

                    if abs(old_start_y - avg_y) > 0.1 or abs(old_end_y - avg_y) > 0.1:
                        changed = True

                elif is_nearly_vertical(line):
                    # 将线段调整为完全竖直
                    avg_x = (points_array[start_idx][0] + points_array[end_idx][0]) / 2
                    old_start_x = points_array[start_idx][0]
                    old_end_x = points_array[end_idx][0]

                    points_array[start_idx][0] = avg_x
                    points_array[end_idx][0] = avg_x

                    if abs(old_start_x - avg_x) > 0.1 or abs(old_end_x - avg_x) > 0.1:
                        changed = True

        # 如果没有发生变化，说明已经收敛
        if not changed:
            break

    # 更新所有点的坐标
    points_list = points_array.tolist()

    # 更新所有线段中的点坐标
    for lines in [red_lines_list, blue_lines_list, green_lines_list, gray_lines_list]:
        for line in lines:
            start_idx = point_mapping[tuple(line[0])]
            end_idx = point_mapping[tuple(line[1])]
            line[0] = points_list[start_idx]
            line[1] = points_list[end_idx]

    return points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理图像的起始和结束ID")
    parser.add_argument("--start_id", type=int, default=0, help="起始ID")
    parser.add_argument("--end_id", type=int, default=100, help="结束ID")
    args = parser.parse_args()

    image_dir = "0_datasets\\L1_7\\test_B"
    save_path = "processed\\L1_7\\test_B\\output_axis"
    os.makedirs(save_path, exist_ok=True)

    # 获取所有图像文件名并排序
    image_names = sorted(os.listdir(image_dir))

    # 根据起始和结束ID筛选要处理的图像
    start_idx = args.start_id
    end_idx = min(args.end_id, len(image_names))
    selected_images = image_names[start_idx:end_idx]

    for image_name in tqdm(selected_images, desc=f"处理图像 {start_idx}-{end_idx}", unit="图像"):
        try:
            image_path = os.path.join(image_dir, image_name)

            # 1. 提取骨架线
            image, skeleton = extract_lines(image_path, save_path)

            # 2. 获取端点坐标
            horizontal_endpoints, vertical_endpoints = find_endpoints(skeleton)
            # 3. 合并接近的水平线段
            horizontal_endpoints = merge_close_horizontal_lines(horizontal_endpoints)
            # 4. 合并接近的竖直线段
            vertical_endpoints = merge_close_vertical_lines(vertical_endpoints)

            # 5. 计算交点
            intersections = find_intersections(horizontal_endpoints, vertical_endpoints)

            # 6. 在获取交点之后，切分线段
            horizontal_endpoints, vertical_endpoints = split_lines_at_intersections(
                horizontal_endpoints, vertical_endpoints, intersections
            )

            # print(f"切分后的水平线段数量：{len(horizontal_endpoints)}")
            # print(f"切分后的竖直线段数量：{len(vertical_endpoints)}")

            # 7. 收集所有端点坐标
            all_endpoints = set()
            for start, end in horizontal_endpoints:
                all_endpoints.add(start)
                all_endpoints.add(end)
            for start, end in vertical_endpoints:
                all_endpoints.add(start)
                all_endpoints.add(end)

            # print(f"原始交点数量：{len(intersections)}")
            # print(f"水平线段数量：{len(horizontal_endpoints)}")
            # print(f"竖直线段数量：{len(vertical_endpoints)}")

            # 8. 合并端点
            updated_horizontal, updated_vertical = merge_close_endpoints(
                horizontal_endpoints, vertical_endpoints, intersections
            )

            # 9. 在合并端点后添加清理步骤
            updated_horizontal, updated_vertical = clean_lines(updated_horizontal, updated_vertical)
            all_merged_endpoints = set()
            for start, end in updated_horizontal + updated_vertical:
                all_merged_endpoints.add(start)
                all_merged_endpoints.add(end)

            # print(f"第一次合并后的端点数量：{len(all_merged_endpoints)}")
            # print(f"第一次清理后的水平线段数量：{len(updated_horizontal)}")
            # print(f"第一次清理后的竖直线段数量：{len(updated_vertical)}")

            # 10. 分析线段颜色
            red_lines, blue_lines, green_lines, gray_lines = analyze_line_colors(
                updated_horizontal, updated_vertical, image
            )

            # 11. 去重
            red_lines, blue_lines, green_lines, gray_lines, all_points = line_and_points_deduplication(
                red_lines, blue_lines, green_lines, gray_lines
            )

            points_list = [[p[0], p[1]] for p in all_points]
            red_lines_list = [[[start[0], start[1]], [end[0], end[1]]] for start, end in red_lines]
            blue_lines_list = [[[start[0], start[1]], [end[0], end[1]]] for start, end in blue_lines]
            green_lines_list = [[[start[0], start[1]], [end[0], end[1]]] for start, end in green_lines]
            gray_lines_list = [[[start[0], start[1]], [end[0], end[1]]] for start, end in gray_lines]

            # 保存合并前的数量
            pre_merge_points = len(points_list)
            pre_merge_red_lines = len(red_lines_list)
            pre_merge_blue_lines = len(blue_lines_list)

            # 12. 执行端点合并
            points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list = (
                merge_close_points(
                    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
                )
            )

            # 13. 最后一次将错误分段的线段合并，蓝红分开
            pre_merge_points_count = len(points_list)
            pre_merge_red_lines_count = len(red_lines_list)
            pre_merge_blue_lines_count = len(blue_lines_list)

            points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list = (
                merge_aligned_lines(
                    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
                )
            )

            # print(f"第二次合并前的节点数量：{pre_merge_points_count} -> 合并后的节点数量：{len(points_list)}")
            # print(
            #     f"第二次合并前的红线数量：{pre_merge_red_lines_count} -> 合并后的红线数量：{len(red_lines_list)}"
            # )
            # print(
            #     f"第二次合并前的蓝线数量：{pre_merge_blue_lines_count} -> 合并后的蓝线数量：{len(blue_lines_list)}"
            # )

            # 14. 删除游离的蓝色线段
            pre_removal_points = len(points_list)
            pre_removal_blue_lines = len(blue_lines_list)

            points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list = (
                remove_isolated_blue_lines(
                    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
                )
            )

            # print(f"删除游离蓝线前的节点数量：{pre_removal_points} -> 删除后的节点数量：{len(points_list)}")
            # print(
            #     f"删除游离蓝线前的蓝线数量：{pre_removal_blue_lines} -> 删除后的蓝线数量：{len(blue_lines_list)}"
            # )

            # 15. 删除短的单边连接线段
            pre_removal_points = len(points_list)
            pre_removal_red_lines = len(red_lines_list)
            pre_removal_blue_lines = len(blue_lines_list)

            points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list = (
                remove_short_single_connected_lines(
                    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
                )
            )

            # print(f"删除短单边线前的节点数量：{pre_removal_points} -> 删除后的节点数量：{len(points_list)}")
            # print(
            #     f"删除短单边线前的红线数量：{pre_removal_red_lines} -> 删除后的红线数量：{len(red_lines_list)}"
            # )
            # print(
            #     f"删除短单边线前的蓝线数量：{pre_removal_blue_lines} -> 删除后的蓝线数量：{len(blue_lines_list)}"
            # )

            # 16. 直线化处理
            # print("开始直线化处理...")
            pre_straighten_points = len(points_list)
            points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list = (
                straighten_lines_and_points(
                    points_list, red_lines_list, blue_lines_list, green_lines_list, gray_lines_list
                )
            )
            # print(f"直线化处理完成，节点数量：{pre_straighten_points} -> {len(points_list)}\n")

            # 创建结果字典
            red_lines_list = [
                {
                    "StartPoint": {"X": line[0][0], "Y": line[0][1]},
                    "EndPoint": {"X": line[1][0], "Y": line[1][1]},
                }
                for line in red_lines_list
            ]
            blue_lines_list = [
                {
                    "StartPoint": {"X": line[0][0], "Y": line[0][1]},
                    "EndPoint": {"X": line[1][0], "Y": line[1][1]},
                }
                for line in blue_lines_list
            ]
            green_lines_list = [
                {
                    "StartPoint": {"X": line[0][0], "Y": line[0][1]},
                    "EndPoint": {"X": line[1][0], "Y": line[1][1]},
                }
                for line in green_lines_list
            ]
            gray_lines_list = [
                {
                    "StartPoint": {"X": line[0][0], "Y": line[0][1]},
                    "EndPoint": {"X": line[1][0], "Y": line[1][1]},
                }
                for line in gray_lines_list
            ]
            result_dict = {
                "points": points_list,
                "shear_wall": red_lines_list,  # 红色 - 剪力墙
                "beam": blue_lines_list,  # 蓝色 - 梁
                "green_lines": green_lines_list,  # 绿色线段
                "gray_lines": gray_lines_list,  # 灰色线段
            }

            # # 输出合并前后的对比
            # #print("\n=== 合并前后对比 ===")
            # #print(f"端点数量: {pre_merge_points} -> {len(points_list)} (减少了 {pre_merge_points - len(points_list)} 个)")
            # #print(f"红线数量: {pre_merge_red_lines} -> {len(red_lines_list)} (减少了 {pre_merge_red_lines - len(red_lines_list)} 条)")
            # #print(f"蓝线数量: {pre_merge_blue_lines} -> {len(blue_lines_list)} (减少了 {pre_merge_blue_lines - len(blue_lines_list)} 条)")

            # 保存为JSON文件
            base_name = image_name.rsplit(".", 1)[0]
            with open(os.path.join(save_path, f"{base_name}.json"), "w") as f:
                json.dump(result_dict, f, indent=2)

            # 创建新的白色画布显示颜色分类结果
            color_canvas = np.ones_like(image) * 255
            linewidth = 2

            # 绘制红色线段 (BGR格式: 0,0,255)
            for line in red_lines_list:
                start = line["StartPoint"]
                end = line["EndPoint"]
                cv2.line(
                    color_canvas,
                    (int(start["X"]), int(start["Y"])),
                    (int(end["X"]), int(end["Y"])),
                    (0, 0, 255),
                    linewidth,
                )

            # 绘制蓝色线段 (BGR格式: 255,0,0)
            for line in blue_lines_list:
                start = line["StartPoint"]
                end = line["EndPoint"]
                cv2.line(
                    color_canvas,
                    (int(start["X"]), int(start["Y"])),
                    (int(end["X"]), int(end["Y"])),
                    (255, 0, 0),
                    linewidth,
                )

            # 绘制绿色线段 (BGR格式: 0,255,0)
            for line in green_lines_list:
                start = line["StartPoint"]
                end = line["EndPoint"]
                cv2.line(
                    color_canvas,
                    (int(start["X"]), int(start["Y"])),
                    (int(end["X"]), int(end["Y"])),
                    (0, 255, 0),
                    linewidth,
                )

            # 绘制灰色线段 (BGR格式: 152,152,152)
            for line in gray_lines_list:
                start = line["StartPoint"]
                end = line["EndPoint"]
                cv2.line(
                    color_canvas,
                    (int(start["X"]), int(start["Y"])),
                    (int(end["X"]), int(end["Y"])),
                    (152, 152, 152),
                    linewidth,
                )
            # color_canvas = cv2.resize(color_canvas, (512, 512))

            # 在所有节点位置绘制黑色小圆圈
            for point in points_list:
                cv2.circle(
                    color_canvas,
                    (int(point[0]), int(point[1])),
                    radius=1,  # 圆圈半径
                    color=(0, 0, 0),  # 黑色
                    thickness=-1,
                )  # -1表示填充圆圈

            color_canvas = cv2.resize(color_canvas, (2048, 1024), interpolation=cv2.INTER_NEAREST)

            # # 在图上标注所有节点的坐标
            # for point in points_list:
            #     cv2.putText(color_canvas, f"({int(point[0])}, {int(point[1])})",
            #                 (int(point[0])*8, int(point[1])*8),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # 在节点位置绘制坐标

            # 保存结果

            cv2.imwrite(os.path.join(save_path, f"{base_name}_color_classified_lines.png"), color_canvas)

            # print(f"最终的红色线段数量：{len(red_lines_list)}")
            # print(f"最终的蓝色线段数量：{len(blue_lines_list)}")
            # print(f"最终的端点数量：{len(points_list)}")
            # print(f"数据已保存到 {base_name}.json")

        except Exception as e:
            # 记录错误信息
            error_message = f"{image_name}: {str(e)}\n"
            with open("wrong.txt", "a") as error_log:
                error_log.write(error_message)

            # print(f"处理 {image_name} 时发生错误: {str(e)}")
            continue
