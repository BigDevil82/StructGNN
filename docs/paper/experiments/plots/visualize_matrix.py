"""
矩阵可视化工具 - 用于论文中展示模型输出结果
通过颜色小方块表示矩阵中的数值 [0,1]
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def visualize_matrix(
    matrix,
    gap=0.05,
    cell_size=1.0,
    cmap="viridis",
    save_path=None,
    dpi=300,
    show_colorbar=True,
    vmin=0,
    vmax=1,
    figsize=None,
):
    """
    可视化矩阵为颜色小方块

    参数:
        matrix: numpy array, 矩阵数据，数值范围 [0, 1]
        gap: float, 小方块之间的间距 (相对于cell_size的比例)
        cell_size: float, 每个小方块的大小
        cmap: str or colormap, 颜色映射方案
                常用科研风格: 'viridis', 'plasma', 'inferno', 'magma',
                            'cividis', 'RdYlBu_r', 'coolwarm', 'seismic'
        save_path: str, 保存路径，None则不保存
        dpi: int, 图片分辨率
        show_colorbar: bool, 是否显示颜色条
        vmin, vmax: float, 颜色映射的数值范围
        figsize: tuple, 图片大小，None则自动计算

    返回:
        fig, ax: matplotlib图形对象
    """

    matrix = np.array(matrix)
    rows, cols = matrix.shape

    # 自动计算图片大小
    if figsize is None:
        width = cols * (cell_size + gap) + gap
        height = rows * (cell_size + gap) + gap
        figsize = (width, height)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 获取颜色映射
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # 归一化数值到[0,1]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 绘制每个小方块
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            color = cmap(norm(value))

            # 计算方块位置 (左下角坐标)
            x = j * (cell_size + gap)
            y = (rows - 1 - i) * (cell_size + gap)  # 翻转y轴使第一行在顶部

            # 绘制方块
            rect = Rectangle((x, y), cell_size, cell_size, facecolor=color, edgecolor="none")
            ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xlim(-gap, cols * (cell_size + gap))
    ax.set_ylim(-gap, rows * (cell_size + gap))
    ax.set_aspect("equal")
    ax.axis("off")

    # 添加颜色条
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"图片已保存至: {save_path}")

    return fig, ax


def create_custom_colormap(colors, name="custom"):
    """
    创建自定义颜色映射

    参数:
        colors: list of colors, 可以是颜色名称或RGB值
        name: str, 颜色映射名称

    返回:
        LinearSegmentedColormap对象
    """
    return LinearSegmentedColormap.from_list(name, colors)


# ==================== 示例用法 ====================

if __name__ == "__main__":
    # 设置matplotlib字体以支持中文（可选）
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 示例1: 随机矩阵
    print("生成示例可视化...")
    # matrix1 = np.random.rand(5, 5)

    # # 使用viridis配色（深蓝-绿-黄）
    # fig1, ax1 = visualize_matrix(
    #     matrix1,
    #     gap=0.1,
    #     cell_size=1.0,
    #     cmap='viridis',
    #     save_path='matrix_viridis.png'
    # )
    # plt.show()

    # # 示例2: 使用plasma配色（紫-红-橙-黄）
    # matrix2 = np.random.rand(6, 8)
    # fig2, ax2 = visualize_matrix(
    #     matrix2,
    #     gap=0.08,
    #     cell_size=0.9,
    #     cmap='plasma',
    #     save_path='matrix_plasma.png'
    # )
    # plt.show()

    # 示例3: 使用coolwarm配色（蓝-白-红）
    matrix3 = np.random.rand(5, 4)
    fig3, ax3 = visualize_matrix(
        matrix3,
        gap=0.15,
        cell_size=1.2,
        cmap="coolwarm",
        save_path="matrix_coolwarm.png",
        show_colorbar=False,
    )
    plt.show()

    # # 示例4: 自定义颜色
    # custom_cmap = create_custom_colormap(['#2c3e50', '#3498db', '#e74c3c', '#f39c12'])
    # matrix4 = np.random.rand(5, 7)
    # fig4, ax4 = visualize_matrix(
    #     matrix4,
    #     gap=0.05,
    #     cell_size=1.0,
    #     cmap=custom_cmap,
    #     save_path='matrix_custom.png',
    #     show_colorbar=True
    # )
    # plt.show()

    # # 示例5: 无间距效果
    # matrix5 = np.random.rand(8, 8)
    # fig5, ax5 = visualize_matrix(
    #     matrix5,
    #     gap=0.0,
    #     cell_size=1.0,
    #     cmap='inferno',
    #     save_path='matrix_no_gap.png'
    # )
    plt.show()

    print("所有示例已生成完成！")


# ==================== 常用配色方案说明 ====================
"""
科研论文中常用的配色方案:

1. 感知均匀配色（推荐用于定量数据）:
   - 'viridis': 深蓝-绿-黄，色盲友好
   - 'plasma': 紫-红-橙-黄
   - 'inferno': 黑-紫-红-橙-黄
   - 'magma': 黑-紫-粉-橙-黄
   - 'cividis': 蓝-黄，色盲友好

2. 发散配色（适合表示正负或偏差）:
   - 'coolwarm': 蓝-白-红
   - 'seismic': 蓝-白-红
   - 'RdYlBu_r': 红-黄-蓝（反向）
   - 'RdBu_r': 红-白-蓝（反向）

3. 单色渐变:
   - 'Blues': 浅蓝-深蓝
   - 'Reds': 浅红-深红
   - 'Greens': 浅绿-深绿
   - 'Greys': 白-黑

使用建议:
- 定量数据: viridis, plasma
- 概率/置信度: viridis, cividis
- 热力图: inferno, magma
- 相关性矩阵: coolwarm, RdBu_r
- 注意力权重: viridis, plasma
"""
