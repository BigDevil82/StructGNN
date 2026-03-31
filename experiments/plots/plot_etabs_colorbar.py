"""
绘制ETABS风格的Colorbar

从ETABS截图中复刻colorbar样式，用于论文插图。
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "outputs/result/paper_plots/etabs_colorbar_2.pdf"

# 指定左右端数值，边界数量 = 色块数 + 1，由 linspace 自动生成
VMIN = -6.68  # 左端数值
VMAX = -0.12  # 右端数值
# VMIN = -11.9  # 左端数值
# VMAX = -0.9  # 右端数值

# ETABS 彩虹色谱：品红 → 红 → 橙 → 黄 → 绿 → 青 → 蓝 → 深蓝
# 色块数 = 13（固定）
colors = [
    "#FF00FF",  # 品红
    "#FF00AA",
    "#FF0055",
    "#FF0000",  # 红
    "#FF5500",  # 橙红
    "#FF8800",  # 橙
    "#FFBB00",  # 橙黄
    "#FFFF00",  # 黄
    "#88FF00",  # 黄绿
    "#00DD00",  # 绿
    "#00DDAA",  # 青绿
    "#00CCFF",  # 青
    "#0066FF",  # 蓝
]


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
        }
    )

    n = len(colors)
    boundaries = np.linspace(VMIN, VMAX, n + 1)

    fig, ax = plt.subplots(figsize=(10, 0.6))
    fig.subplots_adjust(bottom=0.05, top=0.55, left=0.02, right=0.98)

    # 绘制色块
    for i in range(n):
        ax.axvspan(i, i + 1, color=colors[i])

    # 数值标注在色块交界处上方
    for i, val in enumerate(boundaries):
        ax.text(
            i, 1.15, f"{val:.2f}", ha="center", va="bottom", fontsize=12, transform=ax.get_xaxis_transform()
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")

    png_path = OUTPUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
