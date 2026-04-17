import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Ellipse

from src.data_engine.postprocess.member_graph import MemberGraphBuilder

COLOR_MAP = {
    "wall": "#d73027",
    "primary_beam": "#4575b4",
    "secondary_beam": "#1a9850",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化检查 FEM 构件归并与图连接结果")
    parser.add_argument("--input-json", required=True, help="FEM 结果 JSON 或 export_to_json 导出的 JSON")
    parser.add_argument("--output", help="可选，输出图片路径")
    parser.add_argument("--coord-tol", type=float, default=0.01, help="坐标带分组与连接容差")
    parser.add_argument(
        "--axis-tol", type=float, default=None, help="水平/竖直判定容差，默认跟 coord-tol 一致"
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show-labels", action="store_true", help="显示归并节点标签")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_member_bounds(raw_members) -> tuple[float, float, float, float]:
    xs = [p for m in raw_members for p in (m.start[0], m.end[0])]
    ys = [p for m in raw_members for p in (m.start[1], m.end[1])]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = span * 0.05
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def draw_graph(ax: plt.Axes, graph: nx.Graph, show_labels: bool):
    positions = {}
    lengths = [attrs["length"] for _, attrs in graph.nodes(data=True)]
    base_minor = max((min(lengths) if lengths else 1.0) * 0.18, 0.05)

    for node_id, attrs in graph.nodes(data=True):
        start = attrs["start"]
        end = attrs["end"]
        positions[node_id] = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        color = COLOR_MAP[attrs["kind"]]
        # ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=3, alpha=0.9, zorder=3)
        angle = 0.0 if attrs["axis"] == "horizontal" else 90.0
        ellipse = Ellipse(
            xy=positions[node_id],
            width=max(attrs["length"], base_minor),
            height=attrs["length"] * 0.1,
            angle=angle,
            fill=False,
            linestyle="--",
            edgecolor=color,
            linewidth=1,
            alpha=0.6,
            zorder=2,
        )
        # print(
        #     f"node={node_id}, kind={attrs['kind']}, length={attrs['length']:.3f}, axis={attrs['axis']}, ellipse_width={ellipse.width:.3f}, ellipse_height={ellipse.height:.3f}"
        # )
        ax.add_patch(ellipse)

    for u, v, attrs in graph.edges(data=True):
        p1 = positions[u]
        p2 = positions[v]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#333333",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            zorder=1,
        )
        for point in attrs.get("intersection_points", []):
            ax.scatter(
                [point[0]],
                [point[1]],
                s=28,
                facecolors="#fefefe",
                edgecolors="#000000",
                linewidths=1.0,
                zorder=5,
            )

    # for node_id, attrs in graph.nodes(data=True):
    #     pos = positions[node_id]
    #     degree = graph.degree[node_id]
    #     face = "white"
    #     edge = "#222222"
    #     size = 55
    #     if degree == 0:
    #         face = "#ffcc00"
    #         edge = "#cc3300"
    #         size = 80
    #     elif degree == 1:
    #         face = "#fff7bc"
    #         edge = "#d95f0e"
    #         size = 70

    #     ax.scatter([pos[0]], [pos[1]], s=size, facecolors=face, edgecolors=edge, linewidths=1.6, zorder=4)
    #     if show_labels:
    #         ax.text(pos[0], pos[1], f"{node_id}/d{degree}", fontsize=8, ha="center", va="bottom")


def apply_extent(ax: plt.Axes, extent):
    minx, miny, maxx, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.margins(0)


def style_axis(ax: plt.Axes, title: str, extent):
    apply_extent(ax, extent)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def print_summary(graph: nx.Graph):
    isolated = [n for n in graph.nodes if graph.degree[n] == 0]
    leaves = [n for n in graph.nodes if graph.degree[n] == 1]
    merged = sorted(
        ((n, len(a["raw_member_ids"]), a["kind"], a["length"]) for n, a in graph.nodes(data=True)),
        key=lambda x: (-x[1], -x[3], x[0]),
    )

    print(f"graph nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    print(f"isolated nodes={isolated}")
    print(f"degree-1 nodes={leaves}")
    print("top merged members:")
    for node_id, raw_count, kind, length in merged[:10]:
        print(f"  node={node_id}, kind={kind}, raw_count={raw_count}, length={length:.3f}")


def main():
    args = parse_args()
    data = load_json(args.input_json)
    builder = MemberGraphBuilder(coord_tol=args.coord_tol, axis_tol=args.axis_tol)
    raw_members = builder.load_members(data)
    graph = builder.build_graph(data)

    if not raw_members:
        raise ValueError("输入数据中没有可用于构图的水平/竖直构件")

    extent = get_member_bounds(raw_members)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    draw_graph(ax, graph, args.show_labels)
    style_axis(ax, "Merged Graph", extent)

    plt.tight_layout(pad=0.6)
    print_summary(graph)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=args.dpi, bbox_inches="tight")
        print(f"saved to: {output}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # python -m scripts.inspect_member_graph --input-json data\dxf\cad_json_data\fem_raw\L1L28_68.json
    main()
