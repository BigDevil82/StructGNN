import json

import matplotlib.pyplot as plt


def visualize_prediction_comparison(builder, predictions, save_path=None):
    """可视化预测结果对比"""
    from src.data_engine.preprocess.room_analyzer import plot_room_analysis, reconstruct_walls

    plt.switch_backend("Agg")

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 8))
    node_ids = list(builder.graph.nodes())

    ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
    for node_id in node_ids:
        node_data = builder.graph.nodes[node_id]
        if node_data.get("sw_vector") is not None:
            plot_room_analysis(
                node_data["poly"],
                node_data["sw_vector"],
                node_data.get("masks", []),
                ax_gt,
                wall_color="green",
            )

    ax_pred.set_title("Model Prediction", fontsize=14, fontweight="bold")
    for i, node_id in enumerate(node_ids):
        node_data = builder.graph.nodes[node_id]
        pred_vec = predictions[i]
        masks = node_data.get("masks", [])
        pred_walls = reconstruct_walls(node_data["poly"], pred_vec, masks)
        plot_room_analysis(node_data["poly"], pred_vec, masks, ax_pred, walls=pred_walls, wall_color="red")

    for ax in [ax_gt, ax_pred]:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def visualize_wall_layout(room_polys, walls, title, save_path=None):
    """可视化后处理后的剪力墙布局。"""
    plt.switch_backend("Agg")

    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    for room_poly in room_polys:
        x_coords, y_coords = room_poly.exterior.xy
        ax.fill(x_coords, y_coords, facecolor=(0.678, 0.847, 0.902, 0.2), edgecolor="lightgray", linewidth=6)

    for wall in walls:
        if wall.is_empty:
            continue
        x_coords, y_coords = wall.xy
        ax.plot(x_coords, y_coords, color="crimson", linewidth=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def export_shearwall_coords(result: dict, output_path: str):
    """导出简化的剪力墙坐标"""
    shearwall_coords = []
    for member in result["members"]:
        if member["type"] == "shearwall":
            shearwall_coords.append(
                [
                    list(member["start_coord"]),
                    list(member["end_coord"]),
                ]
            )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shearwall_coords, f, indent=2)
    print(f"  Exported coords to: {output_path}")
