import os
from pathlib import Path

from pipelines.case_study.adapters import export_structural_input_json, fem_result_to_fem_input
from pipelines.case_study.fem_builder import FEMTopologyBuilder, visualize_fem_result
from pipelines.case_study.symmetry_postprocess import (
    detect_left_right_symmetry,
    load_layout_reference_geometries,
    save_symmetry_detection_debug_plot,
    symmetrize_raw_walls,
)
from pipelines.case_study.visualization import visualize_prediction_comparison, visualize_wall_layout
from src.shearwall_pred.inference import load_ensemble_model, predict_shear_walls, prepare_case_graph


def build_fem_result(
    room_polys,
    masks_list,
    predictions,
    symmetry_info,
    symmetry_mode: str,
    symmetry_threshold: float,
    symmetry_debug_path: str | None = None,
):
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=100.0)

    for i, room_poly in enumerate(room_polys):
        masks = masks_list[i] if masks_list else None
        fem_builder.add_room(room_poly, predictions[i], masks)

    raw_wall_count_before = len(fem_builder.raw_walls)
    symmetry_applied = False
    if symmetry_mode != "none":
        if symmetry_info["is_symmetric"] and symmetry_info["axis_x"] is not None:
            fem_builder.raw_walls = symmetrize_raw_walls(
                fem_builder.raw_walls,
                fem_builder.room_edges,
                axis_x=symmetry_info["axis_x"],
                mode=symmetry_mode,
                min_length=fem_builder.min_length,
                debug_path=symmetry_debug_path,
            )
            symmetry_applied = True
            print(
                "  对称化后处理: "
                f"mode={symmetry_mode}, raw_walls {raw_wall_count_before} -> {len(fem_builder.raw_walls)}"
            )
        else:
            print(
                "  对称化后处理: 跳过, "
                f"mode={symmetry_mode}, confidence={symmetry_info['confidence']:.3f} < {symmetry_threshold:.3f}"
            )

    result = fem_builder.build()
    result["symmetry"] = {
        **symmetry_info,
        "mode": symmetry_mode,
        "threshold": symmetry_threshold,
        "applied": symmetry_applied,
        "raw_wall_count_before": raw_wall_count_before,
        "raw_wall_count_after": len(fem_builder.raw_walls),
    }
    return result, fem_builder, symmetry_applied


def predict_structural_result(
    dxf_path: str,
    model_dir: str,
    category: int = None,
    device: str = "cuda",
    symmetry_mode: str = "none",
    symmetry_threshold: float = 0.85,
    use_gt: bool = False,
    symmetry_debug_path: str | None = None,
):
    """从 DXF 直接生成结构构件结果，供结构后端复用。"""
    layout_geometries = load_layout_reference_geometries(dxf_path)
    symmetry_info = detect_left_right_symmetry(
        layout_geometries["infill_geometries"],
        layout_geometries["room_geometries"],
        threshold=symmetry_threshold,
    )
    builder_graph, data_batch, room_polys, masks_list = prepare_case_graph(dxf_path, category, device)

    if symmetry_info["axis_x"] is None:
        print("  左右对称检测: 无有效几何，跳过")
    else:
        print(
            "  左右对称检测: "
            f"source={symmetry_info['source']}, axis_x={symmetry_info['axis_x']:.2f}, "
            f"confidence={symmetry_info['confidence']:.3f}, "
            f"is_symmetric={symmetry_info['is_symmetric']}"
        )

    if use_gt:
        print("  使用 Ground Truth 标注")
        node_ids = list(builder_graph.graph.nodes())
        predictions = []
        for node_id in node_ids:
            sw_vector = builder_graph.graph.nodes[node_id].get("sw_vector")
            if sw_vector is None:
                raise ValueError(f"节点 {node_id} 缺少 sw_vector，该DXF可能没有GT标注")
            predictions.append(sw_vector)
    else:
        print("  使用模型预测")
        model = load_ensemble_model(model_dir, device)
        predictions = predict_shear_walls(model, data_batch)

    result, fem_builder, symmetry_applied = build_fem_result(
        room_polys=room_polys,
        masks_list=masks_list,
        predictions=predictions,
        symmetry_info=symmetry_info,
        symmetry_mode=symmetry_mode,
        symmetry_threshold=symmetry_threshold,
        symmetry_debug_path=symmetry_debug_path,
    )

    return {
        "result": result,
        "builder_graph": builder_graph,
        "room_polys": room_polys,
        "predictions": predictions,
        "fem_builder": fem_builder,
        "symmetry_applied": symmetry_applied,
        "symmetry_info": symmetry_info,
        "layout_geometries": layout_geometries,
    }


def run_case_study(
    dxf_path: str,
    model_dir: str,
    output_dir: str,
    category: int = None,
    device: str = "cuda",
    symmetry_mode: str = "none",
    symmetry_threshold: float = 0.85,
):
    os.makedirs(output_dir, exist_ok=True)
    file_name = Path(dxf_path).stem
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("=" * 60)
    print("Engineering Case Study (Shapely-Enhanced)")
    print("=" * 60)
    print(f"Input: {dxf_path}")
    print(f"Output: {output_dir}")

    print("\n[1/5] 加载模型...")
    print("\n[2/5] 读取DXF并构建图...")
    print("\n[3/5] 模型预测...")
    print("\n[4/5] 解析为FEM构件 (Shapely Topology)...")
    prediction_bundle = predict_structural_result(
        dxf_path=dxf_path,
        model_dir=model_dir,
        category=category,
        device=device,
        symmetry_mode=symmetry_mode,
        symmetry_threshold=symmetry_threshold,
        symmetry_debug_path=os.path.join(debug_dir, f"{file_name}_symmetry_postprocess.png"),
        use_gt=False,
    )
    layout_geometries = prediction_bundle["layout_geometries"]
    symmetry_info = prediction_bundle["symmetry_info"]
    builder_graph = prediction_bundle["builder_graph"]
    room_polys = prediction_bundle["room_polys"]
    result = prediction_bundle["result"]
    predictions = prediction_bundle["predictions"]
    fem_builder = prediction_bundle["fem_builder"]
    symmetry_applied = prediction_bundle["symmetry_applied"]
    save_symmetry_detection_debug_plot(
        layout_geometries["infill_geometries"],
        layout_geometries["room_geometries"],
        symmetry_info,
        os.path.join(debug_dir, f"{file_name}_symmetry_detection.png"),
    )
    print(f"  房间数量: {len(room_polys)}")

    print(f"  节点数量: {result['statistics']['num_nodes']}")
    print(f"  构件总数: {result['statistics']['num_members']}")
    print(f"    - 剪力墙: {result['statistics']['num_shearwalls']}")
    print(f"    - 梁: {result['statistics']['num_beams']}")

    print("\n[5/5] 可视化和导出...")
    visualize_fem_result(
        result,
        room_polys,
        title=f"FEM Topology (Shapely) - {file_name}",
        save_path=os.path.join(output_dir, f"{file_name}_fem_members.png"),
        show_exceptions=False,
    )
    visualize_prediction_comparison(
        builder_graph,
        predictions,
        save_path=os.path.join(output_dir, f"{file_name}_raw_pred.png"),
    )

    if symmetry_applied:
        visualize_wall_layout(
            room_polys,
            fem_builder.raw_walls,
            title=f"Symmetrized Shear Wall Layout - {file_name}",
            save_path=os.path.join(output_dir, f"{file_name}_pred_symmetrized.png"),
        )

    structural_json_path = os.path.join(output_dir, f"{file_name}_structural_input.json")
    export_structural_input_json(result, structural_json_path)
    result["structural_input"] = fem_result_to_fem_input(result)
    result["structural_input_json_path"] = structural_json_path

    print("\n" + "=" * 60)
    print("案例研究完成！")
    return result
