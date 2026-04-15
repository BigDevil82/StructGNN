import json
import os
from pathlib import Path
from typing import Optional, Tuple

from shapely.geometry import GeometryCollection, Polygon, box

from src.data_engine.postprocess.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from src.data_engine.postprocess.symmetry_postprocess import (
    detect_left_right_symmetry,
    load_layout_reference_geometries,
)
from src.misc.parallel import run_batch
from src.misc.timer import Timer
from src.shearwall_modeling.core.domain import FEMInput
from src.shearwall_modeling.geometry.scaling import choose_scale_factor
from src.shearwall_pred.utils import build_graph_from_dxf

DEFAULT_SYMMETRY_THRESHOLD = 0.85
LEFT_AUG_SUFFIX = "_left_aug"


def _is_left_right_separable(
    layout_geometries: dict, axis_x: float, area_tol: float = 1.0e6
) -> tuple[bool, int]:
    """
    判断布局是否“左右可分”。

    含义：对称轴不切穿房间内部，即任一房间不应同时占据轴线两侧的显著面积。
    """
    room_geometries = layout_geometries.get("room_geometries", [])
    if not room_geometries:
        return False, 0

    union_geom = GeometryCollection()
    for geom in room_geometries:
        if geom is None or geom.is_empty:
            continue
        union_geom = union_geom.union(geom)

    if union_geom.is_empty:
        return False, 0

    minx, miny, maxx, maxy = union_geom.bounds
    pad = 5.0
    left_half = box(minx - pad, miny - pad, axis_x, maxy + pad)
    right_half = box(axis_x, miny - pad, maxx + pad, maxy + pad)

    split_room_count = 0
    for geom in room_geometries:
        if geom is None or geom.is_empty:
            continue

        left_area = geom.intersection(left_half).area
        right_area = geom.intersection(right_half).area
        if left_area > area_tol and right_area > area_tol:
            split_room_count += 1

    return split_room_count == 0, split_room_count


def _extract_left_half_result(result: dict, axis_x: float, x_tol: float = 100.0) -> dict:
    """从完整 FEM 结果提取左半部分，并重建节点编号与统计信息。"""
    nodes = result.get("nodes", [])
    members = result.get("members", [])
    slabs = result.get("slabs", [])

    keep_old_node_ids = [idx for idx, (x, _) in enumerate(nodes) if x <= axis_x + x_tol]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_old_node_ids)}
    new_nodes = [nodes[old_id] for old_id in keep_old_node_ids]

    new_members = []
    for member in members:
        u_old = member["start_node"]
        v_old = member["end_node"]
        if u_old not in old_to_new or v_old not in old_to_new:
            continue

        u_new = old_to_new[u_old]
        v_new = old_to_new[v_old]
        m = dict(member)
        m["id"] = len(new_members)
        m["start_node"] = u_new
        m["end_node"] = v_new
        m["start_coord"] = new_nodes[u_new]
        m["end_coord"] = new_nodes[v_new]
        new_members.append(m)

    new_slabs = []
    if slabs and new_nodes:
        xs = [pt[0] for pt in new_nodes]
        ys = [pt[1] for pt in new_nodes]
        pad = max(100.0, 0.01 * max(max(xs) - min(xs), max(ys) - min(ys)))
        left_half = box(min(xs) - pad, min(ys) - pad, axis_x, max(ys) + pad)

        for slab in slabs:
            if len(slab) < 3:
                continue
            poly = Polygon(slab)
            if poly.is_empty or not poly.is_valid:
                continue

            clipped = poly.intersection(left_half)
            if clipped.is_empty:
                continue

            if isinstance(clipped, Polygon):
                new_slabs.append(list(clipped.exterior.coords)[:-1])
            elif hasattr(clipped, "geoms"):
                for g in clipped.geoms:
                    if isinstance(g, Polygon) and not g.is_empty and g.area > 1.0:
                        new_slabs.append(list(g.exterior.coords)[:-1])

    if new_nodes:
        min_xy = [float(min(pt[0] for pt in new_nodes)), float(min(pt[1] for pt in new_nodes))]
        max_xy = [float(max(pt[0] for pt in new_nodes)), float(max(pt[1] for pt in new_nodes))]
    else:
        min_xy = [0.0, 0.0]
        max_xy = [0.0, 0.0]

    left_result = {
        "nodes": new_nodes,
        "members": new_members,
        "slabs": new_slabs,
        "statistics": {
            "num_nodes": len(new_nodes),
            "min_xy": min_xy,
            "max_xy": max_xy,
            "num_members": len(new_members),
            "num_shearwalls": sum(1 for m in new_members if m["type"] == "shearwall"),
            "num_beams": sum(1 for m in new_members if m["type"] == "beam"),
            "num_primary_beams": sum(
                1 for m in new_members if m["type"] == "beam" and m.get("beam_role") == "primary"
            ),
            "num_secondary_beams": sum(
                1 for m in new_members if m["type"] == "beam" and m.get("beam_role") == "secondary"
            ),
            "num_slabs": len(new_slabs),
            "total_sw_length": sum(m["length"] for m in new_members if m["type"] == "shearwall"),
            "total_beam_length": sum(m["length"] for m in new_members if m["type"] == "beam"),
        },
    }
    return left_result


def _extract_left_room_polys(room_polys: list, axis_x: float, area_tol: float = 1.0e6) -> list:
    """裁剪得到左半房间轮廓，用于左半结果可视化。"""
    if not room_polys:
        return []

    valid_rooms = [room for room in room_polys if room is not None and not room.is_empty]
    if not valid_rooms:
        return []

    minx = min(room.bounds[0] for room in valid_rooms)
    miny = min(room.bounds[1] for room in valid_rooms)
    maxx = max(room.bounds[2] for room in valid_rooms)
    maxy = max(room.bounds[3] for room in valid_rooms)
    pad = 5.0
    left_half = box(minx - pad, miny - pad, axis_x, maxy + pad)

    left_rooms = []
    for room in valid_rooms:
        clipped = room.intersection(left_half)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            if clipped.area > area_tol:
                left_rooms.append(clipped)
        elif hasattr(clipped, "geoms"):
            for geom in clipped.geoms:
                if isinstance(geom, Polygon) and geom.area > area_tol:
                    left_rooms.append(geom)

    return left_rooms


def _prepare_graph_inputs(dxf_path: str):
    layout_geometries = load_layout_reference_geometries(dxf_path)
    symmetry_info = detect_left_right_symmetry(
        layout_geometries["infill_geometries"],
        layout_geometries["room_geometries"],
        threshold=DEFAULT_SYMMETRY_THRESHOLD,
    )

    # 1. 读取DXF并构建图
    try:
        builder_graph = build_graph_from_dxf(dxf_path, mode="none")
    except Exception as e:
        print(f"Error processing {dxf_path}: {e}")
        raise e

    # 提取房间信息
    node_ids = list(builder_graph.graph.nodes())
    room_polys = [builder_graph.graph.nodes[n]["poly"] for n in node_ids]
    masks_list = [builder_graph.graph.nodes[n].get("masks", []) for n in node_ids]

    sw_vectors = []
    for n in node_ids:
        sv = builder_graph.graph.nodes[n].get("sw_vector")
        if sv is None:
            raise ValueError(f"节点 {n} 缺少 sw_vector，该DXF可能没有GT标注")
        sw_vectors.append(sv)

    return room_polys, masks_list, sw_vectors, symmetry_info


def _build_fem_result(room_polys: list, masks_list: list, sw_vectors: list) -> dict:
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=200.0)
    for i, room_poly in enumerate(room_polys):
        fem_builder.add_room(room_poly, sw_vectors[i], masks_list[i])
    return fem_builder.build()


def add_scale_factor(result_json_path):
    json_path = Path(result_json_path)
    input_data = FEMInput.from_json(json_path)
    if not input_data.all_members():
        raise ValueError("No beams/walls found in JSON.")

    factor = choose_scale_factor(input_data)
    with open(result_json_path, "r") as f:
        data = json.load(f)
        data["statistics"]["scale_factor"] = factor
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_dxf_to_fem_topology(dxf_path: str, output_path: str) -> dict:
    room_polys, masks_list, sw_vectors, _ = _prepare_graph_inputs(dxf_path)

    # 3. 构建FEM拓扑
    print("processing:", os.path.basename(dxf_path))
    result = _build_fem_result(room_polys, masks_list, sw_vectors)
    export_to_json(result, output_path)
    add_scale_factor(output_path)

    # check if short members exist
    # short_members = [mem for mem in result["members"] if mem["length"] < 200.0]
    # if short_members:
    #     for mem in short_members:
    #         print(
    #             "  [WARNING] Short member detected: ID={}, Type={}, Length={:.2f}mm, start: {}, end: {}".format(
    #                 mem["id"], mem["type"], mem["length"], mem["start_coord"], mem["end_coord"]
    #             )
    #         )
    save_path = Path(output_path).with_suffix(".png")
    visualize_fem_result(
        result, room_polys, title=os.path.basename(dxf_path), save_path=save_path, show_node=False
    )

    ## for debug
    if not result["validation"]["ok"]:
        visualize_fem_result(
            result,
            room_polys,
            title=os.path.basename(dxf_path),
        )
    return result


def convert_dxf_to_left_half_topology(
    dxf_path: str,
    output_path: str,
    symmetry_threshold: float = DEFAULT_SYMMETRY_THRESHOLD,
) -> Optional[dict]:
    """
    并列功能：仅在布局“左右对称且左右可分”时导出左半 FEM 结果。
    """
    layout_geometries = load_layout_reference_geometries(dxf_path)
    symmetry_info = detect_left_right_symmetry(
        layout_geometries["infill_geometries"],
        layout_geometries["room_geometries"],
        threshold=symmetry_threshold,
    )

    axis_x = symmetry_info.get("axis_x")
    is_separable = False
    split_room_count = 0
    if symmetry_info.get("is_symmetric") and axis_x is not None:
        is_separable, split_room_count = _is_left_right_separable(layout_geometries, axis_x)

    if axis_x is None:
        print("  left-half-check: no valid geometry")
        return None

    print(
        "  left-half-check: "
        f"symmetric={symmetry_info['is_symmetric']}, "
        f"confidence={symmetry_info['confidence']:.3f}, "
        f"axis_x={axis_x:.2f}, "
        f"left-right-separable={is_separable}, split_rooms={split_room_count}"
    )

    if (not symmetry_info.get("is_symmetric")) or (not is_separable):
        print("  left-half-check: skip export (layout not eligible)")
        return None

    room_polys, masks_list, sw_vectors, _ = _prepare_graph_inputs(dxf_path)
    result = _build_fem_result(room_polys, masks_list, sw_vectors)

    left_result = _extract_left_half_result(result, axis_x=axis_x)
    left_room_polys = _extract_left_room_polys(room_polys, axis_x=axis_x)

    left_output_path = output_path
    if not left_output_path.endswith(".json"):
        left_output_path = str(
            Path(output_path).with_name(Path(output_path).stem + LEFT_AUG_SUFFIX + ".json")
        )

    export_to_json(left_result, left_output_path)
    add_scale_factor(left_output_path)

    save_path = Path(left_output_path).with_suffix(".png")
    visualize_fem_result(
        left_result, left_room_polys, title=os.path.basename(dxf_path), save_path=save_path, show_node=False
    )
    print("  left-half exported: " f"{left_output_path}, members={left_result['statistics']['num_members']}")

    return left_result


def _convert_task(task: Tuple[str, str]) -> Optional[str]:
    """并行工作函数：成功返回 None，失败返回错误信息。"""
    dxf_path, output_path = task
    try:
        convert_dxf_to_fem_topology(dxf_path, output_path)
        return None
    except Exception as e:
        return str(e)


def _convert_left_task(task: Tuple[str, str]) -> Optional[str]:
    """并行工作函数（左半导出）：成功返回 None，跳过返回 SKIPPED，失败返回错误信息。"""
    dxf_path, output_path = task
    try:
        result = convert_dxf_to_left_half_topology(dxf_path, output_path)
        if result is None:
            return "SKIPPED_NOT_ELIGIBLE"
        return None
    except Exception as e:
        return str(e)


def convert_folder(dxf_folder: str, output_folder: str, max_workers: Optional[int] = None) -> None:
    tasks = []
    for root, _, files in os.walk(dxf_folder):
        for filename in files:
            if filename.lower().endswith(".dxf"):
                dxf_path = os.path.join(root, filename)
                json_filename = os.path.splitext(filename)[0] + ".json"
                output_path = os.path.join(output_folder, json_filename)
                tasks.append((dxf_path, output_path))

    if not tasks:
        print(f"No DXF files found in: {dxf_folder}")
        return

    with Timer(prefix="DXF batch conversion time:"):
        outcomes = run_batch(tasks, _convert_task, max_workers=max_workers, backend="process")

    failed = [
        (item[0], outcome.error if outcome.error is not None else (outcome.result or "Unknown error"))
        for outcome in outcomes
        for item in [outcome.item]
        if not outcome.ok or outcome.result is not None
    ]

    print(f"Finished converting {len(tasks)} DXF files. Failed: {len(failed)}")
    for dxf_path, err in failed:
        print(f"  [FAILED] {dxf_path}: {err}")


def convert_folder_left_half(dxf_folder: str, output_folder: str, max_workers: Optional[int] = None) -> None:
    """批量导出左半增广样本：仅对“对称且左右可分”的布局输出结果。"""
    tasks = []
    for root, _, files in os.walk(dxf_folder):
        for filename in files:
            if filename.lower().endswith(".dxf"):
                dxf_path = os.path.join(root, filename)
                json_filename = os.path.splitext(filename)[0] + LEFT_AUG_SUFFIX + ".json"
                output_path = os.path.join(output_folder, json_filename)
                tasks.append((dxf_path, output_path))

    if not tasks:
        print(f"No DXF files found in: {dxf_folder}")
        return

    with Timer(prefix="DXF left-half batch conversion time:"):
        outcomes = run_batch(tasks, _convert_left_task, max_workers=max_workers, backend="process")

    failed = []
    skipped = []
    for outcome in outcomes:
        dxf_path, _ = outcome.item
        if outcome.ok and outcome.result is None:
            continue
        if outcome.ok and outcome.result == "SKIPPED_NOT_ELIGIBLE":
            skipped.append(dxf_path)
            continue

        err = outcome.error if outcome.error is not None else (outcome.result or "Unknown error")
        failed.append((dxf_path, err))

    succeeded = len(tasks) - len(skipped) - len(failed)
    print(
        "Finished left-half conversion: "
        f"total={len(tasks)}, exported={succeeded}, skipped={len(skipped)}, failed={len(failed)}"
    )
    for dxf_path in skipped:
        print(f"  [SKIPPED] {dxf_path}: not symmetric/separable")
    for dxf_path, err in failed:
        print(f"  [FAILED] {dxf_path}: {err}")


if __name__ == "__main__":
    dxf_folder = r"data/dxf/fem_raw"
    output_folder = r"data/dxf/cad_json_data/fem_raw"
    os.makedirs(output_folder, exist_ok=True)
    convert_folder(dxf_folder, output_folder)

    # dxf_folder = r"data/dxf/fem_raw"
    # output_folder = r"data/dxf/cad_json_data/fem_raw_left_aug"
    # os.makedirs(output_folder, exist_ok=True)
    # convert_folder_left_half(dxf_folder, output_folder)

    # dxf = r"data/dxf/fem_raw/L27_211.dxf"
    # output = r"data\dxf\cad_json_data\fem_raw" + os.sep + os.path.splitext(os.path.basename(dxf))[0] + ".json"
    # # convert_dxf_to_fem_topology(dxf, output)
    # convert_dxf_to_left_half_topology(dxf, output)
