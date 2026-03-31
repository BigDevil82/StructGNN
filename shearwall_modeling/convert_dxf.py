import os
from typing import Optional, Tuple

from experiments.pipelines.case_study.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from misc.parallel import run_batch
from misc.timer import Timer
from shearwall_pred.utils import build_graph_from_dxf


def convert_dxf_to_fem_topology(dxf_path: str, output_path: str) -> FEMTopologyBuilder:
    # 1. 读取DXF并构建图
    try:
        builder_graph = build_graph_from_dxf(dxf_path, mode="none")
    except Exception as e:
        print(f"Error processing {dxf_path}: {e}")
        return

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

    # 3. 构建FEM拓扑
    print("processing:", os.path.basename(dxf_path))
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=100.0)
    for i, room_poly in enumerate(room_polys):
        fem_builder.add_room(room_poly, sw_vectors[i], masks_list[i])
    result = fem_builder.build()
    export_to_json(result, output_path)
    if not result["validation"]["ok"]:
        visualize_fem_result(
            result,
            room_polys,
            title=os.path.basename(dxf_path),
        )


def _convert_task(task: Tuple[str, str]) -> Optional[str]:
    """并行工作函数：成功返回 None，失败返回错误信息。"""
    dxf_path, output_path = task
    try:
        convert_dxf_to_fem_topology(dxf_path, output_path)
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


if __name__ == "__main__":
    dxf_folder = r"data/dxf/fem_raw"
    output_folder = r"data/dxf/cad_json_data/fem_raw"
    os.makedirs(output_folder, exist_ok=True)
    convert_folder(dxf_folder, output_folder)

    # dxf = r"data/dxf/fem_raw/L1L28_190.dxf"
    # output = r"data\data\dxf\cad_json_data\shearwall_split_8_2\L1L28_10.json"
    # convert_dxf_to_fem_topology(dxf, output)
