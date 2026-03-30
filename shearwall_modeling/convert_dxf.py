import os

from experiments.case_study.fem_builder import FEMTopologyBuilder, export_to_json
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
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=100.0)
    for i, room_poly in enumerate(room_polys):
        fem_builder.add_room(room_poly, sw_vectors[i], masks_list[i])
    result = fem_builder.build()
    export_to_json(result, output_path)


def convert_folder(dxf_folder: str, output_folder: str) -> None:

    for root, dirs, files in os.walk(dxf_folder):
        for filename in files:
            if filename.lower().endswith(".dxf"):
                dxf_path = os.path.join(root, filename)
                json_filename = os.path.splitext(filename)[0] + ".json"
                output_path = os.path.join(output_folder, json_filename)
                convert_dxf_to_fem_topology(dxf_path, output_path)


if __name__ == "__main__":
    dxf_folder = r"dxf/shearwall_split_8_2"
    output_folder = r"dxf/cad_json_data/shearwall_split_8_2"
    os.makedirs(output_folder, exist_ok=True)
    convert_folder(dxf_folder, output_folder)
