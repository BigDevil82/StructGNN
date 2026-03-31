import json
from pathlib import Path

from src.shearwall_modeling.domain import FEMInput, PlanMember


def fem_result_to_serializable(fem_result: dict) -> dict:
    """将 case_study 的拓扑结果转换为共享的结构输入 JSON 语义。"""
    return {
        "shearwalls": [
            {
                "start": list(member["start_coord"]),
                "end": list(member["end_coord"]),
                "start_node": member["start_node"],
                "end_node": member["end_node"],
                "length": member["length"],
            }
            for member in fem_result["members"]
            if member["type"] == "shearwall"
        ],
        "beams": [
            {
                "start": list(member["start_coord"]),
                "end": list(member["end_coord"]),
                "start_node": member["start_node"],
                "end_node": member["end_node"],
                "length": member["length"],
            }
            for member in fem_result["members"]
            if member["type"] == "beam"
        ],
        "slabs": [
            [[float(point[0]), float(point[1])] for point in slab]
            for slab in fem_result.get("slabs", [])
        ],
        "statistics": fem_result.get("statistics", {}),
        "symmetry": fem_result.get("symmetry", {}),
        "validation": fem_result.get("validation", {}),
    }


def fem_result_to_fem_input(fem_result: dict, xy_scale_to_m: float = 0.001) -> FEMInput:
    """将 case_study 结果转为结构分析通用的 FEMInput。"""
    walls = [
        PlanMember(
            start=(float(member["start_coord"][0]) * xy_scale_to_m, float(member["start_coord"][1]) * xy_scale_to_m),
            end=(float(member["end_coord"][0]) * xy_scale_to_m, float(member["end_coord"][1]) * xy_scale_to_m),
        )
        for member in fem_result["members"]
        if member["type"] == "shearwall"
    ]
    beams = [
        PlanMember(
            start=(float(member["start_coord"][0]) * xy_scale_to_m, float(member["start_coord"][1]) * xy_scale_to_m),
            end=(float(member["end_coord"][0]) * xy_scale_to_m, float(member["end_coord"][1]) * xy_scale_to_m),
        )
        for member in fem_result["members"]
        if member["type"] == "beam"
    ]
    slabs = [
        [(float(point[0]) * xy_scale_to_m, float(point[1]) * xy_scale_to_m) for point in slab]
        for slab in fem_result.get("slabs", [])
    ]
    return FEMInput(walls=walls, beams=beams, slabs=slabs)


def export_structural_input_json(fem_result: dict, output_path: str | Path) -> None:
    """导出共享结构输入 JSON，供 ETABS/OpenSees/YJK 等后端复用。"""
    output_path = Path(output_path)
    export_data = fem_result_to_serializable(fem_result)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"  Exported structural input to: {output_path}")
