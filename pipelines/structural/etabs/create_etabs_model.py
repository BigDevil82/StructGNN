"""
ETABS自动建模脚本

流程：
1. 读取DXF建筑图纸，调用预测模型获取剪力墙和梁的坐标
2. 启动/连接ETABS，定义材料、截面、楼层
3. 按楼层循环创建剪力墙、梁、楼板构件
4. 对楼板施加恒载和活载
5. 保存模型（时程函数、荷载工况、质量源由用户在ETABS中手动设置）
6. 运行分析并提取结果
"""

import os
import sys
from pathlib import Path
from typing import Any

import comtypes
import numpy as np
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.case_study.symmetry_postprocess import (
    detect_left_right_symmetry,
    load_layout_reference_geometries,
    symmetrize_raw_walls,
)
from pipelines.case_study.adapters import fem_result_to_fem_input
from pipelines.structural.etabs.etabs_util import (
    ShellType,
    SlabType,
    create_ETABS_instance,
    define_beam_sec,
    define_Conc_Mat,
    define_slab_sec,
    define_wall_sec,
)
from pipelines.structural.etabs.unit import GPa, M, kN, mm
from src.shearwall_modeling.domain import FEMInput
from src.shearwall_pred.config import model_config, viz_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.utils import build_graph_from_dxf

# 初始化COM
comtypes.CoInitialize()


################################## 用户配置 ##################################

DXF_PATH = "data/dxf/shearwall_split_8_2/test/L27_136.dxf"  # 输入DXF文件路径
MODEL_DIR = "outputs/result/shearwall_pred/0126_cond_kfold"  # 训练好的模型目录
CATEGORY = None  # 建筑类别 (0/1/2/None)
DEVICE = "cuda"  # 推理设备
SYMMETRY_MODE = "none"  # 对称后处理模式: none/union/intersection
SYMMETRY_THRESHOLD = 0.85  # 对称检测阈值


MODEL_SAVE_DIR = "outputs/result/case_study/etabs_file"  # ETABS模型保存目录
MODEL_NAME = "model_001"  # 模型文件名（不含扩展名）

# ETABS程序路径；设为空字符串则使用COM注册路径
ETABS_PROGRAM_PATH = r"C:\Program Files\Computers and Structures\ETABS 18\ETABS.exe"
# True: 附加到已运行的ETABS实例；False: 启动新实例
ATTACH_TO_INSTANCE = True

######################## 结构参数 ########################
SCALE_FACTOR = 1.0  # 坐标缩放比例（根据DXF实际尺寸调整，确保建模尺寸合理）
floor_num = 18  # 楼层数
story_height = 3.0 * M  # 层高（米）
wall_thick = 200 * mm  # 剪力墙厚度
slab_thick = 120 * mm  # 楼板厚度
beam_h = 500 * mm  # 梁高
beam_b = 300 * mm  # 梁宽

FC = 30  # 混凝土强度等级 (C40)
# 荷载参数（kN/m²）
DEAD_LOAD = 5.0  # 附加恒载（面层、隔墙等，不含自重）
LIVE_LOAD = 2.0  # 活载（住宅）
APPLY_RIGID_DIAPHRAGM = True  # 是否为每层施加刚性楼板约束
DIAPHRAGM_Z_TOL = 1e-4  # 节点归层容差（m）

##############################################################################


def load_ensemble_model(model_dir: str, device: str) -> EnsembleShearWallGNN:
    """加载K-Fold Ensemble预测模型"""
    cv_path = Path(model_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))
    if not model_paths:
        raise FileNotFoundError(f"在 {cv_path} 下未找到fold模型")

    print(f"  加载 {len(model_paths)} 个fold模型...")
    model = EnsembleShearWallGNN(model_paths, model_config)
    model.to(device)
    model.eval()
    return model


def predict_fem_members(
    dxf_path: str,
    model_dir: str,
    category: int,
    device: str,
    use_gt: bool = False,
    symmetry_mode: str = "none",
    symmetry_threshold: float = 0.85,
) -> dict:
    """
    从DXF图纸获取剪力墙和梁的布置，返回FEM构件结果。

    Args:
        use_gt: True 使用Ground Truth标注，False 使用模型预测

    Returns:
        result dict，包含 'members' 列表，每项含 'type', 'start_coord', 'end_coord'
        坐标单位为毫米（mm）
    """
    from pipelines.case_study.fem_builder import FEMTopologyBuilder

    # 1. 读取DXF并构建图
    builder_graph = build_graph_from_dxf(dxf_path, mode="none")
    layout_geometries = load_layout_reference_geometries(dxf_path)
    symmetry_info = detect_left_right_symmetry(
        layout_geometries["infill_geometries"],
        layout_geometries["room_geometries"],
        threshold=symmetry_threshold,
    )

    if symmetry_info["axis_x"] is None:
        print("  左右对称检测: 无有效几何，跳过")
    else:
        print(
            "  左右对称检测: "
            f"source={symmetry_info['source']}, axis_x={symmetry_info['axis_x']:.2f}, "
            f"confidence={symmetry_info['confidence']:.3f}, "
            f"is_symmetric={symmetry_info['is_symmetric']}"
        )

    # 提取房间信息
    node_ids = list(builder_graph.graph.nodes())
    room_polys = [builder_graph.graph.nodes[n]["poly"] for n in node_ids]
    masks_list = [builder_graph.graph.nodes[n].get("masks", []) for n in node_ids]

    # 2. 获取剪力墙向量
    if use_gt:
        print("  使用 Ground Truth 标注")
        sw_vectors = []
        for n in node_ids:
            sv = builder_graph.graph.nodes[n].get("sw_vector")
            if sv is None:
                raise ValueError(f"节点 {n} 缺少 sw_vector，该DXF可能没有GT标注")
            sw_vectors.append(sv)
    else:
        print("  使用模型预测")
        data = builder_graph.to_pyg_data()
        if category is not None:
            cate_one_hot = np.zeros((1, 3))
            cate_one_hot[:, category] = 1.0
            data.condition = torch.tensor(cate_one_hot, dtype=torch.float)
        data_batch = Batch.from_data_list([data]).to(device)

        model = load_ensemble_model(model_dir, device)
        model.eval()
        with torch.no_grad():
            pred_prob, pred_ratio = model(data_batch)
            pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
            predictions = pred_combined.cpu().numpy()
            predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)
        sw_vectors = [predictions[i] for i in range(len(node_ids))]

    # 3. 构建FEM拓扑
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=100.0)
    for i, room_poly in enumerate(room_polys):
        fem_builder.add_room(room_poly, sw_vectors[i], masks_list[i])

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

    source = "GT" if use_gt else "预测"
    stats = result["statistics"]
    print(
        f"  {source}完成：{stats['num_shearwalls']} 剪力墙，{stats['num_beams']} 梁，"
        f"{stats['num_slabs']} 楼板，{stats['num_nodes']} 节点"
    )
    return result


def load_structural_input(input_path: str, xy_scale_to_m: float = 0.001) -> FEMInput:
    """从共享结构输入 JSON 加载结构建模输入。"""
    return FEMInput.from_json(Path(input_path), xy_scale_to_m=xy_scale_to_m)


def _normalize_structural_input(
    fem_result: dict | None = None,
    structural_input: FEMInput | None = None,
    structural_input_path: str | None = None,
    use_gt: bool = False,
) -> tuple[FEMInput, dict | None]:
    """统一 ETABS 后端的结构输入来源。"""
    if structural_input is not None:
        return structural_input, fem_result

    if structural_input_path is not None:
        return load_structural_input(structural_input_path), fem_result

    if fem_result is None:
        source = "GT标注" if use_gt else "模型预测"
        print(f"\n[{'GT' if use_gt else '预测'}] 从DXF获取FEM构件（{source}）...")
        device = DEVICE if torch.cuda.is_available() else "cpu"
        fem_result = predict_fem_members(
            DXF_PATH,
            MODEL_DIR,
            CATEGORY,
            device,
            use_gt=use_gt,
            symmetry_mode=SYMMETRY_MODE,
            symmetry_threshold=SYMMETRY_THRESHOLD,
        )

    return fem_result_to_fem_input(fem_result), fem_result


def print_structural_input_diagnostics(structural_input: FEMInput):
    """打印标准结构输入的几何诊断信息。"""
    members = structural_input.all_members()
    walls = structural_input.walls
    beams = structural_input.beams

    all_coords = []
    for member in members:
        all_coords.append(member.start)
        all_coords.append(member.end)

    if not all_coords:
        print("[诊断] 无构件数据")
        return

    xs = [coord[0] for coord in all_coords]
    ys = [coord[1] for coord in all_coords]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    print("\n" + "=" * 60)
    print("[诊断] 模型几何信息")
    print("=" * 60)
    print(f"  平面范围: {x_range / mm:.0f} x {y_range / mm:.0f} mm  ({x_range:.1f} x {y_range:.1f} m)")
    print(f"  构件数量: {len(walls)} 剪力墙, {len(beams)} 梁")

    if walls:
        wall_lens = [member.length for member in walls]
        print(
            f"  剪力墙长度: min={min(wall_lens)/mm:.0f}mm, max={max(wall_lens)/mm:.0f}mm, "
            f"avg={sum(wall_lens)/len(wall_lens)/mm:.0f}mm"
        )
        print(f"             ({min(wall_lens):.2f}m ~ {max(wall_lens):.2f}m)")

    if beams:
        beam_lens = [member.length for member in beams]
        print(
            f"  梁跨度:     min={min(beam_lens)/mm:.0f}mm, max={max(beam_lens)/mm:.0f}mm, "
            f"avg={sum(beam_lens)/len(beam_lens)/mm:.0f}mm"
        )
        print(f"             ({min(beam_lens):.2f}m ~ {max(beam_lens):.2f}m)")

    slabs = structural_input.slabs
    print(f"  楼板数量: {len(slabs)}")

    print("\n[诊断] 合理性检查:")
    if x_range > 60 or y_range > 60:
        print(f"  平面尺寸偏大（>{60}m），可能缩放比例有误")
    elif x_range < 10 or y_range < 10:
        print("  平面尺寸偏小（<10m），可能缩放比例有误")
    else:
        print("  平面尺寸在合理范围内")

    if beams:
        max_span = max(beam_lens)
        if max_span > 12:
            print(f"  最大梁跨 {max_span:.1f}m 偏大（住宅一般 3-6m）")
        else:
            print("  梁跨在合理范围内")

    if len(slabs) == 0:
        print("  无楼板！模型缺少质量和面荷载")
    print("=" * 60)


def print_model_diagnostics(fem_result: dict):
    """打印模型几何尺寸诊断信息，用于检查建模是否合理"""
    from pipelines.structural.etabs.unit import mm

    members = fem_result["members"]
    walls = [m for m in members if m["type"] == "shearwall"]
    beams = [m for m in members if m["type"] == "beam"]

    # 所有构件端点坐标（mm）
    all_coords = []
    for m in members:
        all_coords.append(m["start_coord"])
        all_coords.append(m["end_coord"])

    if not all_coords:
        print("[诊断] 无构件数据")
        return

    xs = [c[0] * SCALE_FACTOR for c in all_coords]
    ys = [c[1] * SCALE_FACTOR for c in all_coords]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    print("\n" + "=" * 60)
    print("[诊断] 模型几何信息")
    print("=" * 60)
    print(f"  平面范围: {x_range:.0f} x {y_range:.0f} mm  ({x_range * mm:.1f} x {y_range * mm:.1f} m)")
    print(f"  构件数量: {len(walls)} 剪力墙, {len(beams)} 梁")

    if walls:
        wall_lens = [m["length"] * SCALE_FACTOR for m in walls]
        print(
            f"  剪力墙长度: min={min(wall_lens):.0f}mm, max={max(wall_lens):.0f}mm, "
            f"avg={sum(wall_lens)/len(wall_lens):.0f}mm"
        )
        print(f"             ({min(wall_lens)*mm:.2f}m ~ {max(wall_lens)*mm:.2f}m)")

    if beams:
        beam_lens = [m["length"] * SCALE_FACTOR for m in beams]
        print(
            f"  梁跨度:     min={min(beam_lens):.0f}mm, max={max(beam_lens):.0f}mm, "
            f"avg={sum(beam_lens)/len(beam_lens):.0f}mm"
        )
        print(f"             ({min(beam_lens)*mm:.2f}m ~ {max(beam_lens)*mm:.2f}m)")

    slabs = fem_result.get("slabs", [])
    print(f"  楼板数量: {len(slabs)}")

    # 合理性检查
    print("\n[诊断] 合理性检查:")
    if x_range * mm > 60 or y_range * mm > 60:
        print(f"  ⚠ 平面尺寸偏大（>{60}m），可能缩放比例有误")
    elif x_range * mm < 10 or y_range * mm < 10:
        print(f"  ⚠ 平面尺寸偏小（<10m），可能缩放比例有误")
    else:
        print(f"  ✓ 平面尺寸在合理范围内")

    if beams:
        max_span = max(beam_lens) * mm
        if max_span > 12:
            print(f"  ⚠ 最大梁跨 {max_span:.1f}m 偏大（住宅一般 3-6m）")
        else:
            print(f"  ✓ 梁跨在合理范围内")

    if len(slabs) == 0:
        print(f"  ⚠ 无楼板！模型缺少质量和面荷载")
    print("=" * 60)


def _fem_input_to_etabs_geometry(structural_input: FEMInput) -> dict[str, Any]:
    all_coords = []
    for member in structural_input.all_members():
        all_coords.append(member.start)
        all_coords.append(member.end)

    min_x = min((coord[0] for coord in all_coords), default=0.0)
    min_y = min((coord[1] for coord in all_coords), default=0.0)

    wall_members = [{"start_coord": member.start, "end_coord": member.end} for member in structural_input.walls]
    beam_members = [{"start_coord": member.start, "end_coord": member.end} for member in structural_input.beams]
    return {
        "wall_members": wall_members,
        "beam_members": beam_members,
        "slab_polys": structural_input.slabs,
        "min_xy": (min_x, min_y),
    }


def check_ret(ret, msg: str = ""):
    """检查ETABS API调用返回值，非0表示失败"""
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0 and msg:
        print(f"警告：{msg} (ret={ret})")
    return ret


def _define_diaphragm(etabs, diaphragm_name: str) -> int:
    """定义楼板约束，优先使用 ETABS Diaphragm API。"""
    if hasattr(etabs, "Diaphragm") and hasattr(etabs.Diaphragm, "SetDiaphragm"):
        try:
            # False 表示刚性楼板（非半刚性）
            return etabs.Diaphragm.SetDiaphragm(diaphragm_name, False)
        except Exception:
            pass

    if hasattr(etabs, "ConstraintDef") and hasattr(etabs.ConstraintDef, "SetDiaphragm"):
        try:
            return etabs.ConstraintDef.SetDiaphragm(diaphragm_name, 3)
        except Exception:
            pass

    return -1


def _assign_point_diaphragm(etabs, point_name: str, diaphragm_name: str) -> int:
    """给节点施加楼板约束，兼容不同 ETABS 版本的接口签名。"""
    if not hasattr(etabs, "PointObj"):
        return -1

    setters = [
        lambda: etabs.PointObj.SetDiaphragm(point_name, 3, diaphragm_name),
        lambda: etabs.PointObj.SetDiaphragm(point_name, diaphragm_name),
        lambda: etabs.PointObj.SetConstraint(point_name, diaphragm_name),
    ]
    for setter in setters:
        try:
            return setter()
        except Exception:
            continue
    return -1


def apply_rigid_diaphragm_constraints(
    etabs,
    floor_num: int,
    story_height: float,
    z_tol: float = 1e-4,
):
    """为每层创建刚性楼板约束，并将该层所有节点施加到对应约束。"""
    print("\n[约束] 创建并施加每层刚性楼板约束...")

    point_ret = etabs.PointObj.GetNameList()
    if check_ret(point_ret, "获取节点列表") != 0:
        print("  跳过刚性楼板约束：无法获取节点列表")
        return

    point_names = point_ret[1]
    points_by_floor = {i: [] for i in range(1, floor_num + 1)}

    for p_name in point_names:
        coord_ret = etabs.PointObj.GetCoordCartesian(p_name)
        if check_ret(coord_ret) != 0:
            continue
        z_val = coord_ret[2]

        for floor_idx in range(1, floor_num + 1):
            target_z = floor_idx * story_height
            if abs(z_val - target_z) <= z_tol:
                points_by_floor[floor_idx].append(p_name)
                break

    total_assigned = 0
    for floor_idx in range(1, floor_num + 1):
        diaphragm_name = f"D{floor_idx}"
        ret = _define_diaphragm(etabs, diaphragm_name)
        if check_ret(ret, f"定义楼板约束 {diaphragm_name}") != 0:
            print(f"  跳过 {diaphragm_name}: 约束定义失败")
            continue

        assigned_count = 0
        for p_name in points_by_floor[floor_idx]:
            ret = _assign_point_diaphragm(etabs, p_name, diaphragm_name)
            if check_ret(ret) == 0:
                assigned_count += 1

        total_assigned += assigned_count
        print(f"  {diaphragm_name}: 施加节点数 {assigned_count}")

    print(f"  刚性楼板约束施加完成，总节点数: {total_assigned}")


def run_analysis_and_extract(etabs):
    """运行分析并提取关键结果（假设用户已手动设置好荷载工况和时程分析）"""
    print("\n[分析] 运行分析...")
    ret = etabs.Analyze.RunAnalysis()
    ret = check_ret(ret, "运行分析")

    if ret != 0:
        print("分析失败，无法提取结果")
        return None

    # 提取模态结果
    print("\n[结果] 提取模态分析结果...")
    etabs.Results.Setup.DeselectAllCasesAndCombosForOutput()
    etabs.Results.Setup.SetCaseSelectedForOutput("Modal")

    ret = etabs.Results.ModalPeriod()
    if check_ret(ret) == 0:
        n_modes = ret[0]
        periods = ret[4]
        print(f"  前 {min(n_modes, 5)} 阶模态周期:")
        for i in range(min(n_modes, 5)):
            print(f"    Mode {i + 1}: T = {periods[i]:.4f} s")

    # 提取层间位移角（选择所有已定义的工况）
    print("\n[结果] 提取层间位移角...")
    ret = etabs.Results.StoryDrifts()
    if check_ret(ret) == 0:
        n = ret[0]
        stories = ret[1]
        load_cases = ret[2]
        drifts_x = ret[5]
        drifts_y = ret[6]
        print(f"  {'楼层':<10} {'工况':<20} {'X位移角':<15} {'Y位移角':<15}")
        for i in range(n):
            print(f"  {stories[i]:<10} {load_cases[i]:<20} {drifts_x[i]:<15.6f} {drifts_y[i]:<15.6f}")

        max_drift_x = max(abs(d) for d in drifts_x) if drifts_x else 0
        max_drift_y = max(abs(d) for d in drifts_y) if drifts_y else 0
        max_drift = max(max_drift_x, max_drift_y)
        print(f"\n  最大层间位移角: 1/{int(1 / max_drift) if max_drift > 0 else 'inf'}")
        if max_drift > 1 / 1000:
            print("  *** 超限！剪力墙结构限值为 1/1000 ***")
        else:
            print("  满足规范要求（限值 1/1000）")

    return ret


def create_etabs_model(
    fem_result: dict = None,
    structural_input: FEMInput | None = None,
    structural_input_path: str | None = None,
    run_analysis: bool = False,
    use_gt: bool = False,
):
    """
    根据FEM预测结果在ETABS中创建结构模型。

    Args:
        fem_result: predict_fem_members() 的返回值。兼容旧接口。
        structural_input: 标准化结构输入，推荐优先使用。
        structural_input_path: 标准化结构输入 JSON 路径。
        run_analysis: 是否在建模后运行分析并提取结果。
                      设为True前需确保已在ETABS中手动设置好时程函数、荷载工况、质量源。
        use_gt: True 使用Ground Truth标注建模，False 使用模型预测建模。
    """

    structural_input, fem_result = _normalize_structural_input(
        fem_result=fem_result,
        structural_input=structural_input,
        structural_input_path=structural_input_path,
        use_gt=use_gt,
    )

    print_structural_input_diagnostics(structural_input)

    geometry_data = _fem_input_to_etabs_geometry(structural_input)
    wall_members = geometry_data["wall_members"]
    beam_members = geometry_data["beam_members"]
    slab_polys = geometry_data["slab_polys"]
    min_x, min_y = geometry_data["min_xy"]

    N_m_C = 10  # ETABS单位系统：N, m, C

    # 荷载值（转换为 N/m²，与ETABS单位系统一致）
    dead_load_val = DEAD_LOAD * kN / M**2  # kN/m² → N/m²
    live_load_val = LIVE_LOAD * kN / M**2

    ######################## 坐标换算 ########################
    # 预测坐标单位为毫米，转换至米
    def to_meters(coord):
        return ((coord[0] - min_x) * SCALE_FACTOR, (coord[1] - min_y) * SCALE_FACTOR)

    ######################## 启动ETABS ########################
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.abspath(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + ".EDB"))

    etabs, etabs_obj, created_new = create_ETABS_instance(ATTACH_TO_INSTANCE, ETABS_PROGRAM_PATH)

    if created_new:
        etabs.InitializeNewModel(N_m_C)
        etabs.File.NewBlank()

    etabs.SetPresentUnits(N_m_C)

    ###################### 定义材料和截面 ####################
    mat_name = f"C{FC}"
    define_Conc_Mat(etabs, mat_name, FC * GPa, 0.2, 1e-5, mat_name)

    wall_sec = "shearwall"
    define_wall_sec(etabs, wall_sec, ShellType.ShellThin.value, mat_name, wall_thick)

    slab_sec = "slab"
    define_slab_sec(etabs, slab_sec, SlabType.Slab.value, ShellType.ShellThin.value, mat_name, slab_thick)

    beam_sec = f"beam_{int(beam_b / mm)}x{int(beam_h / mm)}"
    define_beam_sec(etabs, beam_sec, mat_name, beam_h, beam_b)

    ###################### 定义楼层 ##########################
    story_names = [f"Story{i}" for i in range(1, floor_num + 1)]
    ret = etabs.Story.SetStories_2(
        0.0,
        floor_num,
        story_names,
        [story_height] * floor_num,
        [True if i == 0 else False for i in range(floor_num)],  # Story1 为主楼层
        ["" if i == 0 else "Story1" for i in range(floor_num)],
        [False] * floor_num,
        [0.0] * floor_num,
        [0] * floor_num,
    )
    check_ret(ret, "定义楼层")

    ###################### 按楼层创建构件 ####################
    # 收集楼板对象名称，用于后续施加荷载
    slab_obj_names = []

    def create_floor_members(floor_z: float):
        z_bot = floor_z - story_height
        z_top = floor_z

        # 剪力墙：四节点面单元
        for member in wall_members:
            p1 = to_meters(member["start_coord"])
            p2 = to_meters(member["end_coord"])
            x = [p1[0], p2[0], p2[0], p1[0]]
            y = [p1[1], p2[1], p2[1], p1[1]]
            z = [z_bot, z_bot, z_top, z_top]
            ret = etabs.AreaObj.AddByCoord(4, x, y, z, "", wall_sec)
            check_ret(ret, f"创建剪力墙 p1={p1} p2={p2}")

        # 梁：线单元（设置在楼层顶面）
        for member in beam_members:
            p1 = to_meters(member["start_coord"])
            p2 = to_meters(member["end_coord"])
            ret = etabs.FrameObj.AddByCoord(
                p1[0],
                p1[1],
                z_top,
                p2[0],
                p2[1],
                z_top,
                "",
                beam_sec,
            )
            check_ret(ret, f"创建梁 p1={p1} p2={p2}")

        # 楼板：多边形面单元（设置在楼层顶面）
        for slab_coords in slab_polys:
            n = len(slab_coords)
            x = [to_meters(pt)[0] for pt in slab_coords]
            y = [to_meters(pt)[1] for pt in slab_coords]
            z = [z_top] * n
            ret = etabs.AreaObj.AddByCoord(n, x, y, z, "", slab_sec)
            check_ret(ret, "创建楼板")
            # AddByCoord 返回 (name, ret_code)，提取楼板名称
            if isinstance(ret, (tuple, list)) and len(ret) >= 2 and ret[-1] == 0:
                slab_obj_names.append(ret[3])

    refresh_freq = 3
    for i in range(1, floor_num + 1):
        create_floor_members(i * story_height)
        if i % refresh_freq == 0 or i == floor_num:
            etabs.View.RefreshView(0, False)

    if APPLY_RIGID_DIAPHRAGM:
        apply_rigid_diaphragm_constraints(
            etabs,
            floor_num=floor_num,
            story_height=story_height,
            z_tol=DIAPHRAGM_Z_TOL,
        )

    ###################### 底层节点固定约束 ####################
    print("\n[约束] 对底层节点施加固定约束...")
    point_ret = etabs.PointObj.GetNameList()
    point_names = []
    if check_ret(point_ret) == 0:
        point_names = point_ret[1]

    fixed_count = 0
    tol = 1e-6  # m
    for p_name in point_names:
        coord_ret = etabs.PointObj.GetCoordCartesian(p_name)
        if check_ret(coord_ret) != 0:
            continue

        x, y, z = coord_ret[0], coord_ret[1], coord_ret[2]
        if abs(z) <= tol:
            # UX, UY, UZ, RX, RY, RZ 全固定
            ret = etabs.PointObj.SetRestraint(p_name, [True, True, True, True, True, True])
            ret = check_ret(ret, f"固定底层节点 {p_name} @ ({x:.3f}, {y:.3f}, {z:.3f})")
            if ret == 0:
                fixed_count += 1

    print(f"  已固定底层节点数量: {fixed_count}")

    #################### 施加楼板荷载 #################
    print(f"\n[荷载] 对 {len(slab_obj_names)} 块楼板施加荷载...")
    print(f"  附加恒载: {DEAD_LOAD} kN/m²，活载: {LIVE_LOAD} kN/m²")

    for slab_name in slab_obj_names:
        # 施加附加恒载（Dir=6 表示面法向，即重力方向）
        ret = etabs.AreaObj.SetLoadUniform(slab_name, "Dead", -dead_load_val, 6)
        check_ret(ret, f"施加恒载 {slab_name}")

        # 施加活载
        ret = etabs.AreaObj.SetLoadUniform(slab_name, "Live", -live_load_val, 6)
        check_ret(ret, f"施加活载 {slab_name}")

    ######################## 保存模型 ########################
    ret = etabs.File.Save(model_path)
    check_ret(ret, "保存ETABS模型")
    if ret == 0:
        print(f"\nETABS模型已保存：{model_path}")
    else:
        print("错误：模型保存失败")
        return ""

    ###################### 运行分析 ##########################
    if run_analysis:
        run_analysis_and_extract(etabs)

    return model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="从DXF预测结果自动创建ETABS有限元模型")
    parser.add_argument("--dxf_path", type=str, default=DXF_PATH)
    parser.add_argument(
        "--structural_input_path",
        type=str,
        default=None,
        help="共享结构输入 JSON 路径；提供后优先使用，不再依赖 DXF 推理流程",
    )
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument(
        "--scale_factor", type=float, default=SCALE_FACTOR, help="坐标缩放比例（默认1.0，单位转换时调整）"
    )
    parser.add_argument("--category", type=int, default=CATEGORY, choices=[0, 1, 2])
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--output_dir", type=str, default=MODEL_SAVE_DIR)
    parser.add_argument(
        "--model_name", type=str, default=None, help="保存的ETABS模型文件名（不含扩展名），默认为DXF文件名"
    )
    parser.add_argument(
        "--run_analysis",
        action="store_true",
        help="建模后运行分析（需先在ETABS中手动设置时程/荷载工况/质量源）",
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        help="使用Ground Truth标注建模（而非模型预测）",
    )
    parser.add_argument(
        "--symmetry_mode",
        type=str,
        default=SYMMETRY_MODE,
        choices=["none", "union", "intersection"],
        help="左右对称后处理模式",
    )
    parser.add_argument(
        "--symmetry_threshold",
        type=float,
        default=SYMMETRY_THRESHOLD,
        help="左右对称检测 IoU 阈值",
    )
    parser.add_argument(
        "--disable_rigid_diaphragm",
        action="store_true",
        help="不施加每层刚性楼板约束",
    )
    parser.add_argument(
        "--diaphragm_z_tol",
        type=float,
        default=DIAPHRAGM_Z_TOL,
        help="节点归层的z坐标容差（m）",
    )
    args = parser.parse_args()

    # 支持命令行覆盖配置
    DXF_PATH = args.dxf_path
    MODEL_DIR = args.model_dir
    CATEGORY = args.category
    SCALE_FACTOR = args.scale_factor
    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_DIR = args.output_dir
    SYMMETRY_MODE = args.symmetry_mode
    SYMMETRY_THRESHOLD = args.symmetry_threshold
    APPLY_RIGID_DIAPHRAGM = not args.disable_rigid_diaphragm
    DIAPHRAGM_Z_TOL = args.diaphragm_z_tol
    MODEL_NAME = args.model_name or os.path.basename(DXF_PATH).split(".")[0]
    if args.use_gt:
        MODEL_NAME += "_GT"

    model_path = create_etabs_model(
        structural_input_path=args.structural_input_path,
        run_analysis=args.run_analysis,
        use_gt=args.use_gt,
    )
    if model_path:
        print(f"建模成功：{model_path}")
    else:
        print("建模失败")


if __name__ == "__main__":
    main()
