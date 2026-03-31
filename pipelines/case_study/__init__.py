"""Compatibility wrapper for the future case study pipeline package."""

from pipelines.case_study.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from pipelines.case_study.run_case_study import main, run_case_study
from pipelines.case_study.symmetry_postprocess import (
    detect_left_right_symmetry,
    load_layout_reference_geometries,
    save_symmetry_detection_debug_plot,
    symmetrize_raw_walls,
)
from pipelines.case_study.unit import GPa, M, MPa, N, Pa, kg, kN, mm, s, ton

__all__ = [
    "FEMTopologyBuilder",
    "export_to_json",
    "visualize_fem_result",
    "detect_left_right_symmetry",
    "load_layout_reference_geometries",
    "save_symmetry_detection_debug_plot",
    "symmetrize_raw_walls",
    "M",
    "s",
    "kg",
    "mm",
    "N",
    "kN",
    "Pa",
    "MPa",
    "GPa",
    "ton",
    "main",
    "run_case_study",
]
