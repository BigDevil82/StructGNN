"""Compatibility entry point for the case study pipeline.

At this stage the concrete implementation still lives under
`experiments.case_study.run_case_study`, but this wrapper is the stable
pipeline-facing import path and should be preferred by new code.
"""

from experiments.case_study.run_case_study import main, run_case_study
from experiments.pipelines.case_study.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from experiments.pipelines.case_study.symmetry_postprocess import (
    detect_left_right_symmetry,
    load_layout_reference_geometries,
    save_symmetry_detection_debug_plot,
    symmetrize_raw_walls,
)

__all__ = [
    "FEMTopologyBuilder",
    "export_to_json",
    "visualize_fem_result",
    "detect_left_right_symmetry",
    "load_layout_reference_geometries",
    "save_symmetry_detection_debug_plot",
    "symmetrize_raw_walls",
    "main",
    "run_case_study",
]


if __name__ == "__main__":
    main()
