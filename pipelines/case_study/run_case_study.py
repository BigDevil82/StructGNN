"""
Engineering Case Study: 从建筑平面图到推理、后处理与可视化
"""

import argparse

import torch

from pipelines.case_study.pipeline_core import run_case_study


def main():
    parser = argparse.ArgumentParser(description="Engineering Case Study")
    parser.add_argument("--dxf_path", type=str, required=True, help="输入DXF文件路径")
    parser.add_argument("--model_dir", type=str, default="outputs/result/shearwall_pred/0126_cond_kfold")
    parser.add_argument("--output_dir", type=str, default="outputs/result/case_study")
    parser.add_argument("--category", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--symmetry_mode",
        type=str,
        default="none",
        choices=["none", "union", "intersection"],
        help="左右对称后处理模式",
    )
    parser.add_argument(
        "--symmetry_threshold",
        type=float,
        default=0.85,
        help="左右对称检测 IoU 阈值",
    )

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_case_study(
        dxf_path=args.dxf_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        category=args.category,
        device=args.device,
        symmetry_mode=args.symmetry_mode,
        symmetry_threshold=args.symmetry_threshold,
    )


if __name__ == "__main__":
    main()
