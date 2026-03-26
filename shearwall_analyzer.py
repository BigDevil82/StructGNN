from pathlib import Path

from shearwall_modeling import ModelConfig, ShearWallAnalysisEngine, load_and_scale_input
from shearwall_modeling.evaluation import SeismicCodeChecker


def main() -> None:
    # User-configurable inputs for parameterized invocation.
    json_path = Path(r"result\case_study\archi_comp_fem_data.json")
    input_unit_scale_to_m = 0.001  # JSON coordinates are in mm.
    stories = 18
    story_height = 3.0
    num_modes = 6
    enable_auto_scale = True
    scale_low = 6.0
    scale_high = 12.0
    scale_seed = 42
    manual_scale_factor = 1
    combine_method = "CQC"
    builder_name = "detailed_shell"  # "equivalent_frame" or "detailed_shell"

    input_data, scale = load_and_scale_input(
        json_path=json_path,
        input_unit_scale_to_m=input_unit_scale_to_m,
        enable_auto_scale=enable_auto_scale,
        low=scale_low,
        high=scale_high,
        seed=scale_seed,
        manual_factor=manual_scale_factor,
    )

    print(f"Geometry scale factor used: {scale:.6f}")

    config = ModelConfig(num_stories=stories, story_height=story_height, num_modes=num_modes)
    config.seismic.combination_method = combine_method

    engine = ShearWallAnalysisEngine()
    # 1. 仅使用 Engine 进行物理建模 (跳过其自带的 analyze 分析方法)
    builder = engine.registry.get(builder_name)
    build_result = builder.build(input_data, config)

    # 2. 挂载综合校核器 (它将接管特征值求解和反应谱迭代)
    checker = SeismicCodeChecker(build_result.master_nodes, config)

    # 一次性返回：普通的质心层间位移角(兼容原有需求) + 规范校核结果
    drifts, checks = checker.run_analysis_and_evaluate()

    # 打印原来需要的位移角
    print(f"\n{config.seismic.combination_method} inter-story drift ratios:")
    for direction in ("X", "Y"):
        print(f"  {direction}: {[round(v, 6) for v in drifts[direction]]}")

    engine.cleanup()


if __name__ == "__main__":
    main()
