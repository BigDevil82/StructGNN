from pathlib import Path

from shearwall_modeling import ModelConfig, ShearWallAnalysisEngine, load_and_scale_input


def main() -> None:
    # User-configurable inputs for parameterized invocation.
    json_path = Path(r"result\case_study\archi_comp_fem_data.json")
    input_unit_scale_to_m = 0.001  # JSON coordinates are in mm.
    stories = 8
    story_height = 3.0
    num_modes = 6
    enable_auto_scale = True
    scale_low = 6.0
    scale_high = 12.0
    scale_seed = 42
    manual_scale_factor = 1
    combine_method = "CQC"
    builder_name = "detailed_shell"

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
    result = engine.analyze(input_data=input_data, config=config, builder_name=builder_name)
    drifts = result.drifts

    print(f"{config.seismic.combination_method} inter-story drift ratios:")
    for direction in ("X", "Y"):
        values = drifts[direction]
        print(f"  {direction}: {[round(v, 6) for v in values]}")

    engine.cleanup()


if __name__ == "__main__":
    main()
