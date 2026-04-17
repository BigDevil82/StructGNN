import argparse

from src.shearwall_modeling.parametric import DatasetGenerationConfig
from src.shearwall_optimization.algorithms import GeneticAlgorithmConfig, RandomSearchConfig
from src.shearwall_optimization.runners import run_shearwall_optimization


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run shearwall optimization on one layout.")
    p.add_argument("--layout-path", required=True)
    p.add_argument("--out", default="outputs/result/optimization/ga_result.json")
    p.add_argument("--algorithm", choices=["ga", "random"], default="ga")

    p.add_argument("--N", type=int, default=28)
    p.add_argument("--hs", type=int, default=120)
    p.add_argument("--h-story", type=float, default=2.9)
    p.add_argument("--intensity", type=float, default=8.0)
    p.add_argument("--site-class", default="II")
    p.add_argument("--seismic-group", type=int, default=1)

    p.add_argument("--num-modes", type=int, default=6)
    p.add_argument("--enable-auto-scale", action="store_true", default=True)
    p.add_argument("--disable-auto-scale", action="store_true")
    p.add_argument("--scale-low", type=float, default=2.0)
    p.add_argument("--scale-high", type=float, default=6.0)
    p.add_argument("--scale-seed", type=int, default=42)
    p.add_argument("--manual-scale-factor", type=float, default=None)

    p.add_argument("--ga-pop", type=int, default=24)
    p.add_argument("--ga-gen", type=int, default=20)
    p.add_argument("--ga-crossover", type=float, default=0.9)
    p.add_argument("--ga-mutation", type=float, default=0.2)
    p.add_argument("--ga-elite", type=int, default=2)
    p.add_argument("--ga-tournament", type=int, default=3)

    p.add_argument("--random-trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    auto_scale = False if args.disable_auto_scale else args.enable_auto_scale

    analysis_cfg = DatasetGenerationConfig(
        samples_per_layout=1,
        samples_per_task=1,
        max_workers=0,
        num_modes=args.num_modes,
        enable_auto_scale=auto_scale,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        scale_seed=args.scale_seed,
        manual_scale_factor=args.manual_scale_factor,
    )

    fixed = {
        "N": args.N,
        "hs": args.hs,
        "h_story": args.h_story,
        "intensity": args.intensity,
        "site_class": args.site_class,
        "seismic_group": args.seismic_group,
    }

    ga_cfg = GeneticAlgorithmConfig(
        population_size=args.ga_pop,
        generations=args.ga_gen,
        crossover_rate=args.ga_crossover,
        mutation_rate=args.ga_mutation,
        elite_size=args.ga_elite,
        tournament_size=args.ga_tournament,
        seed=args.seed,
    )
    random_cfg = RandomSearchConfig(n_trials=args.random_trials, seed=args.seed)

    result = run_shearwall_optimization(
        layout_path=args.layout_path,
        out_path=args.out,
        algorithm=args.algorithm,
        fixed_params=fixed,
        analysis_cfg=analysis_cfg,
        ga_cfg=ga_cfg,
        random_cfg=random_cfg,
    )

    print("best_objective:", result.best_objective)
    print("best_feasible:", result.best_feasible)
    print("best_solution:", result.best_solution)


if __name__ == "__main__":
    main()
