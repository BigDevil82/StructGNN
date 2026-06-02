import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shearwall_modeling.parametric import DatasetGenerationConfig
from src.shearwall_optimization.algorithms import (
    GeneticAlgorithmConfig,
    NSGA2Config,
    OptunaBayesConfig,
    ParticleSwarmConfig,
    RandomSearchConfig,
    RandomPreselectionConfig,
)
from src.shearwall_optimization.problems import ShearWallLimitConfig, ShearWallObjectiveConfig
from src.shearwall_optimization.runners import run_shearwall_optimization
from src.shearwall_optimization.steel_ranking import SteelRankingConfig
from src.shearwall_optimization.surrogate_evaluation import SurrogateAcceptanceConfig
from src.shearwall_optimization.surrogate_screening import SurrogateScreeningConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run shearwall optimization on one layout.")
    p.add_argument("--layout-path", required=True)
    p.add_argument("--out", default="outputs/result/optimization/ga_result.json")
    p.add_argument("--algorithm", choices=["ga", "pso", "nsga2", "optuna", "random"], default="ga")

    p.add_argument("--N", type=int, default=18)
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
    p.add_argument("--optimizer-workers", type=int, default=0, help="0 means sequential evaluation.")

    p.add_argument("--steel-price-per-kg", type=float, default=0.005)
    p.add_argument("--infeasible-penalty", type=float, default=1e6)
    p.add_argument("--limit-max-torsion", type=float, default=1.5)
    p.add_argument("--limit-max-drift", type=float, default=1.0 / 1000.0)
    p.add_argument("--limit-min-shear-weight", type=float, default=0.016)
    p.add_argument("--limit-min-stiffness", type=float, default=0.7)
    p.add_argument("--limit-max-period-ratio", type=float, default=0.9)

    p.add_argument("--ga-pop", type=int, default=24)
    p.add_argument("--ga-gen", type=int, default=20)
    p.add_argument("--ga-crossover", type=float, default=0.9)
    p.add_argument("--ga-mutation", type=float, default=0.3)
    p.add_argument("--ga-elite", type=int, default=2)
    p.add_argument("--ga-tournament", type=int, default=3)
    p.add_argument("--ga-verbose", action="store_true")
    p.add_argument("--ga-log-every", type=int, default=1)
    p.add_argument("--ga-surrogate-screen", action="store_true")
    p.add_argument(
        "--ga-surrogate-artifact",
        default=r"data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt",
    )
    p.add_argument(
        "--ga-surrogate-graph-cache",
        default=r"data\parametric\surrogate_dataset\gnn_room_graph_cache",
    )
    p.add_argument(
        "--ga-surrogate-layout-features",
        default=r"data\parametric\surrogate_dataset\layout_features.parquet",
    )
    p.add_argument("--ga-surrogate-threshold", type=float, default=None)
    p.add_argument("--ga-surrogate-batch-size", type=int, default=512)
    p.add_argument("--ga-surrogate-accept", action="store_true")
    p.add_argument(
        "--ga-steel-artifact-dir",
        default=r"data\parametric\surrogate_dataset\steel_quantile_lightgbm",
    )
    p.add_argument("--ga-accept-pass-threshold", type=float, default=0.99)
    p.add_argument("--ga-accept-max-steel-rel-gap", type=float, default=0.75)
    p.add_argument("--ga-accept-audit-rate", type=float, default=0.0)
    p.add_argument("--ga-steel-ranking", action="store_true")
    p.add_argument(
        "--ga-steel-ranking-artifact",
        default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\gnn_steel.pt",
    )
    p.add_argument(
        "--ga-steel-ranking-graph-cache",
        default=r"data\parametric\cache\gnn_room_graph_cache",
    )
    p.add_argument(
        "--ga-steel-ranking-layout-features",
        default=r"data\parametric\surrogate_dataset\layout_features.parquet",
    )
    p.add_argument("--ga-steel-ranking-eval-ratio", type=float, default=0.4)
    p.add_argument("--ga-steel-ranking-min-eval", type=int, default=8)
    p.add_argument("--ga-steel-ranking-random-ratio", type=float, default=0.1)
    p.add_argument("--ga-steel-ranking-batch-size", type=int, default=512)
    p.add_argument("--ga-steel-ranking-use-feasibility", action="store_true")
    p.add_argument(
        "--ga-steel-ranking-feasibility-artifact",
        default=r"data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt",
    )
    p.add_argument(
        "--ga-steel-ranking-feasibility-graph-cache",
        default=r"data\parametric\cache\gnn_room_graph_cache",
    )
    p.add_argument(
        "--ga-steel-ranking-feasibility-layout-features",
        default=r"data\parametric\surrogate_dataset\layout_features.parquet",
    )
    p.add_argument("--ga-steel-ranking-feasibility-threshold", type=float, default=None)
    p.add_argument("--ga-steel-ranking-feasibility-penalty-kg", type=float, default=200000.0)
    p.add_argument("--ga-random-preselect", action="store_true")
    p.add_argument("--ga-random-preselect-eval-ratio", type=float, default=0.5)
    p.add_argument("--ga-random-preselect-min-eval", type=int, default=8)

    p.add_argument("--pso-swarm", type=int, default=24)
    p.add_argument("--pso-iter", type=int, default=20)
    p.add_argument("--pso-inertia", type=float, default=0.72)
    p.add_argument("--pso-cognitive", type=float, default=1.49)
    p.add_argument("--pso-social", type=float, default=1.49)
    p.add_argument("--pso-vclamp", type=float, default=0.25)

    p.add_argument("--random-trials", type=int, default=200)
    p.add_argument("--optuna-trials", type=int, default=50)
    p.add_argument("--optuna-startup-trials", type=int, default=10)
    p.add_argument("--optuna-progress", action="store_true")
    p.add_argument("--nsga2-trials", type=int, default=80)
    p.add_argument("--nsga2-pop", type=int, default=24)
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
        max_workers=(args.optimizer_workers if args.optimizer_workers > 0 else None),
        verbose=args.ga_verbose,
        log_every=args.ga_log_every,
        seed=args.seed,
        surrogate_screening=SurrogateScreeningConfig(
            enabled=args.ga_surrogate_screen,
            artifact_path=args.ga_surrogate_artifact,
            graph_cache_dir=args.ga_surrogate_graph_cache,
            layout_features_path=args.ga_surrogate_layout_features,
            screening_threshold=args.ga_surrogate_threshold,
            batch_size=args.ga_surrogate_batch_size,
        ),
        surrogate_acceptance=SurrogateAcceptanceConfig(
            enabled=args.ga_surrogate_accept,
            steel_artifact_dir=args.ga_steel_artifact_dir,
            pass_probability_threshold=args.ga_accept_pass_threshold,
            max_steel_rel_upper_gap=args.ga_accept_max_steel_rel_gap,
            audit_rate=args.ga_accept_audit_rate,
            seed=args.seed,
        ),
        steel_ranking=SteelRankingConfig(
            enabled=args.ga_steel_ranking,
            artifact_path=args.ga_steel_ranking_artifact,
            graph_cache_dir=args.ga_steel_ranking_graph_cache,
            layout_features_path=args.ga_steel_ranking_layout_features,
            eval_ratio=args.ga_steel_ranking_eval_ratio,
            min_eval_count=args.ga_steel_ranking_min_eval,
            random_ratio=args.ga_steel_ranking_random_ratio,
            batch_size=args.ga_steel_ranking_batch_size,
            use_feasibility_penalty=args.ga_steel_ranking_use_feasibility,
            feasibility_artifact_path=args.ga_steel_ranking_feasibility_artifact,
            feasibility_graph_cache_dir=args.ga_steel_ranking_feasibility_graph_cache,
            feasibility_layout_features_path=args.ga_steel_ranking_feasibility_layout_features,
            feasibility_threshold=args.ga_steel_ranking_feasibility_threshold,
            feasibility_penalty_kg=args.ga_steel_ranking_feasibility_penalty_kg,
        ),
        random_preselection=RandomPreselectionConfig(
            enabled=args.ga_random_preselect,
            eval_ratio=args.ga_random_preselect_eval_ratio,
            min_eval_count=args.ga_random_preselect_min_eval,
        ),
    )
    random_cfg = RandomSearchConfig(n_trials=args.random_trials, seed=args.seed)
    pso_cfg = ParticleSwarmConfig(
        swarm_size=args.pso_swarm,
        iterations=args.pso_iter,
        inertia=args.pso_inertia,
        cognitive=args.pso_cognitive,
        social=args.pso_social,
        velocity_clamp=args.pso_vclamp,
        max_workers=(args.optimizer_workers if args.optimizer_workers > 0 else None),
        seed=args.seed,
    )
    optuna_cfg = OptunaBayesConfig(
        n_trials=args.optuna_trials,
        n_startup_trials=args.optuna_startup_trials,
        seed=args.seed,
        show_progress_bar=args.optuna_progress,
    )
    nsga2_cfg = NSGA2Config(
        n_trials=args.nsga2_trials,
        population_size=args.nsga2_pop,
        seed=args.seed,
    )
    objective_cfg = ShearWallObjectiveConfig(
        steel_price_per_kg=args.steel_price_per_kg,
        infeasible_penalty=args.infeasible_penalty,
    )
    limit_cfg = ShearWallLimitConfig(
        max_torsion_ratio=args.limit_max_torsion,
        max_drift_ratio=args.limit_max_drift,
        min_shear_weight_ratio=args.limit_min_shear_weight,
        min_stiffness_ratio=args.limit_min_stiffness,
        max_period_ratio=args.limit_max_period_ratio,
    )
    print("Starting optimization...")
    result = run_shearwall_optimization(
        layout_path=args.layout_path,
        out_path=args.out,
        algorithm=args.algorithm,
        fixed_params=fixed,
        analysis_cfg=analysis_cfg,
        objective_cfg=objective_cfg,
        limit_cfg=limit_cfg,
        ga_cfg=ga_cfg,
        nsga2_cfg=nsga2_cfg,
        optuna_cfg=optuna_cfg,
        pso_cfg=pso_cfg,
        random_cfg=random_cfg,
    )

    print("best_objective:", result.best_objective)
    print("best_objectives:", result.best_objectives)
    print("best_feasible:", result.best_feasible)
    print("best_solution:", result.best_solution)


if __name__ == "__main__":
    main()
