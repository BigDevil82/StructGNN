import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.surrogate_dataset.design_space import sample_designs
from experiments.surrogate_dataset.opensees_analyzer import OpenSeesEquivalentAnalyzer
from experiments.surrogate_dataset.topology_io import load_topology_summary


def _load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict]):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_meta(path: Path, meta: Dict):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def generate(config: Dict) -> Dict:
    topology_paths = sorted(glob(config["topology_glob"]))
    if not topology_paths:
        raise FileNotFoundError(f"No topology file matched: {config['topology_glob']}")

    n_limit = int(config.get("num_topologies_limit", len(topology_paths)))
    topology_paths = topology_paths[:n_limit]

    analyzer = OpenSeesEquivalentAnalyzer(constraints=config["constraints"])
    spp = int(config["samples_per_topology"])
    seed_base = int(config.get("seed", 42))
    context = dict(config["context"])

    rows = []
    for topo_idx, topo_path in enumerate(topology_paths):
        topo = load_topology_summary(Path(topo_path))
        topo_feat = topo.to_feature_dict()
        designs = sample_designs(config["design_space"], spp, seed=seed_base + topo_idx)

        for sample_idx, design in enumerate(designs):
            design_dict = design.to_dict()
            response = analyzer.evaluate(topo_feat=topo_feat, design=design_dict, context=context)
            rows.append(
                {
                    "topology_id": topo.topology_id,
                    "topology_path": topo.source_path,
                    "sample_id": sample_idx,
                    **topo_feat,
                    **design_dict,
                    **response,
                }
            )

    _write_csv(Path(config["output_csv"]), rows)
    _write_jsonl(Path(config["output_jsonl"]), rows)
    feasible_count = sum(int(r.get("feasible", 0.0)) for r in rows)
    fail_count = sum(int(r.get("analysis_failed", 0.0)) for r in rows)
    meta = {
        "num_topologies": len(topology_paths),
        "samples_per_topology": spp,
        "num_samples_total": len(rows),
        "feasible_ratio": (feasible_count / len(rows)) if rows else 0.0,
        "analysis_failure_ratio": (fail_count / len(rows)) if rows else 0.0,
        "config": config,
    }
    _write_meta(Path(config["output_meta"]), meta)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset with OpenSees equivalent analyzer.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/surrogate_dataset/config.opensees.example.json",
        help="Path to generation config JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(Path(args.config))
    meta = generate(config)
    print("OpenSees dataset generated:")
    print(f"  num_topologies:        {meta['num_topologies']}")
    print(f"  samples_total:         {meta['num_samples_total']}")
    print(f"  feasible_ratio:        {meta['feasible_ratio']:.3f}")
    print(f"  analysis_failure_ratio:{meta['analysis_failure_ratio']:.3f}")
    print(f"  csv:                   {config['output_csv']}")
    print(f"  jsonl:                 {config['output_jsonl']}")
    print(f"  meta:                  {config['output_meta']}")


if __name__ == "__main__":
    main()

