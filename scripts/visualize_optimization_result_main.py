import argparse
import json
from pathlib import Path

from src.shearwall_optimization.visualization import generate_optimization_plots


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize optimization result JSON.")
    p.add_argument("--result-json", required=True)
    p.add_argument("--out-dir", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result_path = Path(args.result_json)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else None
    saved = generate_optimization_plots(payload=payload, result_path=result_path, out_dir=out_dir)
    if not saved:
        raise ValueError("Result JSON has empty history.")

    for path in saved:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
