# Surrogate Dataset Bootstrap

This folder provides a minimal, script-only pipeline for generating surrogate-model training data.

Current goal:
- Build a large, consistent dataset quickly.
- Keep ETABS out of the batch loop.
- Reserve ETABS/OpenSees high-fidelity analysis for later calibration.

## What is included

- `generate_dataset.py`: end-to-end data generation entrypoint.
- `topology_io.py`: load and summarize topology JSON files.
- `design_space.py`: grouped design variables and LHS sampling.
- `analyzer_proxy.py`: fast proxy evaluator (replaceable by OpenSees evaluator later).
- `config.example.json`: example config.

## Input topology format

Use JSON exported by `experiments/case_study/fem_builder.py -> export_to_json`.

Required keys:
- `nodes`
- `shearwalls`
- `beams`
- `statistics` (optional but recommended)

## Quick start

```powershell
python experiments/surrogate_dataset/generate_dataset.py `
  --config experiments/surrogate_dataset/config.example.json
```

Outputs:
- CSV dataset (`output_csv`)
- JSONL dataset (`output_jsonl`)
- metadata JSON (`output_meta`)

## Next step: OpenSees integration

Replace `ProxyAnalyzer` in `generate_dataset.py` with your `OpenSeesAnalyzer`.
The interface is intentionally simple:

- input: `topology_features`, `design_variables`, `context`
- output: response dict (`drift`, `period`, `axial_ratio`, `feasible`, ...)

This lets you keep the same sampling/aggregation pipeline while switching analyzers.

