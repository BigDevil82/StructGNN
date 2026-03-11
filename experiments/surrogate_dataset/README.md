# Surrogate Dataset Bootstrap

This folder provides a single OpenSees-based pipeline for generating surrogate-model training data.

## What is included

- `generate_dataset.py`: end-to-end dataset generation entrypoint.
- `opensees_wall_analyzer.py`: linear-elastic OpenSees analyzer using real wall topology.
- `topology_io.py`: topology loading and summary extraction.
- `design_space.py`: grouped design variables and LHS sampling.
- `config.example.json`: example config.

## Input topology format

Use JSON exported by `experiments/case_study/fem_builder.py -> export_to_json`.

Required keys:
- `nodes`
- `shearwalls`
- `beams`

## Model notes

- Analysis type: linear elastic static + modal.
- X and Y directions are analyzed separately.
- Walls are grouped by orientation and assembled as equivalent parallel wall systems.
- Drift limit in config is set to `1/1000` (`0.001`).

## Quick start (conda env `dl`)

```powershell
conda run -n dl pip install openseespy
conda run -n dl python experiments/surrogate_dataset/generate_dataset.py `
  --config experiments/surrogate_dataset/config.example.json
```

Outputs:
- CSV dataset (`output_csv`)
- JSONL dataset (`output_jsonl`)
- metadata JSON (`output_meta`)
