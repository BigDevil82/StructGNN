# OpenSees-based analysis and design workflow

## Purpose

The OpenSees workflow provides the high-fidelity evaluator used for both dataset generation and optimization verification. It maps a layout and a parameter vector to structural response metrics, reinforcement quantities, feasibility labels, and material cost.

In the paper, this workflow should be presented as the physics-based ground-truth generator.

## Input geometry

The geometry input is a preprocessed layout JSON derived from CAD floor plans. Each layout contains shear wall and beam line segments, node connectivity, and member roles. Geometry is scaled to metric units before modeling.

The workflow supports automatic geometric scaling. The default input unit scale is:

```text
input_unit_scale_to_m = 0.001
```

An additional automatic scale factor can be sampled or assigned to map layout drawings into plausible building dimensions.

## Parametric structural model

For each sampled design, the workflow constructs a structural model using:

- story number `N`;
- story height `h_story`;
- wall thicknesses for bottom, middle, and top story groups;
- main and secondary beam sections;
- concrete grade;
- seismic design parameters.

Stories are split into three standard groups:

```text
bottom / middle / top
```

The current story grouping rule uses approximately 30% lower stories and 70% middle-story boundary, with at least one top group. Each group receives its own wall thickness, while the current baseline uses the same concrete grade for all groups.

## Model builder

The default parametric dataset generation uses the `mvlem_frame` builder. The codebase also contains alternative builders such as equivalent-frame and detailed-shell builders, but these should be treated as available implementation variants rather than central paper contributions.

For paper clarity, describe the baseline model as an OpenSees-based equivalent frame/MVLEM-style shear wall model suitable for large-scale parametric analysis.

## Analysis workflow

For each candidate design, the workflow executes:

1. model configuration from parameters;
2. OpenSees model construction;
3. modal analysis;
4. response-spectrum or seismic response evaluation;
5. extraction of global structural response metrics;
6. member demand extraction;
7. reinforcement design and material quantity calculation.

The main response outputs include:

- first six modal periods;
- period ratio;
- torsion ratio;
- maximum interstory drift ratio;
- minimum shear-weight ratio;
- minimum adjacent-story stiffness ratio;
- wall axial compression ratio and limit;
- wall shear ratio and limit;
- beam shear ratio and limit;
- total concrete mass;
- total steel mass.

## Feasibility labeling

The high-fidelity analysis returns three key boolean states:

```text
converged
analysis_feasible
design_passed
```

The dataset label used for final feasibility prediction is:

```text
final_pass = converged AND analysis_feasible AND design_passed
```

This label is intentionally stricter than response-level feasibility alone because optimization candidates must pass both structural response checks and reinforcement design checks.

## Material quantity calculation

Concrete and steel quantities are obtained from the design pipeline after structural analysis. In optimization acceleration, concrete quantity can also be estimated rapidly from:

- wall total length;
- beam total length;
- story count;
- wall thickness;
- beam sections;
- concrete density.

Steel quantity is more difficult to estimate directly because it depends on structural demand and reinforcement design. This motivates the steel surrogate model.

## Ground-truth role in the paper

The paper should emphasize the following:

- OpenSees analysis and reinforcement design define the ground truth;
- machine learning models are trained against this workflow;
- all optimization methods are ultimately evaluated by the same high-fidelity evaluator;
- surrogate-assisted optimization only reduces the number of calls to this evaluator.

