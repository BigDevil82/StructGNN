# Problem formulation

## Design task

The design task is section-level optimization of shear wall structures. For a given building layout and seismic design condition, the optimizer selects wall thicknesses, beam sections, and concrete grade to minimize material cost while satisfying structural response and member design requirements.

The layout is treated as fixed in this study. The optimization acts on section and material parameters rather than wall layout generation. This separates the present work from the preceding layout-generation research and positions it as the downstream structural design completion stage.

## Inputs

For each optimization case, the fixed inputs are:

- `layout_id`: the shear wall layout represented by a preprocessed CAD/JSON floor plan;
- `N`: number of stories;
- `h_story`: story height, currently 2.9 m in the parametric baseline;
- `hs`: floor load-related parameter used by the parametric model;
- seismic design condition: intensity, site class, and seismic group;
- modeling configuration: OpenSees model builder, number of modes, and geometric scaling settings.

The current primary experiment maps layout families to seismic intensities:

| Layout family | Intensity |
| --- | --- |
| `L17` | 7.0 |
| `L27` | 7.5 |
| `L1L28` | 8.0 |

Site class and seismic group are sampled per layout-seed pair and kept identical across compared optimization methods.

## Design variables

The optimization variables are discrete:

| Variable | Meaning | Candidate values |
| --- | --- | --- |
| `tw_bot` | wall thickness in bottom stories | 200, 250, 300, 350, 400 mm |
| `tw_mid` | wall thickness in middle stories | 160, 180, 200, 250, 300 mm |
| `tw_top` | wall thickness in top stories | 160, 180, 200, 250 mm |
| `hb_main` | main beam depth | 400, 500, 550, 600, 650, 700 mm |
| `bb_main` | main beam width | 200, 250, 300, 350 mm |
| `hb_sec` | secondary beam depth | 300, 400, 450, 500 mm |
| `bb_sec` | secondary beam width | 200, 250, 300 mm |
| `conc_bot` | concrete grade, also used for middle and top groups in the current baseline | C30, C35, C40, C45, C50 |

The repair rule enforces a monotonic wall-thickness hierarchy:

```text
tw_top <= tw_mid <= tw_bot
```

This avoids unrealistic upper-story wall thicknesses exceeding lower-story ones.

## Structural feasibility

A design is feasible only if it passes both response-level and member-level checks:

```text
feasible_design = converged AND analysis_feasible AND design_passed
```

The response-level checks include:

- torsion ratio;
- maximum interstory drift ratio;
- period ratio;
- minimum shear-weight ratio;
- minimum adjacent-story stiffness ratio;
- wall axial compression utilization;
- wall shear utilization;
- beam shear utilization.

The current optimization constraints use the following representative limits:

| Constraint | Limit |
| --- | --- |
| torsion ratio | <= 1.5 |
| maximum drift ratio | <= 1/1000 |
| minimum shear-weight ratio | >= 0.016, with intensity-specific limits available in dataset labeling |
| minimum stiffness ratio | >= 0.7 |
| period ratio | <= 0.9 |

Member-level feasibility is determined by the reinforcement design pipeline. A design with failed beam or wall reinforcement checks is not considered feasible.

## Objective function

The main objective is material cost:

```text
C(x) = C_concrete(x) + C_steel(x)
```

Concrete cost is computed from estimated or analyzed concrete mass and concrete-grade-specific unit price. Steel cost is computed from steel mass and a specified steel unit price:

```text
C_steel = steel_kg * steel_price_per_kg
```

During direct FEA evaluation, the optimizer uses a penalized objective:

```text
F(x) = C(x) * (1 + violation_mean(x)) + infeasible_penalty * I[infeasible]
```

where `violation_mean` is the average normalized violation of selected response and member-utilization checks.

However, when ranking optimization candidates, feasibility is treated as a primary ordering criterion:

```text
rank(x) = (not feasible, objective)
```

Thus any feasible design is preferred over infeasible designs before comparing material cost.

## Role of surrogate models

The optimization problem is expensive because each objective and constraint evaluation requires:

1. generating an OpenSees model;
2. running structural analysis;
3. evaluating seismic response;
4. performing member reinforcement design.

The surrogate models are introduced to reduce the number of candidates sent to this full evaluation. They do not redefine feasibility and do not replace final verification.

