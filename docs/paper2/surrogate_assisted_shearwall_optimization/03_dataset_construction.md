# Dataset construction

## Layout set

The dataset is constructed from 143 shear wall layouts. Each layout corresponds to a preprocessed CAD/JSON floor plan and belongs to one of the layout families used in previous layout-generation research:

- `L17`;
- `L27`;
- `L1L28`.

The layout geometry provides the fixed topology and spatial distribution of shear walls and beams. The present study treats layout as fixed and samples structural section, material, and design-condition parameters on top of it.

## Parameter sampling

For each layout, a large number of structural design samples are generated. The nominal dataset scale is:

```text
143 layouts x 5000 samples per layout
```

The parameter space includes:

| Parameter | Values |
| --- | --- |
| `N` | 18 to 33 stories |
| `tw_bot` | 200, 250, 300, 350, 400 mm |
| `tw_mid` | 160, 180, 200, 250, 300 mm |
| `tw_top` | 160, 180, 200, 250 mm |
| `hb_main` | 400, 500, 550, 600, 650, 700 mm |
| `bb_main` | 200, 250, 300, 350 mm |
| `hb_sec` | 300, 400, 450, 500 mm |
| `bb_sec` | 200, 250, 300 mm |
| `conc_bot` | C30, C35, C40, C45, C50 |
| `intensity` | 6.0, 7.0, 7.5, 8.0 |
| `site_class` | I0, I, II, III, IV |
| `seismic_group` | 1, 2, 3 |

Sampling uses Latin hypercube sampling by default. Each layout receives an independent layout-specific random seed, so samples are reproducible while remaining decorrelated across layouts.

## Data generation record

Each generated sample stores:

- layout identifier and sample identifier;
- sampled parameters;
- geometric scale;
- convergence state;
- response-level feasibility;
- reinforcement design pass/fail state;
- failed beam/wall counts;
- material concrete mass;
- material steel mass;
- modal periods and derived response metrics;
- wall and beam utilization-related metrics;
- error message if the analysis fails.

This makes the dataset suitable for both:

- classification or screening of feasible designs;
- regression of material quantities;
- diagnostic analysis of governing constraints.

## Label definition

The primary feasibility label is:

```text
final_pass = converged AND feasible AND design_passed
```

Additional margin labels are computed during dataset preparation:

```text
margin_drift = drift_limit - max_drift_ratio
margin_torsion = torsion_limit - torsion_ratio
margin_shear = min_shear_weight_ratio - shear_weight_ratio_limit
margin_stiffness = min_stiffness_ratio - stiffness_limit
```

These margins are useful for error analysis and for explaining why direct binary classification is difficult near code-limit boundaries.

## Cleaning and deduplication

Dataset preparation performs:

- normalization of categorical values such as site class and concrete grade;
- optional removal of non-converged samples;
- removal of rows with invalid numeric values, negative material mass, or nonpositive first-period values;
- optional deduplication by `layout_id` and parameter vector.

The cleaned dataset is saved as a surrogate-model training table and later merged with layout-level features.

## Layout-level features

A set of layout statistics is extracted from each floor plan. These features encode engineering priors that are difficult for a model to learn from only 143 unique layouts:

- bounding-box dimensions and aspect ratio;
- number and total length of shear walls;
- wall length distribution;
- wall length in X and Y directions;
- wall direction imbalance and `wall_xy_ratio`;
- wall centroid and eccentricity;
- quadrant-wise wall distribution;
- main and secondary beam counts and lengths;
- beam-wall count and length ratios;
- joint degree and connected component statistics;
- simplified wall second-moment measures;
- boundary wall ratio.

These features are used as tabular baselines and are also available to the surrogate-assisted optimization module for concrete and cost estimation.

## Train/validation/test split

The dataset is split by layout group, not by individual sample. This is important because samples from the same layout share topology and many layout features. A sample-level random split would overestimate generalization.

The default split ratios are:

```text
train : validation : test = 70% : 15% : 15%
```

The split file preserves all samples belonging to a layout in the same partition. This allows test performance to better reflect generalization to unseen layouts.

## Paper description guidance

In the paper, the dataset should be introduced as a physics-informed supervised dataset generated from a high-fidelity structural analysis pipeline, rather than as a generic machine-learning table. The key point is that each label and regression target comes from a full OpenSees analysis and design calculation.

