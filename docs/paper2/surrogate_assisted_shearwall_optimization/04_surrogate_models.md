# Surrogate models

## Modeling objective

The surrogate models are designed for optimization acceleration, not final structural approval. Their roles are:

1. detect candidates that are very likely infeasible and can be skipped;
2. estimate steel usage for candidate ranking and cost-budget allocation;
3. provide cheap but conservative information before deciding whether to run FEA.

The final selected design is always verified by the high-fidelity OpenSees workflow.

## Input representation

Each candidate design combines:

- discrete design parameters;
- seismic design conditions;
- fixed layout identity;
- layout graph representation;
- optional global layout statistics.

Categorical parameters include:

```text
conc_bot, intensity, site_class, seismic_group
```

Numerical parameters include story number and section dimensions. Categorical variables are encoded through embeddings in neural models and through categorical/one-hot mechanisms in tabular baselines.

## Graph representation

Two graph representations were considered during development.

### Member graph

The member graph treats structural members as nodes:

- node: wall, primary beam, or secondary beam;
- edge: member connectivity;
- node features: type, axis, normalized length, normalized center coordinate, boundary indicator, distance to plan center;
- edge features: member-type relation, normalized distance, orthogonality.

This representation directly encodes structural components and their connectivity.

### Room graph

The room graph treats rooms or spatial cells as graph nodes:

- node: room polygon;
- edge: room adjacency;
- node features: geometric room features, layout constraints, wall-distribution vector;
- edge features: adjacency and shared-boundary descriptors;
- graph features: normalized graph size, density, normalized area, and wall-distribution summaries.

The room graph is aligned with previous layout-generation work and better reflects the spatial organization of the floor plan. It is the more relevant representation for the current paper narrative.

## Neural architecture

The GNN surrogate uses a layout-parameter fusion architecture:

1. project node features to hidden dimension;
2. apply multiple message-passing layers, currently GraphSAGE or GINE variants;
3. aggregate node states with global mean and max pooling;
4. aggregate edge features through a small MLP and global pooling;
5. encode design parameters with a parameter encoder;
6. concatenate graph embedding, edge embedding, graph-level features, and parameter embedding;
7. pass the fused representation through an MLP head.

For classification, the head outputs a logit for `final_pass`. For steel regression, the same architecture outputs a normalized log steel mass.

## Feasibility screening model

The feasibility surrogate is trained on:

```text
target = final_pass
```

The direct binary classification metrics are useful for model development, but they are not the final operational objective. The optimization task needs a conservative screener:

- false rejection of feasible designs must be rare;
- uncertain cases should be retained for FEA;
- the model should reject as many clearly infeasible designs as possible under a high feasible-recall constraint.

Therefore, the deployed threshold is selected on the validation set by:

```text
maximize reject_rate
subject to feasible_recall >= target_recall
```

The default conservative target is:

```text
screening_target_recall = 0.995
```

In optimization, a candidate is skipped only when:

```text
predicted_probability < screening_threshold
```

This converts a classifier into a risk-controlled pre-screening tool.

## Steel usage prediction model

The steel surrogate predicts:

```text
material_steel_kg
```

The target is transformed as:

```text
y = standardize(log1p(material_steel_kg))
```

Training uses a Smooth L1 loss in the transformed space. During inference, the prediction is transformed back to kilograms:

```text
steel_kg = expm1(y_pred * std + mean)
```

The steel model is used because steel mass depends on structural demand and reinforcement design, while concrete mass can be estimated more directly from geometry and section dimensions.

## Cost prediction from steel surrogate

During optimization, predicted steel mass is converted to material cost:

```text
predicted_material_cost = estimated_concrete_cost + predicted_steel_cost
```

Concrete mass is estimated from:

- wall total length;
- beam total length;
- wall and beam sections;
- story count;
- story height;
- concrete density.

The ranking score for cost preselection can include feasibility risk:

```text
score = material_cost + penalty_cost * max(0, hinge_target - feasible_probability)
```

This avoids ranking low-cost but likely infeasible designs too aggressively.

## Baselines and model-selection narrative

For the paper, baselines should be organized into a small number of meaningful categories:

1. **Parameter-only MLP**: uses only section, material, and design-condition variables.
2. **Parameter + layout statistics model**: uses manually engineered global layout features.
3. **Room-graph GNN**: uses graph topology plus design parameters.
4. **Surrogate-assisted optimization**: uses the trained models inside the optimization loop.

This is enough to show why both design parameters and layout information matter. Avoid listing every exploratory variant. Variants such as GINE, attention pooling, Set2Set, or pairwise ranking can be mentioned only if they directly support a specific methodological choice.

## Important interpretation

The GNN does not need to dominate every tabular baseline to justify the research. In this work, the stronger contribution is the system-level integration:

- high-fidelity data generation;
- conservative feasibility screening;
- material-cost-aware candidate ranking;
- online local calibration;
- reduction of FEA calls during optimization.

