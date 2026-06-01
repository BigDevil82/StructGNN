# Steel Regression Experiment Log

# 2024-06-30: 钢筋量预测任务 Ablation Study

1. 测试 GNN 是否发挥作用

背景：之前的完整 GNN 模型（参数 + graph_feat + 图消息传递）在钢筋量预测任务上表现不错，但不清楚性能提升主要来自于 `graph_feat` 还是图结构消息传递。为了验证这一点，我们设计了两个 Baseline：

训练结果：

| 模型 | 输入 | Test MAE | Test RMSE | Test R2 | Test MAPE |
|---|---|---:|---:|---:|---:|
| Baseline 1 | 参数 `p_num + p_cat` | 32300 kg | 45770 kg | 0.351 | 24.68% |
| Baseline 2 | 参数 + `graph_feat` | 13833 kg | 18750 kg | 0.891 | 11.23% |
| 完整 GNN | 参数 + graph_feat + 图消息传递 | 13119 kg | 16876 kg | 0.912 |11.03% |

结论很明确：

`param_encoder` 单独远远不够；`graph_feat` 确实承担了很大预测能力，加入后性能接近可用水平。看来目前GNN确实没发挥什么太大作用.

输出位置：

```text
data\parametric\ckpt\steel_ablation_param_only
data\parametric\ckpt\steel_ablation_param_graph_feat
```

使用命令示例：

```powershell
.\.venv\Scripts\python.exe scripts\experiments\train_steel_ablation.py `
  --baseline param_graph_feat `
  --graph-repr room `
  --graph-cache-dir data\parametric\cache\gnn_room_graph_cache `
  --output-dir data\parametric\ckpt\steel_ablation_param_graph_feat `
  --batch-size 512 `
  --max-epochs 2 `
  --early-stop-rounds 1 `
  --lr 0.0005 `
  --weight-decay 0.0001 `
  --hidden-dim 128 `
  --gnn-layers 3 `
  --conv-type sage `
  --dropout 0.1 `
  --log-interval 1
```

2. 测试不同 GNN backbone 的效果

背景：之前的 GNN 模型使用了 SAGEConv 作为 backbone，但不清楚是否是这个特定架构带来了性能提升。为了验证这一点，我们测试了以下几种常见的 GNN 架构：
- GINEConv
- GATConv
- GraphSAGE
- SAGEConv（原模型）

3. 2026-04-28 GNN readout and fusion variants

Goal: check whether the room-graph GNN is adding useful information beyond `param_encoder` and `graph_feat`.

Common setup:

```powershell
--graph-repr room
--graph-cache-dir data\parametric\cache\gnn_room_graph_cache
--batch-size 512
--max-epochs 2
--early-stop-rounds 1
--lr 0.0005
--weight-decay 0.0001
--hidden-dim 128
--gnn-layers 3
--conv-type sage
--dropout 0.1
--log-interval 1
```

### Results

| Model | Output dir | Val MAE | Test MAE | Test RMSE | Test R2 | Test MAPE |
|---|---|---:|---:|---:|---:|---:|
| Param only MLP | `data\parametric\ckpt\steel_ablation_param_only` | 35427 | 32300 | 45770 | 0.351 | 24.68% |
| Param + graph_feat MLP | `data\parametric\ckpt\steel_ablation_param_graph_feat` | 12453 | 13833 | 18750 | 0.891 | 11.23% |
| GNN mean/max | `data\parametric\ckpt\steel_gnn_meanmax_e2` | 18504 | 14457 | 20434 | 0.871 | 11.50% |
| GNN attention pooling | `data\parametric\ckpt\steel_gnn_attention_e2` | 18290 | 14127 | 19248 | 0.885 | 11.38% |
| GNN FiLM fusion | `data\parametric\ckpt\steel_gnn_film_e2` | 18742 | 15102 | 21500 | 0.857 | 11.99% |
| GNN Set2Set pooling | `data\parametric\ckpt\steel_gnn_set2set_e2` | 18260 | 18976 | 24506 | 0.814 | 16.11% |
| GNN virtual node | `data\parametric\ckpt\steel_gnn_virtual_node_e2` | 17209 | 13995 | 18722 | 0.891 | 11.28% |

### Observations

1. `param_only` is clearly insufficient. Steel usage depends strongly on layout information.
2. `param + graph_feat` is already very strong. This supports the concern that global scalar graph features explain most of the current model performance.
3. Under the same 2-epoch quick setting, the tested full GNN variants do not beat `param + graph_feat MLP`.
4. Attention pooling and virtual node are the best GNN variants in this group, but their test MAE remains slightly worse than `param + graph_feat`.
5. Set2Set is unstable in this short run and generalizes worse.
6. FiLM fusion in the current simple form does not improve the result.


## 2026-04-29 High-steel under-prediction fixes

Based on the error analysis above, the largest practical issue is systematic under-prediction in high steel-usage samples. Since previous ablations showed that `param + graph_feat MLP` is already competitive with the current GNN, these experiments use `param_graph_feat` as the baseline and stop using message passing for this branch.

Common setup:

```powershell
--baseline param_graph_feat
--graph-repr room
--graph-cache-dir data\parametric\cache\gnn_room_graph_cache
--batch-size 512
--max-epochs 2
--early-stop-rounds 1
--lr 0.0005
--weight-decay 0.0001
--hidden-dim 128
--dropout 0.1
```

### Training and calibration variants

| Variant | Output dir | Test MAE | Test RMSE | Test R2 | Test MAPE | Test bias |
|---|---|---:|---:|---:|---:|---:|
| Standard log loss | `data\parametric\ckpt\steel_ablation_param_graph_feat` | 13833 | 18750 | 0.891 | 11.23% | -1664 |
| kg auxiliary loss, alpha=0.2 | `data\parametric\ckpt\steel_mlp_graphfeat_kg_aux_a02` | 13998 | 19074 | 0.887 | 11.15% | n/a |
| q80 high-steel weighted loss | `data\parametric\ckpt\steel_mlp_graphfeat_q80_weight` | 14305 | 19165 | 0.886 | 11.55% | n/a |
| continuous steel-weighted loss | `data\parametric\ckpt\steel_mlp_graphfeat_cont_weight` | 13750 | 18475 | 0.894 | 11.26% | 819 |
| standard + linear calibration | `data\parametric\ckpt\steel_mlp_graphfeat_linear_cal` | 14607 | 19069 | 0.887 | 12.28% | 2166 |
| standard + 3-bin quantile bias calibration | `data\parametric\ckpt\steel_mlp_graphfeat_quantile_cal` | 14612 | 19000 | 0.888 | 12.32% | 2223 |

Notes:

- `kg_aux` does not improve this 2-epoch setting. It slightly worsens MAE/RMSE compared with standard log loss.
- `q80_weight` improves validation RMSE/R2 but worsens test performance, suggesting it overfits the validation layout distribution.
- `continuous_weight = clamp(y / y_median, 0.5, 3.0)` is the best of this group. It slightly improves test MAE/RMSE/R2 and strongly reduces high-steel under-prediction.
- Linear and quantile-bias calibration both improve validation metrics, but worsen test MAE/MAPE and flip bias positive. This means simple global validation calibration is not reliable under layout-group distribution shift.

### High steel-usage bucket comparison

Highest true steel-usage bucket: 80-100%, `163924-534227 kg`.

| Model | High-bucket MAE | High-bucket RMSE | High-bucket MAPE | High-bucket bias |
|---|---:|---:|---:|---:|
| GNN best analyzed earlier | 20651 | 26589 | 9.07% | -17399 |
| Param + graph_feat standard | 21375 | 27344 | 10.00% | -10962 |
| Param + graph_feat continuous weight | 20607 | 26524 | 9.65% | -6705 |

The continuous weighting strategy does not reduce high-bucket MAE dramatically, but it reduces the systematic under-prediction by about 39% relative to the standard `param + graph_feat` baseline.

### Current conclusion

For steel usage prediction, `param + graph_feat MLP` should be treated as the main baseline. The best quick fix so far is continuous target-dependent weighting, because it reduces high-steel under-prediction without hurting overall metrics.

However, layout-level bias remains the dominant issue. Even after continuous weighting, the worst layouts still show large one-direction errors. The next useful direction is likely not more global calibration, but layout-aware correction, such as:

- predicting residuals over a simple layout/material baseline;
- adding layout embedding or layout-cluster calibration;
- training a conservative upper-bound/quantile model for optimization screening use;
- using conformal calibration grouped by layout complexity or steel-demand regime.

## 2026-06-01 Pairwise ranking for steel usage

Motivation: exact steel-usage regression has layout-level bias and high-steel under-prediction. For optimization, relative order may be more useful than exact kg prediction, so we tested a pairwise ranker.

Training setup:

- Backbone: `Param + graph_feat MLP`
- Score convention: lower score means lower predicted steel usage
- Pair construction: same `layout_id + N + hs + h_story + intensity + site_class + seismic_group`
- Training pairs ignore small steel gaps: `abs(diff) < 3000 kg`
- Pair weights: `clip(abs(diff) / 20000, 0.5, 3.0)`
- Loss: `BCEWithLogits(score_j - score_i, y_i < y_j)`

Command:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\train_steel_pairwise_ranker.py `
  --output-dir data\parametric\ckpt\steel_pairwise_ranker_v1 `
  --graph-cache-dir data\parametric\cache\gnn_room_graph_cache `
  --batch-size 2048 `
  --max-epochs 2 `
  --pairs-per-epoch 300000 `
  --eval-pairs 200000 `
  --min-pair-gap-kg 3000 `
  --large-gap-kg 10000 `
  --lr 0.0005 `
  --weight-decay 0.0001 `
  --hidden-dim 128 `
  --dropout 0.1 `
  --top-frac 0.1
```

Pairwise ranker result:

| Split | pair_acc | large_gap_acc | spearman_group_mean | top10 recall | mean regret |
|---|---:|---:|---:|---:|---:|
| val | 0.969 | 0.999 | 0.966 | 0.917 | 171 kg |
| test | 0.965 | 0.999 | 0.961 | 0.912 | 202 kg |

Comparison with using existing regression predictions as ranking scores:

| Ranker | pair_acc | large_gap_acc | top10 recall | mean regret |
|---|---:|---:|---:|---:|
| GNN regression prediction | 0.979 | 1.000 | 0.948 | 66 kg |
| Param + graph_feat regression | 0.965 | 0.999 | 0.914 | 169 kg |
| Continuous-weight MLP regression | 0.960 | 0.999 | 0.900 | 235 kg |
| Pairwise ranker | 0.964 | 0.999 | 0.912 | 202 kg |

Conclusion:

- Pairwise ranking is feasible and gives very high comparison accuracy, especially for pairs with steel difference above 10 t.
- However, direct pairwise training does not beat using the best regression model's prediction as a ranking score.
- The best current ranker is still the previous GNN regression model when evaluated only as a sorter.
- The pairwise formulation remains useful conceptually for optimization, but it should probably be used as an auxiliary loss or evaluation objective rather than replacing regression outright.

Next recommended experiment:

- Train a multitask `Param + graph_feat` model with both kg regression loss and pairwise ranking loss.
- Evaluate by optimization-facing metrics: top-k recall, mean regret, and FEA-call reduction under GA preselection.
