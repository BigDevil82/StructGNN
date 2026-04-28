# Steel Regression Experiment Log

## 2026-04-28 GNN readout and fusion variants

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

### Current conclusion

For steel-usage regression, the current room-graph message passing is not yet providing a clear advantage over `param + graph_feat`.

The next experiments should focus on either:

- improving node/edge features so message passing has more useful local information to aggregate;
- training longer with repeated seeds before declaring small differences meaningful;
- adding stronger layout descriptors to the MLP baseline and treating it as a competitive production baseline;
- trying target decomposition, such as predicting steel ratio or residual over a rule-based/material baseline instead of raw steel weight.
