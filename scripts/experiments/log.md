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

### Current conclusion

For steel-usage regression, the current room-graph message passing is not yet providing a clear advantage over `param + graph_feat`.

The next experiments should focus on either:

- improving node/edge features so message passing has more useful local information to aggregate;
- training longer with repeated seeds before declaring small differences meaningful;
- adding stronger layout descriptors to the MLP baseline and treating it as a competitive production baseline;
- trying target decomposition, such as predicting steel ratio or residual over a rule-based/material baseline instead of raw steel weight.

## 2026-04-29 Steel GNN error analysis

Model analyzed:

```text
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\predictions_test.parquet
```

Overall test performance:

| count | MAE | RMSE | MAPE | R2 | bias |
|---:|---:|---:|---:|---:|---:|
| 109996 | 11482.8 kg | 15674.7 kg | 9.26% | 0.924 | -5024.9 kg |

The mean bias is negative, so the model is systematically under-predicting steel usage on the test set.

### By true steel usage quantile

| y_true bucket | true range kg | MAE | RMSE | MAPE | R2 | bias |
|---|---:|---:|---:|---:|---:|---:|
| 0-20% | 31471-80267 | 7382.7 | 8877.2 | 11.41% | 0.264 | 2126.5 |
| 20-40% | 80268-102840 | 8922.2 | 10753.4 | 9.76% | -1.735 | -1439.9 |
| 40-60% | 102840-127469 | 10033.1 | 12310.7 | 8.78% | -2.023 | -3052.1 |
| 60-80% | 127469-163924 | 10425.3 | 13248.4 | 7.26% | -0.623 | -5359.9 |
| 80-100% | 163924-534227 | 20650.7 | 26589.0 | 9.07% | 0.727 | -17399.2 |

Observations:

- Absolute error grows with steel usage, especially in the highest 20% interval.
- The highest steel-usage interval is strongly under-predicted.
- R2 inside middle quantile buckets is poor because each bucket has a narrow target range; MAE/MAPE are more interpretable for these slices.

### By graph scale quantile

Node count:

| node_count bucket | node range | MAE | RMSE | MAPE | R2 | bias |
|---|---:|---:|---:|---:|---:|---:|
| 0-20% | 17-20 | 9824.9 | 11805.6 | 11.94% | 0.783 | 963.2 |
| 20-40% | 20-23 | 10089.9 | 12397.6 | 10.30% | 0.882 | -1669.9 |
| 40-60% | 23-29 | 6041.6 | 7669.8 | 4.96% | 0.970 | -3402.0 |
| 60-80% | 29-32 | 11559.0 | 14572.8 | 9.16% | 0.865 | -1718.7 |
| 80-100% | 32-46 | 19898.7 | 25772.5 | 9.93% | 0.869 | -19297.2 |

Edge count:

| edge_count bucket | edge range | MAE | RMSE | MAPE | R2 | bias |
|---|---:|---:|---:|---:|---:|---:|
| 0-20% | 40-50 | 8649.0 | 10289.8 | 10.19% | 0.853 | -206.4 |
| 20-40% | 50-56 | 11893.6 | 13752.1 | 12.89% | 0.811 | -689.1 |
| 40-60% | 56-78 | 9329.5 | 12591.1 | 7.50% | 0.889 | 767.5 |
| 60-80% | 78-88 | 7643.3 | 10522.9 | 5.79% | 0.942 | -5699.3 |
| 80-100% | 88-136 | 19898.7 | 25772.5 | 9.93% | 0.869 | -19297.2 |

Layout area (`bbox_area`):

| bbox_area bucket | area range | MAE | RMSE | MAPE | R2 | bias |
|---|---:|---:|---:|---:|---:|---:|
| 0-20% | 481-561 | 10784.6 | 12731.4 | 10.89% | 0.918 | -4619.5 |
| 20-40% | 561-603 | 9720.5 | 14107.3 | 6.65% | 0.953 | -8012.1 |
| 40-60% | 603-1045 | 6217.4 | 9571.2 | 5.02% | 0.963 | -4661.6 |
| 60-80% | 1045-1339 | 20155.9 | 24812.0 | 14.68% | 0.885 | -5265.4 |
| 80-100% | 1339-2238 | 10535.7 | 12654.5 | 9.05% | 0.898 | -2565.9 |

Observations:

- The largest node/edge-count bucket has the largest error and very strong under-prediction.
- The worst area bucket is not the largest one, but the 60-80% area bucket, so error is not driven by area alone.
- Graph complexity appears to matter more through certain layouts than through a smooth monotonic scale effect.

### Parameter buckets

Worst parameter buckets by MAE:

| param | bucket | count | MAE | RMSE | MAPE | R2 | bias |
|---|---|---:|---:|---:|---:|---:|---:|
| N | 80-100% | 21999 | 14532.8 | 19276.8 | 9.31% | 0.905 | -5833.2 |
| tw_mid | 300 | 13150 | 14155.8 | 19905.5 | 8.42% | 0.919 | -6481.3 |
| tw_bot | 400 | 22000 | 13812.5 | 19127.0 | 8.43% | 0.918 | -6487.6 |
| tw_top | 250 | 8741 | 13585.5 | 19135.1 | 8.57% | 0.922 | -6037.9 |
| N | 60-80% | 21999 | 12752.4 | 17557.2 | 8.97% | 0.907 | -5937.9 |
| hb_main | 700 | 18340 | 12038.9 | 16120.6 | 9.35% | 0.922 | -5345.3 |
| intensity | 8.0 | 27498 | 11943.5 | 16200.8 | 9.34% | 0.923 | -5351.7 |
| site_class | IV | 22000 | 11931.3 | 16201.3 | 9.34% | 0.922 | -5263.4 |

Observations:

- Larger wall thickness and larger main beam depth buckets have higher absolute error, which is consistent with the high steel-usage under-prediction.
- Higher floors (`N` upper buckets), intensity 8.0, and site class IV also show higher errors.
- These are structurally demanding cases, so a direct raw steel-weight regression tends to smooth them downward.

### Layout-level error

Worst layouts by MAE:

| layout_id | count | MAE | RMSE | MAPE | R2 | bias |
|---|---:|---:|---:|---:|---:|---:|
| L1L28_25 | 5000 | 41255.5 | 43061.0 | 17.40% | 0.642 | -41255.5 |
| L1L28_11 | 5000 | 27981.8 | 29669.5 | 12.61% | 0.807 | -27981.8 |
| L27_59 | 5000 | 20949.7 | 21591.5 | 17.29% | 0.665 | 20949.7 |
| L1L28_206 | 5000 | 20228.1 | 20727.4 | 24.62% | 0.362 | 20228.1 |
| L17_226 | 5000 | 18779.0 | 19478.9 | 14.87% | 0.714 | -18779.0 |

Observations:

- Layout-level bias is very strong: some layouts are almost uniformly under-predicted or over-predicted.
- The worst case, `L1L28_25`, is under-predicted by about 41 t on average.
- This suggests the model is still missing layout-specific factors or distribution shift between layout groups.

### Outputs

```text
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_y_true_bucket.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_node_count_bucket.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_edge_count_bucket.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_bbox_area_bucket.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_param_bucket.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\metrics_by_layout.csv
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\param_error_bars.png
data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis\report.md
```

Current conclusion:

The model is accurate enough on average, but unsafe for directly replacing FEA/material calculation in high-demand regions because it under-predicts heavy steel cases. For optimization usage, this supports using either a calibrated upper-bound model, residual correction by layout group, or a conservative post-calibration layer before using predictions to skip FEA.
