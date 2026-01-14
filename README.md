# Png2Dxf - 建筑结构智能设计系统

基于深度学习的建筑结构构件自动生成系统，从CAD图纸自动识别并预测剪力墙和梁的布置方案。

## 项目概述

本项目通过图神经网络（GNN）技术，实现从DXF建筑图纸到结构构件布置的自动化设计，支持剪力墙分布预测和梁位置预测两大核心功能。

## 功能模块

### 🏗️ axis_engine - 轴线提取引擎

CAD布局解析与几何重构核心引擎

- **cad_processor.py**: CAD布局处理器
  - 将散乱线段构建为闭合墙体多边形
  - 提取墙体中轴线（支持混合厚度）
  - 生成房间区域划分
- **wall_centerline.py**: 墙体中心线提取算法
  - 自动推断墙体厚度（扫描线法）
  - 基于边缘配对提取骨架线
  - 支持混合厚度墙体剥离法
- **line_network_calibrator.py**: 线网络校准器
  - 坐标网格化对齐
  - 构件吸附到主导网格
  - T型/L型连接点拓扑缝合
- **rect_decomposer.py**: 矩形分解器

### 🧱 shearwall_pred - 剪力墙预测

基于图神经网络的剪力墙分布预测

- **model.py**: 剪力墙GNN模型
  - 3层GATv2图注意力网络
  - FiLM条件编码模块
  - 双输出：分类（是否存在墙）+ 回归（长度比例）
- **trainer.py**: 训练与推理框架
  - 支持train/test/visualize三种模式
  - 混合损失函数（分类+回归+一致性）
  - 全局密度约束
- **dataset.py**: 剪力墙数据集构建
- **augmentor.py**: 数据增强（翻转、旋转）
- **losses.py**: 损失函数定义

### 📏 beam_pred - 梁布置预测

基于图神经网络的梁位置预测

- **model.py**: 梁预测GNN模型
  - 3层GINEConv图卷积网络
  - 链接预测解码器
  - 显式几何特征（dx, dy）
- **train.py**: 训练流程
- **predict.py**: 预测推理与可视化
- **beam_dataset.py**: 梁数据集
- **data_aug.py**: 数据增强
- **split_dataset.py**: 数据集划分

### 🔄 preprocess - 数据预处理

DXF文件解析与结构化数据生成

- **dxf_extractor.py**: DXF构件提取
  - 提取剪力墙、填充墙、门、窗、梁、房间
  - 支持LINE、LWPOLYLINE、POLYLINE实体
- **beam_ir_builder.py**: 结构图构建
  - 房间校准与对齐
  - 构件映射到房间边
  - 梁Ground Truth生成（剪力墙互补规则）
  - 全局节点管理与校准
- **room_analyzer.py**: 房间边缘提取
- **room_calibrator.py**: 房间坐标校准
- **layout_graph.py**: 布局图构建
- **merge_rooms.py**: 房间合并处理

## 技术栈

- **深度学习框架**: PyTorch, PyTorch Geometric
- **几何处理**: Shapely, NetworkX
- **CAD解析**: ezdxf
- **可视化**: Matplotlib

## 快速开始

### 环境安装

```bash
pip install torch torch-geometric shapely networkx ezdxf matplotlib
```

### 数据准备

1. 准备DXF格式建筑图纸，按图层分类：
   - `SHEAR_WALLS`: 剪力墙
   - `INFILL_WALLS`: 填充墙
   - `DOORS`: 门
   - `WINDOWS`: 窗
   - `BEAMS`: 梁
   - `ROOM`: 房间

2. 数据目录结构：
```
dxf/
├── train/
│   ├── *.dxf
├── test/
│   └── *.dxf
```

### 训练剪力墙预测模型

```bash
python -m shearwall_pred.trainer
```

### 训练梁预测模型

```bash
python -m beam_pred.train
```

### 预测推理

```bash
# 剪力墙预测可视化
python -m shearwall_pred.train --mode visualize --ckpt checkpoints/shearwall_predictor.pth

# 梁位置预测
python -m beam_pred.predict --model checkpoints/beam_predictor.pth --output result/
```

## 工作流程

```
DXF文件
  ↓
[dxf_extractor] 提取构件信息
  ↓
[beam_ir_builder] 构建结构化图
  ├─ 房间校准
  ├─ 构件映射
  └─ 节点管理
  ↓
[axis_engine] 轴线提取（可选）
  ↓
┌───────────────┬───────────────┐
│               │               │
[shearwall_pred] [beam_pred]
  │               │
GNN预测        GNN预测
  │               │
剪力墙布置      梁位置
  └───────┬───────┘
          ↓
    结果可视化
```

## 核心算法

### 墙体中轴线提取
- 扫描线法估算墙厚
- 边缘配对算法提取骨架
- 剥离法处理混合厚度墙体
- 拐角自动修复与延伸

### 图神经网络
- **GATv2**: 捕获房间间的空间依赖关系
- **GINEConv**: 处理带边特征的图结构
- **FiLM**: 条件注入，支持不同建筑类型
- **混合损失**: 分类+回归+一致性约束

### 数据增强
- 水平/垂直翻转
- 90°/180°/270°旋转
- 保持拓扑结构不变

## 项目结构

```
Png2Dxf/
├── axis_engine/          # 轴线提取引擎
├── beam_pred/            # 梁预测模块
├── shearwall_pred/       # 剪力墙预测模块
├── preprocess/           # 数据预处理
├── dxf/                  # DXF数据目录
├── data_cache/           # 处理后的数据缓存
├── result/               # 结果输出目录
```

## 配置说明

剪力墙预测主要配置（`shearwall_pred/config.py`）：

```python
# 模型参数
NODE_FEATURE_DIM = 25      # 节点特征维度
EDGE_FEATURE_DIM = 8        # 边特征维度
HIDDEN_DIM = 256           # 隐藏层维度
OUTPUT_DIM = 32            # 输出维度

# 训练参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 100

# 损失权重
CLS_WEIGHT = 1.0           # 分类损失
REG_WEIGHT = 2.0           # 回归损失
CONSISTENCY_WEIGHT = 0.5   # 一致性损失
```

## 输出说明

训练完成后，模型会保存在 `result/` 目录下：
- `best_model.pth`: 验证集最优模型
- `final_model.pth`: 最终训练模型
- `loss_curve.png`: 训练损失曲线
- `test_set_results/`: 测试集评估结果
  - `metrics.json`: 评估指标（IoU等）

## 性能指标

📈 测试结果统计:
  平均 IoU: 0.6455
  最大 IoU: 0.8754
  最小 IoU: 0.3400

Configuration:
  - loss: HybridLoss + DensityLoss
  - k-fold Cross Validation: k=5
  - data augmentation: ("none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270")

## 许可证

MIT License


