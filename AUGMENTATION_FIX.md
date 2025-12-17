# 数据增广泄露问题修复

## 问题描述

在进行数据增广后，原先使用 `random_split()` 划分数据集的方式会导致**数据泄露**问题：

- 假设有文件 `file_A.dxf`，经过增广后生成 3 个样本：
  - `file_A.dxf` (原始)
  - `file_A.dxf` (flip_x)
  - `file_A.dxf` (flip_y)
  
- 使用 `random_split()` 会随机将这 3 个样本分配到训练集、验证集、测试集
- 这导致**同一个文件的不同增广版本可能出现在不同的数据集中**
- 模型在训练集上见过 `file_A` 的某个增广版本，在测试集上又遇到它的另一个增广版本
- 这会导致测试结果过于乐观，无法真实反映模型的泛化能力

## 解决方案

### 核心思路
按**原始文件**进行分组划分，确保同一个文件的所有增广版本都在同一个集合中。

### 修改内容

#### 1. `train/dataset.py` - 添加元数据跟踪

**修改点：**
- 在 `__init__` 中添加三个列表来跟踪元数据
- 在 `process()` 中记录每个样本对应的原始文件索引和增广模式
- 保存元数据到独立文件 `*_metadata.pt`

```python
# __init__ 添加的属性
self.file_indices = []  # 每个样本对应的原始文件索引
self.aug_modes = []     # 每个样本的增广模式
self.dxf_files = []     # 原始DXF文件列表

# process() 中的循环修改
for file_idx, dxf_file in enumerate(dxf_files):
    for mode in aug_modes:
        # ... 处理数据 ...
        data_list.append(data)
        file_indices_list.append(file_idx)  # 记录原始文件索引
        aug_modes_list.append(mode)         # 记录增广模式

# 保存元数据
metadata = {
    'file_indices': file_indices_list,
    'aug_modes': aug_modes_list,
    'dxf_files': dxf_files
}
torch.save(metadata, metadata_path)
```

#### 2. `train/utils.py` - 更新文档字符串

**修改点：**
- `build_graph_from_dxf()` 函数已支持 `mode` 参数，用于数据增广
- 更新文档字符串，明确说明支持的增广模式

```python
def build_graph_from_dxf(dxf_path: str, mode: str = "none"):
    """
    流程：提取 → 增广 → 校准 → 分析 → 构图
    
    Args:
        mode: 数据增广模式，可选值: "none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"
    """
```

#### 3. `train/trainer.py` - 新增文件分组函数

**新增函数：`split_dataset_by_files()`**

```python
def split_dataset_by_files(dataset, train_ratio, val_ratio, test_ratio):
    """
    按原始文件分组划分数据集，避免数据泄露
    
    步骤：
    1. 加载元数据，获取 file_indices
    2. 按 file_idx 分组：file_idx -> [sample_indices]
    3. 对唯一文件索引进行随机打乱和划分
    4. 将文件索引展开为样本索引
    
    返回：(train_indices, val_indices, test_indices)
    """
```

**修改函数：`prepare_data()`**
- 使用 `split_dataset_by_files()` 替代 `random_split()`
- 确保同一文件的所有增广版本在同一集合中

**修改函数：`visualize_test_set()`**
- 加载元数据获取文件映射关系
- 使用 `file_indices[sample_idx]` 获取原始文件
- 保存文件名包含增广模式信息（如 `file_A_flip_x.png`）

**修改函数：`evaluate_model_on_test_set()`**
- 使用 `split_dataset_by_files()` 划分数据集
- 加载元数据并正确映射样本到文件
- 可视化时显示增广模式信息

## 使用示例

### 训练模型
```python
from train.trainer import train

# 自动使用文件分组划分
train()
```

### 评估模型
```python
from train.trainer import evaluate_model_on_test_set

evaluate_model_on_test_set(
    model_path="result/ckpt/best_model.pth",
    save_visualizations=True
)
```

## 验证正确性

运行训练时会看到类似输出：
```
文件划分: 训练60个, 验证10个, 测试10个
样本划分: 训练180个, 验证30个, 测试30个
```

- **文件数量**：按原始DXF文件计数
- **样本数量**：文件数 × 增广倍数（例如 60 × 3 = 180）

确保测试集中的文件在训练集和验证集中都不存在。

## 注意事项

1. **删除旧缓存**：修改后需要删除 `data_cache/processed/` 目录，重新处理数据
   ```bash
   rm -rf data_cache/processed/*
   ```

2. **元数据文件**：自动保存为 `data_cache/processed/data_metadata.pt`

3. **可视化命名**：测试集可视化结果会包含增广模式，例如：
   - `L17_101.png` (原始)
   - `L17_101_flip_x.png` (水平翻转)
   - `L17_101_flip_y.png` (垂直翻转)

4. **随机种子**：确保训练和评估使用相同的随机种子，保证数据划分一致

## 测试清单

- [x] `dataset.py` - 元数据跟踪和保存
- [x] `utils.py` - 文档字符串更新
- [x] `trainer.py` - 文件分组划分函数
- [x] `prepare_data()` - 使用新的划分方式
- [x] `visualize_test_set()` - 正确映射文件
- [x] `evaluate_model_on_test_set()` - 正确映射文件
- [ ] 实际运行测试（需要删除缓存重新处理）
