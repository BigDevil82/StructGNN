## 目标

做一个二分类模型，预测：

* `final_pass`

输入：

* 当前已有的表格特征
* 不用 `layout_id` 当输入
* 继续保持 **按 layout 分组划分 train/val/test**

---

## 方案选择

### 第一版就做：**Tabular MLP**

不要一上来搞太复杂。先做一个适合表格数据的 MLP baseline。

输入分两类：

* **数值特征**：直接标准化后输入
* **类别特征**：做 embedding
  例如：

  * `conc_bot`
  * `site_class`
  * `seismic_group`
  * `intensity` 也可以当类别试一版

---

## 模型结构

一个够用的版本：

* 数值输入：BatchNorm
* 类别输入：Embedding 后拼接
* 主干：3 层 MLP
  例如：

  * 256
  * 128
  * 64
* 每层：

  * Linear
  * ReLU 或 GELU
  * BatchNorm
  * Dropout(0.2~0.3)

输出：

* 1 个 logit
* 用 sigmoid 得到概率

---

## 损失函数

你这个是类别不平衡问题，优先这样做：

### 首选

* `BCEWithLogitsLoss(pos_weight=...)`

其中：

[
pos_weight = \frac{N_{neg}}{N_{pos}}
]

### 可选第二版

* Focal Loss

但第一版先别上，先用加权 BCE。

---

## 数据处理

### 数值特征

* 用 train 集均值方差做标准化
* val/test 复用同一套 scaler

### 类别特征

* 转成从 0 开始的整数索引
* 每个字段单独建 vocab

### 缺失值

* 数值缺失填 train 均值
* 类别缺失单独一个 unknown 类

---

## 训练配置

先用这套：

* optimizer: `AdamW`
* learning rate: `1e-3`
* batch size: `4096` 或 `8192`
* epochs: `50`
* early stopping: `10`
* weight decay: `1e-4`

学习率调度：

* 可先不用
* 或简单用 `ReduceLROnPlateau`

---

## 评估指标

保持和树模型一致，重点看：

* ROC-AUC
* PR-AUC
* F1
* Precision
* Recall
* Balanced Accuracy
* Brier

并且：

* 在 val 集上选 threshold
* 再固定 threshold 到 test 集评估

---

## 你真正要做的实验顺序

### 实验 1：纯 MLP

* 所有特征都数值化
* 类别直接 one-hot 或整数编码
* 看能否接近树模型

### 实验 2：Embedding MLP

* 类别特征改 embedding
* 数值特征标准化
* 这是主实验

### 实验 3：Residual MLP

如果实验 2 不错，再把主干改成残差结构，稍微加深一点。

---

## 成败判断标准

你的树模型现在大概是：

* ROC-AUC ≈ 0.963
* PR-AUC ≈ 0.84
* F1 ≈ 0.74

所以深度学习这边不要幻想明显超过。
重点看：

### 成功

* PR-AUC 接近树模型（比如差距 < 0.01~0.02）
* F1 接近 0.74
* 概率校准不差

### 失败

* 明显低于树模型很多
* 训练波动大
* 对 threshold 很敏感

如果只是接近，那也算有价值，因为后面你还能继续接图结构模型。

---

## 我建议你直接做的版本

### 输入

* 数值特征：标准化
* 类别特征：embedding

### 模型

* MLP: 256 → 128 → 64
* BatchNorm + ReLU + Dropout(0.2)

### 损失

* `BCEWithLogitsLoss(pos_weight=neg/pos)`

### 训练

* AdamW
* lr=1e-3
* batch_size=4096
* max_epoch=50
* early_stop=10

### 输出

* 保存：

  * best model
  * val 上最优 threshold
  * test 指标
  * test 预测概率表
