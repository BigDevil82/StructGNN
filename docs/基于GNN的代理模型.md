
## 目标

做一个 **GNN 分类模型**，输入：

* 布局图
* 12 个参数变量

输出：

* `final_pass` 概率

也就是：

[
(\text{graph layout}, \text{design params}) \rightarrow P(final_pass)
]

---

# 一、先定最小可执行方案

## 1. 图是“布局级”的，不是“样本级”的

同一个布局的 5000 个样本，**图结构是一样的**，变的是：

* `N`
* `tw_bot/tw_mid/tw_top`
* `hb_main/bb_main`
* `hb_sec/bb_sec`
* `conc_bot`
* `intensity`
* `site_class`
* `seismic_group`

所以最自然的做法是：

* 每个 `layout_id` 存一个图
* 每个样本引用对应的图，再带上自己的参数向量

---

## 2. 模型结构用“双路输入”

最推荐的第一版结构：

### 路 1：GNN 编码布局图

输入图的节点、边特征，得到一个 graph embedding：

[
z_g = \text{GNN}(G)
]

### 路 2：MLP 编码参数

把 12 个参数做成一个向量，编码成 parameter embedding：

[
z_p = \text{MLP}(x_{param})
]

### 融合

拼接后做分类：

[
\hat{y} = \text{MLP}([z_g, z_p])
]

这是第一版最稳的结构。

---

# 二、图里放什么特征

## 1. 节点特征

先设计一个统一的节点特征模板，不同构件在这个模板里填写各自相关的信息；不适用的部分补 0

每个节点代表一个构件，建议至少放这些：

### 通用几何特征

* 构件类型：墙 / 主梁 / 次梁
* 长度
* 方向：水平/竖向
* 中心点坐标 `(x, y)`
* 两端点坐标或归一化端点
* 是否边界构件
* 是否靠近平面中心

### 构件专属特征

#### 墙

* 是否外圈墙
* 是否角部墙

#### 梁

* 主梁/次梁
* 是否连接两片墙
* 是否边界梁

### 强烈建议

所有坐标和长度都做**布局内归一化**，例如除以平面外包框尺寸，避免模型记尺度绝对值。

---

## 2. 边特征

如果两个构件连接，就连边。边特征建议至少有：

* 连接类型：墙-墙 / 墙-主梁 / 墙-次梁 / 梁-梁
* 连接点相对位置
* 两节点中心距
* 是否正交连接
* 是否端点连接 / 搭接连接

第一版简单点也可以只放：

* 边类型
* 两节点中心距

---

## 3. 图级特征

除了图本身，建议保留一些全局图特征，直接拼到图 embedding 后面：

* 平面长宽比
* 节点数
* 边数
* 墙节点数
* 主梁节点数
* 次梁节点数
* 墙总长度
* X/Y 向墙总长度比

也就是：**GNN + 少量全局手工特征**，通常比纯 GNN 更稳。

---

# 三、参数怎么输入

12 个参数别直接生硬拼原值，建议分开处理：

## 数值型

* `N`
* `tw_bot, tw_mid, tw_top`
* `hb_main, bb_main`
* `hb_sec, bb_sec`

做标准化后输入 MLP。

## 类别型

* `conc_bot`
* `intensity`
* `site_class`
* `seismic_group`

做 embedding 后拼接。
`intensity` 也可以先按类别处理。

最终得到一个 `param embedding`。

---

# 四、先做什么任务

## 第一阶段：只做分类

先只做：

* `final_pass`

损失函数：

* `BCEWithLogitsLoss(pos_weight=...)`

---

## 第二阶段：再做多任务

如果分类有效，再扩展成：

* 分类：`final_pass`
* 回归：`max_drift_ratio`
* 回归：`material_steel_kg`

共享一个 GNN backbone，再接多个 head。

---

# 五、推荐的 GNN 结构

第一版不要搞太花，直接用一个成熟方案。

## 首选：GraphSAGE / GIN / GAT

我建议顺序：

### 第一版

* **GraphSAGE** 或 **GIN**
* 3 层 message passing
* hidden dim = 128
* global mean pooling + global max pooling
* 得到 graph embedding

---

# 六、数据组织方式

## 一个关键点

这里是**同一个图对应 5000 个不同参数样本**。
所以要避免重复存图造成浪费。

先通过 `src\data_engine\postprocess\member_graph.py` 把每个 `layout_id` 的图结构处理好，保存成一个json文件，key 是 `layout_id`，value 是图的节点、边和特征。

训练时：

* 根据 `layout_id` 取图
* 再读该样本的参数和标签（前面已经处理为parquet格式）

---

## 划分规则

* 按之前已经完成的 train / val / test 的 split 划分样本

---

# 七、训练流程

## 最小训练设置

* optimizer: `AdamW`
* lr: `1e-3`
* batch size: 64 或 128 个样本
* epoch: 50
* early stopping: 10
* hidden dim: 128
* GNN layers: 3
* dropout: 0.2

---

# 十、最小落地版本

如果只给一个最简执行方案，就是：

### 输入

* 布局图：

  * 节点特征：构件类型、长度、方向、中心坐标、是否边界
  * 边特征：连接类型、中心距
* 参数向量：

  * 12 个设计变量

### 模型

* GraphSAGE/GIN 3 层
* global mean + max pooling
* 参数走一个 2 层 MLP
* 拼接后接分类头

### 输出

* `final_pass` 概率
