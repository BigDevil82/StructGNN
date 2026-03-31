# shearwall_pred

本目录是当前项目中最核心的预测模型模块，负责房间级剪力墙布局预测。

## 主要职责

- 数据集定义与缓存
- 数据增强
- 模型结构定义
- 损失函数
- 训练、测试、可视化
- 交叉验证与集成

## 主要文件

- `dataset.py`: 数据集定义
- `augmentor.py`: 数据增强
- `model.py`: 模型结构
- `losses.py`: 损失函数
- `trainer.py`: 训练 / 测试 / 可视化入口
- `cross_validate.py`: 交叉验证与集成

## 主要入口

推荐入口：

```bash
python -m shearwall_pred.trainer --mode train
python -m shearwall_pred.trainer --mode test --ckpt path\to\model.pth
python -m shearwall_pred.trainer --mode visualize --ckpt path\to\model.pth
```

## 边界建议

- 这里应只承载预测模型相关逻辑
- 不建议混入外部建模软件接口
- 案例流程应通过 `pipelines` 或包装入口调用本目录，而不是反过来
