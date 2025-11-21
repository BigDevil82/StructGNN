# PNG to DXF Converter

将语义化剪力墙结构平面布局PNG图像转换为CAD DXF文件的工具。

## 功能特性

- 🎯 **多元素识别**: 自动识别剪力墙、填充墙、窗户、门、梁等构件
- 🎨 **颜色语义化**: 基于HSV颜色空间进行精确分割
- 📐 **精确轮廓**: 智能轮廓检测和多边形简化
- 🗂️ **分层管理**: 自动创建DXF图层，便于CAD软件管理
- ⚡ **高效处理**: 批量处理多个构件类型

## 支持的构件类型

| 构件类型 | 颜色 | DXF图层 | 说明 |
|---------|------|---------|------|
| 剪力墙 (sw) | 红色 | SHEAR_WALLS | 结构主要承重墙 |
| 填充墙 (iw) | 灰色 | INFILL_WALLS | 非承重填充墙 |
| 窗户 (window) | 绿色 | WINDOWS | 窗户开口 |
| 门 (door) | 蓝色 | DOORS | 门洞开口 |
| 梁 (beam) | 黑色 | BEAMS | 结构梁 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本用法

```python
from png2dxf_converter import PNG2DXFConverter

# 创建转换器
converter = PNG2DXFConverter("input.png", scale=1.0)

# 预览轮廓检测结果（可选）
converter.preview_contours()

# 执行转换
converter.convert("output.dxf")
```

### 高级用法

```python
# 自定义缩放比例（1像素 = 10mm）
converter = PNG2DXFConverter("input.png", scale=10.0)

# 只预览特定类型构件
converter.preview_contours("sw")  # 只显示剪力墙

# 处理单一构件类型
converter.process_element_type("window")
```

## 技术原理

### 1. 图像预处理
- HSV颜色空间分割
- 形态学操作去噪
- 二值化处理

### 2. 轮廓检测
- 外轮廓提取
- 多边形近似简化
- 面积过滤

### 3. 坐标转换
- 像素坐标到世界坐标
- Y轴翻转处理
- 缩放变换

### 4. DXF生成
- 创建图层结构
- 生成填充多边形
- 设置颜色属性

## 参数配置

### 转换器参数
- `scale`: 像素到实际尺寸的缩放比例
- `epsilon_factor`: 轮廓简化程度（默认0.02）
- `min_area`: 最小轮廓面积阈值（默认100）

### 构件配置
每种构件类型可配置：
- DXF颜色代码
- 图层名称
- 是否填充

## 文件结构

```
├── png2dxf_converter.py    # 主转换器类
├── element_extractor.py    # 元素提取器（现有）
├── img_util.py            # 图像工具函数（现有）
├── requirements.txt       # 依赖包列表
└── README.md             # 说明文档
```

## 注意事项

1. **图像质量**: 输入PNG图像应具有清晰的颜色边界
2. **颜色标准**: 确保各构件颜色符合预定义标准
3. **尺寸缩放**: 根据实际需要调整scale参数
4. **DXF兼容性**: 生成的DXF文件兼容AutoCAD 2010及以上版本

## DXF构件信息提取

在工程师完成梁的绘制后，使用`dxf_extractor.py`提取构件信息：

### 基本用法

```python
from dxf_extractor import DXFExtractor

# 提取单个文件
extractor = DXFExtractor()
data = extractor.extract_from_file("completed_drawing.dxf")

# 批量提取
extractor.extract_batch("dxf_files/", "output_json/")
```

### 输出格式

```json
{
  "walls": [
    {
      "StartPoint": {"X": 1000.0, "Y": 1000.0},
      "EndPoint": {"X": 1000.0, "Y": 3000.0}
    }
  ],
  "beams": [
    {
      "StartPoint": {"X": 1500.0, "Y": 1500.0},
      "EndPoint": {"X": 4500.0, "Y": 1500.0}
    }
  ]
}
```

### 支持的构件类型
- **剪力墙**: 从`SHEAR_WALLS`图层提取多段线并拆分为线段
- **梁**: 从`BEAMS`图层提取直线和多段线

## 完整工作流程

1. **PNG转DXF**: `python png2dxf_converter.py`
2. **工程师绘制梁**: 在`BEAMS`图层绘制直线
3. **提取构件信息**: `python dxf_extractor.py`
4. **获得JSON结果**: 包含所有墙和梁的坐标信息

## 扩展开发

### 添加新构件类型
1. 在`element_extractor.py`中添加HSV分割逻辑
2. 在`PNG2DXFConverter`中添加对应配置
3. 在`DXFExtractor`中添加提取逻辑

### 优化轮廓检测
- 调整形态学操作参数
- 修改多边形简化系数
- 增加轮廓后处理逻辑
