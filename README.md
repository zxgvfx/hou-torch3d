# hou-torch3d

Houdini 与 PyTorch3D 数据互转和深度学习项目

## 项目简介

本项目提供了 Houdini 几何体与 PyTorch3D 结构之间的高效数据转换，支持：

- **双向转换**: Houdini Geometry ↔ PyTorch3D Meshes/Pointclouds
- **属性支持**: 点属性、面属性、全局属性的完整转换
- **扩展功能**: 支持额外属性的 ExtendedMeshes 类
- **训练集成**: 专门的属性损失函数用于深度学习
- **完整测试**: 全面的测试覆盖和文档

## 快速开始

### 环境要求

- Python 3.11+
- Houdini 20.5+
- PyTorch 2.1+
- PyTorch3D 0.7.4+

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd hou-torch3d

# 安装依赖
pip install -r requirements.txt

# 或安装最小依赖
pip install -r requirements-minimal.txt
```

### 基本使用

```python
from pyLib.toolLib import dataConvert as dc
from pyLib.toolLib.extended_meshes import ExtendedMeshes
import torch

# 从Houdini几何体创建转换器
converter = dc.Convert(hou_geo=houdini_geometry)

# 添加自定义属性
converter.addAttrib('custom_attr', torch.rand(num_vertices, 3))

# 转换为ExtendedMeshes（支持额外属性）
extended_mesh = converter.toMeshes()

# 转换回Houdini
houdini_geo = converter.toHoudini()
```

## 项目结构

```
hou-torch3d/
├── pyLib/                    # 🎯 核心生产代码
│   ├── toolLib/             # 工具库（数据转换、扩展网格等）
│   ├── lossLib/             # 损失函数库
│   ├── houLib/              # Houdini集成库
│   └── ...
├── net/                     # 🧠 神经网络模块
├── train/                   # 🏃 训练脚本和示例
├── tests/                   # 🧪 所有测试代码
├── scripts/                 # 🛠️ 工具脚本
├── docs/                    # 📚 项目文档
└── file/                    # 📁 示例文件
```

## 核心功能

### 1. 数据转换 (`pyLib.toolLib.dataConvert`)

- **Convert 类**: 主要的转换器类
- **双向转换**: Houdini ↔ PyTorch3D
- **属性管理**: 完整的属性添加、获取、验证功能

### 2. 扩展网格 (`pyLib.toolLib.extended_meshes`)

- **ExtendedMeshes 类**: 支持额外属性的 PyTorch3D 网格
- **属性存储**: 任意数量和类型的顶点属性
- **兼容性**: 与标准 PyTorch3D Meshes 完全兼容

### 3. 属性损失函数 (`pyLib.lossLib.attribute_loss`)

- **一致性损失**: 属性值匹配损失
- **平滑性损失**: 相邻顶点属性平滑性约束
- **组合损失**: 多种损失的灵活组合

### 4. 训练示例 (`train/attribute_training_example.py`)

- **完整训练流程**: 从数据加载到损失计算
- **属性优化**: 使用额外属性进行网格优化
- **实际应用**: 可直接用于生产的训练代码

## 测试

### 运行所有测试

```bash
# 使用Houdini环境
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/ -v

# 使用系统Python（需要安装相应依赖）
pytest tests/ -v
```

### 运行特定测试

```bash
# 测试基础功能
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_basic_pytest.py -v

# 测试ExtendedMeshes
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_extended_meshes.py -v

# 测试数据转换
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_data_convert.py -v
```

### 检查环境

```bash
# 运行版本检查
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe scripts/check_versions.py
```

## 文档

- **[测试指南](docs/README_TESTING.md)**: 完整的测试设置和使用指南
- **[pytest手册](docs/PYTEST_MANUAL.md)**: pytest的详细使用说明
- **[Python兼容性](docs/PYTHON_COMPATIBILITY.md)**: Python 3.11兼容性说明
- **[项目结构](docs/PROJECT_STRUCTURE.md)**: 详细的项目结构说明

## 示例

### 基础数据转换

```python
import hou
from pyLib.toolLib import dataConvert as dc

# 获取Houdini几何体
geo = hou.pwd().geometry()

# 转换为PyTorch3D
converter = dc.Convert(hou_geo=geo)
mesh = converter.toMeshes()

print(f"顶点数: {mesh.verts_packed().shape[0]}")
print(f"面数: {mesh.faces_packed().shape[0]}")
```

### 使用额外属性

```python
import torch
from pyLib.toolLib.extended_meshes import ExtendedMeshes

# 创建带属性的网格
verts = torch.rand(100, 3)
faces = torch.randint(0, 100, (50, 3))
attributes = {
    'color': torch.rand(100, 3),
    'weight': torch.rand(100, 1)
}

mesh = ExtendedMeshes(verts=[verts], faces=[faces], attributes=attributes)

# 访问属性
color = mesh.get_attribute('color')
print(f"颜色属性形状: {color.shape}")
```

### 训练中使用属性损失

```python
from pyLib.lossLib.attribute_loss import CombinedAttributeLoss

# 创建损失函数
loss_fn = CombinedAttributeLoss(
    consistency_weight=1.0,
    smoothness_weight=0.1
)

# 在训练循环中使用
loss = loss_fn(predicted_mesh, target_attributes)
loss.backward()
```

## 开发

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v
```

### 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

本项目采用 [LICENSE](LICENSE) 许可证。

## 作者

[项目作者信息]

## 更新日志

- **v1.0.0**: 初始版本，支持基础数据转换
- **v1.1.0**: 添加ExtendedMeshes和属性损失函数
- **v1.2.0**: 完整的测试覆盖和文档
- **v1.3.0**: 项目结构优化，生产代码和测试代码分离
