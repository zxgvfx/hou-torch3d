# 项目测试指南

## pytest 简介

**pytest** 是 Python 中最流行的测试框架之一，具有以下特点：

- **简单易用**：使用简单的 `assert` 语句进行断言
- **自动发现**：自动发现和运行测试文件
- **丰富的插件**：大量第三方插件扩展功能
- **详细报告**：提供详细的测试报告和失败信息
- **参数化测试**：支持数据驱动测试
- **夹具系统**：强大的 `fixture` 系统用于测试设置

## 项目测试结构

```
hou-torch3d/
├── tests/                          # 测试目录
│   ├── __init__.py
│   ├── conftest.py                 # pytest配置文件，包含共享fixtures
│   ├── test_extended_meshes.py     # ExtendedMeshes测试
│   ├── test_attribute_loss.py      # 属性损失函数测试
│   └── test_data_convert.py        # dataConvert测试
├── pytest.ini                      # pytest配置文件
├── requirements.txt                 # 项目依赖
├── run_tests.py                    # 测试运行脚本
└── test_pytest_compatibility.py    # pytest兼容性测试
```

## 安装和设置

### 1. 安装依赖

```bash
# 安装pytest和相关插件
pip install pytest pytest-cov pytest-mock pytest-xdist

# 或者使用requirements.txt
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 检查pytest是否可用
python -m pytest --version
```

## 运行测试

### 方法1：使用测试运行脚本

```bash
# 使用Houdini环境运行测试
python run_tests.py --houdini

# 使用系统Python运行测试
python run_tests.py --system
```

### 方法2：直接使用pytest

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_extended_meshes.py

# 运行特定测试类
pytest tests/test_extended_meshes.py::TestExtendedMeshes

# 运行特定测试方法
pytest tests/test_extended_meshes.py::TestExtendedMeshes::test_creation
```

### 方法3：使用标记运行测试

```bash
# 运行单元测试
pytest -m unit

# 运行集成测试
pytest -m integration

# 运行ExtendedMeshes相关测试
pytest -m extended_meshes

# 运行属性损失函数测试
pytest -m attribute_loss

# 运行数据转换测试
pytest -m data_convert
```

## 测试标记说明

项目定义了以下测试标记：

- `unit`: 单元测试
- `integration`: 集成测试
- `slow`: 慢速测试
- `houdini`: 需要Houdini环境的测试
- `extended_meshes`: ExtendedMeshes相关测试
- `attribute_loss`: 属性损失函数测试
- `data_convert`: 数据转换测试

## 生成测试报告

### 覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=pyLib tests/

# 生成详细的HTML覆盖率报告
pytest --cov=pyLib --cov-report=html tests/
```

### 并行运行测试

```bash
# 使用多进程运行测试
pytest -n auto tests/
```

## 测试文件说明

### 1. conftest.py

包含共享的 fixtures，为所有测试提供通用的测试数据：

- `device`: 测试设备（CPU/CUDA）
- `sample_verts`: 示例顶点数据
- `sample_faces`: 示例面数据
- `sample_attributes`: 示例属性数据
- `sample_mesh`: 示例PyTorch3D网格
- `sample_extended_mesh`: 示例ExtendedMeshes
- `sample_houdini_geometry`: 示例Houdini几何体
- `mock_houdini_geometry`: 模拟的Houdini几何体（用于单元测试）

### 2. test_extended_meshes.py

测试 ExtendedMeshes 类的功能：

- 创建和初始化
- 属性添加、访问、检查
- 属性移除和清除
- 设备处理
- 错误处理

### 3. test_attribute_loss.py

测试属性损失函数：

- 一致性损失
- 平滑性损失
- 梯度损失
- 组合损失
- 训练模拟

### 4. test_data_convert.py

测试 dataConvert 模块：

- 转换器创建
- 属性添加和验证
- 状态管理
- 错误处理
- 集成测试

## 编写新测试

### 1. 创建测试文件

```python
# tests/test_new_feature.py
import pytest
import torch
from pyLib.toolLib.extended_meshes import ExtendedMeshes

class TestNewFeature:
    """新功能测试类"""
    
    @pytest.mark.unit
    def test_basic_functionality(self, sample_extended_mesh):
        """测试基本功能"""
        # 你的测试代码
        assert True
    
    @pytest.mark.integration
    def test_integration(self, sample_mesh):
        """测试集成功能"""
        # 集成测试代码
        assert True
```

### 2. 使用 fixtures

```python
def test_with_fixtures(sample_extended_mesh, device):
    """使用共享fixtures的测试"""
    mesh = sample_extended_mesh.to(device)
    assert mesh.device == device
```

### 3. 参数化测试

```python
@pytest.mark.parametrize("attr_name,attr_shape", [
    ("color", (4, 3)),
    ("weight", (4, 1)),
    ("feature", (4, 4))
])
def test_attribute_shapes(self, sample_extended_mesh, attr_name, attr_shape):
    """参数化测试属性形状"""
    attr = sample_extended_mesh.get_attribute(attr_name)
    assert attr.shape == attr_shape
```

## 最佳实践

### 1. 测试命名

- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试方法：`test_*`

### 2. 测试组织

- 单元测试：测试单个函数或方法
- 集成测试：测试多个组件的交互
- 端到端测试：测试完整的工作流程

### 3. 测试数据

- 使用 fixtures 提供测试数据
- 避免硬编码测试数据
- 使用模拟对象隔离外部依赖

### 4. 断言

- 使用明确的断言
- 测试边界条件
- 测试错误情况

## 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保项目路径正确
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Houdini环境问题**
   ```bash
   # 使用Houdini Python环境
   "C:\Program Files\Side Effects Software\Houdini 20.5.613\bin\hython.exe" -m pytest tests/
   ```

3. **设备问题**
   ```bash
   # 强制使用CPU
   CUDA_VISIBLE_DEVICES="" pytest tests/
   ```

### 调试测试

```bash
# 详细输出
pytest -v tests/

# 显示本地变量
pytest -l tests/

# 在失败时停止
pytest -x tests/

# 显示最慢的测试
pytest --durations=10 tests/
```

## 持续集成

可以在CI/CD管道中使用以下命令：

```yaml
# GitHub Actions示例
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=pyLib --cov-report=xml tests/
```

这样设置后，你的项目就有了完整的测试框架，可以确保代码质量和功能正确性！ 