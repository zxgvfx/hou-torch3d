# pytest 手动测试指南

## 正确的命令格式

在你的项目中，使用Houdini的Python环境时，正确的命令格式是：

```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest [选项] [测试文件]
```

**注意**：双引号的位置很重要！

## 基本测试命令

### 1. 测试单个函数
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::test_simple_math -v -s
```

### 2. 测试单个类
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::TestCalculator -v -s
```

### 3. 测试单个文件
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py -v
```

### 4. 测试整个tests目录
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/ -v
```

### 5. 测试项目的ExtendedMeshes
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_extended_meshes.py -v
```

## 常用选项说明

### 输出选项
- `-v` 或 `--verbose`: 详细输出
- `-s`: 显示print输出（不捕获stdout）
- `-q` 或 `--quiet`: 简化输出
- `--tb=short`: 简短的错误追踪
- `--tb=long`: 详细的错误追踪

### 运行选项
- `-x`: 遇到第一个失败就停止
- `--maxfail=2`: 失败2次后停止
- `-k "test_name"`: 只运行名称包含"test_name"的测试
- `--lf`: 只运行上次失败的测试
- `--ff`: 先运行失败的测试

### 覆盖率选项
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest --cov=pyLib tests/
```

## 实际测试示例

### 测试基础功能
```bash
# 测试基本数学运算
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::test_simple_math -v -s

# 测试PyTorch操作
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::test_torch_operations -v -s

# 测试NumPy操作
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::test_numpy_operations -v -s
```

### 测试项目代码
```bash
# 测试ExtendedMeshes创建
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_extended_meshes.py::TestExtendedMeshes::test_creation -v -s

# 测试属性操作
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_extended_meshes.py::TestExtendedMeshes::test_attribute_access -v -s

# 测试数据转换
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_data_convert.py::TestDataConvert::test_add_attrib -v -s
```

### 参数化测试
```bash
# 测试乘法（会运行4次不同的参数）
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest test_basic_pytest.py::test_multiply_by_two -v -s
```

## 测试结果解读

### 成功的测试
```
test_basic_pytest.py::test_simple_math PASSED                [100%]
```

### 失败的测试
```
test_basic_pytest.py::test_will_fail FAILED                  [100%]
================================ FAILURES ================================
```

### 测试统计
```
============== 15 passed in 2.08s ==============
```

## 常见问题和解决方案

### 1. 命令不工作
**问题**: 命令执行没有输出或报错
**解决**: 检查双引号位置，使用正确格式：
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest
```

### 2. 找不到模块
**问题**: `No module named 'pyLib'`
**解决**: 确保在项目根目录下运行命令

### 3. 测试失败
**问题**: 某些测试失败
**解决**: 查看详细错误信息，使用 `-v` 和 `--tb=long` 选项

### 4. 警告信息
**问题**: `PytestUnknownMarkWarning`
**解决**: 这是正常的，表示使用了自定义标记

## 调试技巧

### 1. 添加打印语句
在测试函数中添加 `print()` 语句，使用 `-s` 选项查看输出：
```python
def test_debug():
    print("调试信息：变量值为", variable)
    assert condition
```

### 2. 使用断点
```python
def test_with_breakpoint():
    import pdb; pdb.set_trace()  # 添加断点
    assert condition
```

### 3. 查看详细错误
```bash
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/ -v --tb=long
```

## 快速开始

1. **创建简单测试**：从 `test_basic_pytest.py` 开始
2. **运行单个测试**：确保基本功能正常
3. **运行项目测试**：测试实际代码
4. **查看结果**：分析通过和失败的测试

## 最佳实践

1. **小步骤测试**：先测试简单功能，再测试复杂功能
2. **使用详细输出**：始终使用 `-v` 选项
3. **保留print输出**：使用 `-s` 选项查看调试信息
4. **逐个测试**：先测试单个函数，再测试整个文件
5. **查看文档**：遇到问题时查看错误信息和文档

这样你就可以有效地使用pytest来测试你的代码了！ 