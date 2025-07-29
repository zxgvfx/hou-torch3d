# 项目结构说明

## 为什么要分离测试代码和生产代码？

### 1. **代码组织清晰**
- 生产代码专注于功能实现
- 测试代码专注于验证功能
- 便于维护和查找

### 2. **部署安全**
- 生产环境不需要测试代码
- 减少部署包大小
- 避免测试代码泄露敏感信息

### 3. **依赖管理**
- 测试依赖不会污染生产环境
- 可以使用不同的依赖版本
- 便于管理开发和生产环境

### 4. **版本控制**
- 可以单独管理测试代码的版本
- 测试代码更改不影响生产代码标签
- 便于代码审查

## 当前项目结构

```
hou-torch3d/                           # 项目根目录
├── pyLib/                             # 🎯 核心生产代码
│   ├── __init__.py
│   ├── toolLib/                       # 工具库
│   │   ├── __init__.py
│   │   ├── dataConvert.py            # 数据转换模块
│   │   ├── extended_meshes.py        # 扩展网格类
│   │   └── hou_plots.py             # Houdini绘图工具
│   ├── lossLib/                      # 损失函数库
│   │   ├── __init__.py
│   │   ├── attribute_loss.py         # 属性损失函数
│   │   └── blendloss.py             # 混合损失函数
│   ├── houLib/                       # Houdini库
│   │   └── ...
│   ├── initLib/                      # 初始化库
│   │   └── ...
│   └── utilLib/                      # 工具库
│       └── ...
├── net/                              # 🧠 神经网络模块
│   ├── __init__.py
│   ├── blendshape/
│   └── topo_landmart/
├── train/                            # 🏃 训练脚本
│   ├── __init__.py
│   ├── blenshape/
│   ├── hou_example.py
│   └── attribute_training_example.py
├── tests/                            # 🧪 所有测试代码
│   ├── __init__.py
│   ├── conftest.py                   # pytest配置和共享fixtures
│   ├── test_extended_meshes.py       # ExtendedMeshes测试
│   ├── test_attribute_loss.py        # 属性损失函数测试
│   ├── test_data_convert.py          # 数据转换测试
│   ├── test_basic_pytest.py          # 基础pytest示例
│   ├── test_real_world_usage.py      # 真实场景测试
│   ├── test_complete_attribute.py    # 完整属性测试
│   ├── test_attribute_transfer.py    # 属性传输测试
│   ├── test_improvements.py          # 改进功能测试
│   └── test_tohoudini_fix.py         # Houdini修复测试
├── scripts/                          # 🛠️ 工具脚本
│   ├── check_versions.py             # 版本检查脚本
│   └── run_tests.py                  # 测试运行脚本
├── docs/                             # 📚 项目文档
│   ├── README_TESTING.md             # 测试指南
│   ├── PYTHON_COMPATIBILITY.md       # Python兼容性
│   ├── PYTEST_MANUAL.md              # pytest手册
│   └── PROJECT_STRUCTURE.md          # 项目结构说明
├── file/                             # 📁 示例文件
│   └── obj/
│       └── dolphin.obj
├── requirements.txt                   # 📦 项目依赖
├── requirements-minimal.txt           # 📦 最小依赖
├── pytest.ini                        # ⚙️ pytest配置
├── README.md                          # 📖 项目说明
├── LICENSE                            # 📄 许可证
└── .gitignore                         # 🚫 Git忽略文件
```

## 各目录职责

### 🎯 生产代码目录
- **`pyLib/`**: 核心库代码，包含所有可重用的模块
- **`net/`**: 神经网络模型定义
- **`train/`**: 训练脚本和示例

### 🧪 测试相关目录
- **`tests/`**: 所有测试代码，包含单元测试、集成测试等
- **`scripts/`**: 工具脚本，如版本检查、测试运行等

### 📚 文档和配置
- **`docs/`**: 项目文档
- **`file/`**: 示例文件和资源
- **配置文件**: requirements.txt, pytest.ini 等

## 测试代码分类

### 1. **单元测试** (Unit Tests)
```
tests/test_extended_meshes.py      # 测试ExtendedMeshes类
tests/test_attribute_loss.py       # 测试损失函数
tests/test_data_convert.py         # 测试数据转换
```

### 2. **集成测试** (Integration Tests)
```
tests/test_real_world_usage.py     # 真实场景测试
tests/test_complete_attribute.py   # 完整属性流程测试
```

### 3. **示例测试** (Example Tests)
```
tests/test_basic_pytest.py         # pytest基础示例
tests/test_improvements.py         # 功能改进测试
```

### 4. **修复测试** (Fix Tests)
```
tests/test_tohoudini_fix.py        # 特定bug修复测试
tests/test_attribute_transfer.py   # 属性传输修复测试
```

## 命令更新

由于文件移动，命令也需要更新：

### 运行测试
```bash
# 运行所有测试
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/ -v

# 运行特定测试
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_extended_meshes.py -v

# 运行基础示例
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe -m pytest tests/test_basic_pytest.py -v
```

### 运行工具脚本
```bash
# 检查版本
C:"\Program Files\Side Effects Software\Houdini 20.5.613\bin"\hython.exe scripts/check_versions.py

# 运行测试脚本
python scripts/run_tests.py --houdini
```

## 最佳实践

### 1. **测试文件命名**
- 测试文件以 `test_` 开头
- 测试类以 `Test` 开头
- 测试方法以 `test_` 开头

### 2. **测试组织**
- 一个模块对应一个测试文件
- 相关的测试放在同一个测试类中
- 使用描述性的测试名称

### 3. **依赖管理**
- 生产依赖放在 `requirements.txt`
- 测试依赖也放在 `requirements.txt` 但标注清楚
- 使用 `requirements-minimal.txt` 只包含核心依赖

### 4. **文档维护**
- 在 `docs/` 目录维护项目文档
- 测试相关文档也放在 `docs/` 中
- 保持文档与代码同步

## 部署时的考虑

### 生产环境部署
```bash
# 只部署生产代码
rsync -av --exclude='tests/' --exclude='scripts/' --exclude='docs/' . production/
```

### 开发环境设置
```bash
# 安装完整依赖（包括测试工具）
pip install -r requirements.txt

# 设置开发环境
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

这样的结构确保了：
✅ 生产代码和测试代码完全分离  
✅ 项目结构清晰易懂  
✅ 便于维护和部署  
✅ 遵循业界最佳实践 