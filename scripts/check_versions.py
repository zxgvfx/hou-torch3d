#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本兼容性检查脚本
"""

import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    print("=== Python版本检查 ===")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11:
        print("✅ Python 3.11+ 兼容性良好")
        return True
    else:
        print("⚠ 建议使用 Python 3.11+")
        return False

def check_package_version(package_name, min_version=None):
    """检查包版本"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"{package_name}: {version}")
        
        if min_version:
            # 简单的版本比较
            current_parts = version.split('.')
            min_parts = min_version.split('.')
            
            for i in range(max(len(current_parts), len(min_parts))):
                current = int(current_parts[i]) if i < len(current_parts) else 0
                minimum = int(min_parts[i]) if i < len(min_parts) else 0
                
                if current > minimum:
                    print(f"✅ {package_name} 版本满足要求")
                    return True
                elif current < minimum:
                    print(f"❌ {package_name} 版本过低，建议升级到 {min_version}+")
                    return False
            
            print(f"✅ {package_name} 版本满足要求")
            return True
        else:
            return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def check_requirements():
    """检查requirements.txt中的包"""
    print("\n=== 包版本检查 ===")
    
    requirements = {
        'torch': '2.1.0',
        'numpy': '1.24.0',
        'pytorch3d': '0.7.4'
    }
    
    all_good = True
    for package, min_version in requirements.items():
        if not check_package_version(package, min_version):
            all_good = False
    
    return all_good

def check_houdini_environment():
    """检查Houdini环境"""
    print("\n=== Houdini环境检查 ===")
    
    try:
        import hou
        print("✅ Houdini环境可用")
        
        # 检查Houdini版本
        houdini_version = hou.applicationVersion()
        print(f"Houdini版本: {houdini_version}")
        
        return True
    except ImportError:
        print("❌ Houdini环境不可用")
        return False

def check_pytest_availability():
    """检查pytest可用性"""
    print("\n=== pytest可用性检查 ===")
    
    try:
        import pytest
        print(f"✅ pytest可用，版本: {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest未安装")
        print("安装命令: pip install pytest>=7.4.0")
        return False

def main():
    """主检查函数"""
    print("开始版本兼容性检查...\n")
    
    checks = [
        ("Python版本", check_python_version),
        ("包版本", check_requirements),
        ("Houdini环境", check_houdini_environment),
        ("pytest可用性", check_pytest_availability)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 检查失败: {e}")
            results.append((name, False))
    
    print("\n=== 检查结果汇总 ===")
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 所有检查通过！你的环境完全兼容。")
        print("\n建议:")
        print("1. 使用 requirements.txt 安装依赖")
        print("2. 运行 pytest tests/ 进行测试")
        print("3. 查看 README_TESTING.md 了解详细用法")
    else:
        print("⚠ 部分检查失败，请根据上述信息进行修复。")
        print("\n修复建议:")
        print("1. 升级Python到3.11+")
        print("2. 安装/升级相关包: pip install -r requirements.txt")
        print("3. 确保Houdini环境正确配置")

if __name__ == "__main__":
    main() 