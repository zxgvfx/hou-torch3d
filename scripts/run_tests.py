#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
"""

import sys
import os
import subprocess
import argparse

def run_pytest_with_houdini():
    """使用Houdini环境运行pytest"""
    houdini_python = r"C:\Program Files\Side Effects Software\Houdini 20.5.613\bin\hython.exe"
    
    if not os.path.exists(houdini_python):
        print("错误：找不到Houdini Python环境")
        return False
    
    cmd = [houdini_python, "-m", "pytest", "tests/", "-v"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False

def run_pytest_with_system_python():
    """使用系统Python运行pytest"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行项目测试")
    parser.add_argument("--houdini", action="store_true", help="使用Houdini环境运行测试")
    parser.add_argument("--system", action="store_true", help="使用系统Python运行测试")
    parser.add_argument("--markers", help="运行特定标记的测试")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    
    args = parser.parse_args()
    
    if args.houdini:
        print("使用Houdini环境运行测试...")
        success = run_pytest_with_houdini()
    elif args.system:
        print("使用系统Python运行测试...")
        success = run_pytest_with_system_python()
    else:
        print("默认使用Houdini环境运行测试...")
        success = run_pytest_with_houdini()
    
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 