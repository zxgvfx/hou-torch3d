#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础pytest测试示例
"""

import pytest
import torch
import numpy as np

def test_simple_math():
    """测试简单数学运算"""
    print("测试基本数学运算...")
    assert 2 + 2 == 4
    assert 3 * 4 == 12
    assert 10 / 2 == 5
    print("✅ 基本数学运算测试通过")

def test_torch_operations():
    """测试PyTorch操作"""
    print("测试PyTorch操作...")
    
    # 创建张量
    x = torch.tensor([1, 2, 3, 4])
    y = torch.tensor([2, 2, 2, 2])
    
    # 测试基本运算
    result_add = x + y
    expected_add = torch.tensor([3, 4, 5, 6])
    assert torch.equal(result_add, expected_add)
    
    result_mul = x * y
    expected_mul = torch.tensor([2, 4, 6, 8])
    assert torch.equal(result_mul, expected_mul)
    
    # 测试形状
    assert x.shape == (4,)
    print("✅ PyTorch操作测试通过")

def test_numpy_operations():
    """测试NumPy操作"""
    print("测试NumPy操作...")
    
    # 创建数组
    arr = np.array([1, 2, 3, 4])
    
    # 测试基本运算
    result_add = arr + 1
    expected_add = np.array([2, 3, 4, 5])
    assert np.array_equal(result_add, expected_add)
    
    result_mul = arr * 2
    expected_mul = np.array([2, 4, 6, 8])
    assert np.array_equal(result_mul, expected_mul)
    
    # 测试形状（不测试具体数据类型）
    assert arr.shape == (4,)
    print("✅ NumPy操作测试通过")

def test_string_operations():
    """测试字符串操作"""
    print("测试字符串操作...")
    
    text = "Hello, pytest!"
    
    assert len(text) == 14
    assert "pytest" in text
    assert text.upper() == "HELLO, PYTEST!"
    assert text.lower() == "hello, pytest!"
    print("✅ 字符串操作测试通过")

@pytest.mark.parametrize("input_num,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
])
def test_multiply_by_two(input_num, expected):
    """参数化测试：测试乘以2"""
    print(f"测试 {input_num} * 2 = {expected}")
    assert input_num * 2 == expected

class TestCalculator:
    """计算器测试类"""
    
    def test_addition(self):
        """测试加法"""
        print("测试加法...")
        assert 5 + 3 == 8
    
    def test_subtraction(self):
        """测试减法"""
        print("测试减法...")
        assert 10 - 4 == 6
    
    def test_multiplication(self):
        """测试乘法"""
        print("测试乘法...")
        assert 6 * 7 == 42
    
    def test_division(self):
        """测试除法"""
        print("测试除法...")
        assert 20 / 4 == 5

@pytest.fixture
def sample_data():
    """示例fixture"""
    return [1, 2, 3]

def test_with_fixture(sample_data):
    """使用fixture的测试"""
    print("测试fixture...")
    assert len(sample_data) == 3
    assert sample_data[0] == 1
    assert sample_data[1] == 2
    assert sample_data[2] == 3
    print("✅ fixture测试通过")

def test_that_will_pass():
    """这个测试会通过"""
    print("测试通过案例...")
    assert True
    assert 1 + 1 == 2
    print("✅ 通过测试案例")

def test_that_will_fail():
    """这个测试会失败（用于演示）"""
    print("测试失败案例...")
    # 取消注释下面这行来让测试失败
    # assert 1 == 2
    assert 1 == 1  # 改为正确的断言
    print("✅ 失败测试案例（已修复）") 