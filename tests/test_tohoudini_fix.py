#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的toHoudini函数
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import hou
import torch
import numpy as np
from pyLib.toolLib.dataConvert import Convert

def test_tohoudini_with_attributes():
    """测试使用默认hou_geo并添加属性后调用toHoudini的情况"""
    print("=== 测试toHoudini函数改进 ===")
    
    # 1. 创建一个简单的Houdini几何体
    geo = hou.Geometry()
    
    # 创建4个点
    for i in range(4):
        pt = geo.createPoint()
        pt.setPosition((i, 0, 0))
    
    # 创建一个面
    poly = geo.createPolygon()
    for pt in geo.points():
        poly.addVertex(pt)
    
    print(f"原始几何体: {len(geo.points())} 个点, {len(geo.prims())} 个面")
    
    # 2. 使用Convert初始化
    cvt = Convert(hou_geo=geo)
    print(f"初始化后 - has_mesh: {cvt.has_mesh}, has_geo: {cvt.has_geo}")
    
    # 3. 添加PyTorch生成的属性
    num_points = len(geo.points())
    attr_val = torch.rand(num_points, 3)  # 3维属性
    cvt.addAttrib('test_attr', attr_val)
    print(f"添加了test_attr属性，形状: {attr_val.shape}")
    
    # 4. 添加面属性
    num_prims = len(geo.prims())
    prim_attr = torch.rand(num_prims, 2)  # 2维面属性
    cvt.addPrimAttrib('prim_attr', prim_attr)
    print(f"添加了prim_attr面属性，形状: {prim_attr.shape}")
    
    # 5. 添加全局属性
    cvt.addDetailAttrib('detail_attr', 42.0)
    print("添加了detail_attr全局属性")
    
    # 6. 调用toHoudini
    result_geo = cvt.toHoudini()
    print(f"toHoudini后几何体: {len(result_geo.points())} 个点, {len(result_geo.prims())} 个面")
    
    # 7. 验证属性是否正确应用
    print("\n=== 验证属性应用 ===")
    
    # 检查点属性
    if result_geo.findPointAttrib('test_attr'):
        print("✓ test_attr点属性存在")
        # 检查前3个点的属性值
        for i in range(min(3, len(result_geo.points()))):
            attr_val_actual = result_geo.point(i).attribValue('test_attr')
            attr_val_expected = attr_val[i].cpu().detach().numpy()
            print(f"  点{i}: 期望 {attr_val_expected}, 实际 {attr_val_actual}")
    else:
        print("✗ test_attr点属性不存在")
    
    # 检查面属性
    if result_geo.findPrimAttrib('prim_attr'):
        print("✓ prim_attr面属性存在")
        # 检查第一个面的属性值
        if len(result_geo.prims()) > 0:
            attr_val_actual = result_geo.prim(0).attribValue('prim_attr')
            attr_val_expected = prim_attr[0].cpu().detach().numpy()
            print(f"  面0: 期望 {attr_val_expected}, 实际 {attr_val_actual}")
    else:
        print("✗ prim_attr面属性不存在")
    
    # 检查全局属性
    if result_geo.findGlobalAttrib('detail_attr'):
        print("✓ detail_attr全局属性存在")
        attr_val_actual = result_geo.attribValue('detail_attr')
        print(f"  全局属性值: {attr_val_actual}")
    else:
        print("✗ detail_attr全局属性不存在")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_tohoudini_with_attributes() 