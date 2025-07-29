#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试属性传递功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import hou
import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pyLib.toolLib.dataConvert import Convert

def test_attribute_transfer():
    """测试属性传递功能"""
    print("=== 测试属性传递功能 ===")
    
    # 1. 创建测试几何体
    print("\n[1] 创建测试几何体")
    geo = hou.Geometry()
    for i in range(4):
        pt = geo.createPoint()
        pt.setPosition((i, 0, 0))
    poly = geo.createPolygon()
    for pt in geo.points():
        poly.addVertex(pt)
    
    print(f"创建了几何体: {len(geo.points())} 个点, {len(geo.prims())} 个面")
    
    # 2. 测试添加属性后toMeshes
    print("\n[2] 测试添加属性后toMeshes")
    cvt = Convert(hou_geo=geo)
    
    # 添加点属性
    point_attr = torch.rand(4, 3)
    cvt.addAttrib('test_point_attr', point_attr)
    print(f"添加了点属性: {point_attr.shape}")
    
    # 添加面属性
    prim_attr = torch.rand(1, 2)
    cvt.addPrimAttrib('test_prim_attr', prim_attr)
    print(f"添加了面属性: {prim_attr.shape}")
    
    # 添加全局属性
    cvt.addDetailAttrib('test_detail_attr', 42.0)
    print("添加了全局属性")
    
    # 转换为PyTorch3D几何体
    mesh = cvt.toMeshes()
    print(f"转换为PyTorch3D几何体: {type(mesh).__name__}")
    
    # 3. 验证属性是否传递到PyTorch3D几何体
    print("\n[3] 验证属性传递")
    
    # 检查点属性
    if hasattr(mesh, 'features_packed') and mesh.features_packed() is not None:
        features = mesh.features_packed()
        print(f"✓ 点云features存在: {features.shape}")
    else:
        print("✗ 点云features不存在")
    
    # 检查面属性（仅对Meshes）
    if isinstance(mesh, Meshes):
        print("✓ 网格几何体，面属性应该已添加")
    else:
        print("ℹ 点云几何体，面属性不适用")
    
    # 4. 测试updateFromGeo的属性传递
    print("\n[4] 测试updateFromGeo的属性传递")
    cvt2 = Convert(hou_geo=geo)
    
    # 添加属性
    cvt2.addAttrib('update_test_attr', torch.rand(4, 3))
    cvt2.addPrimAttrib('update_prim_attr', torch.rand(1, 2))
    cvt2.addDetailAttrib('update_detail_attr', 123.0)
    
    # 使用updateFromGeo
    cvt2.updateFromGeo(geo)
    mesh2 = cvt2.t3d_geo
    print(f"updateFromGeo转换成功: {type(mesh2).__name__}")
    
    # 5. 测试属性获取
    print("\n[5] 测试属性获取")
    
    # 获取点属性
    point_attr_retrieved = cvt.getAttrib('test_point_attr')
    if point_attr_retrieved is not None:
        print(f"✓ 成功获取点属性: {point_attr_retrieved.shape}")
    else:
        print("✗ 获取点属性失败")
    
    # 获取面属性
    prim_attr_retrieved = cvt.getPrimAttrib('test_prim_attr')
    if prim_attr_retrieved is not None:
        print(f"✓ 成功获取面属性: {prim_attr_retrieved.shape}")
    else:
        print("✗ 获取面属性失败")
    
    # 获取全局属性
    detail_attr_retrieved = cvt.getDetailAttrib('test_detail_attr')
    if detail_attr_retrieved is not None:
        print(f"✓ 成功获取全局属性: {detail_attr_retrieved}")
    else:
        print("✗ 获取全局属性失败")
    
    # 6. 测试状态信息
    print("\n[6] 测试状态信息")
    info = cvt.get_state_info()
    print("状态信息:", info)
    
    # 7. 测试属性验证
    print("\n[7] 测试属性验证")
    try:
        # 测试错误的属性大小
        wrong_attr = torch.rand(10, 3)  # 错误的点数
        cvt.addAttrib('wrong_size_attr', wrong_attr)
        print("✗ 应该报错但没有")
    except ValueError as e:
        print(f"✓ 属性验证正确: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_attribute_transfer() 