#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的属性传递功能测试
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import hou
import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pyLib.toolLib.dataConvert import Convert

def test_complete_attribute_transfer():
    """完整的属性传递测试"""
    print("=== 完整属性传递测试 ===")
    
    # 1. 测试网格几何体
    print("\n[1] 测试网格几何体")
    geo_mesh = hou.Geometry()
    # 创建一个简单的四边形
    for i in range(4):
        pt = geo_mesh.createPoint()
        pt.setPosition((i, 0, 0))
    poly = geo_mesh.createPolygon()
    for pt in geo_mesh.points():
        poly.addVertex(pt)
    
    cvt_mesh = Convert(hou_geo=geo_mesh)
    
    # 添加各种属性
    point_attr = torch.rand(4, 3)
    cvt_mesh.addAttrib('point_attr', point_attr)
    
    prim_attr = torch.rand(1, 2)
    cvt_mesh.addPrimAttrib('prim_attr', prim_attr)
    
    cvt_mesh.addDetailAttrib('detail_attr', 42.0)
    
    # 转换为PyTorch3D
    mesh = cvt_mesh.toMeshes()
    print(f"✓ 网格转换成功: {type(mesh).__name__}")
    print(f"  顶点数: {mesh.verts_packed().shape[0]}")
    print(f"  面数: {mesh.faces_packed().shape[0]}")
    
    # 2. 测试点云几何体
    print("\n[2] 测试点云几何体")
    geo_points = hou.Geometry()
    for i in range(10):
        pt = geo_points.createPoint()
        pt.setPosition((i, i*0.5, 0))
    
    cvt_points = Convert(hou_geo=geo_points)
    
    # 添加点属性
    point_attr_cloud = torch.rand(10, 4)
    cvt_points.addAttrib('point_attr_cloud', point_attr_cloud)
    
    cvt_points.addDetailAttrib('detail_attr_cloud', 123.0)
    
    # 转换为PyTorch3D
    pointcloud = cvt_points.toMeshes()
    print(f"✓ 点云转换成功: {type(pointcloud).__name__}")
    print(f"  点数: {pointcloud.points_packed().shape[0]}")
    
    # 3. 测试属性获取
    print("\n[3] 测试属性获取")
    
    # 网格属性
    mesh_point_attr = cvt_mesh.getAttrib('point_attr')
    mesh_prim_attr = cvt_mesh.getPrimAttrib('prim_attr')
    mesh_detail_attr = cvt_mesh.getDetailAttrib('detail_attr')
    
    print(f"网格点属性: {mesh_point_attr.shape if mesh_point_attr is not None else 'None'}")
    print(f"网格面属性: {mesh_prim_attr.shape if mesh_prim_attr is not None else 'None'}")
    print(f"网格全局属性: {mesh_detail_attr}")
    
    # 点云属性
    cloud_point_attr = cvt_points.getAttrib('point_attr_cloud')
    cloud_detail_attr = cvt_points.getDetailAttrib('detail_attr_cloud')
    
    print(f"点云点属性: {cloud_point_attr.shape if cloud_point_attr is not None else 'None'}")
    print(f"点云全局属性: {cloud_detail_attr}")
    
    # 4. 测试状态信息
    print("\n[4] 测试状态信息")
    mesh_info = cvt_mesh.get_state_info()
    points_info = cvt_points.get_state_info()
    
    print("网格状态:", mesh_info)
    print("点云状态:", points_info)
    
    # 5. 测试属性验证
    print("\n[5] 测试属性验证")
    
    # 测试正确的属性
    try:
        correct_attr = torch.rand(4, 3)
        cvt_mesh.addAttrib('correct_attr', correct_attr)
        print("✓ 正确属性添加成功")
    except Exception as e:
        print(f"✗ 正确属性添加失败: {e}")
    
    # 测试错误的属性类型
    try:
        cvt_mesh.addAttrib('wrong_type', [1, 2, 3])
        print("✗ 应该报错但没有")
    except TypeError:
        print("✓ 错误类型检测正确")
    
    # 测试错误的属性维度
    try:
        cvt_mesh.addAttrib('wrong_dim', torch.tensor(42))
        print("✗ 应该报错但没有")
    except ValueError:
        print("✓ 错误维度检测正确")
    
    # 测试错误的属性大小
    try:
        cvt_mesh.addAttrib('wrong_size', torch.rand(10, 3))
        print("✗ 应该报错但没有")
    except ValueError:
        print("✓ 错误大小检测正确")
    
    # 6. 测试双向转换
    print("\n[6] 测试双向转换")
    
    # 网格: Houdini -> PyTorch3D -> Houdini
    mesh_back = cvt_mesh.toHoudini()
    print(f"✓ 网格双向转换成功: {len(mesh_back.points())} 点, {len(mesh_back.prims())} 面")
    
    # 点云: Houdini -> PyTorch3D -> Houdini
    points_back = cvt_points.toHoudini()
    print(f"✓ 点云双向转换成功: {len(points_back.points())} 点")
    
    # 7. 测试属性覆盖
    print("\n[7] 测试属性覆盖")
    
    # 添加同名属性，不覆盖
    try:
        cvt_mesh.addAttrib('point_attr', torch.rand(4, 3), overlay=False)
        print("✗ 应该报错但没有")
    except:
        print("✓ 属性覆盖检测正确")
    
    # 添加同名属性，覆盖
    try:
        cvt_mesh.addAttrib('point_attr', torch.rand(4, 3), overlay=True)
        print("✓ 属性覆盖成功")
    except Exception as e:
        print(f"✗ 属性覆盖失败: {e}")
    
    # 8. 测试缓存清理
    print("\n[8] 测试缓存清理")
    
    print(f"清理前属性数量: {len(cvt_mesh._attribs)}")
    cvt_mesh.clearAttribs()
    print(f"清理后属性数量: {len(cvt_mesh._attribs)}")
    
    if len(cvt_mesh._attribs) == 0:
        print("✓ 缓存清理正确")
    else:
        print("✗ 缓存清理失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_complete_attribute_transfer() 