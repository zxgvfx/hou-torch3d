#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的dataConvert功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import hou
import torch
import numpy as np
from pyLib.toolLib.dataConvert import Convert

def test_improvements():
    """测试改进后的功能"""
    print("=== 测试改进后的功能 ===")
    
    # 1. 测试状态验证
    print("\n[1] 测试状态验证")
    try:
        # 创建空的转换器
        cvt = Convert(hou_geo=hou.Geometry())
        cvt.validate_state()
        print("✓ 状态验证通过")
    except Exception as e:
        print(f"✗ 状态验证失败: {e}")
    
    # 2. 测试状态信息
    print("\n[2] 测试状态信息")
    geo = hou.Geometry()
    for i in range(4):
        pt = geo.createPoint()
        pt.setPosition((i, 0, 0))
    poly = geo.createPolygon()
    for pt in geo.points():
        poly.addVertex(pt)
    
    cvt = Convert(hou_geo=geo)
    info = cvt.get_state_info()
    print("状态信息:", info)
    
    # 3. 测试属性验证
    print("\n[3] 测试属性验证")
    try:
        # 正确的属性
        attr_val = torch.rand(4, 3)
        cvt.addAttrib('test_attr', attr_val)
        print("✓ 正确属性添加成功")
        
        # 错误的属性类型
        try:
            cvt.addAttrib('wrong_type', [1, 2, 3])
            print("✗ 应该报错但没有")
        except TypeError:
            print("✓ 错误类型检测正确")
        
        # 错误的属性维度
        try:
            cvt.addAttrib('wrong_dim', torch.tensor(42))
            print("✗ 应该报错但没有")
        except ValueError:
            print("✓ 错误维度检测正确")
        
        # 错误的属性大小
        try:
            cvt.addAttrib('wrong_size', torch.rand(10, 3))
            print("✗ 应该报错但没有")
        except ValueError:
            print("✓ 错误大小检测正确")
            
    except Exception as e:
        print(f"✗ 属性验证测试失败: {e}")
    
    # 4. 测试转换一致性
    print("\n[4] 测试转换一致性")
    try:
        # 使用toMeshes转换
        mesh1 = cvt.toMeshes()
        print(f"✓ toMeshes转换成功: {type(mesh1).__name__}")
        
        # 使用updateFromGeo转换
        cvt2 = Convert(hou_geo=geo)
        cvt2.updateFromGeo(geo)
        mesh2 = cvt2.t3d_geo
        print(f"✓ updateFromGeo转换成功: {type(mesh2).__name__}")
        
        # 比较两种方法的结果
        if isinstance(mesh1, type(mesh2)):
            print("✓ 转换方法一致性验证通过")
        else:
            print("✗ 转换方法一致性验证失败")
            
    except Exception as e:
        print(f"✗ 转换一致性测试失败: {e}")
    
    # 5. 测试错误处理
    print("\n[5] 测试错误处理")
    try:
        # 空几何体
        try:
            cvt.updateFromGeo(None)
            print("✗ 应该报错但没有")
        except ValueError:
            print("✓ 空几何体检测正确")
        
        # 空点几何体
        empty_geo = hou.Geometry()
        try:
            cvt.updateFromGeo(empty_geo)
            print("✗ 应该报错但没有")
        except ValueError:
            print("✓ 空点几何体检测正确")
            
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
    
    # 6. 测试缓存清理
    print("\n[6] 测试缓存清理")
    try:
        cvt.addAttrib('cache_test', torch.rand(4, 3))
        cvt.addPrimAttrib('prim_test', torch.rand(1, 2))
        cvt.addDetailAttrib('detail_test', 42.0)
        
        print(f"添加属性前: {len(cvt._attribs)} 个点属性, {len(cvt._prim_attribs)} 个面属性")
        
        cvt._init_data()
        
        print(f"清理后: {len(cvt._attribs)} 个点属性, {len(cvt._prim_attribs)} 个面属性")
        
        if len(cvt._attribs) == 0 and len(cvt._prim_attribs) == 0:
            print("✓ 缓存清理正确")
        else:
            print("✗ 缓存清理失败")
            
    except Exception as e:
        print(f"✗ 缓存清理测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_improvements() 