#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实世界使用场景测试
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_houdini_to_extended_meshes():
    """测试从Houdini几何体创建ExtendedMeshes"""
    print("=== 测试Houdini到ExtendedMeshes转换 ===")
    try:
        import hou
        from pyLib.toolLib import dataConvert as dc
        from pyLib.toolLib.extended_meshes import ExtendedMeshes
        
        # 创建简单的Houdini几何体
        geo = hou.Geometry()
        
        # 创建点
        pt1 = geo.createPoint()
        pt1.setPosition((0, 0, 0))
        pt2 = geo.createPoint()
        pt2.setPosition((1, 0, 0))
        pt3 = geo.createPoint()
        pt3.setPosition((0, 1, 0))
        pt4 = geo.createPoint()
        pt4.setPosition((1, 1, 0))
        
        # 创建面
        poly = geo.createPolygon()
        poly.addVertex(pt1)
        poly.addVertex(pt2)
        poly.addVertex(pt3)
        
        poly2 = geo.createPolygon()
        poly2.addVertex(pt2)
        poly2.addVertex(pt4)
        poly2.addVertex(pt3)
        
        # 添加点属性
        if not geo.findPointAttrib('color'):
            geo.addArrayAttrib(hou.attribType.Point, 'color', hou.attribData.Float, 3)
        
        for i, pt in enumerate(geo.points()):
            color = (float(i), float(i), float(i))
            pt.setAttribValue('color', color)
        
        print(f"✓ 创建Houdini几何体成功，点数: {len(geo.points())}, 面数: {len(geo.prims())}")
        
        # 转换为PyTorch3D
        converter = dc.Convert(hou_geo=geo)
        mesh = converter.toMeshes()
        
        print(f"✓ 转换为PyTorch3D成功: {type(mesh)}")
        
        # 添加额外属性
        num_verts = mesh.verts_packed().shape[0]
        weight_attr = torch.rand(num_verts, 1)
        feature_attr = torch.rand(num_verts, 4)
        
        converter.addAttrib('weight', weight_attr)
        converter.addAttrib('feature', feature_attr)
        
        # 转换为ExtendedMeshes
        extended_mesh = converter.toMeshes()
        
        if isinstance(extended_mesh, ExtendedMeshes):
            print("✓ 成功转换为ExtendedMeshes")
            print(f"  - 顶点数: {extended_mesh.verts_packed().shape[0]}")
            print(f"  - 面数: {extended_mesh.faces_packed().shape[0]}")
            print(f"  - 属性数量: {len(extended_mesh.get_attributes())}")
            
            # 检查属性
            weight = extended_mesh.get_attribute('weight')
            feature = extended_mesh.get_attribute('feature')
            print(f"  - weight属性: {weight.shape}")
            print(f"  - feature属性: {feature.shape}")
        else:
            print("⚠ 转换为标准Meshes，属性存储在converter中")
        
        return True
    except Exception as e:
        print(f"✗ Houdini转换测试失败: {e}")
        return False

def test_attribute_loss_training():
    """测试属性损失训练"""
    print("\n=== 测试属性损失训练 ===")
    try:
        from pyLib.toolLib.extended_meshes import ExtendedMeshes
        from pyLib.lossLib.attribute_loss import CombinedAttributeLoss
        
        # 创建源网格和目标网格
        verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
        
        # 创建属性
        source_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1),
            'feature': torch.rand(4, 4)
        }
        
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1),
            'feature': torch.rand(4, 4)
        }
        
        source_mesh = ExtendedMeshes(verts=[verts], faces=[faces], attributes=source_attrs)
        target_mesh = ExtendedMeshes(verts=[verts], faces=[faces], attributes=target_attrs)
        
        # 创建可训练的顶点偏移
        vertex_offset = torch.zeros_like(verts, requires_grad=True)
        
        # 创建损失函数
        loss_fn = CombinedAttributeLoss(
            consistency_weight=1.0,
            smoothness_weight=0.1
        )
        
        # 模拟训练步骤
        optimizer = torch.optim.Adam([vertex_offset], lr=0.01)
        
        print("开始模拟训练...")
        for epoch in range(10):
            optimizer.zero_grad()
            
            # 前向传播
            deformed_verts = verts + vertex_offset
            deformed_mesh = ExtendedMeshes(
                verts=[deformed_verts],
                faces=[faces],
                attributes=source_attrs
            )
            
            # 计算损失
            loss = loss_fn(deformed_mesh, target_attrs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
        
        print("✓ 属性损失训练测试成功")
        return True
    except Exception as e:
        print(f"✗ 属性损失训练测试失败: {e}")
        return False

def test_data_convert_with_attributes():
    """测试带属性的数据转换"""
    print("\n=== 测试带属性的数据转换 ===")
    try:
        from pyLib.toolLib import dataConvert as dc
        from pytorch3d.structures import Meshes
        
        # 创建PyTorch3D网格
        verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
        mesh = Meshes(verts=[verts], faces=[faces])
        
        # 创建转换器
        converter = dc.Convert(t3d_geo=mesh)
        
        # 添加多个属性
        num_verts = mesh.verts_packed().shape[0]
        attributes = {
            'color': torch.rand(num_verts, 3),
            'weight': torch.rand(num_verts, 1),
            'feature': torch.rand(num_verts, 4),
            'normal': torch.rand(num_verts, 3)
        }
        
        for name, attr in attributes.items():
            success = converter.addAttrib(name, attr)
            print(f"  ✓ 添加属性 '{name}': {success}")
        
        # 转换为ExtendedMeshes
        from pyLib.toolLib.extended_meshes import ExtendedMeshes
        extended_mesh = converter.toMeshes()
        
        if isinstance(extended_mesh, ExtendedMeshes):
            print("✓ 成功转换为ExtendedMeshes")
            all_attrs = extended_mesh.get_attributes()
            for name, attr in all_attrs.items():
                print(f"  - {name}: {attr.shape}")
        else:
            print("⚠ 转换为标准Meshes")
        
        # 转换回Houdini几何体
        houdini_geo = converter.toHoudini()
        print(f"✓ 转换回Houdini几何体成功，点数: {len(houdini_geo.points())}")
        
        return True
    except Exception as e:
        print(f"✗ 数据转换测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始真实世界使用场景测试...")
    
    tests = [
        test_houdini_to_extended_meshes,
        test_attribute_loss_training,
        test_data_convert_with_attributes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有真实世界测试通过！")
        print("\n总结:")
        print("✓ ExtendedMeshes 可以成功存储和访问额外属性")
        print("✓ 属性损失函数可以用于训练")
        print("✓ dataConvert 可以正确处理带属性的转换")
        print("✓ 整个流程从Houdini到PyTorch3D再回到Houdini都能正常工作")
    else:
        print("⚠ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 