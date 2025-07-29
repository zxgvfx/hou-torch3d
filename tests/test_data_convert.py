#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataConvert 的 pytest 测试
"""

import pytest
import torch
from pyLib.toolLib import dataConvert as dc
from pyLib.toolLib.extended_meshes import ExtendedMeshes
from pytorch3d.structures import Meshes, Pointclouds

class TestDataConvert:
    """dataConvert 测试类"""
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_convert_creation_with_mesh(self, sample_mesh):
        """测试使用PyTorch3D网格创建转换器"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        assert converter.has_mesh == True
        assert converter.has_geo == False
        assert converter.t3d_geo is not None
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_convert_creation_with_geometry(self, mock_houdini_geometry):
        """测试使用Houdini几何体创建转换器"""
        converter = dc.Convert(hou_geo=mock_houdini_geometry)
        
        assert converter.has_mesh == False
        assert converter.has_geo == True
        assert converter.geo is not None
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_convert_creation_invalid(self):
        """测试无效的转换器创建"""
        with pytest.raises(ValueError):
            dc.Convert()  # 没有提供任何几何体
        
        with pytest.raises(ValueError):
            # 同时提供两种几何体
            converter = dc.Convert(t3d_geo=Meshes(verts=[torch.rand(3, 3)], faces=[torch.rand(2, 3)]))
            converter.geo = "mock_geo"  # 模拟同时拥有两种几何体
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_add_attrib(self, sample_mesh):
        """测试添加点属性"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        attr_val = torch.rand(4, 3)
        success = converter.addAttrib('test_attr', attr_val)
        
        assert success == True
        assert 'test_attr' in converter._attribs
        assert converter.getAttrib('test_attr').shape == (4, 3)
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_add_prim_attrib(self, sample_mesh):
        """测试添加面属性"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        attr_val = torch.rand(2, 1)  # 2个面
        success = converter.addPrimAttrib('test_prim_attr', attr_val)
        
        assert success == True
        assert 'test_prim_attr' in converter._prim_attribs
        assert converter.getPrimAttrib('test_prim_attr').shape == (2, 1)
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_add_detail_attrib(self, sample_mesh):
        """测试添加全局属性"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        success = converter.addDetailAttrib('test_detail_attr', 42.0)
        
        assert success == True
        assert 'test_detail_attr' in converter._detail_attribs
        assert converter.getDetailAttrib('test_detail_attr') == 42.0
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_add_attrib_validation(self, sample_mesh):
        """测试添加属性的验证"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        # 测试非tensor输入
        with pytest.raises(TypeError):
            converter.addAttrib('test', [1, 2, 3])
        
        # 测试维度不匹配
        wrong_size_attr = torch.rand(3, 3)  # 应该是4个顶点
        with pytest.raises(ValueError):
            converter.addAttrib('test', wrong_size_attr)
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_to_meshes_with_attributes(self, sample_mesh):
        """测试带属性转换为Meshes"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        # 添加属性
        attr_val = torch.rand(4, 3)
        converter.addAttrib('test_attr', attr_val)
        
        # 转换为ExtendedMeshes
        extended_mesh = converter.toMeshes()
        
        assert isinstance(extended_mesh, ExtendedMeshes)
        assert extended_mesh.has_attribute('test_attr')
        assert extended_mesh.get_attribute('test_attr').shape == (4, 3)
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_get_state_info(self, sample_mesh):
        """测试获取状态信息"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        state_info = converter.get_state_info()
        
        assert isinstance(state_info, dict)
        assert 'has_mesh' in state_info
        assert 'has_geo' in state_info
        assert 'device' in state_info
        assert 'attribs_count' in state_info
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_clear_attribs(self, sample_mesh):
        """测试清除属性"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        # 添加一些属性
        converter.addAttrib('test1', torch.rand(4, 3))
        converter.addPrimAttrib('test2', torch.rand(2, 1))
        converter.addDetailAttrib('test3', 42.0)
        
        # 清除属性
        converter.clearAttribs()
        
        assert len(converter._attribs) == 0
        assert len(converter._prim_attribs) == 0
        assert len(converter._detail_attribs) == 0
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_validate_state(self, sample_mesh):
        """测试状态验证"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        # 正常状态应该不抛出异常
        converter.validate_state()
        
        # 异常状态应该抛出异常
        converter.geo = "mock_geo"  # 模拟同时拥有两种几何体
        with pytest.raises(RuntimeError):
            converter.validate_state()
    
    @pytest.mark.integration
    @pytest.mark.data_convert
    def test_mesh_to_houdini_conversion(self, sample_mesh):
        """测试网格到Houdini的转换（集成测试）"""
        converter = dc.Convert(t3d_geo=sample_mesh)
        
        # 添加属性
        converter.addAttrib('color', torch.rand(4, 3))
        converter.addPrimAttrib('prim_id', torch.rand(2, 1))
        converter.addDetailAttrib('global_val', 42.0)
        
        # 转换为Houdini几何体
        houdini_geo = converter.toHoudini()
        
        # 验证转换结果
        assert houdini_geo is not None
        # 注意：这里无法直接验证Houdini几何体的内容，因为需要Houdini环境
    
    @pytest.mark.integration
    @pytest.mark.data_convert
    def test_houdini_to_mesh_conversion(self, mock_houdini_geometry):
        """测试Houdini到网格的转换（集成测试）"""
        converter = dc.Convert(hou_geo=mock_houdini_geometry)
        
        # 转换为PyTorch3D网格
        mesh = converter.toMeshes()
        
        # 验证转换结果
        assert mesh is not None
        assert hasattr(mesh, 'verts_packed')
        assert hasattr(mesh, 'faces_packed')
    
    @pytest.mark.unit
    @pytest.mark.data_convert
    def test_pointcloud_conversion(self):
        """测试点云转换"""
        # 创建点云
        points = torch.rand(10, 3)
        pointcloud = Pointclouds(points=[points])
        
        converter = dc.Convert(t3d_geo=pointcloud)
        
        # 添加属性
        features = torch.rand(10, 4)
        converter.addAttrib('features', features)
        
        # 转换为ExtendedMeshes（应该是Pointclouds）
        result = converter.toMeshes()
        
        assert isinstance(result, Pointclouds)
        # Pointclouds的属性通过features存储 