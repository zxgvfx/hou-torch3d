#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtendedMeshes 的 pytest 测试
"""

import pytest
import torch
import numpy as np
from pyLib.toolLib.extended_meshes import ExtendedMeshes
from pytorch3d.structures import Meshes

class TestExtendedMeshes:
    """ExtendedMeshes 测试类"""
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_creation(self, sample_verts, sample_faces, sample_attributes):
        """测试ExtendedMeshes创建"""
        mesh = ExtendedMeshes(
            verts=[sample_verts], 
            faces=[sample_faces], 
            attributes=sample_attributes
        )
        
        assert isinstance(mesh, ExtendedMeshes)
        assert mesh.verts_packed().shape[0] == 4
        assert mesh.faces_packed().shape[0] == 2
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_attribute_access(self, sample_extended_mesh):
        """测试属性访问"""
        # 测试获取属性
        color = sample_extended_mesh.get_attribute('color')
        weight = sample_extended_mesh.get_attribute('weight')
        feature = sample_extended_mesh.get_attribute('feature')
        
        assert color.shape == (4, 3)
        assert weight.shape == (4, 1)
        assert feature.shape == (4, 4)
        
        # 测试属性检查
        assert sample_extended_mesh.has_attribute('color') == True
        assert sample_extended_mesh.has_attribute('nonexistent') == False
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_add_attribute(self, sample_extended_mesh):
        """测试添加属性"""
        new_attr = torch.rand(4, 2)
        sample_extended_mesh.add_attribute('new_attr', new_attr)
        
        assert sample_extended_mesh.has_attribute('new_attr') == True
        assert sample_extended_mesh.get_attribute('new_attr').shape == (4, 2)
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_remove_attribute(self, sample_extended_mesh):
        """测试移除属性"""
        sample_extended_mesh.remove_attribute('color')
        assert sample_extended_mesh.has_attribute('color') == False
        assert sample_extended_mesh.get_attribute('color') is None
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_get_attributes(self, sample_extended_mesh):
        """测试获取所有属性"""
        all_attrs = sample_extended_mesh.get_attributes()
        assert len(all_attrs) == 3
        assert 'color' in all_attrs
        assert 'weight' in all_attrs
        assert 'feature' in all_attrs
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_to_meshes(self, sample_extended_mesh):
        """测试转换为标准Meshes"""
        std_mesh = sample_extended_mesh.to_meshes()
        assert isinstance(std_mesh, Meshes)
        assert std_mesh.verts_packed().shape == sample_extended_mesh.verts_packed().shape
        assert std_mesh.faces_packed().shape == sample_extended_mesh.faces_packed().shape
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_from_meshes(self, sample_mesh, sample_attributes):
        """测试从标准Meshes创建"""
        extended_mesh = ExtendedMeshes.from_meshes(sample_mesh, sample_attributes)
        assert isinstance(extended_mesh, ExtendedMeshes)
        assert extended_mesh.verts_packed().shape == sample_mesh.verts_packed().shape
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_device_handling(self, sample_extended_mesh, device):
        """测试设备处理"""
        if device.type == "cuda":
            mesh_on_device = sample_extended_mesh.to(device)
            assert mesh_on_device.device == device
            
            # 检查属性是否也在正确的设备上
            for attr_name, attr_value in mesh_on_device.get_attributes().items():
                assert attr_value.device == device
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_invalid_attribute_dimension(self, sample_verts, sample_faces):
        """测试无效属性维度"""
        mesh = ExtendedMeshes(verts=[sample_verts], faces=[sample_faces])
        
        # 测试顶点数不匹配
        wrong_size_attr = torch.rand(3, 3)  # 应该是4个顶点
        with pytest.raises(ValueError):
            mesh.add_attribute('wrong_size', wrong_size_attr)
    
    @pytest.mark.unit
    @pytest.mark.extended_meshes
    def test_clear_attributes(self, sample_extended_mesh):
        """测试清除所有属性"""
        sample_extended_mesh.clear_attributes()
        assert len(sample_extended_mesh.get_attributes()) == 0 