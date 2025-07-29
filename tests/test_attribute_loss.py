#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
属性损失函数的 pytest 测试
"""

import pytest
import torch
import torch.nn.functional as F
from pyLib.lossLib.attribute_loss import AttributeLoss, CombinedAttributeLoss
from pyLib.toolLib.extended_meshes import ExtendedMeshes

class TestAttributeLoss:
    """属性损失函数测试类"""
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_consistency_loss(self, sample_extended_mesh):
        """测试一致性损失"""
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1)
        }
        
        loss = AttributeLoss.attribute_consistency_loss(
            sample_extended_mesh, target_attrs
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
        assert loss.item() >= 0
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_consistency_loss_with_weights(self, sample_extended_mesh):
        """测试带权重的一致性损失"""
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1)
        }
        weights = {'color': 2.0, 'weight': 1.0}
        
        loss = AttributeLoss.attribute_consistency_loss(
            sample_extended_mesh, target_attrs, weights
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_smoothness_loss(self, sample_extended_mesh):
        """测试平滑性损失"""
        loss = AttributeLoss.attribute_smoothness_loss(
            sample_extended_mesh, 'color', weight=1.0
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
        assert loss.item() >= 0
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_smoothness_loss_nonexistent_attribute(self, sample_extended_mesh):
        """测试不存在的属性的平滑性损失"""
        loss = AttributeLoss.attribute_smoothness_loss(
            sample_extended_mesh, 'nonexistent', weight=1.0
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_gradient_loss(self, sample_extended_mesh):
        """测试梯度损失"""
        loss = AttributeLoss.attribute_gradient_loss(
            sample_extended_mesh, 'color', weight=1.0
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
        assert loss.item() >= 0
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_gradient_loss_with_target(self, sample_extended_mesh):
        """测试带目标梯度的梯度损失"""
        target_gradients = torch.rand(6, 3)  # 假设有6条边
        
        loss = AttributeLoss.attribute_gradient_loss(
            sample_extended_mesh, 'color', target_gradients, weight=1.0
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_standard_meshes_consistency_loss(self, sample_mesh):
        """测试标准Meshes的一致性损失（应该返回0）"""
        target_attrs = {'color': torch.rand(4, 3)}
        
        loss = AttributeLoss.attribute_consistency_loss(sample_mesh, target_attrs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

class TestCombinedAttributeLoss:
    """组合属性损失测试类"""
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_creation(self):
        """测试创建组合损失函数"""
        loss_fn = CombinedAttributeLoss(
            consistency_weight=1.0,
            smoothness_weight=0.1,
            gradient_weight=0.01
        )
        
        assert loss_fn.consistency_weight == 1.0
        assert loss_fn.smoothness_weight == 0.1
        assert loss_fn.gradient_weight == 0.01
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_call(self, sample_extended_mesh):
        """测试调用组合损失函数"""
        loss_fn = CombinedAttributeLoss()
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1)
        }
        
        loss = loss_fn(sample_extended_mesh, target_attrs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
        assert loss.item() >= 0
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_call_with_weights(self, sample_extended_mesh):
        """测试带权重的调用"""
        loss_fn = CombinedAttributeLoss()
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1)
        }
        attr_weights = {'color': 2.0, 'weight': 1.0}
        
        loss = loss_fn(sample_extended_mesh, target_attrs, attr_weights)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad == True
    
    @pytest.mark.unit
    @pytest.mark.attribute_loss
    def test_training_simulation(self, sample_extended_mesh):
        """测试训练模拟"""
        loss_fn = CombinedAttributeLoss(
            consistency_weight=1.0,
            smoothness_weight=0.1
        )
        
        target_attrs = {
            'color': torch.rand(4, 3),
            'weight': torch.rand(4, 1)
        }
        
        # 创建可训练的顶点偏移
        vertex_offset = torch.zeros_like(sample_extended_mesh.verts_packed(), requires_grad=True)
        optimizer = torch.optim.Adam([vertex_offset], lr=0.01)
        
        # 模拟几个训练步骤
        for _ in range(3):
            optimizer.zero_grad()
            
            # 前向传播
            deformed_verts = sample_extended_mesh.verts_packed() + vertex_offset
            deformed_mesh = ExtendedMeshes(
                verts=[deformed_verts],
                faces=sample_extended_mesh.faces_list(),
                attributes=sample_extended_mesh.get_attributes()
            )
            
            # 计算损失
            loss = loss_fn(deformed_mesh, target_attrs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            assert loss.item() >= 0 