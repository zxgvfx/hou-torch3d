#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持额外属性的Loss函数
"""

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from typing import Dict, Optional, Union
from ..toolLib.extended_meshes import ExtendedMeshes

class AttributeLoss:
    """
    支持额外属性的Loss函数集合
    """
    
    @staticmethod
    def attribute_consistency_loss(mesh: Union[Meshes, ExtendedMeshes], 
                                 target_attributes: Dict[str, torch.Tensor],
                                 weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        属性一致性损失
        :param mesh: 网格对象
        :param target_attributes: 目标属性字典
        :param weights: 各属性权重
        :return: 损失值
        """
        total_loss = torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        if isinstance(mesh, ExtendedMeshes):
            for attr_name, target_attr in target_attributes.items():
                if mesh.has_attribute(attr_name):
                    pred_attr = mesh.get_attribute(attr_name)
                    weight = weights.get(attr_name, 1.0) if weights else 1.0
                    
                    # L2损失
                    loss = F.mse_loss(pred_attr, target_attr)
                    total_loss = total_loss + weight * loss
        else:
            # 对于标准Meshes，属性存储在Convert类的_attribs中
            # 这里需要从外部传入当前属性
            pass
        
        return total_loss
    
    @staticmethod
    def attribute_smoothness_loss(mesh: Union[Meshes, ExtendedMeshes],
                                attribute_name: str,
                                weight: float = 1.0) -> torch.Tensor:
        """
        属性平滑性损失
        :param mesh: 网格对象
        :param attribute_name: 属性名
        :param weight: 权重
        :return: 损失值
        """
        if not isinstance(mesh, ExtendedMeshes):
            return torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        if not mesh.has_attribute(attribute_name):
            return torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        # 获取属性
        attr = mesh.get_attribute(attribute_name)
        
        # 获取边的顶点索引
        edges = mesh.edges_packed()
        
        # 计算相邻顶点间的属性差异
        edge_attrs = attr[edges]
        attr_diff = edge_attrs[:, 0] - edge_attrs[:, 1]
        
        # 平滑性损失：相邻顶点属性差异的平方和
        loss = torch.mean(attr_diff ** 2)
        
        return weight * loss
    
    @staticmethod
    def attribute_gradient_loss(mesh: Union[Meshes, ExtendedMeshes],
                              attribute_name: str,
                              target_gradients: Optional[torch.Tensor] = None,
                              weight: float = 1.0) -> torch.Tensor:
        """
        属性梯度损失
        :param mesh: 网格对象
        :param attribute_name: 属性名
        :param target_gradients: 目标梯度（可选）
        :param weight: 权重
        :return: 损失值
        """
        if not isinstance(mesh, ExtendedMeshes):
            return torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        if not mesh.has_attribute(attribute_name):
            return torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        # 获取属性
        attr = mesh.get_attribute(attribute_name)
        
        # 计算属性梯度（简化版本）
        edges = mesh.edges_packed()
        edge_attrs = attr[edges]
        gradients = edge_attrs[:, 0] - edge_attrs[:, 1]
        
        if target_gradients is not None:
            # 与目标梯度比较
            loss = F.mse_loss(gradients, target_gradients)
        else:
            # 梯度正则化（鼓励平滑）
            loss = torch.mean(gradients ** 2)
        
        return weight * loss

class CombinedAttributeLoss:
    """
    组合属性损失
    """
    
    def __init__(self, 
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 0.1,
                 gradient_weight: float = 0.01):
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
    
    def __call__(self, 
                 mesh: Union[Meshes, ExtendedMeshes],
                 target_attributes: Dict[str, torch.Tensor],
                 attribute_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        计算组合属性损失
        :param mesh: 网格对象
        :param target_attributes: 目标属性
        :param attribute_weights: 属性权重
        :return: 总损失
        """
        total_loss = torch.tensor(0.0, device=mesh.device, requires_grad=True)
        
        # 一致性损失
        consistency_loss = AttributeLoss.attribute_consistency_loss(
            mesh, target_attributes, attribute_weights
        )
        total_loss = total_loss + self.consistency_weight * consistency_loss
        
        # 平滑性损失
        if isinstance(mesh, ExtendedMeshes):
            for attr_name in target_attributes.keys():
                if mesh.has_attribute(attr_name):
                    smoothness_loss = AttributeLoss.attribute_smoothness_loss(
                        mesh, attr_name, self.smoothness_weight
                    )
                    total_loss = total_loss + smoothness_loss
        
        return total_loss 