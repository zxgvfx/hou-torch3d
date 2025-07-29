#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用额外属性进行训练的示例
"""

import torch
import hou
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pyLib.toolLib import dataConvert as dc
from pyLib.toolLib.extended_meshes import ExtendedMeshes
from pyLib.lossLib.attribute_loss import CombinedAttributeLoss

class AttributeTrainingExample:
    """
    使用额外属性进行训练的示例类
    """
    
    def __init__(self, 
                 source_mesh: Meshes,
                 target_mesh: Meshes,
                 source_attributes: dict,
                 target_attributes: dict,
                 device: torch.device = None):
        """
        初始化训练示例
        :param source_mesh: 源网格
        :param target_mesh: 目标网格
        :param source_attributes: 源属性
        :param target_attributes: 目标属性
        :param device: 设备
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 转换为ExtendedMeshes并添加属性
        self.source_mesh = ExtendedMeshes.from_meshes(source_mesh, source_attributes).to(self.device)
        self.target_mesh = ExtendedMeshes.from_meshes(target_mesh, target_attributes).to(self.device)
        
        # 可训练的顶点偏移
        self.vertex_offset = torch.zeros_like(self.source_mesh.verts_packed(), 
                                            requires_grad=True, 
                                            device=self.device)
        
        # 属性损失函数
        self.attribute_loss = CombinedAttributeLoss(
            consistency_weight=1.0,
            smoothness_weight=0.1,
            gradient_weight=0.01
        )
    
    def forward(self) -> ExtendedMeshes:
        """
        前向传播
        :return: 变形后的网格
        """
        # 应用顶点偏移
        deformed_verts = self.source_mesh.verts_packed() + self.vertex_offset
        
        # 创建新的ExtendedMeshes，保持属性
        deformed_mesh = ExtendedMeshes(
            verts=[deformed_verts],
            faces=self.source_mesh.faces_list(),
            attributes=self.source_mesh.get_attributes()
        ).to(self.device)
        
        return deformed_mesh
    
    def compute_loss(self, deformed_mesh: ExtendedMeshes) -> torch.Tensor:
        """
        计算总损失
        :param deformed_mesh: 变形后的网格
        :return: 总损失
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 1. 几何损失（Chamfer距离）
        src_points = deformed_mesh.verts_packed().unsqueeze(0)
        tgt_points = self.target_mesh.verts_packed().unsqueeze(0)
        chamfer_loss, _ = chamfer_distance(src_points, tgt_points)
        total_loss = total_loss + chamfer_loss
        
        # 2. 边缘损失
        edge_loss = mesh_edge_loss(deformed_mesh)
        total_loss = total_loss + 0.1 * edge_loss
        
        # 3. 属性一致性损失
        target_attrs = self.target_mesh.get_attributes()
        if target_attrs:
            attribute_loss = self.attribute_loss(deformed_mesh, target_attrs)
            total_loss = total_loss + attribute_loss
        
        return total_loss
    
    def train(self, num_epochs: int = 100, lr: float = 0.01) -> ExtendedMeshes:
        """
        训练
        :param num_epochs: 训练轮数
        :param lr: 学习率
        :return: 最终网格
        """
        optimizer = torch.optim.Adam([self.vertex_offset], lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # 前向传播
            deformed_mesh = self.forward()
            
            # 计算损失
            loss = self.compute_loss(deformed_mesh)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        return self.forward()

def example_usage():
    """
    使用示例
    """
    # 1. 从Houdini几何体创建网格
    node = hou.pwd()
    source_geo = node.geometry()
    target_geo = node.inputs()[1].geometry()
    
    # 2. 转换为PyTorch3D网格
    source_convert = dc.Convert(hou_geo=source_geo)
    target_convert = dc.Convert(hou_geo=target_geo)
    
    source_mesh = source_convert.toMeshes()
    target_mesh = target_convert.toMeshes()
    
    # 3. 创建示例属性（这里用随机数据）
    num_verts = source_mesh.verts_packed().shape[0]
    source_attributes = {
        'color': torch.rand(num_verts, 3),
        'weight': torch.rand(num_verts, 1),
        'feature': torch.rand(num_verts, 4)
    }
    
    target_attributes = {
        'color': torch.rand(num_verts, 3),
        'weight': torch.rand(num_verts, 1),
        'feature': torch.rand(num_verts, 4)
    }
    
    # 4. 创建训练器
    trainer = AttributeTrainingExample(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_attributes=source_attributes,
        target_attributes=target_attributes
    )
    
    # 5. 训练
    final_mesh = trainer.train(num_epochs=50, lr=0.01)
    
    # 6. 转换回Houdini
    final_convert = dc.Convert(t3d_geo=final_mesh)
    final_geo = final_convert.toHoudini()
    
    # 7. 更新几何体
    source_geo.clear()
    source_geo.copy(final_geo)
    
    print("训练完成！")

if __name__ == "__main__":
    example_usage() 