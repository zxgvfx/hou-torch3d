#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展的Meshes类，支持额外属性
"""

import torch
from pytorch3d.structures import Meshes
from typing import Dict, Optional, Union, List

class ExtendedMeshes(Meshes):
    """
    扩展的Meshes类，支持存储和使用额外属性
    """
    
    def __init__(self, 
                 verts: Union[List[torch.Tensor], torch.Tensor],
                 faces: Union[List[torch.Tensor], torch.Tensor],
                 attributes: Optional[Dict[str, torch.Tensor]] = None,
                 **kwargs):
        """
        初始化扩展的Meshes
        :param verts: 顶点坐标
        :param faces: 面索引
        :param attributes: 额外属性字典
        """
        super().__init__(verts, faces, **kwargs)
        
        # 存储额外属性
        self._attributes: Dict[str, torch.Tensor] = {}
        if attributes:
            for name, attr in attributes.items():
                self.add_attribute(name, attr)
    
    def add_attribute(self, name: str, attribute: torch.Tensor) -> None:
        """
        添加点属性
        :param name: 属性名
        :param attribute: 属性值 (N, D) 其中N是顶点数
        """
        if attribute.shape[0] != self.verts_packed().shape[0]:
            raise ValueError(f"属性顶点数({attribute.shape[0]})与网格顶点数({self.verts_packed().shape[0]})不匹配")
        
        self._attributes[name] = attribute.to(self.device)
    
    def get_attribute(self, name: str) -> Optional[torch.Tensor]:
        """
        获取点属性
        :param name: 属性名
        :return: 属性值
        """
        return self._attributes.get(name, None)
    
    def has_attribute(self, name: str) -> bool:
        """
        检查是否有指定属性
        :param name: 属性名
        :return: 是否存在
        """
        return name in self._attributes
    
    def get_attributes(self) -> Dict[str, torch.Tensor]:
        """
        获取所有属性
        :return: 属性字典
        """
        return self._attributes.copy()
    
    def remove_attribute(self, name: str) -> None:
        """
        移除属性
        :param name: 属性名
        """
        if name in self._attributes:
            del self._attributes[name]
    
    def clear_attributes(self) -> None:
        """
        清除所有属性
        """
        self._attributes.clear()
    
    def to_meshes(self) -> Meshes:
        """
        转换为标准Meshes对象
        :return: 标准Meshes对象
        """
        return Meshes(verts=self.verts_list(), faces=self.faces_list())
    
    @classmethod
    def from_meshes(cls, meshes: Meshes, attributes: Optional[Dict[str, torch.Tensor]] = None):
        """
        从标准Meshes创建ExtendedMeshes
        :param meshes: 标准Meshes对象
        :param attributes: 额外属性
        :return: ExtendedMeshes对象
        """
        return cls(verts=meshes.verts_list(), faces=meshes.faces_list(), attributes=attributes) 