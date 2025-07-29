#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest 配置文件，包含共享的 fixtures
"""

import pytest
import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def device():
    """返回测试设备"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_verts():
    """返回示例顶点数据"""
    return torch.tensor([
        [0, 0, 0],
        [1, 0, 0], 
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=torch.float32)

@pytest.fixture
def sample_faces():
    """返回示例面数据"""
    return torch.tensor([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=torch.int64)

@pytest.fixture
def sample_attributes():
    """返回示例属性数据"""
    return {
        'color': torch.rand(4, 3),
        'weight': torch.rand(4, 1),
        'feature': torch.rand(4, 4)
    }

@pytest.fixture
def sample_mesh(sample_verts, sample_faces):
    """返回示例PyTorch3D网格"""
    from pytorch3d.structures import Meshes
    return Meshes(verts=[sample_verts], faces=[sample_faces])

@pytest.fixture
def sample_extended_mesh(sample_verts, sample_faces, sample_attributes):
    """返回示例ExtendedMeshes"""
    from pyLib.toolLib.extended_meshes import ExtendedMeshes
    return ExtendedMeshes(
        verts=[sample_verts], 
        faces=[sample_faces], 
        attributes=sample_attributes
    )

@pytest.fixture
def sample_houdini_geometry():
    """返回示例Houdini几何体（如果可用）"""
    try:
        import hou
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
        
        return geo
    except ImportError:
        pytest.skip("Houdini环境不可用")

@pytest.fixture
def mock_houdini_geometry():
    """返回模拟的Houdini几何体（用于单元测试）"""
    class MockGeometry:
        def __init__(self):
            self.points_data = [
                {'position': (0, 0, 0), 'number': 0},
                {'position': (1, 0, 0), 'number': 1},
                {'position': (0, 1, 0), 'number': 2},
                {'position': (1, 1, 0), 'number': 3}
            ]
            self.prims_data = [
                {'points': [0, 1, 2]},
                {'points': [1, 3, 2]}
            ]
        
        def points(self):
            return [MockPoint(data) for data in self.points_data]
        
        def prims(self):
            return [MockPrim(data) for data in self.prims_data]
        
        def __len__(self):
            return len(self.points_data)
    
    class MockPoint:
        def __init__(self, data):
            self.data = data
        
        def position(self):
            return self.data['position']
        
        def number(self):
            return self.data['number']
    
    class MockPrim:
        def __init__(self, data):
            self.data = data
        
        def points(self):
            return [MockPoint({'position': (0, 0, 0), 'number': idx}) for idx in self.data['points']]
    
    return MockGeometry() 