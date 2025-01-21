#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/20 15:38
# @Author  : zxgvfx
# @File    : meshes_attr_interpolate.py
# @Software: PyCharm
import hou
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pyLib.houLib.node.sop.sop_verb import SopNode
from pyLib.toolLib import dataConvert


def sample_points_from_geo(temp_point:hou.Geometry,
                           temp_mesh: Meshes,
                           trg_point:hou.Geometry,
                           trg_mesh: Meshes=None, )\
        ->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    导入外部定位点数据，使用定位点将两个物体对应的区域标记，定位点需要包含 sourceprim，sourceprimuv 属性
    通过sourceprim，sourceprimuv 属性，使用Attrib_Interpolate节点将输入的模板点插值。
    :param temp_point: 模板Mesh定位点,可以从houdini sop中的topo landmark节点创建。也可以通过卷积识别产生。
    :param trg_point: 目标Mesh定位点,可以从houdini sop中的topo landmark节点创建。也可以通过卷积识别产生。
    :param temp_mesh: 模板Mesh
    :param trg_mesh: 目标Mesh
    :return: 返回一个含有四个元组的列表，
    第一个是模板点返回的Tensor张量，
    第二个是目标点的张量，
    第三个是模板Geometry面的序号，
    第四个是目标Geometry面的序号
    """
    #先进行变形
    temp_geo = dataConvert.Convert(t3d_geo=temp_mesh).toHoudini()
    sn = SopNode()
    sn.attrib_interpolate(temp_point, temp_geo)
    temp_mesh_out = dataConvert.Convert(hou_geo=sn.geo).toMeshes()
    if trg_mesh:
        trg_geo = dataConvert.Convert(t3d_geo=trg_mesh).toHoudini()
        sn = SopNode()
        sn.attrib_interpolate(trg_point, trg_geo)
        trg_mesh_out = dataConvert.Convert(hou_geo=sn.geo).toMeshes()
    else:
        trg_mesh_out = dataConvert.Convert(hou_geo=trg_point).toMeshes()
    if isinstance(temp_mesh_out,Pointclouds) and isinstance(trg_mesh_out,Pointclouds) :
        temp_p = temp_mesh_out.points_packed()
        trg_p = trg_mesh_out.points_packed()
    else:
        temp_p = temp_mesh_out.verts_packed()
        trg_p = trg_mesh_out.verts_packed()
    temp_srcprim = torch.tensor(list(map(lambda x:x.attribValue('sourceprim'),temp_point.points())),device=temp_p.device)
    trg_srcprim = torch.tensor(list(map(lambda x:x.attribValue('sourceprim'),trg_point.points())),device=trg_p.device)

    return (temp_p.reshape((1,-1,3)).cuda(),
            trg_p.reshape((1,-1,3)).cuda(),
            temp_srcprim,
            trg_srcprim)