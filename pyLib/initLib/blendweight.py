#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/21 11:21
# @Author  : zxgvfx
# @File    : blendweight.py
# @Software: PyCharm
"""
目前使用 topo transfer 节点进行初始化3d数据
"""
from net.blendshape.blend import BlendMeshNet
from pyLib.houLib.node.sop import sop_verb
from pyLib.toolLib import dataConvert as dc
import imp
imp.reload(sop_verb)
imp.reload(dc)


def initialize_topotransfer(net:BlendMeshNet):
    """
    使用 topo transfer 初始化3d模型
    :param net:
    :return:
    """
    # 没有geo 就不优化
    if net.temp_points and net.trg_points:

        sop = sop_verb.SopNode()
        sop.topotransfer(net.temp_geos,
                         net.trg_geos,
                         net.temp_points,
                         net.trg_points)
        mesh_cvt = dc.Convert(t3d_geo=net.temp_mesh)
        mesh_cvt.updateFromGeo(sop.geo)
        new_mash = mesh_cvt.t3d_geo
        weight = new_mash.verts_packed()-net.temp_mesh.verts_packed()
        parms = net.parameters()
        for parm in parms:
            parm.data = weight
        net.temp_mesh = new_mash

