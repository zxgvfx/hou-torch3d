#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/15 16:06
# @Author  : zxgvfx
# @File    : sop_verb.py
# @Software: PyCharm
"""
存放常用的功能节点
"""
import hou
class SopNode:
    """
    创建一小段节点流，作用与第一个输入的 Geometry。
    """
    def __init__(self):
        self.geo = hou.Geometry()
    def attrib_interpolate(self,input1: hou.Geometry,
                           input2: hou.Geometry,
                           param: dict = dict()):
        """
        sop node Attrib Interpolate
        根据输入的 sourceprim，sourceprimuv 属性，进行插值。
        :param input1: 第一个输入端的物体
        :param input2: 第二个输入端的物体
        :param param: 节点的参数字典
        :return:
        """
        node_verb = hou.sopNodeTypeCategory().nodeVerb("attribinterpolate")
        node_verb.setParms(param)
        node_verb.execute(self.geo, [input1,input2])
    def load_cache(self,file_path: str,param: dict = dict()):
        """
        加载缓存
        :param file_path: 缓存文件路径
        :param param: 节点的参数字典
        :return:
        """
        node_verb = hou.sopNodeTypeCategory().nodeVerb("file")
        per_set_param = {"file": file_path}
        #updata param
        for key,value in param.items():
            per_set_param[key] = value
        node_verb.setParms(per_set_param)
        node_verb.execute(self.geo, [])

