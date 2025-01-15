import imp
import hou
from train.blenshape import blend_by_points as bps
from lib.toolLib import dataConvert as dc

from pytorch3d.utils import ico_sphere
import torch

imp.reload(dc)
imp.reload(bps)

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()


node_geo_cvt = dc.Convert(hou_geo = geo)

src_mesh = node_geo_cvt.toMeshes()
trg_mesh = ico_sphere(4,torch.device("cuda:0"))

bs = bps.Blendshape(trg_mesh,src_mesh) # bs

optimizer = bs.optimizer()  # 优化器
train_geo = bps.train(bs,optimizer,680) #训练

geo_cvt = dc.Convert(train_geo.cpu()) #转换器
h_geo =  geo_cvt.toHoudini()  # 转换成geo
geo.copy(h_geo)  # 拉取到 houdini