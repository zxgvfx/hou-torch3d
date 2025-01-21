import torch
import hou
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from net.topo_landmart.landmarks import load_landmark
import pyLib.toolLib.dataConvert as dc #数据转换
from pyLib.houLib.tools import meshes_attr_interpolate as att_interpolate

class BlendMeshNet(nn.Module):
    def __init__(self,temp_mesh: Meshes):
        super(BlendMeshNet, self).__init__()
        self.temp_mesh = temp_mesh
        self.temp_points = None
        self.trg_points = None
        self.temp_geos = None
        self.trg_geos = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.zeros(temp_mesh.verts_packed().shape,
                                               device=self.device,
                                               dtype=torch.float32,
                                               requires_grad=True))

    def forward(self,x: Meshes):
        trg_mesh = x.to(self.device)
        new_src_mesh_id = torch.tensor([],dtype=torch.float32,device=self.device)
        if not self.temp_points:
            trg_points = self.sample_points(trg_mesh)
            temp_points = self.sample_points(self.temp_mesh)
        else:
            # 由temp 面生成点
            # todo : 需要生成多密度点去添加权重属性，然后更新weigh，目前是考虑拖拽点
            temp_points,trg_points,new_src_mesh_id,_ = att_interpolate.sample_points_from_geo(self.temp_points,
                                                                                              self.temp_mesh,
                                                                                              self.trg_points)

        self.temp_mesh = self.temp_mesh.offset_verts(self.weight)
        return self.temp_mesh,temp_points,trg_points,new_src_mesh_id

    def sample_points(self,meshes: Meshes,num_points: int=5000)->torch.Tensor:
        return sample_points_from_meshes(meshes, num_points)

    def set_landmark(self,
                     temp_geos: hou.Geometry,
                     trg_geos: hou.Geometry,
                     temp_points: hou.Geometry,
                     trg_points: hou.Geometry):
        """
        导入外部定位点数据，使用定位点将两个物体对应的区域标记，定位点需要包含 sourceprim，sourceprimuv 属性
        通过sourceprim，sourceprimuv 属性，使用Attrib_Interpolate节点将输入的模板点插值。
        :param temp_geos:
        :param trg_geos:
        :param temp_points:
        :param trg_points:
        :return:
        """
        self.temp_points = temp_points
        self.trg_points = trg_points
        self.temp_geos = temp_geos
        self.trg_geos = trg_geos


if __name__ == "__main__":
    # landmark from disk
    landmark_geo = load_landmark()
    device=torch.device("cuda:0")
    temp_points = dc.Convert(hou_geo=landmark_geo["temp_landmarks"],device=device).toMeshes().points_packed()
    trg_points = dc.Convert(hou_geo=landmark_geo["trg_landmarks"],device=device).toMeshes().points_packed()

    # houdini python node
    if __name__ == "builtins":
        import pyLib.lossLib.blendloss as bl
        import net.blendshape.blend as blend
        node = hou.pwd()
        geo = node.geometry()
        in_geo1 = node.inputs()[1].geometry()
        in_geo2 = node.inputs()[2].geometry()
        in_geo3 = node.inputs()[3].geometry()

        device = torch.device("cuda:0")
        temp_m = dc.Convert(hou_geo=geo, device=device).toMeshes()
        trg_m = dc.Convert(hou_geo=in_geo1, device=device).toMeshes()

        temp_points = dc.Convert(hou_geo=in_geo2, device=device).toMeshes().points_packed()
        trg_points = dc.Convert(hou_geo=in_geo3, device=device).toMeshes().points_packed()

    net = blend.BlendMeshNet(temp_m)
    net.set_landmark(in_geo2, in_geo3)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)

    attrib_interpolate_loss = bl.BlendLoss.attrib_interpolate_loss

    temp_mesh, temp_points, trg_points, new_src_mesh_id = net(trg_m)
    loss = attrib_interpolate_loss(temp_points, trg_points, temp_mesh, new_src_mesh_id)
    loss.backward()
    optimizer.step()

    ngeo = dc.Convert(t3d_geo=temp_mesh).toHoudini()
    geo.clear()
    geo.copy(ngeo)