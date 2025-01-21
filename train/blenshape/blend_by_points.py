import hou
import torch
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pyLib.toolLib import dataConvert
import pyLib.lossLib.blendloss as bl
import imp
imp.reload(bl)
imp.reload(dataConvert)
from pyLib.houLib.tools import meshes_attr_interpolate
imp.reload(meshes_attr_interpolate)

class Blendshape:
    def __init__(self, temp_mesh: Meshes,
                 trg_mesh: Meshes,
                 w_chamfer = 1.0,
                 w_edge = 1.0,
                 w_normal=0.01,
                 w_laplacian = 0.1):
        self.temp_mesh = temp_mesh
        self.trg_mesh = trg_mesh
        self.w_chamfer = w_chamfer
        self.w_edge = w_edge
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian
        self.init_data()

    def init_data(self):
        """初始化"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")
        self.temp_mesh = self.temp_mesh.to(self.device)
        self.trg_mesh = self.trg_mesh.to(self.device)
        self.deform_verts = torch.full(self.temp_mesh.verts_packed().shape,
                                       0.0,
                                       device=self.device,
                                       requires_grad=True)

    def optimizer(self,lr = 1.0,momentum=0.9):
        """优化器"""
        return torch.optim.SGD([self.deform_verts],
                               lr=lr,
                               momentum=momentum)

    def sample_points(self, temp_mesh: Meshes,
                      trg_mesh=None,
                      num_points = 5000)->tuple[torch.Tensor,torch.Tensor]:
        if trg_mesh:
            trg_points = sample_points_from_meshes(trg_mesh, num_points)
        else:
            trg_points = sample_points_from_meshes(self.trg_mesh, num_points)
        src_points = sample_points_from_meshes(temp_mesh, num_points)
        return src_points.to(self.device),trg_points.to(self.device)

    def update(self):
        return self.temp_mesh.offset_verts(self.deform_verts)

def train_with_geo(net:Blendshape,
          optimizer: torch.optim,
          num_epoch: int ,
          temp_geo: hou.Geometry,
          trg_geo: hou.Geometry):
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    new_src_mesh = None
    attrib_interpolate_loss = bl.BlendLoss.attrib_interpolate_loss
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        new_src_mesh = net.update()
        temp_points,trg_points,new_src_mesh_id,_= meshes_attr_interpolate.sample_points_from_geo(temp_geo,new_src_mesh,trg_geo)
        attloss = attrib_interpolate_loss(temp_points,trg_points,new_src_mesh,new_src_mesh_id)
        trg_points, src_points = net.sample_points(new_src_mesh, num_points=5000)
        loss_chamfer, _ = chamfer_distance(trg_points, src_points)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss = attloss*0.1 + loss_chamfer +loss_edge * net.w_edge #+ loss_laplacian * net.w_laplacian + loss_normal * net.w_normal
        print(f"loss:  {loss}")
        # chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        # edge_losses.append(float(loss_edge.detach().cpu()))
        # normal_losses.append(float(loss_normal.detach().cpu()))
        # laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        loss.backward()
        optimizer.step()
    return new_src_mesh

def train(net:Blendshape,
          optimizer: torch.optim,
          num_epoch: int = 1):
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    new_src_mesh = None
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        new_src_mesh = net.update()
        trg_points,src_points = net.sample_points(new_src_mesh, num_points=5000)
        loss_chamfer, _ = chamfer_distance(trg_points,src_points)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss = loss_chamfer * net.w_chamfer + loss_edge * net.w_edge + loss_normal * net.w_normal + loss_laplacian * net.w_laplacian
        print('total_loss = %.6f' % loss)
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        loss.backward()
        optimizer.step()

    return new_src_mesh




