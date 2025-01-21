#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/20 14:23
# @Author  : zxgvfx
# @File    : blendloss.py
# @Software: PyCharm
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import ball_query

class BlendLoss:

    @classmethod
    def attrib_interpolate_loss(cls,src_points_data:torch.Tensor,
                                trg_points_data:torch.Tensor,
                                meshes: Meshes,
                                meshes_id: torch.Tensor)->torch.Tensor:
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )
        loss_dis = torch.norm_except_dim(src_points_data - trg_points_data, dim=1)[0]
        N = len(meshes)
        verts_packed = meshes.verts_packed()
        edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
        edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
        num_edges_per_mesh = meshes.num_edges_per_mesh()
        weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
        weights = 1.0 / weights.float()
        updata_verts_packed = verts_packed.clone().detach()
        verts_id_packed = meshes.faces_packed()
        for index,i in enumerate(meshes_id):
            v_ids = verts_id_packed[i]
            a = src_points_data[0][index] - trg_points_data[0][index]
            for v_index in v_ids:
                verts_packed[v_index].data += a

        verts_edges = verts_packed[edges_packed]
        updata_verts_edges = updata_verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        updata_v0, updata_v1 = updata_verts_edges.unbind(1)
        # loss = torch.norm_except_dim(verts_packed-updata_verts_packed ,dim=1)[0] ** 2.0
        loss = (v0 - updata_v0).norm(dim=1, p=2) ** 2 +  (v1 - updata_v1).norm(dim=1, p=2) ** 2
        return loss.sum()



