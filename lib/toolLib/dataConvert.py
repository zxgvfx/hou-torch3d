import numpy as np
import hou
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj

class CoreFuntion:

    def __init__(self, hou_geo: hou.Geometry,):
        self.geo = hou_geo

    def create_hou_points(self, p: np.ndarray, ):
        """
        To create houdini point geometry.
        :param p: verts data.
        :return None
        """
        pt = self.geo.createPoint()
        pt.setPosition((float(p[0]), float(p[1]), float(p[2])))

    def create_hou_prim(self, p: np.ndarray, ):
        """
        To create houdini prim geometry.
        :param p: verts_id data.
        :return:
        """
        poly = self.geo.createPolygon()
        for pt in p[::-1]:
            pts = self.geo.point(int(pt))
            poly.addVertex(pts)

    def create_hou_normal(self,p,):
        """
        Todo: To create houdini normal.
        :param p: normal data
        :return:
        """
        pass

    def create_hou_color(self,p,):
        """
        Todo: To create houdini color
        :param p: texture data
        :return:
        """
        pass

    def create_np_verts(self,p, ) -> np.ndarray:
        """
        hou point(verts) position to numpy data。
        :param p: numpy data with index, index -> p[-1]
        :return: numpy data shape(3,)
        """
        return np.array(self.geo.point(int(p[-1])).position())

    def create_np_verts_id(self,p, )-> np.ndarray:
        """
        hou face point(face_verts) index to numpy data。
        :param p: numpy data with index, index -> p[-1]
        :return: numpy data shape(triangles or quads,)
        """
        return np.array([pt.number() for pt in self.geo.prim([int(p[-1])]).points()])

    def create_np_normal(self,p,) -> np.ndarray:
        """
        hou normal to numpy data
        :param p:
        :return:
        """
        pass

    def create_np_texture(self,p,) ->np.ndarray:
        """
        hou color to numpy data
        :param p:
        :return:
        """
        pass

# Meshes.verts_list() # meshs ————> point position list
# Meshes.faces_list() # meshs ————> faces idx list

class Convert:
    def __init__(self,verts_data: tuple[np.ndarray]= None,
                 geo: hou.Geometry = None):
        self.verts = verts_data[0]
        self.verts_id = verts_data[1]
        self.geo = geo
        self.convert_type = -1
        self._init_data_()

    def _init_data_(self):
        if (len(self.verts)==0)and(self.geo == None ) :
            raise IOError("确保输入是否正确！！")
        elif (len(self.verts)>0) and (self.geo == None ):
            self.geo = hou.Geometry()
            self.convert_type = 0
        elif (self.geo != None ) and (len(self.verts)==0):
            point_id = np.arange(len(self.geo.points()),dtype=np.float32)
            prim_id = np.arange(len(self.geo.prims()),dtype=np.float32)
            self.verts_index = np.concatenate(np.ndarray((point_id.shape[0],3),dtype=np.float32),point_id.reshape((-1,1)),1)
            self.verts_id_index = np.concatenate(np.ndarray((prim_id.shape[0],3),dtype=np.float32),prim_id.reshape((-1,1)),1)
            self.convert_type = 1
        else :
            point_id = np.arange(len(self.geo.points()), dtype=np.float32)
            prim_id = np.arange(len(self.geo.prims()), dtype=np.float32)
            self.verts_index = np.concatenate(np.ndarray((point_id.shape[0], 3), dtype=np.float32),
                                              point_id.reshape((-1, 1)), 1)
            self.verts_id_index = np.concatenate(np.ndarray((prim_id.shape[0], 3), dtype=np.float32),
                                                 prim_id.reshape((-1, 1)), 1)
            self.convert_type = 2

    def toHoudini(self):
        if self.convert_type == 2 or self.convert_type == 0:
            c = CoreFuntion(self.geo)
            np.apply_along_axis(c.create_hou_points, 1,self.verts)
            np.apply_along_axis(c.create_hou_prim, 1,self.verts_id)
            return self.geo
        else:
            print("No input pytorch3d (verts,verts_id) Data !")

    def toPytorch3d(self):
        if self.convert_type == 2 or self.convert_type == 1:
            c = CoreFuntion(self.geo)
            self.verts = np.apply_along_axis(c.create_np_verts, 1, self.verts_index)
            self.verts_id = np.apply_along_axis(c.create_np_verts_id, 1, self.verts_id_index)

            return self.verts,self.verts_id
        else:
            raise RuntimeError("Input geometry error!")

if __name__ == "__main__":
    import os,sys
    dir_path = os.path.dirname(__file__)
    trg_obj = os.path.abspath(f'{dir_path}/../../file/obj/dolphin.obj')
    verts, faces, aux = load_obj(trg_obj)

    convert = Convert([verts.detach().numpy(), faces.verts_idx.detach().numpy()])
    geo = convert.toHoudini()
    print(geo.boundingBox())
    # [-0.141481, 0.139833, 0.031976, 0.493277, -0.283368, 0.430819]
