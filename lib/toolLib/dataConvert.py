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
        for pt in p[:-1][::-1]:
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
        return np.array(self.geo.point(int(p[-1])).position(),dtype=np.float32)

    def create_np_verts_id(self,p, )-> np.ndarray:
        """
        hou face point(face_verts) index to numpy data。
        :param p: numpy data with index, index -> p[-1]
        :return: numpy data shape(triangles or quads,)
        """
        return np.array([pt.number() for pt in np.array(self.geo.prim(int(p[-1])).points())[::-1]],dtype=np.int64)

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
    def __init__(self,
                 t3d_geo: Meshes= None,
                 hou_geo: hou.Geometry = None):
        self.t3d_geo = t3d_geo
        self.geo = hou_geo

        self._init_data_()

    def createIndex(self,verts_num: int,
                    face_num: int,
                    verts_data: np.ndarray = np.array([]),
                    face_data:  np.ndarray =np.array([])):
        point_id = np.arange(verts_num, dtype=np.float32)
        prim_id = np.arange(face_num, dtype=np.float32)
        if verts_data.any():
            verts_data = verts_data.astype(np.float32)
        else:
            verts_data = np.ndarray((verts_num, 3), dtype=np.float32)
        if face_data.any():
            face_data = face_data.astype(np.float32)
        else:
            face_data = np.ndarray((face_num, 3), dtype=np.float32)

        verts_index = np.concatenate( (verts_data,
                                      point_id.reshape((-1, 1))), 1)
        verts_id_index = np.concatenate((face_data,
                                         prim_id.reshape((-1, 1))), 1)
        return verts_index, verts_id_index
    def createIndexFromMeshes(self):

        v = self.t3d_geo.verts_packed().detach().numpy()
        f = self.t3d_geo.faces_packed().detach().numpy()
        verts_index, verts_id_index = self.createIndex(v.shape[0],
                                                       f.shape[0],
                                                       v,
                                                       f)
        return verts_index, verts_id_index

    def createIndexFromGeo(self):
        v_num = len(self.geo.points())
        f_num = len(self.geo.prims())
        verts_index, verts_id_index = self.createIndex(v_num,f_num)
        return verts_index,verts_id_index

    def genMeshesByData(self,verts,face_id,*args,**kwargs):
        return Meshes(verts =[torch.from_numpy(verts)],faces=[torch.from_numpy(face_id)])

    def _init_data_(self):
        if not self.t3d_geo and not self.geo:
            raise IOError("确保输入是否正确！！")
        if self.t3d_geo and not self.geo:
            self.t3d_geo.cpu()
            self.verts_index, self.verts_id_index = self.createIndexFromMeshes()
            self.geo = hou.Geometry()
        if self.geo and not self.t3d_geo :
            self.verts_index,self.verts_id_index = self.createIndexFromGeo()

    def toHoudini(self):
        geo = hou.Geometry()
        cvt = CoreFuntion(geo)
        np.apply_along_axis(cvt.create_hou_points, 1,self.verts_index)
        np.apply_along_axis(cvt.create_hou_prim, 1,self.verts_id_index)
        return geo

    def toMeshes(self)->Meshes:
        cvt = CoreFuntion(self.geo)
        # error: 如果模型既有三角面，又有四边面，需要将其全部转化为三角面或四边面
        verts = np.apply_along_axis(cvt.create_np_verts, 1, self.verts_index)
        verts_id = np.apply_along_axis(cvt.create_np_verts_id, 1, self.verts_id_index)
        return self.genMeshesByData(verts,verts_id)

if __name__ == "__main__":
    import os
    dir_path = os.path.dirname(__file__)
    trg_obj = os.path.abspath(f'{dir_path}/../../file/obj/dolphin.obj')
    verts, faces, aux = load_obj(trg_obj)
    # ----------------
    trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
    trg_mesh.cpu()
    obj_cvt = Convert(trg_mesh)
    geo = obj_cvt.toHoudini()
    print(geo.boundingBox())

    # [-0.141481, 0.139833, 0.031976, 0.493277, -0.283368, 0.430819]
    #----------------
    geo_hou = hou.Geometry()
    box_verb = hou.sopNodeTypeCategory().nodeVerb("box")
    box_verb.setParms({
        "t": hou.Vector3(0.5, -0.5, 2.0),
        "scale": 0.5,
    })
    box_verb.execute(geo_hou , [])
    convert = Convert(hou_geo = geo_hou)
    verts,verts_id =  convert.toMeshes()
    print(verts,verts_id)
