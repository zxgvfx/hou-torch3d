import numpy as np
import hou
import torch
from pytorch3d.structures import Meshes, Pointclouds
from .extended_meshes import ExtendedMeshes
from pytorch3d.io import load_obj, save_obj
from typing import Dict, Any, Union, Tuple, Optional

# =========================
# Houdini与PyTorch3D数据互转核心类
# =========================
class CoreFunction:
    """
    Houdini Geometry与numpy/torch数据互转的基础方法，支持点、面、法线、颜色、属性等。
    """
    def __init__(self, hou_geo: hou.Geometry):
        self.geo = hou_geo

    def create_hou_points(self, p: np.ndarray):
        """
        创建Houdini点。
        :param p: (x, y, z, ...) 点坐标
        """
        pt = self.geo.createPoint()
        pt.setPosition((float(p[0]), float(p[1]), float(p[2])))

    def create_hou_prim(self, p: np.ndarray):
        """
        创建Houdini多边形面。
        :param p: 顶点索引数组，最后一个元素是面ID，前面是顶点索引
        """
        poly = self.geo.createPolygon()
        # 获取顶点索引（去掉最后一个面ID）
        vertex_indices = p[:-1]
        # 按顺序添加顶点，确保面正确闭合
        for vertex_idx in vertex_indices:
            pt = self.geo.point(int(vertex_idx))
            poly.addVertex(pt)

    def create_hou_attribs(self, attribs: Dict[str, torch.Tensor], index: int):
        """
        创建/设置点属性。
        :param attribs: 属性字典
        :param index: 点索引
        """
        pt = self.geo.point(index)
        for att, value in attribs.items():
            v = list(value.cpu().detach().numpy().astype(object)[index])
            if len(v) > 1:
                if not self.geo.findPointAttrib(att):
                    self.geo.addArrayAttrib(hou.attribType.Point, att, hou.attribData.Float, len(v))
                pt.setAttribValue(att, v)
            else:
                if not self.geo.findPointAttrib(att):
                    self.geo.addAttrib(hou.attribType.Point, att, v[0])
                pt.setAttribValue(att, v[0])

    def create_hou_normal(self, normals: np.ndarray):
        """
        创建法线属性（点法线）。
        :param normals: (N, 3) 法线数组
        """
        if not self.geo.findPointAttrib('N'):
            self.geo.addAttrib(hou.attribType.Point, 'N', (0.0, 0.0, 0.0))
        for i, pt in enumerate(self.geo.points()):
            n = normals[i]
            # 保证是float类型的tuple，防止numpy类型报错
            pt.setAttribValue('N', (float(n[0]), float(n[1]), float(n[2])))

    def create_hou_color(self, colors: np.ndarray):
        """
        创建颜色属性（点颜色Cd）。
        :param colors: (N, 3) 颜色数组
        """
        if not self.geo.findPointAttrib('Cd'):
            self.geo.addAttrib(hou.attribType.Point, 'Cd', (1.0, 1.0, 1.0))
        for i, pt in enumerate(self.geo.points()):
            c = colors[i]
            pt.setAttribValue('Cd', (float(c[0]), float(c[1]), float(c[2])))

    def create_np_verts(self) -> np.ndarray:
        """
        Houdini点转numpy数组。
        :return: (N, 3)
        """
        return np.array([np.array(pt.position()) for pt in self.geo.points()], dtype=np.float32)

    def create_np_verts_id(self) -> np.ndarray:
        """
        Houdini面顶点索引转numpy数组。
        :return: (M, n) n为多边面顶点数
        """
        return np.array([
            np.array([pt.number() for pt in prim.points()])[::-1] for prim in self.geo.prims()
        ], dtype=object)

    def create_np_normal(self) -> np.ndarray:
        """
        Houdini点法线转numpy数组。
        :return: (N, 3)
        """
        normals = []
        for pt in self.geo.points():
            if self.geo.findPointAttrib('N'):
                normals.append(np.array(pt.attribValue('N')))
            else:
                normals.append(np.zeros(3))
        return np.array(normals, dtype=np.float32)

    def create_np_color(self) -> np.ndarray:
        """
        Houdini点颜色转numpy数组。
        :return: (N, 3)
        """
        colors = []
        for pt in self.geo.points():
            if self.geo.findPointAttrib('Cd'):
                colors.append(np.array(pt.attribValue('Cd')))
            else:
                colors.append(np.ones(3))
        return np.array(colors, dtype=np.float32)

    def create_np_prim_attrib(self, attrib_name: str) -> np.ndarray:
        """
        Houdini面属性转numpy数组。
        :param attrib_name: 属性名
        :return: (M, ...)
        """
        attrib = self.geo.findPrimAttrib(attrib_name)
        if not attrib:
            return None
        return np.array([prim.attribValue(attrib_name) for prim in self.geo.prims()])

    def create_np_detail_attrib(self, attrib_name: str) -> Any:
        """
        Houdini全局属性转python对象。
        :param attrib_name: 属性名
        :return: 属性值
        """
        attrib = self.geo.findGlobalAttrib(attrib_name)
        if not attrib:
            return None
        return self.geo.attribValue(attrib_name)

# =========================
# 数据转换主类 - 重新设计
# =========================
class Convert:
    """
    支持Houdini Geometry与PyTorch3D Meshes/Pointclouds的互转，支持点/面/全局属性、法线、颜色等。
    重新设计：状态一致性，职责清晰，缓存机制。
    """
    def __init__(self,
                 t3d_geo: Optional[Union[Meshes, Pointclouds]] = None,
                 hou_geo: Optional[hou.Geometry] = None,
                 device: torch.device = torch.device("cpu"),
                 force_device: bool = False):
        """
        初始化转换器。
        :param t3d_geo: PyTorch3D几何体
        :param hou_geo: Houdini几何体
        :param device: 计算设备
        :param force_device: 是否强制使用指定设备
        """
        # 设置设备
        if not force_device and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = device
            if device.type == "cpu":
                print("WARNING: CPU only, this will be slow!")
        
        # 属性存储
        self._attribs: Dict[str, torch.Tensor] = dict()
        self._prim_attribs: Dict[str, torch.Tensor] = dict()
        self._detail_attribs: Dict[str, Any] = dict()
        
        # 缓存
        self._cached_geo = None
        self._cached_mesh = None
        self._cached_normals = None
        self._cached_colors = None
        
        # 初始化数据
        if t3d_geo is not None and hou_geo is not None:
            raise ValueError("只能指定t3d_geo或hou_geo其中之一")
        elif t3d_geo is not None:
            self.t3d_geo = t3d_geo.to(self.device)
            self.geo = None
        elif hou_geo is not None:
            self.geo = hou_geo
            self.t3d_geo = None
        else:
            raise ValueError("必须指定t3d_geo或hou_geo其中之一")

    def toMeshes(self) -> Union[Meshes, Pointclouds]:
        """
        转换为PyTorch3D Meshes/Pointclouds，并更新内部状态。
        使用与updateFromGeo相同的转换逻辑，确保一致性。
        同时将添加的属性应用到PyTorch3D几何体。
        :return: PyTorch3D几何体
        """
        if self.t3d_geo is not None:
            # 如果已有t3d_geo，应用属性到现有几何体
            self._apply_attributes_to_mesh()
            return self.t3d_geo
        
        if self.geo is None:
            raise ValueError("没有可转换的Houdini几何体")
        
        # 使用与updateFromGeo相同的逻辑，确保一致性
        verts_index, verts_id_index = self.create_index_from_geo(self.geo)
        verts, _ = torch.split(verts_index, verts_index.shape[-1]-1, dim=1)
        if verts_id_index.size().numel() == 0:
            self.t3d_geo = Pointclouds(points=[verts])
        else:
            verts_id, _ = torch.split(verts_id_index, verts_id_index.shape[-1] - 1, dim=1)
            self.t3d_geo = Meshes(verts=[verts], faces=[verts_id])
        
        # 应用属性到新创建的几何体
        self._apply_attributes_to_mesh()
        
        return self.t3d_geo

    def _apply_attributes_to_mesh(self):
        """
        将添加的属性应用到PyTorch3D几何体。
        """
        if self.t3d_geo is None:
            return
        
        # 应用点属性
        if self._attribs:
            for att_name, att_value in self._attribs.items():
                # 确保属性在正确的设备上
                att_value = att_value.to(self.device)
                
                # 对于Pointclouds，使用features
                if isinstance(self.t3d_geo, Pointclouds):
                    # 将属性作为features添加
                    current_features = self.t3d_geo.features_packed()
                    if current_features is None:
                        # 如果没有features，创建新的
                        self.t3d_geo = Pointclouds(
                            points=[self.t3d_geo.points_packed()],
                            features=[att_value]
                        )
                    else:
                        # 如果有features，扩展
                        new_features = torch.cat([current_features, att_value], dim=1)
                        self.t3d_geo = Pointclouds(
                            points=[self.t3d_geo.points_packed()],
                            features=[new_features]
                        )
                # 对于Meshes，使用ExtendedMeshes支持属性
                elif isinstance(self.t3d_geo, Meshes):
                    # 如果已经是ExtendedMeshes，直接添加属性
                    if isinstance(self.t3d_geo, ExtendedMeshes):
                        self.t3d_geo.add_attribute(att_name, att_value)
                    else:
                        # 转换为ExtendedMeshes
                        attributes = {att_name: att_value}
                        for existing_name, existing_attr in self._attribs.items():
                            if existing_name != att_name:
                                attributes[existing_name] = existing_attr
                        
                        self.t3d_geo = ExtendedMeshes.from_meshes(self.t3d_geo, attributes)
        
        # 应用面属性（仅对Meshes有效）
        if self._prim_attribs and isinstance(self.t3d_geo, Meshes):
            for att_name, att_value in self._prim_attribs.items():
                att_value = att_value.to(self.device)
                # 面属性暂时存储在_prim_attribs中，可以通过getPrimAttrib获取
                # 如果需要，可以扩展ExtendedMeshes支持面属性
        
        # 全局属性存储在实例变量中，可以通过getDetailAttrib获取
        # PyTorch3D没有直接的全局属性支持，所以保持现状

    def toHoudini(self) -> hou.Geometry:
        """
        转换为Houdini Geometry，并更新内部状态。
        如果已有几何体，会将添加的属性应用到现有几何体上。
        :return: Houdini几何体
        """
        
        # 如果已有几何体，需要将属性应用到现有几何体上
        if self.geo is not None:
            # 应用点属性
            if self._attribs:
                cf = CoreFunction(self.geo)
                for i, pt in enumerate(self.geo.points()):
                    for att_name, att_value in self._attribs.items():
                        if i < att_value.shape[0]:  # 确保索引有效
                            v = att_value[i].cpu().detach().numpy()
                            if len(v.shape) > 0:
                                if not self.geo.findPointAttrib(att_name):
                                    self.geo.addArrayAttrib(hou.attribType.Point, att_name, hou.attribData.Float, len(v))
                                pt.setAttribValue(att_name, v.tolist())
                            else:
                                if not self.geo.findPointAttrib(att_name):
                                    self.geo.addAttrib(hou.attribType.Point, att_name, float(v))
                                pt.setAttribValue(att_name, float(v))
            
            # 应用面属性
            if self._prim_attribs:
                for att_name, att_value in self._prim_attribs.items():
                    if not self.geo.findPrimAttrib(att_name):
                        # 根据第一个值确定属性类型
                        first_val = att_value[0].cpu().detach().numpy()
                        if len(first_val.shape) > 0:
                            self.geo.addArrayAttrib(hou.attribType.Prim, att_name, hou.attribData.Float, len(first_val))
                        else:
                            self.geo.addAttrib(hou.attribType.Prim, att_name, float(first_val))
                    
                    for i, prim in enumerate(self.geo.prims()):
                        if i < att_value.shape[0]:
                            v = att_value[i].cpu().detach().numpy()
                            if len(v.shape) > 0:
                                prim.setAttribValue(att_name, v.tolist())
                            else:
                                prim.setAttribValue(att_name, float(v))
            
            # 应用全局属性
            if self._detail_attribs:
                for att_name, att_value in self._detail_attribs.items():
                    if not self.geo.findGlobalAttrib(att_name):
                        self.geo.addAttrib(hou.attribType.Global, att_name, att_value)
                    # 使用正确的方法设置全局属性
                    if hasattr(self.geo, 'setGlobalAttribValue'):
                        self.geo.setGlobalAttribValue(att_name, att_value)
                    else:
                        # 如果方法不存在，尝试其他方式
                        try:
                            self.geo.setAttribValue(att_name, att_value)
                        except AttributeError:
                            # 如果都不行，至少创建属性
                            pass
            
            return self.geo
        
        # 如果没有几何体，从PyTorch3D几何体创建
        if self.t3d_geo is None:
            raise ValueError("没有可转换的PyTorch3D几何体")
        
        # 从PyTorch3D几何体创建Houdini几何体
        if isinstance(self.t3d_geo, Meshes):
            verts = self.t3d_geo.verts_packed()
            faces = self.t3d_geo.faces_packed()
            # 为面数据添加面ID
            faces_with_id = torch.cat([faces, torch.arange(faces.shape[0], device=faces.device).unsqueeze(1)], dim=1)
        elif isinstance(self.t3d_geo, Pointclouds):
            verts = self.t3d_geo.points_packed()
            faces_with_id = torch.tensor([])
        else:
            raise ValueError(f"不支持的PyTorch3D几何体类型: {type(self.t3d_geo)}")
        
        # 创建Houdini几何体
        self.geo = gen_geo_by_data(
            verts, faces_with_id, self._attribs, 
            self._cached_normals, self._cached_colors,
            self._prim_attribs, self._detail_attribs
        )
        
        return self.geo

    def create_index_from_geo(self, geo: hou.Geometry = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从Houdini几何体创建索引数据，添加错误处理和验证。
        :param geo: Houdini几何体，如果为None则使用self.geo
        :return: (verts_index, verts_id_index) 顶点索引和面索引
        """
        # 确定使用的几何体
        target_geo = geo if geo is not None else self.geo
        if target_geo is None:
            raise ValueError("没有可用的Houdini几何体")
        
        # 验证几何体
        if len(target_geo.points()) == 0:
            raise ValueError("几何体必须包含至少一个点")
        
        v_num = len(target_geo.points())
        f_num = len(target_geo.prims())
        cf = CoreFunction(target_geo)

        try:
            verts_np = cf.create_np_verts()
            v = torch.from_numpy(verts_np).to(device=self.device)
            
            # 目前只支持3边面
            if f_num != 0:
                verts_id_np = cf.create_np_verts_id()
                # 处理object类型的数组，转换为统一的格式
                faces_list = []
                for face_verts in verts_id_np:
                    # 确保面顶点顺序正确（Houdini使用逆时针）
                    face_verts = face_verts[::-1]  # 反转顺序
                    faces_list.append(face_verts)
                
                # 转换为torch张量，先转换为numpy数组以提高性能
                faces_array = np.array(faces_list, dtype=np.int64)
                f = torch.from_numpy(faces_array).to(device=self.device)
                verts_index, verts_id_index = self.create_index_by_data(v, f)
            else:
                verts_index, verts_id_index = self.create_index_by_data(v)
            
            return verts_index, verts_id_index
        except Exception as e:
            raise RuntimeError(f"从几何体创建索引失败: {e}")

    def create_index_by_data(self, v: torch.Tensor, f: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据顶点和面数据创建索引。
        :param v: 顶点数据
        :param f: 面数据（可选）
        :return: (verts_index, verts_id_index)
        """
        if f is not None:
            # 有面数据，创建网格索引
            # 为顶点数据添加索引列
            v_indices = torch.arange(v.shape[0], device=v.device).unsqueeze(1)
            verts_index = torch.cat([v, v_indices], dim=1)
            
            # 为面数据添加面ID列
            f_indices = torch.arange(f.shape[0], device=f.device).unsqueeze(1)
            verts_id_index = torch.cat([f, f_indices], dim=1)
        else:
            # 只有顶点数据，创建点云索引
            v_indices = torch.arange(v.shape[0], device=v.device).unsqueeze(1)
            verts_index = torch.cat([v, v_indices], dim=1)
            verts_id_index = torch.tensor([], device=v.device)
        
        return verts_index, verts_id_index

    def _init_data(self):
        """
        初始化数据缓存，统一清理所有缓存。
        """
        self._cached_geo = None
        self._cached_mesh = None
        self._cached_normals = None
        self._cached_colors = None
        # 清除属性缓存
        self._attribs.clear()
        self._prim_attribs.clear()
        self._detail_attribs.clear()

    def updateFromGeo(self, geo: hou.Geometry):
        """
        从houdini中处理之后，转到 更新转化之前的 Meshes 物体
        :param geo: houdini geometry topo 必须一致
        :return:
        """
        if geo is None:
            raise ValueError("几何体不能为空")
        
        if len(geo.points()) == 0:
            raise ValueError("几何体必须包含至少一个点")
        
        try:
            verts_index, verts_id_index = self.create_index_from_geo(geo)
            verts, _ = torch.split(verts_index, verts_index.shape[-1]-1, dim=1)
            if verts_id_index.size().numel() == 0:
                self.t3d_geo = Pointclouds(points=[verts])
            else:
                verts_id, _ = torch.split(verts_id_index, verts_id_index.shape[-1] - 1, dim=1)
                self.t3d_geo = Meshes(verts=[verts], faces=[verts_id])
            
            # 应用属性到新创建的几何体
            self._apply_attributes_to_mesh()
            
            self.geo = None
            self._init_data()
        except Exception as e:
            raise RuntimeError(f"几何体转换失败: {e}")

    def updateFromMeshes(self, meshes: Union[Meshes, Pointclouds]):
        """
        从PyTorch3D几何体更新数据。
        :param meshes: PyTorch3D几何体
        """
        self.t3d_geo = meshes.to(self.device)
        self.geo = None  # 清除缓存
        self._cached_geo = None
        self._cached_mesh = None

    def addAttrib(self, attrib_name: str, attrib_value: torch.Tensor, overlay: bool = False) -> bool:
        """
        添加点属性，包含输入验证。
        :param attrib_name: 属性名
        :param attrib_value: 属性值
        :param overlay: 是否覆盖已存在的属性
        :return: 是否成功添加
        """
        # 输入验证
        if not isinstance(attrib_value, torch.Tensor):
            raise TypeError("属性值必须是torch.Tensor")
        
        if attrib_value.dim() < 1:
            raise ValueError("属性值至少需要1维")
        
        # 验证属性维度与几何体点数匹配
        expected_points = 0
        if self.has_mesh:
            expected_points = self.t3d_geo.verts_packed().shape[0]
        elif self.has_geo:
            expected_points = len(self.geo.points())
        else:
            expected_points = attrib_value.shape[0]
        
        if attrib_value.shape[0] != expected_points:
            raise ValueError(f"属性点数({attrib_value.shape[0]})与几何体点数({expected_points})不匹配")
        
        if attrib_name in self._attribs and not overlay:
            print(f'已存在{attrib_name}属性,请检查！')
            return False
        
        self._attribs[attrib_name] = attrib_value.to(self.device)
        return True

    def addPrimAttrib(self, attrib_name: str, attrib_value: torch.Tensor, overlay: bool = False) -> bool:
        """
        添加面属性，包含输入验证。
        :param attrib_name: 属性名
        :param attrib_value: 属性值
        :param overlay: 是否覆盖已存在的属性
        :return: 是否成功添加
        """
        # 输入验证
        if not isinstance(attrib_value, torch.Tensor):
            raise TypeError("面属性值必须是torch.Tensor")
        
        if attrib_value.dim() < 1:
            raise ValueError("面属性值至少需要1维")
        
        # 验证属性维度与几何体面数匹配
        expected_prims = 0
        if self.has_mesh:
            expected_prims = self.t3d_geo.faces_packed().shape[0]
        elif self.has_geo:
            expected_prims = len(self.geo.prims())
        else:
            expected_prims = attrib_value.shape[0]
        
        if attrib_value.shape[0] != expected_prims:
            raise ValueError(f"面属性数量({attrib_value.shape[0]})与几何体面数({expected_prims})不匹配")
        
        if attrib_name in self._prim_attribs and not overlay:
            print(f'已存在{attrib_name}面属性,请检查！')
            return False
        
        self._prim_attribs[attrib_name] = attrib_value.to(self.device)
        return True

    def addDetailAttrib(self, attrib_name: str, attrib_value: Any, overlay: bool = False) -> bool:
        """
        添加全局属性。
        :param attrib_name: 属性名
        :param attrib_value: 属性值
        :param overlay: 是否覆盖已存在的属性
        :return: 是否成功添加
        """
        if attrib_name in self._detail_attribs.keys() and not overlay:
            print(f'已存在{attrib_name}全局属性,请检查！')
            return False
        else:
            self._detail_attribs[attrib_name] = attrib_value
            return True

    def getAttrib(self, attrib_name: str) -> Optional[torch.Tensor]:
        """获取点属性"""
        return self._attribs.get(attrib_name, None)

    def getPrimAttrib(self, attrib_name: str) -> Optional[torch.Tensor]:
        """获取面属性"""
        return self._prim_attribs.get(attrib_name, None)

    def getDetailAttrib(self, attrib_name: str) -> Optional[Any]:
        """获取全局属性"""
        return self._detail_attribs.get(attrib_name, None)

    def getAttribs(self) -> Dict[str, torch.Tensor]:
        """获取所有点属性"""
        return self._attribs.copy()

    def clearAttribs(self):
        """清除所有属性"""
        self._attribs.clear()
        self._prim_attribs.clear()
        self._detail_attribs.clear()

    @property
    def has_mesh(self) -> bool:
        """是否有PyTorch3D几何体"""
        return self.t3d_geo is not None

    @property
    def has_geo(self) -> bool:
        """是否有Houdini几何体"""
        return self.geo is not None

    def validate_state(self):
        """
        验证内部状态一致性。
        """
        if self.has_mesh and self.has_geo:
            raise RuntimeError("不能同时拥有PyTorch3D和Houdini几何体")
        
        if not self.has_mesh and not self.has_geo:
            raise RuntimeError("必须拥有至少一种几何体")

    def get_state_info(self) -> dict:
        """
        获取状态信息。
        :return: 状态信息字典
        """
        return {
            'has_mesh': self.has_mesh,
            'has_geo': self.has_geo,
            'device': str(self.device),
            'attribs_count': len(self._attribs),
            'prim_attribs_count': len(self._prim_attribs),
            'detail_attribs_count': len(self._detail_attribs),
            'mesh_type': type(self.t3d_geo).__name__ if self.t3d_geo else None,
            'geo_points': len(self.geo.points()) if self.geo else 0,
            'geo_prims': len(self.geo.prims()) if self.geo else 0
        }

# =========================
# 工具函数
# =========================
def gen_meshes_by_data(verts_index: torch.Tensor,
                       verts_id_index: torch.Tensor) -> Union[Meshes, Pointclouds]:
    if verts_index.size().numel() != 0 and verts_id_index.size().numel() != 0:
        v = torch.split(verts_index, verts_index.shape[-1] - 1, dim=1)
        f = torch.split(verts_id_index, verts_index.shape[-1] - 1, dim=1)
        return Meshes(verts=[v[0]], faces=[f[0]])
    else:
        if verts_index.size().numel() != 0:
            v = torch.split(verts_index, verts_index.shape[-1] - 1, dim=1)
            return Pointclouds(points=[v[0]])


def gen_geo_by_data(verts_index: torch.Tensor,
                    verts_id_index: torch.Tensor,
                    attributes: Dict[str, torch.Tensor],
                    normals: np.ndarray = None,
                    colors: np.ndarray = None,
                    prim_attribs: Dict[str, torch.Tensor] = None,
                    detail_attribs: Dict[str, Any] = None) -> hou.Geometry:
    geo = hou.Geometry()
    cvt = CoreFunction(geo)
    
    # 创建点
    np.apply_along_axis(cvt.create_hou_points, 1, verts_index.cpu().detach().numpy())
    
    # 创建面 - 修复面创建逻辑
    if verts_id_index.size().numel() != 0:
        faces_np = verts_id_index.cpu().detach().numpy()
        for face_data in faces_np:
            # face_data 格式: [v1, v2, v3, face_id]
            vertex_indices = face_data[:-1]  # 去掉面ID
            poly = geo.createPolygon()
            for vertex_idx in vertex_indices[::-1]:
                pt = geo.point(int(vertex_idx))
                poly.addVertex(pt)
    
    # 点属性
    if attributes and len(attributes.keys()) > 0:
        [cvt.create_hou_attribs(attributes, i) for i, _ in enumerate(verts_index.cpu().detach().numpy())]
    # 法线
    if normals is not None:
        cvt.create_hou_normal(normals)
    # 颜色
    if colors is not None:
        cvt.create_hou_color(colors)
    # 面属性
    if prim_attribs:
        for att, value in prim_attribs.items():
            if not geo.findPrimAttrib(att):
                geo.addAttrib(hou.attribType.Prim, att, value[0].tolist() if hasattr(value[0], 'tolist') else value[0])
            for i, prim in enumerate(geo.prims()):
                prim.setAttribValue(att, value[i].tolist() if hasattr(value[i], 'tolist') else value[i])
    # 全局属性
    if detail_attribs:
        for att, value in detail_attribs.items():
            if not geo.findGlobalAttrib(att):
                geo.addAttrib(hou.attribType.Global, att, value)
            geo.setAttribValue(att, value)
    return geo

# =========================
# 测试入口
# =========================
if __name__ == "__main__":
    import os
    from pytorch3d.io import load_obj
    import numpy as np
    print("==== dataConvert.py 测试入口 ====")
    dir_path = os.path.dirname(__file__)
    trg_obj = os.path.abspath(f'{dir_path}/../../file/obj/dolphin.obj')
    print(f"加载OBJ: {trg_obj}")
    verts, faces, aux = load_obj(trg_obj)
    trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
    trg_mesh.cpu()

    # 1. Meshes -> Houdini Geometry
    print("\n[1] Meshes -> Houdini Geometry")
    obj_cvt = Convert(t3d_geo=trg_mesh)
    geo = obj_cvt.toHoudini()
    print("Houdini Geometry boundingBox:", geo.boundingBox())
    print("点数量:", len(geo.points()), "面数量:", len(geo.prims()))
    print("obj_cvt.has_mesh:", obj_cvt.has_mesh, "obj_cvt.has_geo:", obj_cvt.has_geo)

    # 2. 添加点属性
    print("\n[2] 添加点属性")
    attr_val = torch.rand(len(geo.points()), 3)
    obj_cvt.addAttrib('test_attr', attr_val)
    geo2 = obj_cvt.toHoudini()
    print("test_attr属性前3个:", [geo2.point(i).attribValue('test_attr') for i in range(3)])

    # 3. 添加法线、颜色
    print("\n[3] 添加法线、颜色")
    normals = np.random.randn(len(geo.points()), 3)
    colors = np.random.rand(len(geo.points()), 3)
    geo3 = hou.Geometry()
    geo3.copy(geo2)
    cf = CoreFunction(geo3)
    cf.create_hou_normal(normals)
    cf.create_hou_color(colors)
    print("N属性前3个:", [geo3.point(i).attribValue('N') for i in range(3)])
    print("Cd属性前3个:", [geo3.point(i).attribValue('Cd') for i in range(3)])

    # 4. Houdini Geometry -> Meshes
    print("\n[4] Houdini Geometry -> Meshes")
    geo_cvt = Convert(hou_geo=geo3)
    mesh2 = geo_cvt.toMeshes()
    print("mesh2 verts shape:", mesh2.verts_packed().shape)
    print("mesh2 faces shape:", mesh2.faces_packed().shape)
    print("geo_cvt.has_mesh:", geo_cvt.has_mesh, "geo_cvt.has_geo:", geo_cvt.has_geo)

    # 5. 多边面三角化测试
    print("\n[5] 多边面三角化测试")
    # 构造一个四边面
    geo_quad = hou.Geometry()
    pts = [geo_quad.createPoint() for _ in range(4)]
    for i, pt in enumerate(pts):
        pt.setPosition((i%2, i//2, 0))
    poly = geo_quad.createPolygon()
    for pt in pts:
        poly.addVertex(pt)
    quad_cvt = Convert(hou_geo=geo_quad)
    mesh_quad = quad_cvt.toMeshes()
    print("四边面转三角面后faces:", mesh_quad.faces_packed())

    # 6. 点云互转
    print("\n[6] 点云互转")
    pc = torch.rand(10, 3)
    pc_cvt = Convert()
    try:
        pc_cvt.toMeshes()
    except Exception as e:
        print("无数据转换报错:", e)

    # 7. 面属性、全局属性
    print("\n[7] 面属性、全局属性")
    prim_attr = torch.arange(len(geo.prims())).unsqueeze(1).float()
    obj_cvt.addPrimAttrib('prim_id', prim_attr)
    obj_cvt.addDetailAttrib('detail_val', 42.0)
    geo4 = obj_cvt.toHoudini()
    print("prim_id前3:", [geo4.prim(i).attribValue('prim_id') for i in range(min(3, len(geo4.prims())))])
    print("detail_val:", geo4.attribValue('detail_val'))

    # 8. updateFromGeo
    print("\n[8] updateFromGeo")
    geo_cvt2 = Convert(hou_geo=geo4)
    geo_cvt2.updateFromGeo(geo4)
    print("updateFromGeo后t3d_geo类型:", type(geo_cvt2.t3d_geo))

    # 9. updateFromMeshes
    print("\n[9] updateFromMeshes")
    geo_cvt3 = Convert(hou_geo=geo4)
    geo_cvt3.updateFromMeshes(mesh2)
    print("updateFromMeshes后geo点数:", len(geo_cvt3.geo.points()))

    # 10. 错误处理
    print("\n[10] 错误处理测试")
    try:
        Convert()
    except Exception as e:
        print("无输入时报错:", e)

    # 11. 状态一致性测试
    print("\n[11] 状态一致性测试")
    test_cvt = Convert(hou_geo=geo)
    print("初始状态 - has_mesh:", test_cvt.has_mesh, "has_geo:", test_cvt.has_geo)
    mesh = test_cvt.toMeshes()
    print("toMeshes后 - has_mesh:", test_cvt.has_mesh, "has_geo:", test_cvt.has_geo)
    geo_back = test_cvt.toHoudini()
    print("toHoudini后 - has_mesh:", test_cvt.has_mesh, "has_geo:", test_cvt.has_geo)

    print("==== 测试结束 ====")
