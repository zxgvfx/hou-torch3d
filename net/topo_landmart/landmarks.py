import hou
from pyLib.houLib.node.sop.sop_verb import SopNode
import os
"""
landmark cache 导入
"""

LANDMARK = os.path.abspath("net/topo_landmart/geo/nz")
FILENAME = ["temp_landmarks.bgeo.sc","trg_landmarks.bgeo.sc"]
def load_landmark():
    geo = dict()
    for file in FILENAME:
        file_path = os.path.join(LANDMARK,file)
        if os.path.exists(file_path):
            if "temp_landmarks" in file:
                sop = SopNode()
                sop.load_cache(file_path)
                geo["temp_landmarks"] = sop.geo
            else:
                sop = SopNode()
                sop.load_cache(file_path)
                geo["trg_landmarks"] = sop.geo
    return geo

