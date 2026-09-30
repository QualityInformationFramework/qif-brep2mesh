from .utils import *


class Points:
    def __init__(self, xml_set):
        count = int(xml_set.get('n'))
        self.index = dict()
        self.points = np.empty((count, 3), dtype=float)
        i = 0
        for xml_point in xml_set:
            id = int(xml_point.get('id'))
            self.index[id] = i
            point = xml_float_array(xml_point, './qif:XYZ')
            self.points[i] = point
            i = i + 1

    def __getitem__(self, id):
        return self.points[self.index[id]]
    
    def bbox_norm(self):
        return np.linalg.norm(np.max(self.points,axis=0) - np.min(self.points,axis=0))