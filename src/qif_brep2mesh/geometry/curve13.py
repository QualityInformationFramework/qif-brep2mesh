from .utils import *
import numpy as np
from scipy.interpolate import BSpline, PPoly

class Curve13:
    def __init__(self, xml):
        if 'id' in xml.attrib:
            self.id = int(xml.get('id'))
        self.domain = xml_attr_float_array(xml, './*','domain')

    def divide(self, lod, loop_count):
        step_count = int(np.ceil(self.length() / lod)) + 1
        if loop_count == 1 and step_count < 4:
            step_count = 4
        if loop_count == 2 and step_count < 3:
            step_count = 3        
        input = np.linspace(self.domain[0], self.domain[1], step_count)
        return self.values(input)
    
    def divide_by(self, step_count):
        input = np.linspace(self.domain[0], self.domain[1], step_count)
        return [self.value(i) for i in input]
    
    def length(self, step_count):
        poly_line = np.array(self.divide_by(step_count))
        return sum(np.linalg.norm(poly_line[1:] - poly_line[:-1], axis=1))
    
    def value(self, point):
        return self.values(np.array([point,]))[0]

class Segment13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.start_point = xml_np_array(xml, './qif:Segment13Core/qif:StartPoint')
        self.end_point = xml_np_array(xml, './qif:Segment13Core/qif:EndPoint')
    
    def length(self):
        return np.linalg.norm(self.end_point - self.start_point)
    
    def values(self, t):
        return self.start_point + t[...,None] * (self.end_point - self.start_point)
    
class Polyline13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        if xml.find('./qif:Polyline13Core/qif:Points', xml_namespace(xml)) is not None:
            self.points = xml_np_array(xml, './qif:Polyline13Core/qif:Points').reshape(-1, 3)
        else:
            self.points = xml_np_binary_array(xml, './qif:Polyline13Core/qif:PointsBinary').reshape(-1, 3)

    def divide(self, lod, loop_count):
        input = np.empty(0, dtype=float)  
        for i in range(len(self.points) - 1):
            length = np.linalg.norm(self.points[i + 1] - self.points[i])
            step_count = int(np.ceil(length / lod)) + 1
            input = np.concatenate((input, np.linspace(i, i + 1, step_count, endpoint = False)))
        input = np.append(input, self.domain[1])
        return self.values(input)
    
    def length(self):
        return sum(np.linalg.norm(self.points[1:] - self.points[:-1], axis=1))
    
    def values(self, t):
        segments = np.floor(t).astype(int).clip(min=0, max=len(self.points)-2)
        return self.points[segments] + (t - segments)[...,None] * (self.points[segments + 1] - self.points[segments])

class ArcCircular13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.radius = xml_float(xml, './qif:ArcCircular13Core/qif:Radius')
        self.center = xml_np_array(xml, './qif:ArcCircular13Core/qif:Center')
        self.dir_beg = xml_np_array(xml, './qif:ArcCircular13Core/qif:DirBeg')
        self.normal = xml_np_array(xml, './qif:ArcCircular13Core/qif:Normal')
        self.dir_y = np.cross(self.normal, self.dir_beg)

    def length(self):
        return (self.domain[1] - self.domain[0]) * self.radius
    
    def values(self, t):
        return self.center + self.radius * (np.cos(t[...,None]) * self.dir_beg + np.sin(t[...,None]) * self.dir_y)

class ArcConic13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.form = xml_attr(xml, './qif:ArcConic13Core','form')
        self.a = xml_float(xml, './qif:ArcConic13Core/qif:A')
        self.b = xml_float(xml, './qif:ArcConic13Core/qif:B')
        self.center = xml_np_array(xml, './qif:ArcConic13Core/qif:Center')
        self.dir_beg = xml_np_array(xml, './qif:ArcConic13Core/qif:DirBeg')
        self.normal = xml_np_array(xml, './qif:ArcConic13Core/qif:Normal')
        self.dir_y = np.cross(self.normal, self.dir_beg)

    def length(self):
        return super().length(5)
            
    def values(self, t):
        t = t[...,None]
        match self.form:
            case 'PARABOLA':
                return self.center + self.a * t * self.dir_beg + self.b * t**2 * self.dir_y
            case 'ELLIPSE':
                return self.center + self.a * np.cos(t) * self.dir_beg + self.b * np.sin(t) * self.dir_y
            case 'HYPERBOLA':
                return self.center + self.a * (1 + (t ** 2) / (self.b ** 2))**0.5 * self.dir_beg + t**2 * self.dir_y


class Spline13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.normalized = xml_attr_bool(xml, './qif:Spline13Core','normalized',False)
        self.knots = xml_np_array(xml, './qif:Spline13Core/qif:Knots')
        self.orders = xml_np_array(xml, './qif:Spline13Core/qif:Orders', int)
        if len(self.orders) == 1:
            coeff = [xml_np_array(xml, './qif:Spline13Core/qif:Coefficients').reshape(-1, 3),]
        else:
            coeff = np.split(xml_np_array(xml, './qif:Spline13Core/qif:Coefficients').reshape(-1, 3), np.cumsum(self.orders)[:-1])
        self.coef = np.flip(np.array([np.pad(co, {0: (0,int(max(self.orders) - len(co)))}) for co in coeff]).swapaxes(0,1), axis=0)
        self.degree = self.orders - 1

    def length(self):
        return super().length(len(self.knots))

    def values(self, t):
        if self.normalized:
            knots = np.arange(len(self.knots))
            k_diff = np.diff(self.knots)
            ti = np.searchsorted(self.knots[1:-1],t)
            t = ti + (t - self.knots[ti])/k_diff[ti]
        else:
            knots = self.knots
        return PPoly(self.coef, knots)(t)    

class Nurbs13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.order = xml_float(xml, './qif:Nurbs13Core/qif:Order')
        self.knots = xml_np_array(xml, './qif:Nurbs13Core/qif:Knots')
        
        if xml.find('./qif:Nurbs13Core/qif:CPs', xml_namespace(xml)) is not None:
            self.cps = xml_np_array(xml, './qif:Nurbs13Core/qif:CPs').reshape(-1, 3)
        else:
            self.cps = xml_np_binary_array(xml, './qif:Nurbs13Core/qif:CPsBinary').reshape(-1, 3)

        if xml.find('./qif:Nurbs13Core/qif:Weights', xml_namespace(xml)) is not None:
            self.weights = xml_np_array(xml, './qif:Nurbs13Core/qif:Weights')
        else:
            self.weights = np.full(self.cps.shape[0], 1)
        self.degree = int(self.order - 1)
        self.domain[0] = max(self.domain[0], self.knots[0]) # clamp the domain to the knot range
        self.domain[1] = min(self.domain[1], self.knots[-1])

    def length(self):
        return super().length(len(self.knots))

    def values(self, t):
        b_splines = BSpline.design_matrix(t, self.knots, self.degree, extrapolate=True).toarray()
        return np.sum((b_splines * self.weights)[:,:,np.newaxis] * self.cps, axis=1) / np.sum(b_splines * self.weights, axis=1)[:,np.newaxis]

class Aggregate13(Curve13):
    def __init__(self, xml):
        super().__init__(xml)
        self.sub_curves = []
        self.turned = []
        self.t_begin = [0]

        xml_sub_curves = xml.find('./qif:Aggregate13Core/qif:SubCurves', xml_namespace(xml))
        for sc in xml_sub_curves:
            core_tag = sc.find('./*').tag.split('}')[1][:-4]
            curve = get[core_tag](sc)
            self.sub_curves.append(curve)
            self.turned.append(xml_attr_bool(xml, './', 'turned', False))
            self.t_begin.append((curve.domain[1] - curve.domain[0]) + self.t_begin[-1])

    def length(self):
        length = 0
        for curve in self.sub_curves:
            length += curve.length()
        return length
    
    def values(self, t):
        t = np.array(t)
        results = np.empty((len(t),3))
        bins = self.t_begin[1:-1] #drop ends
        curves = np.digitize(t, bins)
        for i in np.unique(curves):
            if not self.turned[i]:
                t_sub = self.sub_curves[i].domain[0] + (t[i == curves] - self.t_begin[i])
            else:
                t_sub = self.sub_curves[i].domain[1] - (t[i == curves] - self.t_begin[i])
            results[i == curves] = self.sub_curves[i].values(t_sub)
        return results

get = {
'Segment13': Segment13,
'Polyline13': Polyline13,
'ArcCircular13': ArcCircular13,
'ArcConic13': ArcConic13,
'Spline13': Spline13,
'Nurbs13': Nurbs13,
'Aggregate13' : Aggregate13
}