from .utils import *
import numpy as np
from scipy.interpolate import BSpline, PPoly

class Curve12:
    def __init__(self, xml):
        if 'id' in xml.attrib:
            self.id = int(xml.get('id'))
        self.domain = xml_attr_float_array(xml, './*','domain')
    
    def divide_by(self, count):
        input = np.linspace(self.domain[0], self.domain[1], count)
        return self.values(input)
    
    def value(self, point):
        return self.values(np.array([point,]))[0]


class Segment12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.start_point = xml_np_array(xml, './qif:Segment12Core/qif:StartPoint')
        self.end_point = xml_np_array(xml, './qif:Segment12Core/qif:EndPoint')

    def values(self, t):
        return self.start_point + t[...,None] * (self.end_point - self.start_point)

class Polyline12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        if xml.find('./qif:Polyline12Core/qif:Points', xml_namespace(xml)) is not None:
            self.points = xml_np_array(xml, './qif:Polyline12Core/qif:Points').reshape(-1, 2)
        else:
            self.points = xml_np_binary_array(xml, './qif:Polyline12Core/qif:PointsBinary').reshape(-1, 2)

    def divide_by(self, count):
        seg_len = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        seg_percent = seg_len/sum(seg_len)
        seg_steps = np.floor(seg_percent * count).astype(int)
        seg_rem = seg_percent * count - seg_steps
        seg_rem[seg_steps == 0] = 0
        seg_steps[seg_steps == 0] = 1

        diff = count - np.sum(seg_steps)
        if diff > 0:
            seg_steps[seg_rem.argsort()[:diff]] += 1


        if diff < 0:
            while count < np.sum(seg_steps):
                for i in seg_rem.argsort()[::-1]:
                    if(seg_steps[i] > 1):
                        seg_steps[i] -= 1
                        if count == np.sum(seg_steps):
                            break

        input = np.empty(0, dtype=float)  
        for i in range(len(self.points) - 1):
            input = np.concatenate((input, np.linspace(i, i + 1, seg_steps[i], endpoint = (i == len(self.points) - 2))))
        
        return self.values(input)
    
    def values(self, t):
        segments = np.floor(t).astype(int).clip(min=0, max=len(self.points)-2)
        return self.points[segments] + (t - segments)[...,None] * (self.points[segments + 1] - self.points[segments])

class ArcCircular12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned = xml_attr_bool(xml, './qif:ArcCircular12Core','turned', False)
        self.radius = xml_float(xml, './qif:ArcCircular12Core/qif:Radius')
        self.center = xml_np_array(xml, './qif:ArcCircular12Core/qif:Center')
        self.dir_beg = xml_np_array(xml, './qif:ArcCircular12Core/qif:DirBeg')
        if self.turned:
            self.dir_y = np.array([self.dir_beg[1], -self.dir_beg[0]])
        else:
            self.dir_y = np.array([-self.dir_beg[1], self.dir_beg[0]])
    
    def values(self, t):
        return self.center + self.radius * (np.cos(t[...,None]) * self.dir_beg + np.sin(t[...,None]) * self.dir_y)

class ArcConic12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.form = xml_attr(xml, './qif:ArcConic12Core','form')
        self.turned = xml_attr_bool(xml, './qif:ArcConic12Core','turned', False)
        self.a = xml_float(xml, './qif:ArcConic12Core/qif:A')
        self.b = xml_float(xml, './qif:ArcConic12Core/qif:B')
        self.center = xml_np_array(xml, './qif:ArcConic12Core/qif:Center')
        self.dir_beg = xml_np_array(xml, './qif:ArcConic12Core/qif:DirBeg')
        if self.turned:
            self.dir_y = np.array([self.dir_beg[1], -self.dir_beg[0]])
        else:
            self.dir_y = np.array([-self.dir_beg[1], self.dir_beg[0]])
            
    def values(self, t):
        t = t[...,None]
        match self.form:
            case 'PARABOLA':
                return self.center + self.a * t * self.dir_beg + self.b * t**2 * self.dir_y
            case 'ELLIPSE':
                return self.center + self.a * np.cos(t) * self.dir_beg + self.b * np.sin(t) * self.dir_y
            case 'HYPERBOLA':
                return self.center + self.a * (1 + (t ** 2) / (self.b ** 2))**0.5 * self.dir_beg + t**2 * self.dir_y

class Spline12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.normalized = xml_attr_bool(xml, './qif:Spline12Core','normalized',False)
        self.knots = xml_np_array(xml, './qif:Spline12Core/qif:Knots')
        self.orders = xml_np_array(xml, './qif:Spline12Core/qif:Orders', int)
        if len(self.orders) == 1:
            self.coefficients = [xml_np_array(xml, './qif:Spline12Core/qif:Coefficients').reshape(-1, 2), ]
        else:
            self.coefficients = np.split(xml_np_array(xml, './qif:Spline12Core/qif:Coefficients').reshape(-1, 2), np.cumsum(self.orders)[:-1])
        coeff = np.split(xml_np_array(xml, './qif:Spline12Core/qif:Coefficients').reshape(-1, 2), np.cumsum(self.orders)[:-1])
        self.coef = np.flip(np.array([np.pad(co, {0: (0,int(max(self.orders) - len(co)))}) for co in coeff]).swapaxes(0,1), axis=0)
        self.degree = self.orders - 1
    
    def values(self, t):
        if self.normalized:
            knots = np.arange(len(self.knots))
            k_diff = np.diff(self.knots)
            ti = np.searchsorted(self.knots[1:-1],t)
            t = ti + (t - self.knots[ti])/k_diff[ti]
        else:
            knots = self.knots
        return PPoly(self.coef, knots)(t)    

class Nurbs12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.order = xml_float(xml, './qif:Nurbs12Core/qif:Order')
        self.knots = xml_np_array(xml, './qif:Nurbs12Core/qif:Knots')
        if xml.find('./qif:Nurbs12Core/qif:CPs', xml_namespace(xml)) is not None:
            cps_array = xml_np_array(xml, './qif:Nurbs12Core/qif:CPs')
        else:
            cps_array = xml_np_binary_array(xml, './qif:Nurbs12Core/qif:CPsBinary')
        self.cps = cps_array.reshape(-1, 2)

        if xml.find('./qif:Nurbs12Core/qif:Weights', xml_namespace(xml)) is not None:
            self.weights = xml_np_array(xml, './qif:Nurbs12Core/qif:Weights')
        else:
            self.weights = np.full(self.cps.shape[0], 1)
        self.degree = int(self.order - 1)
    
    def values(self, t):
        b_splines = BSpline.design_matrix(t, self.knots, self.degree, extrapolate=True).toarray()
        return np.sum((b_splines * self.weights)[:,:,np.newaxis] * self.cps, axis=1) / np.sum(b_splines * self.weights, axis=1)[:,np.newaxis]   

class Aggregate12(Curve12):
    def __init__(self, xml):
        super().__init__(xml)
        self.sub_curves = []
        self.turned = []
        self.t_begin = [0]

        xml_sub_curves = xml.find('./qif:Aggregate12Core/qif:SubCurves', xml_namespace(xml))
        for sc in xml_sub_curves:
            core_tag = sc.find('./*').tag.split('}')[1][:-4] #drop namespace and 'Core'
            curve = get[core_tag](sc)
            self.sub_curves.append(curve)
            self.turned.append(xml_attr_bool(xml, './', 'turned', False))
            self.t_begin.append((curve.domain[1] - curve.domain[0]) + self.t_begin[-1])
    
    def values(self, t):
        results = np.empty((len(t),2))
        bins = self.t_begin[1:-1] #drop ends
        curves = np.searchsorted(bins, t)
        for i in np.unique(curves):
            if not self.turned[i]:
                t_sub = self.sub_curves[i].domain[0] + (t[i == curves] - self.t_begin[i])
            else:
                t_sub = self.sub_curves[i].domain[1] - (t[i == curves] - self.t_begin[i])
            results[i == curves] = self.sub_curves[i].values(t_sub)
        return results
    
get = {
'Segment12': Segment12,
'Polyline12': Polyline12,
'ArcCircular12': ArcCircular12,
'ArcConic12': ArcConic12,
'Spline12': Spline12,
'Nurbs12': Nurbs12,
'Aggregate12' : Aggregate12
}