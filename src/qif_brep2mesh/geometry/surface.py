import numpy as np
from .utils import *
from . import curve13
from scipy.interpolate import NdPPoly, BSpline
from scipy.optimize import minimize

class Surface:
    def __init__(self, xml):
        self.id = xml.get('id')

    def normal(self, u, v):
        u = min(max(u, self.domain_u[0]), self.domain_u[1])
        v = min(max(v, self.domain_v[0]), self.domain_v[1])
        u_step = (self.domain_u[1] - self.domain_u[0]) / 1000
        u_lower = self.value(max(u - u_step, self.domain_u[0]), v)
        u_upper = self.value(min(u + u_step, self.domain_u[1]), v)
        u_dir = u_upper - u_lower
        v_step = (self.domain_v[1] - self.domain_v[0]) / 1000
        v_lower = self.value(u, max(v - v_step, self.domain_v[0]))
        v_upper = self.value(u, min(v + v_step, self.domain_v[1]))
        v_dir = v_upper - v_lower
        return np.cross(u_dir, v_dir) / np.linalg.norm(np.cross(u_dir, v_dir))
    
    def normals(self, points):
        points[:,0] = np.clip(points[:,0], self.domain_u[0], self.domain_u[1])
        points[:,1] = np.clip(points[:,1], self.domain_v[0], self.domain_v[1])

        u_step = (self.domain_u[1] - self.domain_u[0]) / 1000
        u_lower = points - [u_step,0]
        u_lower[:,0] = np.clip(u_lower[:,0], self.domain_u[0], self.domain_u[1])
        u_upper = points + [u_step,0]
        u_upper[:,0] = np.clip(u_upper[:,0], self.domain_u[0], self.domain_u[1])
        u_dir = self.values(u_upper) - self.values(u_lower)

        v_step = (self.domain_v[1] - self.domain_v[0]) / 1000
        v_lower = points - [0, v_step]
        v_lower[:,0] = np.clip(v_lower[:,0], self.domain_v[0], self.domain_v[1])
        v_upper = points + [0, v_step]
        v_upper[:,0] = np.clip(v_upper[:,0], self.domain_v[0], self.domain_v[1])
        v_dir = self.values(v_upper) - self.values(v_lower)

        return np.cross(u_dir, v_dir) / np.linalg.norm(np.cross(u_dir, v_dir), axis=1)[...,None]
    
    def value(self, point):
        return self.values(np.array([point,]))[0]    
    
    def get_coord(self, value, start_point = None):
        if start_point is None:
            u = np.linspace(self.domain_u[0], self.domain_u[1], 7)
            v = np.linspace(self.domain_v[0], self.domain_v[1], 7)
            U, V = np.meshgrid(u, v)
            grid = np.column_stack((U.ravel(), V.ravel()))
            grid_values = np.sum((self.values(grid) - value)**2, axis=1)
            start_point = grid[np.argmin(grid_values)]
            
        def min_value(point2d, find_point):
                return np.sum((self.value(point2d) - find_point)**2)
        return minimize(min_value, start_point, value, bounds=(self.domain_u, self.domain_v)).x
    
    def make_grid(self, dim):
        u = np.linspace(self.domain_u[0], self.domain_u[1], dim)
        v = np.linspace(self.domain_v[0], self.domain_v[1], dim)
        U, V = np.meshgrid(u, v)
        grid = np.column_stack((U.ravel(), V.ravel()))
        values = self.values(grid)
        return grid, values

    def get_coords(self, values):
        grid_c, grid_v = self.make_grid(9)

        # distance from each grid point to its nearest neighbor
        diff = grid_v[:, None, :] - grid_v[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.min(dist_sq, axis=1)

        # closest grid point for each value
        val_diff = values[:, None, :] - grid_v[None, :, :]
        val_dist_sq = np.sum(val_diff ** 2, axis=-1)
        closest_point = np.argmin(val_dist_sq, axis=1)

        def min_value(point2d, find_point):
            return np.sum((self.value(point2d) - find_point)**2)
        
        coords = np.empty((len(values), 2))

        if np.all(min_dist[closest_point] > 0) and np.max(min_dist[closest_point]) / np.min(min_dist[closest_point]) <= 2:  # grid points are distinct and evenly spread
            current_point = grid_c[closest_point[0]]
            for i in np.arange(len(values)):
                current_point = minimize(min_value, current_point, values[i], bounds=(self.domain_u, self.domain_v)).x
                coords[i] = current_point
            return coords

        solved = np.zeros(len(values))
        next_index = np.roll(np.arange(len(solved)), -1)
        last_index = np.roll(np.arange(len(solved)), 1)

        max_index = np.nonzero(np.max(min_dist[closest_point]) == min_dist[closest_point])[0]
        for i in max_index:
            coords[i] = minimize(min_value, grid_c[closest_point[i]], values[i], bounds=(self.domain_u, self.domain_v)).x
            solved[i] = 1

        while np.any(solved == 0):
            next_list = np.nonzero((solved == 0) & (solved[next_index] != 0))[0]
            for i in next_list:
                coords[i] = minimize(min_value, coords[next_index[i]], values[i], bounds=(self.domain_u, self.domain_v)).x
                solved[i] = 1
            last_list = np.nonzero((solved == 0) & (solved[last_index] != 0))[0]
            for i in last_list:
                coords[i] = minimize(min_value, coords[last_index[i]], values[i], bounds=(self.domain_u, self.domain_v)).x
                solved[i] = 1

        return coords



class Plane23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.domain_u = xml_attr_float_array(xml, './qif:Plane23Core','domainU')
        self.domain_v = xml_attr_float_array(xml, './qif:Plane23Core','domainV')
        self.origin = xml_np_array(xml, './qif:Plane23Core/qif:Origin')
        self.dir_u = xml_np_array(xml, './qif:Plane23Core/qif:DirU')
        self.dir_v = xml_np_array(xml, './qif:Plane23Core/qif:DirV')
        self.scale_u = np.linalg.norm(self.dir_u)
        self.scale_v = np.linalg.norm(self.dir_v)
        self.direction = np.array([self.dir_u, self.dir_v])

    def scale(self):
        return [self.scale_u, self.scale_v]

    def values(self, points):
        return self.origin + (points @ self.direction)
    
class Cylinder23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned_v = xml_attr_bool(xml, './qif:Cylinder23Core', 'turnedV', False)
        self.scale_u = xml_attr_float(xml, './qif:Cylinder23Core','scaleU', 1) 
        self.scale_v = xml_attr_float(xml, './qif:Cylinder23Core','scaleV', 1)
        self.diameter = xml_float(xml, './qif:Cylinder23Core/qif:Diameter')
        self.length = xml_float(xml, './qif:Cylinder23Core/qif:Length')
        self.axis_point = xml_np_array(xml, './qif:Cylinder23Core/qif:Axis/qif:AxisPoint')
        self.direction = xml_np_array(xml, './qif:Cylinder23Core/qif:Axis/qif:Direction')
        self.dir_beg = xml_np_array(xml, './qif:Cylinder23Core/qif:Sweep/qif:DirBeg')
        self.domain_angle = xml_np_array(xml, './qif:Cylinder23Core/qif:Sweep/qif:DomainAngle')

        self.domain_u = self.domain_angle / self.scale_u
        self.domain_v = [0, self.length / self.scale_v]
        self.radius = self.diameter / 2
        self.dir_y = np.cross(self.direction, self.dir_beg)

    def scale(self):
        return [self.scale_u * self.radius , self.scale_v]

    def values(self, points):
        points = points * [self.scale_u, self.scale_v]
        if self.turned_v:
            points[:,1] = self.length - points[:,1]
        return self.axis_point + self.radius * (np.cos(points[:,0,None]) * self.dir_beg + np.sin(points[:,0,None]) * self.dir_y) + points[:,1,None] * self.direction

class Cone23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned_v = xml_attr(xml, './qif:Cone23Core', 'turnedV', 'false') == 'true'
        self.scale_u = xml_attr_float(xml, './qif:Cone23Core','scaleU', 1)
        self.scale_v = xml_attr_float(xml, './qif:Cone23Core','scaleV', 1)
        self.diameter_bottom = xml_float(xml, './qif:Cone23Core/qif:DiameterBottom')
        self.diameter_top = xml_float(xml, './qif:Cone23Core/qif:DiameterTop')
        self.length = xml_float(xml, './qif:Cone23Core/qif:Length')
        self.axis_point = xml_np_array(xml, './qif:Cone23Core/qif:Axis/qif:AxisPoint')
        self.direction = xml_np_array(xml, './qif:Cone23Core/qif:Axis/qif:Direction')
        self.dir_beg = xml_np_array(xml, './qif:Cone23Core/qif:Sweep/qif:DirBeg')
        self.domain_angle = xml_np_array(xml, './qif:Cone23Core/qif:Sweep/qif:DomainAngle')

        self.domain_u = self.domain_angle / self.scale_u
        self.domain_v = [0, self.length / self.scale_v]
        self.radius_top = self.diameter_top / 2
        self.radius_bottom = self.diameter_bottom / 2        
        self.dir_y = np.cross(self.direction, self.dir_beg)

    def scale(self):
        return [self.scale_u * (self.radius_top + self.radius_bottom) / 2 , self.scale_v]
    
    def values(self, points):
        points = points * [self.scale_u, self.scale_v]
        if self.turned_v:
            points[:,1] = self.length - points[:,1]
        radius = self.radius_bottom + points[:,1,None] * (self.radius_top - self.radius_bottom) / self.length
        return self.axis_point + radius * (np.cos(points[:,0,None]) * self.dir_beg + np.sin(points[:,0,None]) * self.dir_y) + points[:,1,None] * self.direction

class Sphere23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned_v = xml_attr(xml, './qif:Sphere23Core', 'turnedV', 'false') == 'true'
        self.scale_u = xml_attr_float(xml, './qif:Sphere23Core','scaleU', 1)
        self.scale_v = xml_attr_float(xml, './qif:Sphere23Core','scaleV', 1)
        self.diameter = xml_float(xml, './qif:Sphere23Core/qif:Diameter')
        self.location = xml_np_array(xml, './qif:Sphere23Core/qif:Location')
        self.dir_mer_pri = xml_np_array(xml, './qif:Sphere23Core/qif:LatitudeLongitudeSweep/qif:DirMeridianPrime')
        self.domain_lat = xml_np_array(xml, './qif:Sphere23Core/qif:LatitudeLongitudeSweep/qif:DomainLatitude')
        self.domain_long= xml_np_array(xml, './qif:Sphere23Core/qif:LatitudeLongitudeSweep/qif:DomainLongitude')
        self.dir_north_pole = xml_np_array(xml, './qif:Sphere23Core/qif:LatitudeLongitudeSweep/qif:DirNorthPole')

        self.domain_u = self.domain_long / self.scale_u
        self.domain_v = self.domain_lat / self.scale_v 
        self.radius = self.diameter / 2
        self.dir_y = np.cross(self.dir_north_pole, self.dir_mer_pri)

    def scale(self):
        return [self.scale_u * self.radius , self.scale_v * self.radius]
    
    def values(self, points):
        points = points * [self.scale_u, self.scale_v]
        if self.turned_v:
            points[:,1] = - points[:,1]
        return self.location + self.radius * np.sin(points[:,1,None]) * self.dir_north_pole + \
        self.radius * np.cos(points[:,1,None]) * (np.cos(points[:,0,None]) * self.dir_mer_pri + np.sin(points[:,0,None]) * self.dir_y)

class Torus23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned_v = xml_attr_bool(xml, './qif:Torus23Core', 'turnedV', False)
        self.offset_v = xml_attr_float(xml, './qif:Torus23Core', 'offsetV', 0)
        self.scale_u = xml_attr_float(xml, './qif:Torus23Core','scaleU', 1)
        self.scale_v = xml_attr_float(xml, './qif:Torus23Core','scaleV', 1)
        self.diameter_minor = xml_float(xml, './qif:Torus23Core/qif:DiameterMinor')
        self.diameter_major = xml_float(xml, './qif:Torus23Core/qif:DiameterMajor')
        self.axis_point = xml_np_array(xml, './qif:Torus23Core/qif:Axis/qif:AxisPoint')
        self.direction = xml_np_array(xml, './qif:Torus23Core/qif:Axis/qif:Direction')
        self.dir_mer_pri = xml_np_array(xml, './qif:Torus23Core/qif:LatitudeLongitudeSweep/qif:DirMeridianPrime')
        self.domain_lat = xml_np_array(xml, './qif:Torus23Core/qif:LatitudeLongitudeSweep/qif:DomainLatitude')
        self.domain_long= xml_np_array(xml, './qif:Torus23Core/qif:LatitudeLongitudeSweep/qif:DomainLongitude')

        self.domain_u = self.domain_long / self.scale_u
        self.domain_v = self.domain_lat / self.scale_v 
        self.radius_minor = self.diameter_minor / 2
        self.radius_major = self.diameter_major / 2
        self.dir_y = np.cross(self.direction, self.dir_mer_pri)

    def scale(self):
        return [self.scale_u * self.radius_major , self.scale_v * self.radius_minor]

    def values(self, points):
        points = points * [self.scale_u, self.scale_v]
        if self.turned_v:
            points[:,1] = self.offset_v - points[:,1]
        else:
            points[:,1] = self.offset_v + points[:,1]
        a = self.radius_minor * np.sin(points[:,1,None]) * self.direction
        r = self.radius_major + self.radius_minor * np.cos(points[:,1,None])
        return self.axis_point + a + r * (np.cos(points[:,0,None]) * self.dir_mer_pri + np.sin(points[:,0,None]) * self.dir_y)

class Extrude23(Surface):
    def __init__(self, xml):
        super().__init__(xml)    
        self.termination_point = xml_np_array(xml, './qif:Extrude23Core/qif:TerminationPoint')
        xml_curve = xml.find('./qif:Extrude23Core/qif:Curve', xml_namespace(xml))
        core_tag = xml_curve.find('./*').tag.split('}')[1][:-4]
        self.curve = curve13.get[core_tag](xml_curve)
        self.curve_start = self.curve.value(self.curve.domain[0])
        self.domain_u = self.curve.domain
        self.domain_v = [0, 1]

    def scale(self):
        return [self.curve.length() / (self.curve.domain[1] - self.curve.domain[0]) , 1]
    
    def values(self, points):
        return self.curve.values(points[:,0]) + (self.termination_point - self.curve_start) * points[:,1,None]

class Ruled23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.turned_second_curve = xml_attr_bool(xml, './qif:Ruled23Core', 'turnedSecondCurve', False)
        self.curve = []
        xml_curves = xml.find('./qif:Ruled23Core', xml_namespace(xml))
        for xml_curve in xml_curves:
            core_tag = xml_curve.find('./*').tag.split('}')[1][:-4]
            curve = curve13.get[core_tag](xml_curve)
            self.curve.append(curve)
        self.domain_u = [0, 1]
        self.domain_v = [0, 1]

    def scale(self):
        return [(self.curve[0].length() / (self.curve[0].domain[1] - self.curve[0].domain[0]) +
                self.curve[1].length() / (self.curve[1].domain[1] - self.curve[1].domain[0]))/2 \
                , 1]
    
    def values(self, points):
        t0 = self.curve[0].domain[0] + points[:,0] * (self.curve[0].domain[1] - self.curve[0].domain[0])
        if self.turned_second_curve:
            t1 = self.curve[1].domain[1] + points[:,0] * (self.curve[1].domain[0] - self.curve[1].domain[1])
        else:
            t1 = self.curve[1].domain[0] + points[:,0] * (self.curve[1].domain[1] - self.curve[1].domain[0])
        return self.curve[0].values(t0) * (1 - points[:,1,None]) + self.curve[1].values(t1) * points[:,1,None]

class Revolution23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.angle = xml_attr_float_array(xml, './qif:Revolution23Core', 'angle')
        self.axis_point = xml_np_array(xml, './qif:Revolution23Core/qif:Axis/qif:AxisPoint')
        self.direction = xml_np_array(xml, './qif:Revolution23Core/qif:Axis/qif:Direction')
        xml_generatrix = xml.find('./qif:Revolution23Core/qif:Generatrix', xml_namespace(xml))
        core_tag = xml_generatrix.find('./*').tag.split('}')[1][:-4]
        self.generatrix = curve13.get[core_tag](xml_generatrix)
        self.domain_u = self.generatrix.domain
        self.domain_v = self.angle

    def scale(self):
        radius_array = []
        for u in np.linspace(self.generatrix.domain[0], self.generatrix.domain[1], 5):
            generatrix_u = self.generatrix.value(u)
            p = self.axis_point + np.dot((generatrix_u - self.axis_point), self.direction) * self.direction 
            r = np.linalg.norm(generatrix_u - p)
            radius_array.append(r)
        radius = np.max(radius_array)
        return [self.generatrix.length() / (self.generatrix.domain[1] - self.generatrix.domain[0]),
                radius * (self.angle[1] - self.angle[0])]
    
    def values(self, points):
        generatrix_u = self.generatrix.values(points[:,0])
        p = self.axis_point + np.dot((generatrix_u - self.axis_point), self.direction)[:,None] * self.direction 
        r = np.linalg.norm(generatrix_u - p)
        dir_x = (generatrix_u - p)/np.linalg.norm(generatrix_u - p)
        dir_y = np.cross(self.direction, dir_x)
        return p + r * (np.cos(points[:,1,None]) * dir_x + np.sin(points[:,1,None]) * dir_y)

class Spline23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.normalized = xml_attr_bool(xml, './qif:Spline23Core','normalized',False)
        self.knots_u = xml_np_array(xml, './qif:Spline23Core/qif:KnotsU')
        self.knots_v = xml_np_array(xml, './qif:Spline23Core/qif:KnotsV')
        self.orders_u = xml_np_array(xml, './qif:Spline23Core/qif:OrdersU', int)
        self.orders_v = xml_np_array(xml, './qif:Spline23Core/qif:OrdersV', int)
        total_u = sum(self.orders_u)
        if len(self.orders_v) == 1:
            self.coefficients = [xml_np_array(xml, './qif:Spline23Core/qif:Coefficients').reshape(-1, 3), ]
        else:  
            self.coefficients = np.split(xml_np_array(xml, './qif:Spline23Core/qif:Coefficients').reshape(-1, 3),np.cumsum(self.orders_v[:-1])*total_u)
        for i in range(len(self.coefficients)):
            if len(self.orders_u) == 1:
                self.coefficients[i] = [self.coefficients[i],]
            else:
                self.coefficients[i] = np.split(self.coefficients[i], np.cumsum(self.orders_u)[:-1] * self.orders_v[i])
            for j in range(len(self.coefficients[i])):
                co = self.coefficients[i][j]
                co = co.reshape(self.orders_v[i], self.orders_u[j], 3)
                co = np.pad(co, {0: (0,int(max(self.orders_v) - co.shape[0])), 1: (0, int(max(self.orders_u) - co.shape[1]))})
                self.coefficients[i][j] = co
        self.coefficients = np.array(self.coefficients)

        self.coef = np.flip(np.transpose(self.coefficients, ( 3, 2, 1, 0, 4)),(0,1)) 
        if self.normalized:
            self.domain_u = [0, 1]
            self.domain_v = [0, 1]
        else:
            self.domain_u = [self.knots_u[0], self.knots_u[-1]]
            self.domain_v = [self.knots_v[0], self.knots_v[-1]]


    def scale(self):
        u_a = np.linspace(self.knots_u[0], self.knots_u[-1], 5)
        v_a = np.linspace(self.knots_v[0], self.knots_v[-1], 5)
        grid = np.meshgrid(u_a, v_a)
        points = np.array(grid).T.reshape(-1, 2)
        results = self.values(points)

        u_dist = np.mean(np.norm(results[1:,:] - results[:-1,:] , axis=0))
        v_dist = np.mean(np.norm(results[:,1:] - results[:,:]-1 , axis=1))
        return [u_dist/(self.knots_u[-1] - self.knots_u[0]), v_dist/(self.knots_v[-1] - self.knots_v[0])]

    def values(self, points):
        if self.normalized:
            knots_u = np.arange(len(self.knots_u))
            k_u_diff = np.diff(self.knots_u)
            ui = np.searchsorted(self.knots_u[1:-1],points[:,0])
            points[:,0] = ui + (points[:,0] - self.knots_u[ui])/k_u_diff[ui]

            knots_v = np.arange(len(self.knots_v))
            k_v_diff = np.diff(self.knots_v)
            vi = np.searchsorted(self.knots_v[1:-1],points[:,1])
            points[:,1] = vi + (points[:,1] - self.knots_v[vi])/k_v_diff[vi]

            knots = [knots_u, knots_v]         
        else:
            knots = [self.knots_u, self.knots_v]
        return NdPPoly(self.coef, knots)(points)    

class Nurbs23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.order_u = int(xml_float(xml, './qif:Nurbs23Core/qif:OrderU'))
        self.order_v = int(xml_float(xml, './qif:Nurbs23Core/qif:OrderV'))
        self.degree_u = int(self.order_u - 1)
        self.degree_v = int(self.order_v - 1)
        self.knots_u = xml_np_array(xml, './qif:Nurbs23Core/qif:KnotsU')
        self.knots_v = xml_np_array(xml, './qif:Nurbs23Core/qif:KnotsV')
        if xml.find('./qif:Nurbs23Core/qif:CPs', xml_namespace(xml)) is not None:
            cps_array = xml_np_array(xml, './qif:Nurbs23Core/qif:CPs')
        else:
            cps_array = xml_np_binary_array(xml, './qif:Nurbs23Core/qif:CPsBinary')
        self.cps = cps_array.reshape(len(self.knots_v) - self.order_v, len(self.knots_u) - self.order_u, 3)

        if xml.find('./qif:Nurbs23Core/qif:Weights', xml_namespace(xml)) is not None:
            self.weights = xml_np_array(xml, './qif:Nurbs23Core/qif:Weights').reshape(self.cps.shape[:-1])
        else:
            self.weights = np.full(self.cps.shape[:-1], 1)
        self.domain_u = [self.knots_u[0], self.knots_u[-1]]
        self.domain_v = [self.knots_v[0], self.knots_v[-1]]

    def scale(self):
        range_u = self.knots_u[-1] - self.knots_u[0]
        range_v = self.knots_v[-1] - self.knots_v[0]
        distance_u = np.sum(np.mean(np.linalg.norm(np.diff(self.cps,axis=1),axis=2),axis=0))
        distance_v = np.sum(np.mean(np.linalg.norm(np.diff(self.cps,axis=0),axis=2),axis=1))
        return [distance_u / range_u, distance_v / range_v]

    def values(self, points):
        b_splines_u = BSpline.design_matrix(points[:,0], self.knots_u, self.degree_u, extrapolate=True).toarray()
        b_splines_v = BSpline.design_matrix(points[:,1], self.knots_v, self.degree_v, extrapolate=True).toarray()
        b_splines = b_splines_u[:,None,:] *  b_splines_v[:,:,None]
        return np.sum((b_splines * self.weights)[:,:,:,np.newaxis] * self.cps, axis=(1,2)) \
            / np.sum(b_splines * self.weights, axis=(1,2))[:,None]

class Offset23(Surface):
    def __init__(self, xml):
        super().__init__(xml)
        self.distance = int(xml_float(xml, './qif:Offset23Core/qif:Distance'))
        xml_surface = xml.find('./qif:Offset23Core/qif:Surface', xml_namespace(xml))
        core_tag = xml_surface.find('./*').tag.split('}')[1][:-4]
        self.surface = get[core_tag](xml_surface)
        self.domain_u = self.surface.domain_u
        self.domain_v = self.surface.domain_v

    def scale(self):
        return self.surface.scale()
    
    def values(self, points):
        return self.surface.values(points) + self.surface.normals(points) * self.distance  


get = {
'Plane23': Plane23,
'Cylinder23': Cylinder23,
'Cone23': Cone23,
'Sphere23': Sphere23,
'Torus23': Torus23,
'Extrude23': Extrude23,
'Ruled23' : Ruled23,
'Revolution23' : Revolution23,
'Spline23' : Spline23,
'Nurbs23' : Nurbs23,
'Offset23' : Offset23
}