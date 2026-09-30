import xml.etree.ElementTree as ET
import numpy as np
import triangle as tr
from .utils import *


def build_qif(qifdoc, lod):
    construct = {'points' : [],'edges':[],'loops':[],'faces':[]}
    for id in qifdoc['vertices']:
        qifdoc['vertices'][id].build(construct) 
    for id in qifdoc['edges']:
        qifdoc['edges'][id].build(construct, lod) 
    for id in qifdoc['faces']:
        qifdoc['faces'][id].build(construct, lod) 
    return construct

class Vertex:
    def __init__(self, qifdoc, xml):
        self.id = int(xml.get('id'))
        point_id = int(xml.find('./qif:Point/qif:Id', xml_namespace(xml)).text)
        self.point = qifdoc['points'][point_id]
    
    def build(self, construct):
        construct['points'].append(self.point)
        self.index =  len(construct['points']) - 1

class Edge:
    def __init__(self, qifdoc, xml):
        ns = xml_namespace(xml)
        self.id = int(xml.get('id'))
        curve_id = xml.find('./qif:Curve/qif:Id', ns).text
        self.curve = qifdoc['curves13'][curve_id]
        vertex_beg_id = xml.find('./qif:VertexBeg/qif:Id', ns).text
        self.vertex_beg = qifdoc['vertices'][vertex_beg_id]
        vertex_end_id = xml.find('./qif:VertexEnd/qif:Id', ns).text
        self.vertex_end = qifdoc['vertices'][vertex_end_id]
        self.min_loops = float('inf')

    def set_min_loops(self, loop_count):
        self.min_loops = min(self.min_loops, loop_count)

    def build(self, construct, lod):
        points = self.curve.divide(lod, self.min_loops)
        points = points[1:-1,:] # the vertices are added separately
        v_list = [self.vertex_beg.index]
        v_list.extend(list(range(len(construct['points']), len(construct['points']) + len(points))))
        construct['points'].extend(points)
        v_list.append(self.vertex_end.index)
        construct['edges'].append(v_list)      
        self.index = len(construct['edges']) - 1

class Loop:
    def __init__(self, qifdoc, xml):
        ns = xml_namespace(xml)
        self.id = int(xml.get('id'))
        self.form = xml.get('form')
        co_edges = xml.find('./qif:CoEdges', ns)
        loop_count = int(co_edges.get('n'))
        self.edge = []
        self.edge_turned = []
        self.curve12 = []
        for co_edge in co_edges:
            edge_id = co_edge.find('./qif:EdgeOriented/qif:Id', ns).text
            self.edge.append(qifdoc['edges'][edge_id])
            self.edge_turned.append(xml_attr_bool(co_edge, './qif:EdgeOriented', 'turned', 'false'))
            qifdoc['edges'][edge_id].set_min_loops(loop_count)
            curve12_element = co_edge.find('./qif:Curve12/qif:Id', ns)
            if curve12_element is None:
                self.curve12.append(None)
            else:
                curve12_id = co_edge.find('./qif:Curve12/qif:Id', ns).text
                self.curve12.append(qifdoc['curves12'][curve12_id])

    def build(self, construct, lod):
        loop = next((i for i, item in enumerate(construct['loops']) if item["id"] == self.id), None)
        if loop is not None:
            return loop
        loop_point_ids = []
        loop2d = []
        for i in range(len(self.edge)):
            edge_id = self.edge[i].index
            edge_point_ids = construct['edges'][edge_id].copy()
            if self.edge_turned[i]:
                edge_point_ids.reverse()
            loop_point_ids.append(edge_point_ids[:-1])
            if self.curve12[i] is None:
                loop2d = None 
            else:
                loop2d.append(self.curve12[i].divide_by(len(edge_point_ids) * 4)[:-1])
        construct['loops'].append({'id':self.id, 'points': loop_point_ids, 'points2d' : loop2d})
        return len(construct['loops']) - 1

class Face:
    def __init__(self, qifdoc, xml):
        ns = xml_namespace(xml)
        self.id = int(xml.get('id'))
        surface_id = xml.find('./qif:Surface/qif:Id', ns).text
        self.surface = qifdoc['surfaces'][surface_id]
        loop_ids = xml.find('./qif:LoopIds', ns)
        self.loops = False
        if loop_ids is not None:
            self.loops = []
            for loop_id in loop_ids:
                self.loops.append(qifdoc['loops'][loop_id.text])

    def build(self, construct, lod):
        points = []
        points2d = []
        segments = []
        holes = []
        if self.loops:
            for loop in self.loops:
                p_start = len(points)
                loop_id = loop.build(construct, lod)
                loop_points = [point_id for points in construct['loops'][loop_id]['points'] for point_id in points]
                p_count = len(loop_points)
                points += loop_points
                loop2d = []
                for edge_id in range(len(construct['loops'][loop_id]['points'])):
                    edge_pt = [construct['points'][point] for point in construct['loops'][loop_id]['points'][edge_id]]
                    if construct['loops'][loop_id]['points2d'] is not None:
                        curve_pt = self.surface.values(construct['loops'][loop_id]['points2d'][edge_id])
                        last_id = 0
                        curve2d = [0]
                        for point_id in range(1, len(edge_pt)):
                            remaining_points = -(len(edge_pt) - point_id -1)
                            if remaining_points == 0: remaining_points = None
                            last_id = np.argmin(np.linalg.norm(edge_pt[point_id]-curve_pt[last_id+1:remaining_points], axis=1)) + last_id + 1
                            curve2d.append(last_id)
                        loop2d += np.array(construct['loops'][loop_id]['points2d'][edge_id])[curve2d].tolist()
                    else:
                        # No Curve12: project the 3d edge points onto the surface. Not reached
                        # in practice, because read_qif rejects CoEdges without a Curve12.
                        edge_points = [construct['points'][point] for edges in construct['loops'][loop_id]['points'] for point in edges]
                        coord2d = self.surface.get_coords(np.array(edge_points))
                        loop2d += coord2d.tolist()
                points2d += loop2d
                segments += [[x, x+1] for x in range(p_start, p_start + p_count)]
                segments[-1][1] = p_start
                if loop.form == 'INNER': # triangle needs a point inside each hole
                    pts = np.array(loop2d)
                    yavg = np.mean(pts[:,1])
                    yuni = np.unique(pts[:,1])
                    yind = np.searchsorted(yuni, yavg)
                    if yind + 1 == len(yuni): # keep yind+2 in range for the midpoint below
                        yind -= 1
                    ycross = np.mean(yuni[yind:yind+2])
                    cross = []
                    for i in range(len(pts)):
                        i_next = i + 1
                        if i_next == len(pts):
                            i_next = 0
                        if (ycross > pts[i][1] and ycross >= pts[i_next][1]) or (ycross < pts[i][1] and ycross <= pts[i_next][1]):
                            continue
                        cross.append(pts[i][0] + (pts[i_next][0] - pts[i][0])*(ycross - pts[i][1])/(pts[i_next][1] - pts[i][1]))
                    cross.sort()
                    holes.append([(cross[1] + cross[0])/2, ycross])
            scale = self.surface.scale()
            scaled_up = np.array(points2d) * scale
            if len(holes):
                holes = np.array(holes) * scale
                t = tr.triangulate({'vertices': scaled_up, 'segments': segments, 'holes':holes}, f'YYpqa{(lod**2)/2}')
            else:
                t = tr.triangulate({'vertices': scaled_up, 'segments': segments}, f'YYpqa{(lod**2)/2}')
            scaled_down =  t['vertices'] / scale
        else:
            scale = self.surface.scale()
            u_step_count = int(np.ceil(((self.surface.domain_u[1] - self.surface.domain_u[0]) * scale[0]) / lod)) + 1
            v_step_count = int(np.ceil(((self.surface.domain_v[1] - self.surface.domain_v[0]) * scale[1]) / lod)) + 1
            u_values = np.linspace(self.surface.domain_u[0], self.surface.domain_u[1], u_step_count)
            v_values = np.linspace(self.surface.domain_v[0], self.surface.domain_v[1], v_step_count)
            scaled_down = np.stack(np.meshgrid(u_values, v_values), -1).reshape(-1,2)
            u_len = len(u_values)
            t = {'triangles' : []}
            for u in range(u_len - 1):
                for v in range(len(v_values) - 1):
                    t['triangles'].append([u + v * u_len, u + v * u_len + 1, u + (v + 1) * u_len])
                    t['triangles'].append([u + v * u_len + 1, u + (v + 1) * u_len + 1, u + (v + 1) * u_len])
        if(len(points2d) < len(scaled_down)):
            new_points = self.surface.values(scaled_down[len(points2d):])
            construct['points'].extend(new_points)
            points.extend(range(len(construct['points']) - len(new_points), len(construct['points'])))
        triangles = np.vectorize(points.__getitem__)(t['triangles'])
        construct['faces'].append({"id": self.id, "triangles": triangles})
