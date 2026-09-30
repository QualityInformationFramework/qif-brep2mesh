import xml.etree.ElementTree as ET
import numpy as np
from ..geometry.utils import xml_namespace


def _ns_helper(root):
    ns_uri = xml_namespace(root)['qif']
    return ns_uri, lambda name: f'{{{ns_uri}}}{name}'


def _array_text(array):
    flat = np.array(array).reshape(-1)
    return ' '.join(str(x) for x in flat)


def write_qif_mesh(qifdoc, face_meshes, output_path):
    """
    Modify the parsed QIF document to replace B-rep faces with mesh faces and write it to output_path.
    face_meshes: dict of face_id -> {'vertices': np.ndarray (n,3), 'triangles': np.ndarray (m,3)}
    """
    root = qifdoc['xml_root']
    tree = qifdoc['xml_tree']
    ns = xml_namespace(root)
    ns_uri, q = _ns_helper(root)

    geometry_set = root.find('.//qif:GeometrySet', ns)
    if geometry_set is None:
        raise ValueError("GeometrySet not found in QIF document.")

    surface_mesh_set = geometry_set.find('qif:SurfaceMeshSet', ns)
    if surface_mesh_set is None:
        surface_mesh_set = ET.SubElement(geometry_set, q('SurfaceMeshSet'))
    else:
        for child in list(surface_mesh_set):
            surface_mesh_set.remove(child)
    surface_mesh_set.set('n', str(len(face_meshes)))

    next_id = int(root.get('idMax', '0')) + 1
    mesh_id_map = {}
    for idx, (face_id, mesh_data) in enumerate(face_meshes.items()):
        mesh_id = next_id
        next_id += 1
        mesh_id_map[face_id] = mesh_id
        mesh_elem = ET.SubElement(surface_mesh_set, q('MeshTriangle'))
        mesh_elem.set('id', str(mesh_id))
        core = ET.SubElement(mesh_elem, q('MeshTriangleCore'))

        triangles_elem = ET.SubElement(core, q('Triangles'))
        triangles = np.array(mesh_data['triangles'], dtype=np.int64)
        triangles_elem.set('count', str(len(triangles)))
        triangles_elem.text = _array_text(triangles)

        vertices_elem = ET.SubElement(core, q('Vertices'))
        vertices = np.array(mesh_data['vertices'], dtype=np.float64)
        vertices_elem.set('count', str(len(vertices)))
        vertices_elem.text = _array_text(vertices)

    topology_set = root.find('.//qif:TopologySet', ns)
    if topology_set is None:
        raise ValueError("TopologySet not found in QIF document.")

    face_set = topology_set.find('qif:FaceSet', ns)
    if face_set is None:
        raise ValueError("FaceSet not found in QIF document.")

    for child in list(face_set):
        face_set.remove(child)
    face_set.set('n', str(len(face_meshes)))

    for face_id, mesh_data in face_meshes.items():
        face_mesh_elem = ET.SubElement(face_set, q('FaceMesh'))
        face_mesh_elem.set('id', str(int(face_id)))

        mesh_ref = ET.SubElement(face_mesh_elem, q('Mesh'))
        mesh_ref_id = ET.SubElement(mesh_ref, q('Id'))
        mesh_ref_id.text = str(mesh_id_map[face_id])

        triangles_elem = ET.SubElement(face_mesh_elem, q('Triangles'))
        tri_count = mesh_data['triangles'].shape[0]
        triangles_elem.set('count', str(tri_count))
        triangles_elem.text = ' '.join(str(i) for i in range(tri_count))

    ET.indent(tree, space="\t", level=0)
    root.set('idMax', str(max(next_id - 1, int(root.get('idMax', '0')))))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
