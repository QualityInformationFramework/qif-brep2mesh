import xml.etree.ElementTree as ET
from .geometry import curve12, curve13, point, surface, topology
from .geometry.utils import xml_namespace

def _require_element(parent, path, ns, file_name, label):
    elem = parent.find(path, ns)
    if elem is None:
        raise ValueError(f"Invalid QIF '{file_name}': missing required element {label} ({path}).")
    return elem


def _require_text(parent, path, ns, file_name, label):
    elem = _require_element(parent, path, ns, file_name, label)
    text = elem.text.strip() if elem.text else ""
    if not text:
        raise ValueError(f"Invalid QIF '{file_name}': empty required text for {label} ({path}).")
    return text


def _require_id(elem, file_name, label):
    elem_id = elem.get('id')
    if elem_id is None or not elem_id.strip():
        raise ValueError(f"Invalid QIF '{file_name}': missing or empty @id on {label}.")
    return elem_id


def _local_tag(xml_elem):
    return xml_elem.tag.split('}')[-1]


def _collect_ids(xml_set, file_name, set_label):
    ids = set()
    for child in xml_set:
        elem_id = _require_id(child, file_name, f"{set_label}/{_local_tag(child)}")
        if elem_id in ids:
            raise ValueError(f"Invalid QIF '{file_name}': duplicate id='{elem_id}' in {set_label}.")
        ids.add(elem_id)
    return ids


def _validate_references(
    file_name,
    ns,
    xml_vertex_set,
    xml_edge_set,
    xml_loop_set,
    xml_face_set,
    point_ids,
    curve12_ids,
    curve13_ids,
    surface_ids,
    vertex_ids,
    edge_ids,
    loop_ids
):
    for vertex in xml_vertex_set:
        vertex_id = _require_id(vertex, file_name, "Vertex")
        point_id = _require_text(vertex, './qif:Point/qif:Id', ns, file_name, f"Vertex id={vertex_id} Point/Id")
        if point_id not in point_ids:
            raise ValueError(f"Invalid QIF '{file_name}': Vertex id={vertex_id} references missing Point id={point_id}.")

    for edge in xml_edge_set:
        edge_id = _require_id(edge, file_name, "Edge")
        curve_id = _require_text(edge, './qif:Curve/qif:Id', ns, file_name, f"Edge id={edge_id} Curve/Id")
        if curve_id not in curve13_ids:
            raise ValueError(f"Invalid QIF '{file_name}': Edge id={edge_id} references missing Curve13 id={curve_id}.")
        vertex_beg = _require_text(edge, './qif:VertexBeg/qif:Id', ns, file_name, f"Edge id={edge_id} VertexBeg/Id")
        if vertex_beg not in vertex_ids:
            raise ValueError(f"Invalid QIF '{file_name}': Edge id={edge_id} references missing VertexBeg id={vertex_beg}.")
        vertex_end = _require_text(edge, './qif:VertexEnd/qif:Id', ns, file_name, f"Edge id={edge_id} VertexEnd/Id")
        if vertex_end not in vertex_ids:
            raise ValueError(f"Invalid QIF '{file_name}': Edge id={edge_id} references missing VertexEnd id={vertex_end}.")

    for loop in xml_loop_set:
        loop_id = _require_id(loop, file_name, "Loop")
        co_edges = _require_element(loop, './qif:CoEdges', ns, file_name, f"Loop id={loop_id} CoEdges")
        for co_edge in co_edges:
            edge_ref = _require_text(co_edge, './qif:EdgeOriented/qif:Id', ns, file_name, f"Loop id={loop_id} CoEdge EdgeOriented/Id")
            if edge_ref not in edge_ids:
                raise ValueError(f"Invalid QIF '{file_name}': Loop id={loop_id} references missing Edge id={edge_ref}.")
            # A CoEdge without a Curve12 is not supported. Reconstructing uv
            # coordinates from the 3d edge is not implemented: a
            # loop mixing present and absent Curve12 raises in Loop.build, and a
            # loop with none at all reaches triangle with degenerate input and
            # segfaults. Reject it here, where we can still say why.
            curve12_elem = co_edge.find('./qif:Curve12/qif:Id', ns)
            if curve12_elem is None:
                raise ValueError(
                    f"Unsupported QIF '{file_name}': Loop id={loop_id} has a CoEdge with no Curve12. "
                    f"Tessellating edges without 2d parameter curves is not implemented.")
            curve12_ref = curve12_elem.text.strip() if curve12_elem.text else ""
            if not curve12_ref:
                raise ValueError(f"Invalid QIF '{file_name}': Loop id={loop_id} has empty CoEdge Curve12/Id.")
            if curve12_ref not in curve12_ids:
                raise ValueError(f"Invalid QIF '{file_name}': Loop id={loop_id} references missing Curve12 id={curve12_ref}.")

    for face in xml_face_set:
        face_id = _require_id(face, file_name, "Face")
        surface_ref = _require_text(face, './qif:Surface/qif:Id', ns, file_name, f"Face id={face_id} Surface/Id")
        if surface_ref not in surface_ids:
            raise ValueError(f"Invalid QIF '{file_name}': Face id={face_id} references missing Surface id={surface_ref}.")
        # LoopIds is optional: a face without loops covers its surface's full
        # parameter domain and is tessellated on a plain uv grid.
        loop_ids_elem = face.find('./qif:LoopIds', ns)
        if loop_ids_elem is None:
            continue
        for loop_ref_elem in loop_ids_elem:
            loop_ref = loop_ref_elem.text.strip() if loop_ref_elem.text else ""
            if not loop_ref:
                raise ValueError(f"Invalid QIF '{file_name}': Face id={face_id} has empty LoopIds/Id.")
            if loop_ref not in loop_ids:
                raise ValueError(f"Invalid QIF '{file_name}': Face id={face_id} references missing Loop id={loop_ref}.")


def read_qif(file_name):
    qifdoc = {"filename": file_name}

    tree = ET.parse(file_name)
    root = tree.getroot()
    ns = xml_namespace(root)
    qifdoc['xml_tree'] = tree
    qifdoc['xml_root'] = root

    qifdoc['qpid'] = _require_text(root, './qif:QPId', ns, file_name, 'QPId')
    qifdoc['linear_unit'] = _require_text(
        root, './qif:FileUnits/qif:PrimaryUnits/qif:LinearUnit/qif:UnitName', ns,
        file_name, 'FileUnits/PrimaryUnits/LinearUnit/UnitName')

    product = _require_element(root, './qif:Product', ns, file_name, 'Product')
    geometry_set = _require_element(product, './qif:GeometrySet', ns, file_name, 'Product/GeometrySet')
    topology_set = _require_element(product, './qif:TopologySet', ns, file_name, 'Product/TopologySet')

    xml_point_set = _require_element(geometry_set, './qif:PointSet', ns, file_name, 'PointSet')
    xml_curve12_set = _require_element(geometry_set, './qif:Curve12Set', ns, file_name, 'Curve12Set')
    xml_curve13_set = _require_element(geometry_set, './qif:Curve13Set', ns, file_name, 'Curve13Set')
    xml_surface_set = _require_element(geometry_set, './qif:SurfaceSet', ns, file_name, 'SurfaceSet')

    xml_vertex_set = _require_element(topology_set, './qif:VertexSet', ns, file_name, 'VertexSet')
    xml_edge_set = _require_element(topology_set, './qif:EdgeSet', ns, file_name, 'EdgeSet')
    xml_loop_set = _require_element(topology_set, './qif:LoopSet', ns, file_name, 'LoopSet')
    xml_face_set = _require_element(topology_set, './qif:FaceSet', ns, file_name, 'FaceSet')

    point_ids = _collect_ids(xml_point_set, file_name, 'PointSet')
    curve12_ids = _collect_ids(xml_curve12_set, file_name, 'Curve12Set')
    curve13_ids = _collect_ids(xml_curve13_set, file_name, 'Curve13Set')
    surface_ids = _collect_ids(xml_surface_set, file_name, 'SurfaceSet')
    vertex_ids = _collect_ids(xml_vertex_set, file_name, 'VertexSet')
    edge_ids = _collect_ids(xml_edge_set, file_name, 'EdgeSet')
    loop_ids = _collect_ids(xml_loop_set, file_name, 'LoopSet')
    _collect_ids(xml_face_set, file_name, 'FaceSet')

    _validate_references(
        file_name,
        ns,
        xml_vertex_set,
        xml_edge_set,
        xml_loop_set,
        xml_face_set,
        point_ids,
        curve12_ids,
        curve13_ids,
        surface_ids,
        vertex_ids,
        edge_ids,
        loop_ids
    )

    qifdoc['points'] = point.Points(xml_point_set)

    qifdoc['curves12'] = {}
    for child in xml_curve12_set:
        tag = _local_tag(child)
        if tag not in curve12.get:
            raise ValueError(f"Invalid QIF '{file_name}': unsupported Curve12 type '{tag}' (id={child.get('id')}).")
        elem_id = _require_id(child, file_name, f"Curve12Set/{tag}")
        qifdoc['curves12'][elem_id] = curve12.get[tag](child)

    qifdoc['curves13'] = {}
    for child in xml_curve13_set:
        tag = _local_tag(child)
        if tag not in curve13.get:
            raise ValueError(f"Invalid QIF '{file_name}': unsupported Curve13 type '{tag}' (id={child.get('id')}).")
        elem_id = _require_id(child, file_name, f"Curve13Set/{tag}")
        qifdoc['curves13'][elem_id] = curve13.get[tag](child)

    qifdoc['surfaces'] = {}
    for child in xml_surface_set:
        tag = _local_tag(child)
        if tag not in surface.get:
            raise ValueError(f"Invalid QIF '{file_name}': unsupported Surface type '{tag}' (id={child.get('id')}).")
        elem_id = _require_id(child, file_name, f"SurfaceSet/{tag}")
        qifdoc['surfaces'][elem_id] = surface.get[tag](child)

    qifdoc['vertices'] = {}
    for child in xml_vertex_set:
        elem_id = _require_id(child, file_name, "Vertex")
        qifdoc['vertices'][elem_id] = topology.Vertex(qifdoc, child)

    qifdoc['edges'] = {}
    for child in xml_edge_set:
        elem_id = _require_id(child, file_name, "Edge")
        qifdoc['edges'][elem_id] = topology.Edge(qifdoc, child)

    qifdoc['loops'] = {}
    for child in xml_loop_set:
        elem_id = _require_id(child, file_name, "Loop")
        qifdoc['loops'][elem_id] = topology.Loop(qifdoc, child)

    qifdoc['faces'] = {}
    for child in xml_face_set:
        elem_id = _require_id(child, file_name, "Face")
        qifdoc['faces'][elem_id] = topology.Face(qifdoc, child)

    return qifdoc
