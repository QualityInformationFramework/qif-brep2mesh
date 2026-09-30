import xml.etree.ElementTree as ET
import base64
import numpy as np
import zlib
import lzma


def write_vtk(construct, file_name, metadata = False, compression = 'zlib', clevel = -1): 

    root = ET.Element("VTKFile")
    root.set("type", "PolyData")
    root.set("byte_order", "LittleEndian")

    if compression == 'zlib':
        format = 'binary'
        root.set("compressor", "vtkZLibDataCompressor")
    elif compression == 'lzma':
        format = 'binary'
        root.set("compressor", "vtkLZMADataCompressor")
    else:
        format = 'ascii'      
    
    pdata = ET.SubElement(root, "PolyData")

    if metadata:
        fdata = ET.SubElement(pdata, "FieldData")
        for key, value in metadata.items():
            fdataarray = ET.SubElement(fdata, "DataArray")
            fdataarray.set('type', 'String')
            fdataarray.set('name', key)
            fdataarray.set('format','ascii')
            fdataarray.text = str(value)

    piece = ET.SubElement(pdata, "Piece")
    piece.set("NumberOfPoints", str(len(construct['points'])))
    piece.set("NumberOfPolys", str(sum(([len(face['triangles']) for face in construct['faces']]))))

    points = ET.SubElement(piece, "Points")
    ptdata = create_data_array(
        np.array(construct['points']),
        {"NumberOfComponents": "3", "format": format}, compression, clevel)
    points.append(ptdata)

    polys = ET.SubElement(piece, "Polys")

    triangles = np.array([triangles for face in construct['faces'] for triangles in face['triangles']])
    pydatac = create_data_array(triangles,{"Name": "connectivity", "format": format}, compression, clevel)
    polys.append(pydatac)

    pydatao = create_data_array(
        np.arange(3, (len(triangles)+1) * 3, 3),
        {"Name": "offsets", "format": format}, compression, clevel)
    polys.append(pydatao)

    ids = [face['id'] for face in construct['faces']]
    tri_count = [len(face['triangles']) for face in construct['faces']]
    cellid = np.repeat(ids, tri_count)
    cell = ET.SubElement(piece, "CellData")
    celldata = create_data_array(cellid, {"Name": "qif_id", "format": format}, compression, clevel)
    cell.append(celldata)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(file_name, encoding="utf-8", xml_declaration=True)

def create_data_array(data, attr, compression = 'zlib', clevel = -1):
    type_map = {'float64': 'Float64', 'int64' : 'Int64'}
    # VTK types: Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64
    data_array = ET.Element("DataArray")
    for key, value in attr.items():
        data_array.set(key, value)
    data_array.set('type', type_map[data.dtype.name])
    if attr['format'] == 'binary':
        blocksize = data.nbytes
        if compression == 'lzma':
            data = lzma.compress(data)
        else:
            data = zlib.compress(data, clevel)
        header = np.array([1, blocksize, blocksize, len(data)], dtype=np.dtype('<u4'))
        data = base64.b64encode(header).decode() + base64.b64encode(data).decode()
    else:
        data = ' '.join([str(out) for out in np.array(data).flatten()])
    data_array.text = data
    return data_array