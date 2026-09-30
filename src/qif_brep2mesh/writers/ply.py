
import numpy as np

def ply_header(metadata, format, vertices, triangles):
    comments = ""
    if metadata:
        for key, value in metadata.items():
            comments += f"comment {key} = {value}\n"
    header = f"""ply
{format}
{comments}element vertex {vertices}
property float x
property float y
property float z
element face {triangles}
property list uchar uint vertex_indices
property uint qif_id
end_header
"""
    return header

def write_ply_ascii(construct, file_name, metadata = False): 

    tri_count = sum(len(face['triangles']) for face in construct['faces'])
    header = ply_header(metadata, "format ascii 1.0", len(construct['points']), tri_count)

    with open(file_name, "w") as file:
        
        file.write(header)
        
        for point in construct['points']:
            file.write(' '.join(map(str, point)) + "\n")

        for face in construct['faces']:
            for tri in face['triangles']:
                file.write(str(len(tri)) + ' ' + ' '.join(map(str, tri)) + ' ' + str(face['id']) + "\n")

def write_ply_binary(construct, file_name, metadata = False): 

    tri_count = sum(len(face['triangles']) for face in construct['faces'])
    header = ply_header(metadata, "format binary_little_endian 1.0", len(construct['points']), tri_count)

    with open(file_name, "wb") as file:
        
        file.write(header.encode('utf-8'))
        
        for point in construct['points']:
            file.write(np.array(point, dtype="<f4").tobytes())

        for face in construct['faces']:
            face_id = int(face['id']).to_bytes(4, 'little', signed=False)
            for tri in face['triangles']:
                file.write(len(tri).to_bytes(1, 'little', signed=False))
                file.write(np.array(tri, dtype="<u4").tobytes())
                file.write(face_id)

def write_ply(construct, file_name, metadata = False, binary=True):
    if binary:
        write_ply_binary(construct, file_name, metadata)
    else:
        write_ply_ascii(construct, file_name, metadata)
