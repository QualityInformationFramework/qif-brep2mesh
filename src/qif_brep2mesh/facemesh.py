import numpy as np


def split_face_meshes(construct):
    """
    Split the shared-vertex construct produced by topology.build_qif into one
    standalone mesh per face.

    build_qif tessellates the whole part into a single point pool so that faces
    share the vertices along their common edges. QIF MeshTriangle elements are
    per-face, so each face needs its own vertex array with triangles reindexed
    into it. Vertices on a shared edge are therefore duplicated once per
    adjoining face in the QIF output, while the VTK output keeps them shared.

    Returns a dict of face id -> {'vertices': (n,3) float64, 'triangles': (m,3) int64}.
    """
    points = np.asarray(construct['points'], dtype=np.float64)
    meshes = {}
    for face in construct['faces']:
        triangles = np.asarray(face['triangles'], dtype=np.int64).reshape(-1, 3)
        if len(triangles) == 0:
            meshes[face['id']] = {
                'vertices': np.empty((0, 3), dtype=np.float64),
                'triangles': np.empty((0, 3), dtype=np.int64)
            }
            continue
        used, reindexed = np.unique(triangles, return_inverse=True)
        meshes[face['id']] = {
            'vertices': points[used],
            'triangles': reindexed.reshape(triangles.shape).astype(np.int64)
        }
    return meshes
