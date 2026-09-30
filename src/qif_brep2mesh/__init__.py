"""Tessellate the B-rep geometry of a QIF MBD document.

The library entry point is `mesh_document`, which returns plain numpy arrays and
touches no files beyond the one it reads. Writers are the caller's choice; see
cli.py for the command line.
"""

import datetime

import numpy as np

from .reader import read_qif
from .geometry.topology import build_qif


def resolve_lod(qifdoc, lod=None, lod_scaled=1):
    """Level of detail, defaulting to the model's bounding-box norm over 140."""
    if lod is None:
        lod = qifdoc['points'].bbox_norm() / 140
    return lod * lod_scaled


def build_document(file_name, lod=None, lod_scaled=1):
    """Read and tessellate, returning (qifdoc, construct, lod).

    `construct` keeps the per-face grouping the QIF and mesh writers need.
    """
    qifdoc = read_qif(file_name)
    lod = resolve_lod(qifdoc, lod, lod_scaled)
    return qifdoc, build_qif(qifdoc, lod), lod


def document_metadata(qifdoc, lod):
    """Metadata the file writers record alongside the mesh.

    The keys are the names written into output files (VTP FieldData, PLY and
    OBJ comments), so they keep the file formats' established spelling rather
    than following Python naming.
    """
    return {'filename': qifdoc['filename'], 'lod': lod, 'qpid': qifdoc['qpid'],
            'linearUnit': qifdoc['linear_unit'], 'generationDate': datetime.datetime.now()}


def flatten(construct):
    """Flatten the per-face construct into one triangle soup.

    Returns points (n,3) float64, triangles (m,3) int64 indexing into them, and
    qif_id (m,) int64 giving the QIF geometry entity id of the face each
    triangle was tessellated from. qif_id is the join key: everything a caller
    draws per-face is a join onto it.
    """
    triangles = [np.asarray(face['triangles'], dtype=np.int64).reshape(-1, 3)
                 for face in construct['faces']]
    qif_id = [np.repeat(np.int64(face['id']), len(tris))
              for face, tris in zip(construct['faces'], triangles)]

    if triangles:
        triangles = np.vstack(triangles).astype(np.int64)
        qif_id = np.concatenate(qif_id).astype(np.int64)
    else:
        triangles = np.empty((0, 3), dtype=np.int64)
        qif_id = np.empty((0,), dtype=np.int64)

    return {'points': np.asarray(construct['points'], dtype=np.float64),
            'triangles': triangles,
            'qif_id': qif_id}


def mesh_document(file_name, lod=None, lod_scaled=1):
    """Tessellate a QIF document and return the mesh in memory.

    Keys: points, triangles, qif_id, plus qpid, linear_unit, lod and filename
    carried through from the document so a caller can label and verify what it
    is holding without re-reading the file.
    """
    qifdoc, construct, lod = build_document(file_name, lod, lod_scaled)
    mesh = flatten(construct)
    mesh.update({'qpid': qifdoc['qpid'], 'linear_unit': qifdoc['linear_unit'],
                 'lod': lod, 'filename': qifdoc['filename']})
    return mesh
