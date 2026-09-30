"""Regression tests. Run with `python tests/test_mesher.py`, or under pytest.

Deliberately dependency-free. The point is to catch a geometry or join-key
regression, not to be a full suite.
"""

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from qif_brep2mesh import mesh_document

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                      'nist_ctc_01_asme1_ap242.qif')
CLI = [sys.executable, '-m', 'qif_brep2mesh.cli']
NS = {'qif': 'http://qifstandards.org/xsd/qif3'}

# Tessellation at the sample's default lod. These are exact: the kernel is
# deterministic, so any change here is a geometry change and wants explaining.
EXPECTED = {'points': 28180, 'triangles': 56394, 'faces': 117}


def test_mesh_counts():
    mesh = mesh_document(SAMPLE)
    assert len(mesh['points']) == EXPECTED['points'], len(mesh['points'])
    assert len(mesh['triangles']) == EXPECTED['triangles'], len(mesh['triangles'])
    assert len(np.unique(mesh['qif_id'])) == EXPECTED['faces']
    assert mesh['linear_unit'] == 'mm'
    assert len(mesh['qpid']) == 36


def test_qif_id_indexes_real_faces():
    """Every qif_id must name a Face in the source document, and every face
    that produced triangles must appear. This is the join the whole pipeline
    rests on."""
    mesh = mesh_document(SAMPLE)
    face_ids = {int(f.get('id'))
               for f in ET.parse(SAMPLE).getroot().find('.//qif:FaceSet', NS)}
    meshed = set(np.unique(mesh['qif_id']).tolist())
    assert meshed <= face_ids, meshed - face_ids
    assert len(mesh['qif_id']) == len(mesh['triangles'])


def test_triangles_index_points():
    mesh = mesh_document(SAMPLE)
    assert mesh['triangles'].min() >= 0
    assert mesh['triangles'].max() < len(mesh['points'])
    assert mesh['triangles'].dtype == np.int64
    assert mesh['points'].dtype == np.float64


def test_lod_scaling_refines():
    coarse = mesh_document(SAMPLE)
    fine = mesh_document(SAMPLE, lod_scaled=0.5)
    assert fine['lod'] < coarse['lod']
    assert len(fine['triangles']) > len(coarse['triangles'])


def test_cli_writes_every_format():
    with tempfile.TemporaryDirectory() as tmp:
        out = {ext: os.path.join(tmp, 'out.' + ext) for ext in ('qif', 'vtp', 'ply', 'obj')}
        subprocess.run(
            CLI + [SAMPLE, '--qif-out', out['qif'], '--vtk-out', out['vtp'],
                   '--ply-out', out['ply'], '--obj-out', out['obj']],
            check=True, capture_output=True)
        for ext, path in out.items():
            assert os.path.getsize(path) > 0, ext

        # the mesh QIF must still be a QIF document carrying its PMI
        src, dst = ET.parse(SAMPLE).getroot(), ET.parse(out['qif']).getroot()
        for path in ('.//qif:CharacteristicNominals', './/qif:FeatureNominals',
                     './/qif:DatumDefinitions', './/qif:FaceSet'):
            a, b = src.find(path, NS), dst.find(path, NS)
            assert (a is None) == (b is None), path
            if a is not None:
                assert len(a) == len(b), f'{path}: {len(a)} -> {len(b)}'
        assert len(dst.find('.//qif:SurfaceMeshSet', NS)) == EXPECTED['faces']


def test_cli_rejects_missing_output():
    r = subprocess.run(CLI + [SAMPLE], capture_output=True, text=True)
    assert r.returncode != 0
    assert 'no output requested' in r.stderr


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        try:
            fn()
            print(f'  ok   {name}')
        except AssertionError as err:
            failed += 1
            print(f'  FAIL {name}: {err}')
    print('all passed' if not failed else f'{failed} failed')
    sys.exit(1 if failed else 0)
