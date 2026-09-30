# qif-brep2mesh

Tessellate the B-rep geometry of a QIF MBD document, to mesh QIF or to VTK, PLY or OBJ.

## Install

```bash
pip install -e .
```

Python 3.10–3.13, enforced by `requires-python` so pip refuses an unsupported interpreter at install time. The floor is `match`/`case` in the curve modules; the ceiling is `triangle`, which publishes no wheel and no sdist for 3.14.

## Command line

```bash
qif-brep2mesh part.qif --qif-out mesh.qif --vtk-out mesh.vtp
```

At least one output is required. Several may be given, and the document is read and tessellated once for all of them.

| Flag | |
|---|---|
| `--qif-out` | mesh QIF: faces become `FaceMesh`, `SurfaceMeshSet` is populated |
| `--vtk-out` | VTK PolyData (`.vtp`) |
| `--ply-out` | PLY |
| `--obj-out` | OBJ |
| `--lod`, `-l` | level of detail; smaller is finer. Defaults to the bounding-box norm over 140 |
| `--lod-scaled`, `-ls` | multiplies the lod; less than 1 increases detail |
| `--ascii`, `-a` | write VTK and PLY as ascii rather than binary |
| `--vtk-compression` | `zlib` (default) or `lzma` for binary VTK |

A bundled sample to try it on:

```bash
qif-brep2mesh tests/data/nist_ctc_01_asme1_ap242.qif --vtk-out out.vtp
```

## Library

The mesh does not have to go through a file:

```python
from qif_brep2mesh import mesh_document

mesh = mesh_document('part.qif', lod=2.0)
mesh['points']     # (n, 3) float64
mesh['triangles']  # (m, 3) int64, indexing into points
mesh['qif_id']     # (m,)   int64, the QIF face each triangle came from
```

Plus `qpid`, `linear_unit`, `lod` and `filename`, carried through from the document. Returns plain numpy, with no VTK or pyvista dependency.

The Python API follows PEP 8: `snake_case` functions, arguments and keys. Names written *into* output files are part of a file format rather than Python, so they keep their established spelling: the VTP `FieldData`, PLY and OBJ comments say `linearUnit` and `generationDate`.

## What each output preserves

`qif_id` is the join key. It carries the QIF geometry entity id of the face each triangle came from, and it is what lets a caller put per-face data (a tolerance, a measured result) back onto the mesh. The formats do not treat it equally.

| Output | `qif_id` | |
|---|---|---|
| **mesh QIF** | n/a | Faces become `FaceMesh`; the grouping *is* the document structure |
| **VTP** | **native** | A cell array. `pyvista.read(...).cell_data['qif_id']` |
| **PLY** | **written, but VTK cannot read it** | `property uint qif_id` on `element face`. Standard, and read by spec-compliant readers; `vtkPLYReader` has no facility for arbitrary properties, so pyvista returns `cell_data=[]` |
| **OBJ** | **recoverable, not native** | Written as `g <qif_id>` per face. VTK exposes an ordinal `GroupIds`, not the id; recover the real ids by reading the `g` lines in order |

So: **use VTP if you want the join for free, or the library API and skip files entirely.** OBJ and PLY are for getting the shape into other tools.

Two further notes on reading a `.vtp` back. **VTK drops String `DataArray`s**, so the `qpid`, `linearUnit` and `lod` written into `FieldData` come back as empty `Unnamed_0..N`. Parse the `.vtp` header as XML instead. And PLY vertices are `float32` where VTP and the library API are `float64`.

The mesh QIF output is modified in place rather than rebuilt, so everything that is not B-rep geometry survives: characteristics, features, datums, units, and any measurement results the input carried.

## Unsupported input

CoEdges with no `Curve12` are rejected. Reconstructing uv coordinates from the 3d edge is not implemented: a loop mixing present and absent `Curve12` raises, and a loop with none at all reaches `triangle` with degenerate input and segfaults. The reader refuses such documents by name and reason before they get that far.

## Validating output

```bash
xmllint --noout --schema qif-xsd/QIFApplications/QIFDocument.xsd out.qif
```

`##other` namespace warnings come from the source files and are not validity errors; look for the `validates` or `fails to validate` line. The schema is not included here. Use a copy of the QIF XSD distribution.

## Tests

```bash
python tests/test_mesher.py
```

Six checks over the bundled sample. Runs under pytest too.

## Layout

```
src/qif_brep2mesh/
├── __init__.py     mesh_document and friends: the public API
├── cli.py          the qif-brep2mesh command
├── reader.py       QIF in, with validation
├── facemesh.py     regroup the tessellation per face
├── geometry/       the tessellation kernel
└── writers/        mesh QIF, VTK, PLY, OBJ
```

## License

MIT. See [`LICENSE`](LICENSE).

Tessellation goes through `triangle`, which wraps Jonathan Richard Shewchuk's Triangle. That library is licensed separately and more restrictively: free for private, research and institutional use, but commercial redistribution requires direct arrangement with the author.
