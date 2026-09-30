import argparse
import sys
import time

from . import build_document, document_metadata
from .facemesh import split_face_meshes
from .writers.obj import write_obj
from .writers.ply import write_ply
from .writers.qifmesh import write_qif_mesh
from .writers.vtk import write_vtk


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog='qif-brep2mesh',
        description='Tessellate the B-rep geometry of a QIF MBD document.',
        epilog='At least one output is required. Several may be given, and the '
               'document is read and tessellated once for all of them.')
    parser.add_argument('in_file', help='QIF file containing B-rep geometry')
    parser.add_argument('--lod', '-l', type=float,
                        help='Level of detail; smaller is finer. Defaults to the '
                             'bounding-box norm over 140.')
    parser.add_argument('--lod-scaled', '-ls', type=float, default=1,
                        help='Multiplies the lod, to adjust the auto-calculated '
                             'value. Less than 1 increases detail.')
    parser.add_argument('--qif-out', metavar='FILE',
                        help='Mesh QIF output. Faces become FaceMesh elements and '
                             'SurfaceMeshSet is populated; all other content survives.')
    parser.add_argument('--vtk-out', metavar='FILE', help='VTK PolyData output (*.vtp)')
    parser.add_argument('--ply-out', metavar='FILE', help='PLY output')
    parser.add_argument('--obj-out', metavar='FILE', help='OBJ output')
    parser.add_argument('--ascii', '-a', action='store_true',
                        help='Write VTK and PLY as ascii rather than binary.')
    parser.add_argument('--vtk-compression', choices=['zlib', 'lzma'], default='zlib',
                        help='Compressor for binary VTK output. Ignored with --ascii.')
    args = parser.parse_args()

    if not any([args.qif_out, args.vtk_out, args.ply_out, args.obj_out]):
        parser.error('no output requested; give at least one of '
                     '--qif-out, --vtk-out, --ply-out, --obj-out')

    try:
        qifdoc, construct, lod = build_document(args.in_file, args.lod, args.lod_scaled)
    except (ValueError, OSError) as err:
        # read_qif rejects malformed or unsupported documents by name and reason;
        # a traceback would bury that.
        sys.exit(f'qif-brep2mesh: {err}')

    metadata = document_metadata(qifdoc, lod)

    if args.qif_out:
        write_qif_mesh(qifdoc, split_face_meshes(construct), args.qif_out)
    if args.vtk_out:
        write_vtk(construct, args.vtk_out, metadata=metadata,
                 compression='none' if args.ascii else args.vtk_compression)
    if args.ply_out:
        write_ply(construct, args.ply_out, metadata=metadata, binary=not args.ascii)
    if args.obj_out:
        write_obj(construct, args.obj_out, metadata=metadata)

    print(f"Execution time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
