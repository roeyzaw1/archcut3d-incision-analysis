"""
Standalone Yoshizawa crest-line detection -- ridges and ravines on a mesh via
the setCurvature binary (Yoshizawa, Belyaev & Seidel 2005).

crest_detection.py has this logic too, but tangled into a GUI class
(CrestDetector) that builds menus/toolbars and needs a MainWindow. This
module exposes it as a plain function so it can run headless -- used as a
labeling accelerator: seed the incision tracer with detected crest lines
that the expert then keeps, deletes, or extends.

Reuses CrestDetector's tested @staticmethod file I/O (_write_object_txt /
_read_crest_file); only the subprocess orchestration is duplicated here
(minus the optional smoothing branch).
"""

import os
import subprocess

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from .resources import get_cache_dir, get_setcurvature_path
from .crest_detection import CrestDetector

DEFAULT_NEIGHBORHOOD = 1
DEFAULT_ALPHA_BETA = np.pi / 3   # crest-detection dialog defaults (Yoshizawa 2005)
DEFAULT_GAMMA = np.pi / 2


def _polydata_from_mesh(vertices, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    n_faces = faces.shape[0]
    cell_data = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64),
                           faces.astype(np.int64)]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(n_faces, numpy_to_vtkIdTypeArray(cell_data, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(cells)
    return pd


def detect_crest_lines(vertices, faces,
                       neighborhood_size=DEFAULT_NEIGHBORHOOD,
                       alpha_beta=DEFAULT_ALPHA_BETA,
                       gamma=DEFAULT_GAMMA,
                       timeout=600):
    """Run setCurvature on the mesh and return:
        {'ridges': data, 'ravines': data}
    where each `data` is {'points': [(x,y,z), ...],
                          'edges_with_comp': [(p1_idx, p2_idx, comp_id), ...]}
    (as produced by CrestDetector._read_crest_file), or None if that file
    was empty/unreadable.
    """
    exe_path = get_setcurvature_path()
    if exe_path is None:
        raise FileNotFoundError("setCurvature binary not found in bin/ -- can't detect crest lines.")

    work_dir = get_cache_dir()
    os.makedirs(work_dir, exist_ok=True)

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(_polydata_from_mesh(vertices, faces))
    tri_filter.Update()
    tri_polydata = tri_filter.GetOutput()

    object_txt = os.path.join(work_dir, "object.txt")
    CrestDetector._write_object_txt(tri_polydata, object_txt, neighborhood_size)

    ridges_txt = os.path.join(work_dir, "ridges.txt")
    ravines_txt = os.path.join(work_dir, "ravines.txt")
    output_txt = os.path.join(work_dir, "output.txt")
    for f in (ridges_txt, ravines_txt, output_txt):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    cmd = [exe_path, "object.txt", "output.txt", str(gamma), str(alpha_beta)]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        raise RuntimeError(f"setCurvature failed:\n{details}")
    if not (os.path.exists(ridges_txt) and os.path.exists(ravines_txt)):
        raise RuntimeError("setCurvature finished but produced no ridges.txt / ravines.txt")

    return {
        'ridges': CrestDetector._read_crest_file(ridges_txt),
        'ravines': CrestDetector._read_crest_file(ravines_txt),
    }
