"""
Port of read_wrl.m: parses the specific VRML flavor MATLAB's write_wrl
produces (and Cut3-D reads back) -- a single IndexedFaceSet with a
`point [ ... ]` vertex block and a `coordIndex [ ... ]` triangle block, one
vertex/face per line, comma-separated. Deliberately a dedicated small parser
(matching MATLAB's own fscanf-based approach) rather than VTK's generic
vtkVRMLImporter, so vertex/face order and count match the MATLAB source
exactly -- every downstream measurement depends on that.
"""

import re

import numpy as np

_NUMBER_RE = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')


def read_wrl(filename):
    """Return (vertices, faces): vertices is (N, 3) float64, faces is
    (M, 3) int64 and 0-indexed (the WRL file itself is already 0-indexed,
    unlike MATLAB's +1'd copy)."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    vertex_numbers = _read_bracketed_numbers(lines, 'point')
    face_numbers = _read_bracketed_numbers(lines, 'coordIndex')

    vertices = np.array(vertex_numbers, dtype=np.float64).reshape(-1, 3)
    faces = np.array(face_numbers, dtype=np.int64).reshape(-1, 4)[:, :3]
    return vertices, faces


def _read_bracketed_numbers(lines, keyword):
    """Find the line containing `keyword`, then collect every number from
    the following lines up to (not including) the line that closes the
    bracket ("]"). Mirrors read_wrl.m's fgets-until-']' pattern."""
    i = 0
    n = len(lines)
    while i < n and keyword not in lines[i]:
        i += 1
    if i >= n:
        raise ValueError(f"WRL file has no '{keyword}' section")
    if '[' not in lines[i]:
        i += 1
    i += 1

    numbers = []
    while i < n and lines[i].strip() != ']':
        numbers.extend(_NUMBER_RE.findall(lines[i]))
        i += 1
    return numbers
