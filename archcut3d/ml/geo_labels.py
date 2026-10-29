"""Geometry-only training labels from crest-line detection.

For a scan, run the Yoshizawa crest detector (crest_lines.py) on the
auto-positioned mesh, keep the long ravine lines (valleys) and project them
onto the scan's 2.5D grid. The result is a LabeledExample whose traces are
purely geometric: no human judgement is involved, only where the surface has
a valley.

If the scan already has a grid (from a hand-labeled .label.npz) that grid is
reused, so crest lines and hand traces live in the same pixel frame;
otherwise the grid is rasterised from the positioned mesh.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .dataset import LabeledExample

MIN_VERTICES = 5000          # coarser scans have too little detail for a 2 mm patch
DEFAULT_MIN_LENGTH_MM = 3.0  # shorter crest lines are texture specks, and give few training anchors
DEFAULT_MAX_TRACES = 80      # keep the longest N lines per scan


def positioned_mesh(wrl_path: str):
    from ..positioning import compute_initial_orientation
    from ..wrl_io import read_wrl
    vertices, faces = read_wrl(wrl_path)
    return compute_initial_orientation(vertices, faces), faces


def crest_traces_on_grid(vertices, faces, grid, which: str = "ravines",
                         min_length_mm: float = DEFAULT_MIN_LENGTH_MM, max_traces: int = DEFAULT_MAX_TRACES):
    """Detect crest lines, project them onto `grid` (a grid_2d.MeshGrid), keep
    those at least `min_length_mm` long and return the longest `max_traces`
    as (N, 2) [col, row] arrays. The length filter is in millimetres, not
    edges: the same number of mesh edges is a very different length on
    scans with different vertex spacing."""
    from ..crest_lines import crest_lines_as_polylines, detect_crest_lines
    result = detect_crest_lines(vertices, faces)
    polylines = crest_lines_as_polylines(result, min_edges=2, which=which)
    h, w = grid.mask.shape
    traces = []
    for pl in polylines:
        colrow = grid.world_xy_to_grid(pl[:, 0], pl[:, 1])
        inside = (colrow[:, 0] >= 0) & (colrow[:, 0] <= w - 1) & (colrow[:, 1] >= 0) & (colrow[:, 1] <= h - 1)
        colrow = colrow[inside]
        if len(colrow) < 4:
            continue
        length_mm = float(np.linalg.norm(np.diff(colrow, axis=0), axis=1).sum() * grid.pixel_size)
        if length_mm >= min_length_mm:
            traces.append((length_mm, colrow.astype(np.float64)))
    traces.sort(key=lambda t: -t[0])
    return [t for _, t in traces[:max_traces]]


def build_crest_example(wrl_path: str, hand_example=None, **kw):
    """LabeledExample of crest-line traces for one scan. `hand_example`, if
    given, supplies the grid (channels, mask, pixel size, origin). Returns
    (example, info dict)."""
    from ..grid_2d import MeshGrid, rasterize_to_grid
    t0 = time.time()
    vertices, faces = positioned_mesh(wrl_path)
    if len(vertices) < MIN_VERTICES:
        raise ValueError(f"{os.path.basename(wrl_path)}: {len(vertices)} vertices, too coarse (< {MIN_VERTICES})")
    if hand_example is not None:
        grid = MeshGrid(channels=hand_example.channels, mask=hand_example.mask,
                        origin=hand_example.origin, pixel_size=hand_example.pixel_size)
    else:
        grid = rasterize_to_grid(vertices, faces)
    traces = crest_traces_on_grid(vertices, faces, grid, **kw)
    ex = LabeledExample(path=wrl_path, channels=grid.channels.astype(np.float32), mask=grid.mask,
                        channel_names=["height", "local_relief", "curvature"], pixel_size=float(grid.pixel_size),
                        origin=(float(grid.origin[0]), float(grid.origin[1])),
                        source_file=os.path.basename(wrl_path), traces=traces)
    info = {"source_file": ex.source_file, "n_vertices": len(vertices), "n_traces": len(traces),
            "traced_mm": round(sum(ex.trace_length_mm(t) for t in traces), 1),
            "pixel_size_mm": ex.pixel_size, "seconds": round(time.time() - t0, 1),
            "grid_from": "hand label file" if hand_example is not None else "rasterised"}
    return ex, info


def save_crest_example(path: str, ex: LabeledExample, info: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lengths = np.array([len(t) for t in ex.traces], dtype=np.int64)
    np.savez_compressed(path, channels=ex.channels, mask=ex.mask, pixel_size=ex.pixel_size,
                        channel_names=np.array(ex.channel_names),
                        origin=np.array(ex.origin), source_file=np.array(ex.source_file),
                        traces_xy=np.concatenate(ex.traces) if ex.traces else np.zeros((0, 2)),
                        trace_lengths=lengths, info=np.array(repr(info)))


def load_crest_example(path: str) -> LabeledExample:
    """Read a cached crest example written by save_crest_example (the same
    layout as a .label.npz, so LabeledExample.load reads it)."""
    return LabeledExample.load(path)
