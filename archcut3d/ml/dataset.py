"""Turn the hand-labeled `.label.npz` files into tracer training samples.

Piece 1 (this file, so far):
  - LabeledExample: load one `.label.npz` and expose its grid + traces.
  - resample_polyline: re-space a traced centerline to uniform arc-length
    steps (labels come in anywhere from 0.03 mm to ~1 mm point spacing; the
    sampler downstream needs them uniform).

A `.label.npz` (written by label_tool.LabelToolDialog) holds:
  channels      (H, W, 3) float32   height / local_relief / curvature
  mask          (H, W) bool         True where the grid covers the mesh
  channel_names (3,) str
  pixel_size    float64             mm per grid cell
  origin        (2,) float64        world (x_min, y_min) of the grid
  source_file   str                 basename of the model it was traced on
  traces_xy     (sum Ni, 2) float64 all traced points, [col, row]
  trace_lengths (T,) int64          points per trace, to split traces_xy
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class LabeledExample:
    path: str
    channels: np.ndarray            # (H, W, C) float32
    mask: np.ndarray               # (H, W) bool
    channel_names: list             # length C
    pixel_size: float              # mm / cell
    origin: tuple                  # (x_min, y_min) world coords
    source_file: str               # model basename
    traces: list                  # list of (Ni, 2) float64 arrays, [col, row]

    @classmethod
    def load(cls, path: str) -> "LabeledExample":
        d = np.load(path, allow_pickle=False)
        lengths = d["trace_lengths"].astype(int)
        splits = np.cumsum(lengths)[:-1]
        traces = [np.asarray(t, dtype=np.float64)
                  for t in np.split(d["traces_xy"].astype(np.float64), splits)]
        return cls(
            path=path,
            channels=d["channels"].astype(np.float32),
            mask=d["mask"].astype(bool),
            channel_names=[str(n) for n in d["channel_names"]],
            pixel_size=float(d["pixel_size"]),
            origin=(float(d["origin"][0]), float(d["origin"][1])),
            source_file=str(d["source_file"]),
            traces=traces,
        )

    @property
    def shape_hw(self) -> tuple:
        return self.channels.shape[:2]

    def trace_length_mm(self, trace: np.ndarray) -> float:
        if len(trace) < 2:
            return 0.0
        seg_px = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        return float(seg_px.sum() * self.pixel_size)

    def summary(self) -> str:
        h, w = self.shape_hw
        lens = [len(t) for t in self.traces]
        mm = [round(self.trace_length_mm(t), 2) for t in self.traces]
        return (f"{os.path.basename(self.path)}: {w}x{h}px @ {self.pixel_size:.4f} mm/px, "
                f"{len(self.traces)} traces, pts={lens}, lengths_mm={mm}")


def load_dataset(paths_or_dir, pattern="*.label.npz"):
    """Load every `.label.npz` under a directory (or an explicit list of
    files) into a list of LabeledExample."""
    if isinstance(paths_or_dir, str):
        if os.path.isdir(paths_or_dir):
            paths = sorted(glob.glob(os.path.join(paths_or_dir, pattern)))
        else:
            paths = [paths_or_dir]
    else:
        paths = list(paths_or_dir)
    return [LabeledExample.load(p) for p in paths]


def _dedup_consecutive(pts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Drop points identical to their predecessor (zero-length segments
    break the arc-length interpolation)."""
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > eps
    return pts[keep]


def resample_polyline(pts_colrow: np.ndarray, spacing_mm: float, pixel_size: float) -> np.ndarray:
    """Re-space a polyline to uniform arc-length steps.

    pts_colrow  : (N, 2) points in grid [col, row]
    spacing_mm  : desired gap between output points, in millimetres
    pixel_size  : mm per grid cell (to convert the polyline's pixel length)

    Returns (M, 2) in the same [col, row] frame. The first and last points
    are preserved exactly; the last gap may be shorter than spacing_mm. A
    polyline shorter than one step comes back as just its two endpoints.
    """
    pts = _dedup_consecutive(np.asarray(pts_colrow, dtype=np.float64))
    if len(pts) < 2:
        return pts.copy()

    seg_px = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_mm = np.concatenate([[0.0], np.cumsum(seg_px) * pixel_size])
    total_mm = cum_mm[-1]

    spacing_mm = float(spacing_mm)
    if spacing_mm <= 0 or total_mm <= spacing_mm:
        return np.stack([pts[0], pts[-1]])

    n_steps = int(np.floor(total_mm / spacing_mm))
    targets = np.arange(n_steps + 1) * spacing_mm
    if targets[-1] < total_mm - 1e-9:
        targets = np.append(targets, total_mm)

    col = np.interp(targets, cum_mm, pts[:, 0])
    row = np.interp(targets, cum_mm, pts[:, 1])
    return np.stack([col, row], axis=1)


# --------------------------------------------------------------------------
# Piece 2: patch extraction
#
# A training/inference step looks at a small square window of the grid,
# rotated so "forward" (the current heading) points up. Cropping at a fixed
# PHYSICAL size and resampling to a fixed pixel resolution makes the 4.6x
# pixel-size spread across the labeled models irrelevant -- an incision of a
# given real width fills the same fraction of every patch.
# --------------------------------------------------------------------------

DEFAULT_PATCH_MM = 2.0
DEFAULT_PATCH_RES = 48


def tangent_at(polyline: np.ndarray, i: int, window: int = 1) -> np.ndarray:
    """Unit forward direction of `polyline` at index i, as a [col, row]
    vector. Central difference over +/- `window` points (clamped at the
    ends). Returns [1, 0] for a degenerate (single-point) line."""
    p = np.asarray(polyline, dtype=np.float64)
    n = len(p)
    if n < 2:
        return np.array([1.0, 0.0])
    a = max(0, i - window)
    b = min(n - 1, i + window)
    v = p[b] - p[a]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # fall back to the whole-line direction
        v = p[-1] - p[0]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return np.array([1.0, 0.0])
    return v / norm


def sample_patch(channels: np.ndarray,
                 mask: np.ndarray,
                 pixel_size: float,
                 center_colrow,
                 heading_colrow,
                 patch_mm: float = DEFAULT_PATCH_MM,
                 out_res: int = DEFAULT_PATCH_RES):
    """Crop a heading-canonicalized patch from a grid.

    channels     : (H, W, C) grid
    mask         : (H, W) bool
    pixel_size   : mm per grid cell
    center_colrow: (col, row) sample centre, may be fractional
    heading_colrow: (dcol, drow) "forward" direction; it maps to patch UP
                   (decreasing patch row). Need not be unit length.
    patch_mm     : physical side length of the crop, in mm
    out_res      : output is (out_res, out_res, C)

    Returns (patch float32 (out_res,out_res,C), pmask bool (out_res,out_res)).
    Samples outside the grid come back as 0 / mask False (bilinear).
    """
    channels = np.asarray(channels)
    if channels.ndim == 2:
        channels = channels[:, :, None]
    C = channels.shape[2]

    f = np.asarray(heading_colrow, dtype=np.float64)
    fn = np.linalg.norm(f)
    if fn < 1e-12:
        raise ValueError("heading_colrow must be non-zero")
    f = f / fn

    # screen frame (row points down): UP = forward, RIGHT = forward rotated
    # 90 deg clockwise as seen on screen -> (x, y) -> (-y, x)
    right = np.array([-f[1], f[0]])
    down = -f

    cx, cy = float(center_colrow[0]), float(center_colrow[1])
    cells_per_patch_px = (patch_mm / out_res) / float(pixel_size)

    idx = np.arange(out_res) - (out_res - 1) / 2.0
    du, dv = np.meshgrid(idx, idx)   # du: rightward (patch col), dv: downward (patch row)

    gc = cx + cells_per_patch_px * (du * right[0] + dv * down[0])
    gr = cy + cells_per_patch_px * (du * right[1] + dv * down[1])
    coords = np.stack([gr.ravel(), gc.ravel()])   # map_coordinates wants (row, col)

    patch = np.empty((out_res, out_res, C), dtype=np.float32)
    for c in range(C):
        patch[:, :, c] = map_coordinates(
            channels[:, :, c].astype(np.float64), coords,
            order=1, mode="constant", cval=0.0).reshape(out_res, out_res)

    pmask = map_coordinates(
        mask.astype(np.float64), coords,
        order=1, mode="constant", cval=0.0).reshape(out_res, out_res) >= 0.5

    return patch, pmask


def sample_patch_from_example(example: "LabeledExample", center_colrow, heading_colrow,
                              patch_mm: float = DEFAULT_PATCH_MM, out_res: int = DEFAULT_PATCH_RES):
    return sample_patch(example.channels, example.mask, example.pixel_size,
                        center_colrow, heading_colrow, patch_mm, out_res)


# --------------------------------------------------------------------------
# Piece 3: training-sample generation
#
# Walk each labeled centerline at a fixed physical step. At every anchor:
#   input  = heading-canonicalized patch centred there
#   step   = (forward, right) vector to the next point ON the line, in the
#            canonicalized frame, in units of `step_mm` (a clean straight
#            step is ~(1, 0))
#   stop   = 1 at/after the true endpoint, else 0
#
# Jitter copies teach recovery: shift the centre off the line and rotate the
# heading; the step target then points from that wrong pose back toward the
# line, so the tracer learns to correct instead of drifting.
# --------------------------------------------------------------------------

DEFAULT_STEP_MM = 0.3


def _to_patch_frame(vec_colrow: np.ndarray, heading_unit: np.ndarray) -> np.ndarray:
    """Express a [col,row] world vector as (forward, right) in the patch
    frame whose UP is `heading_unit` (matches sample_patch's convention:
    right = heading rotated 90 deg clockwise on screen)."""
    f = heading_unit
    right = np.array([-f[1], f[0]])
    return np.array([float(vec_colrow @ f), float(vec_colrow @ right)])


@dataclass
class TracerSamples:
    patches: np.ndarray      # (N, R, R, C) float32
    masks: np.ndarray        # (N, R, R) bool
    steps: np.ndarray        # (N, 2) float32  (forward, right) in step_mm units
    stops: np.ndarray        # (N,) float32    0 / 1
    groups: np.ndarray       # (N,) int        index into group_names (scanned model) for splitting
    meta: list               # length N: dicts {source_file, trace, point, jitter}
    group_names: list = field(default_factory=list)   # distinct source_file values, sorted; group g = group_names[g]

    def __len__(self):
        return len(self.stops)


def make_samples(example: "LabeledExample",
                 examples_group: int = 0,
                 step_mm: float = DEFAULT_STEP_MM,
                 patch_mm: float = DEFAULT_PATCH_MM,
                 out_res: int = DEFAULT_PATCH_RES,
                 tangent_window: int = 2,
                 n_jitter: int = 4,
                 jitter_pos_mm: float = 0.25,
                 jitter_deg: float = 15.0,
                 end_overshoot: int = 2,
                 min_trace_pts: int = 4,
                 rng=None) -> TracerSamples:
    """Generate tracer training tuples from one LabeledExample.

    n_jitter        : extra perturbed copies per anchor (0 = clean only)
    jitter_pos_mm   : max centre offset (uniform in a disc), mm
    jitter_deg      : max heading rotation, degrees
    end_overshoot   : how many step_mm past each endpoint to add stop=1 anchors
    min_trace_pts   : skip resampled traces shorter than this
    """
    rng = np.random.default_rng(rng)
    px = example.pixel_size
    step_px = step_mm / px
    pos_jit_px = jitter_pos_mm / px

    patches, masks, steps, stops, meta = [], [], [], [], []

    def emit(center, heading_unit, target_world, stop, tinfo):
        patch, pmask = sample_patch(example.channels, example.mask, px,
                                    center, heading_unit, patch_mm, out_res)
        patches.append(patch)
        masks.append(pmask)
        steps.append((_to_patch_frame(np.asarray(target_world), heading_unit) / step_px).astype(np.float32))
        stops.append(np.float32(stop))
        meta.append(tinfo)

    for ti, raw in enumerate(example.traces):
        line = resample_polyline(raw, step_mm, px)
        M = len(line)
        if M < min_trace_pts:
            continue

        for i in range(M):
            base_heading = tangent_at(line, min(i, M - 2), window=tangent_window)
            # lookahead point on the line: next point, or (for the last) a
            # straight extrapolation so "keep going" still has a direction
            ahead = line[i + 1] if i < M - 1 else line[i] + base_heading * step_px
            stop = 1.0 if i >= M - 1 else 0.0

            variants = [(np.asarray(line[i], float), base_heading, 0)]
            for j in range(n_jitter):
                ang = np.radians(rng.uniform(-jitter_deg, jitter_deg))
                ca, sa = np.cos(ang), np.sin(ang)
                hj = np.array([ca * base_heading[0] - sa * base_heading[1],
                               sa * base_heading[0] + ca * base_heading[1]])
                r = pos_jit_px * np.sqrt(rng.random())
                phi = rng.uniform(0, 2 * np.pi)
                cj = np.asarray(line[i], float) + r * np.array([np.cos(phi), np.sin(phi)])
                variants.append((cj, hj, j + 1))

            for center, heading, jid in variants:
                emit(center, heading, np.asarray(ahead) - center, stop,
                     {"source_file": example.source_file, "trace": ti, "point": i, "jitter": jid})

        # a few anchors stepping past the endpoint, all stop=1, so the tracer
        # learns to halt rather than run off the mesh
        for end_idx, sgn in ((M - 1, +1), (0, -1)):
            h = tangent_at(line, min(end_idx, M - 2), window=tangent_window) * sgn
            for k in range(1, end_overshoot + 1):
                center = np.asarray(line[end_idx], float) + h * step_px * k
                emit(center, h, h * step_px, 1.0,
                     {"source_file": example.source_file, "trace": ti,
                      "point": end_idx, "jitter": -1})

    n = len(stops)
    return TracerSamples(
        patches=np.asarray(patches, dtype=np.float32) if n else np.zeros((0, out_res, out_res, example.channels.shape[2]), np.float32),
        masks=np.asarray(masks, dtype=bool) if n else np.zeros((0, out_res, out_res), bool),
        steps=np.asarray(steps, dtype=np.float32) if n else np.zeros((0, 2), np.float32),
        stops=np.asarray(stops, dtype=np.float32),
        groups=np.full(n, examples_group, dtype=int),
        meta=meta,
    )


def make_dataset(examples, rng=None, **kw) -> TracerSamples:
    """make_samples over several LabeledExamples, concatenated.

    `groups` indexes distinct scanned models (`source_file`), not label
    files: several label files of the same artifact share one group, so a
    group-based split can never put the same model on both sides. Group ids
    follow the sorted order of distinct `source_file` values, so they do not
    depend on the order of `examples`."""
    rng = np.random.default_rng(rng)
    names = sorted({e.source_file for e in examples})
    group_of = {name: g for g, name in enumerate(names)}
    parts = [make_samples(e, examples_group=group_of[e.source_file], rng=rng, **kw) for e in examples]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("no usable traces in the given examples")
    return TracerSamples(
        patches=np.concatenate([p.patches for p in parts]),
        masks=np.concatenate([p.masks for p in parts]),
        steps=np.concatenate([p.steps for p in parts]),
        stops=np.concatenate([p.stops for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        meta=[m for p in parts for m in p.meta],
        group_names=names,
    )
