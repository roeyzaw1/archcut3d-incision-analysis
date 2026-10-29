"""Per-point geometry of hand-drawn lines, and "departure" segments.

For every resampled point of a hand trace the geometric minimum is located
on the line perpendicular to the trace (within +/- half_width_mm, tilt
corrected, as in policies.GeometricMinimumPolicy). The lateral offset
between the hand point and that minimum, and the depth of the dip there,
say how well the hand line follows the surface minimum at that point. A
point where the hand line is far from the minimum, or where there is hardly
any dip at all, is a departure from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import (DEFAULT_PATCH_MM, DEFAULT_PATCH_RES, DEFAULT_STEP_MM,
                      resample_polyline, sample_patch, tangent_at)
from .policies import GeometricMinimumPolicy

DEPARTURE_OFFSET_MM = 0.1
DEPARTURE_DEPTH_PERCENTILE = 10.0


@dataclass
class AnchorTable:
    """One row per resampled hand point that has a successor (the last point
    of each trace is excluded: it has no next point)."""
    source_file: str
    trace: np.ndarray            # (K,) index of the hand trace
    point: np.ndarray            # (K,) index of the point along the resampled trace
    pos: np.ndarray              # (K, 2) [col, row]
    heading: np.ndarray          # (K, 2) unit tangent
    next_pos: np.ndarray         # (K, 2) next resampled hand point
    geo_offset_mm: np.ndarray    # (K,) signed lateral offset of the minimum (right positive); NaN if off-mesh
    geo_depth: np.ndarray        # (K,) tilt-corrected dip depth; NaN if off-mesh
    geo_pos: np.ndarray          # (K, 2) [col, row] of the geometric minimum
    pixel_size: float
    departure: np.ndarray = None  # (K,) bool, filled by label_departures

    def __len__(self):
        return len(self.point)


def build_anchor_table(example, step_mm: float = DEFAULT_STEP_MM, half_width_mm: float = 0.5,
                       patch_mm: float = DEFAULT_PATCH_MM, out_res: int = DEFAULT_PATCH_RES,
                       tangent_window: int = 2) -> AnchorTable:
    geo = GeometricMinimumPolicy(step_mm=step_mm, half_width_mm=half_width_mm,
                                 patch_mm=patch_mm, out_res=out_res)
    px = example.pixel_size
    rows = {k: [] for k in ("trace", "point", "pos", "heading", "next_pos", "off", "depth", "geo_pos")}
    for ti, raw in enumerate(example.traces):
        line = resample_polyline(raw, step_mm, px)
        for i in range(len(line) - 1):
            heading = tangent_at(line, i, window=tangent_window)
            patch, mask = sample_patch(example.channels, example.mask, px, line[i], heading, patch_mm, out_res)
            res = geo.profile_minimum(patch, mask, forward_mm=0.0)
            if res is None:
                off, depth, gpos = np.nan, np.nan, np.array([np.nan, np.nan])
            else:
                off, depth = res
                right = np.array([-heading[1], heading[0]])
                gpos = line[i] + right * (off / px)
            rows["trace"].append(ti)
            rows["point"].append(i)
            rows["pos"].append(line[i])
            rows["heading"].append(heading)
            rows["next_pos"].append(line[i + 1])
            rows["off"].append(off)
            rows["depth"].append(depth)
            rows["geo_pos"].append(gpos)

    def arr(k, shape=None):
        a = np.array(rows[k], dtype=np.float64)
        return a if len(a) else np.zeros((0,) + (shape or ()))

    return AnchorTable(source_file=example.source_file,
                       trace=np.array(rows["trace"], dtype=int), point=np.array(rows["point"], dtype=int),
                       pos=arr("pos", (2,)), heading=arr("heading", (2,)), next_pos=arr("next_pos", (2,)),
                       geo_offset_mm=arr("off"), geo_depth=arr("depth"), geo_pos=arr("geo_pos", (2,)),
                       pixel_size=px)


def depth_threshold(tables, percentile: float = DEPARTURE_DEPTH_PERCENTILE) -> float:
    """The given percentile of dip depth pooled over all points of all tables."""
    depths = np.concatenate([t.geo_depth[np.isfinite(t.geo_depth)] for t in tables])
    return float(np.percentile(depths, percentile))


def label_departures(tables, offset_mm: float = DEPARTURE_OFFSET_MM,
                     depth_percentile: float = DEPARTURE_DEPTH_PERCENTILE) -> float:
    """Set `.departure` on every table: a point is a departure if the hand
    line is more than `offset_mm` from the geometric minimum, or the dip
    there is shallower than the pooled `depth_percentile` of dip depths (no
    clear minimum to follow). Points with no profile on the mesh count as
    departures. Returns the depth threshold used (same units as the
    local_relief channel)."""
    thr = depth_threshold(tables, depth_percentile)
    for t in tables:
        with np.errstate(invalid="ignore"):
            t.departure = ~((np.abs(t.geo_offset_mm) <= offset_mm) & (t.geo_depth >= thr))
    return thr
