"""Evaluate a step policy against the hand traces of one model.

Two views, each reported on all hand points and separately on "departure"
and "normal" points (see anchors.py):

One-step: for every resampled hand point (an AnchorTable row) put the
policy at the hand point with the hand heading, take one step, and compare
with the next hand point:
    step_dist_mm        mean distance between predicted and hand next point
    step_angle_deg      mean angle between predicted and hand step vectors
    step_lateral_abs_mm mean |sideways error| in the heading frame
    step_to_geo_mm      mean distance between the predicted next point and the
                        geometric tracer's next point from the same pose
    frac_step_closer_to_hand
                        fraction of points where the predicted next point is
                        closer to the hand's next point than to the geometric
                        tracer's (no closed-loop drift involved)

Closed loop: trace from the start of every hand line (rollout.trace_hand_line)
and measure, per hand point, the distance to the predicted path:
    cl_hand_to_pred_mm  mean distance from hand points to the predicted path
On departure points, also:
    geo_to_pred_mm      mean distance from the geometric-minimum position
                        (at that hand point) to the predicted path
    frac_closer_to_hand fraction of departure points where the predicted
                        path is closer to the hand point than to the
                        geometric-minimum position, counting only points
                        where the path passes within NEARBY_MM of either
                        (once it has drifted further away the comparison says
                        nothing about which one it was following);
                        n_path_nearby is the number of points counted
Path-level metrics (segment "all" only, hand-length weighted over traces):
chamfer, both one-way distances, coverage, length ratio, endpoint error.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .anchors import AnchorTable
from .dataset import DEFAULT_PATCH_MM, DEFAULT_PATCH_RES, DEFAULT_STEP_MM, sample_patch
from .policies import GeometricMinimumPolicy
from .rollout import (COVERAGE_THRESHOLDS_MM, hand_point_distances, points_to_polyline_distance,
                      polyline_length_mm, trace_hand_line)

NEARBY_MM = 0.5

PATH_METRICS = ["pred_to_hand_mm", "hand_to_pred_mm", "chamfer_mm"] + \
               [f"coverage_{t:g}" for t in COVERAGE_THRESHOLDS_MM] + ["length_ratio", "endpoint_error_mm"]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    cos = np.full(len(a), np.nan)
    cos[ok] = np.clip((a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok]), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def one_step_errors(policy, example, table: AnchorTable, step_mm: float = DEFAULT_STEP_MM,
                    patch_mm: float = DEFAULT_PATCH_MM, out_res: int = DEFAULT_PATCH_RES,
                    geo_ref=None):
    """Per-hand-point one-step errors; returns dict of (K,) arrays.
    `geo_ref` is the geometric tracer whose step is the reference for
    step_to_geo_mm (default: GeometricMinimumPolicy with its default band)."""
    if geo_ref is None:
        geo_ref = GeometricMinimumPolicy(step_mm=step_mm, patch_mm=patch_mm, out_res=out_res)
    px = example.pixel_size
    step_px = step_mm / px
    K = len(table)
    dist, lat, to_geo, closer = (np.full(K, np.nan) for _ in range(4))
    pred_step_vec = np.zeros((K, 2))
    for k in range(K):
        h = table.heading[k]
        patch, mask = sample_patch(example.channels, example.mask, px, table.pos[k], h, patch_mm, out_res)
        rvec = np.array([-h[1], h[0]])
        (fwd, right), _ = policy(patch, mask)
        pred_vec = (fwd * h + right * rvec) * step_px
        (gf, gr), _ = geo_ref(patch, mask)
        geo_vec = (gf * h + gr * rvec) * step_px
        pred_step_vec[k] = pred_vec
        err = (table.pos[k] + pred_vec) - table.next_pos[k]
        dist[k] = np.linalg.norm(err) * px
        lat[k] = abs(float(err @ rvec)) * px
        to_geo[k] = np.linalg.norm(pred_vec - geo_vec) * px
        closer[k] = float(dist[k] < to_geo[k])
    ang = _angle_deg(pred_step_vec, table.next_pos - table.pos)
    return {"step_dist_mm": dist, "step_angle_deg": ang, "step_lateral_abs_mm": lat,
            "step_to_geo_mm": to_geo, "step_closer_to_hand": closer}


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def evaluate_policy(policy, example, table: AnchorTable, can_stop: bool,
                    step_mm: float = DEFAULT_STEP_MM, **trace_kw):
    """Evaluate one policy on one model. `table.departure` must be set.

    Returns (segments, traces, paths): segments maps "all" / "departure" /
    "normal" to a dict of metrics; traces is a list of per-trace dicts;
    paths is the list of predicted [col, row] paths (one per hand trace that
    could be traced)."""
    px = example.pixel_size
    K = len(table)
    one = one_step_errors(policy, example, table, step_mm)

    d_hand = np.full(K, np.nan)
    d_geo = np.full(K, np.nan)
    traces, paths, weights = [], [], []
    for ti, raw in enumerate(example.traces):
        rows = np.flatnonzero(table.trace == ti)
        if len(rows) == 0:
            continue
        try:
            res, m, hand_line = trace_hand_line(policy, example, raw, can_stop=can_stop, step_mm=step_mm, **trace_kw)
        except ValueError:
            continue
        d_hand[rows] = hand_point_distances(res.path, table.pos[rows], px)
        geo_ok = rows[np.isfinite(table.geo_pos[rows, 0])]
        d_geo[geo_ok] = points_to_polyline_distance(table.geo_pos[geo_ok], res.path) * px
        traces.append({"trace": ti, "n_steps": res.n_steps, "stop_reason": res.stop_reason,
                       "hand_length_mm": polyline_length_mm(hand_line, px), **m})
        paths.append(res.path)
        weights.append(polyline_length_mm(hand_line, px))

    dep = np.asarray(table.departure, dtype=bool)
    seg_masks = {"all": np.ones(K, bool), "departure": dep, "normal": ~dep}
    segments = {}
    for name, sel in seg_masks.items():
        row = {"n_points": int(sel.sum()),
               "step_dist_mm": _nanmean(one["step_dist_mm"][sel]),
               "step_angle_deg": _nanmean(one["step_angle_deg"][sel]),
               "step_lateral_abs_mm": _nanmean(one["step_lateral_abs_mm"][sel]),
               "step_to_geo_mm": _nanmean(one["step_to_geo_mm"][sel]),
               "frac_step_closer_to_hand": _nanmean(one["step_closer_to_hand"][sel]),
               "cl_hand_to_pred_mm": _nanmean(d_hand[sel])}
        if name == "departure":
            row["geo_to_pred_mm"] = _nanmean(d_geo[sel])
            near = sel & np.isfinite(d_hand) & np.isfinite(d_geo) & (np.fmin(d_hand, d_geo) <= NEARBY_MM)
            row["n_path_nearby"] = int(near.sum())
            row["frac_closer_to_hand"] = float((d_hand[near] < d_geo[near]).mean()) if near.any() else float("nan")
        if name == "all" and traces:
            w = np.array(weights)
            for k in PATH_METRICS:
                vals = np.array([t[k] for t in traces], dtype=float)
                ok = np.isfinite(vals)
                row[k] = float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
            row["stop_reasons"] = ";".join(f"{r}:{n}" for r, n in sorted(Counter(t["stop_reason"] for t in traces).items()))
            row["n_traces"] = len(traces)
        segments[name] = row
    return segments, traces, paths
