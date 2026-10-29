"""Closed-loop tracing: run a step policy along a grid and score the path.

A policy maps a heading-canonicalised patch to a step and a stop probability:

    policy(patch, mask) -> ((forward, right), stop_prob)

`patch` is the raw (R, R, C) crop returned by dataset.sample_patch (rotated
so the current heading points up), `mask` the matching (R, R) bool array.
The step is in units of `step_mm`: (1, 0) is one full step straight ahead,
(0, 1) one step to the right of the heading. Learned models and baselines
plug in through the same interface.

`trace` repeatedly samples a patch at the current pose, asks the policy for
a step, moves, and sets the heading to the direction just travelled
(optionally smoothed). It ends when the policy's stop probability exceeds a
threshold, when the position leaves the mask, or after `max_steps`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .dataset import (DEFAULT_PATCH_MM, DEFAULT_PATCH_RES, DEFAULT_STEP_MM,
                      resample_polyline, sample_patch)

Policy = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, float]"]

# Multiple of the hand-trace length a stoppable policy may travel before it
# is cut off; policies that cannot stop get exactly the hand-trace length.
DEFAULT_BUDGET_FACTOR = 1.5


@dataclass
class TraceResult:
    path: np.ndarray                  # (N, 2) [col, row]; path[0] is the start
    stop_reason: str                  # "stop", "left_mask", "zero_step" or "max_steps"
    stop_probs: list = field(default_factory=list)   # stop probability at each visited pose

    @property
    def n_steps(self) -> int:
        return len(self.path) - 1


def _inside_mask(mask: np.ndarray, colrow) -> bool:
    h, w = mask.shape
    c = int(round(float(colrow[0])))
    r = int(round(float(colrow[1])))
    return 0 <= c < w and 0 <= r < h and bool(mask[r, c])


def default_max_steps(hand_length_mm: float, step_mm: float, budget_factor: float = DEFAULT_BUDGET_FACTOR) -> int:
    return max(1, int(np.ceil(budget_factor * hand_length_mm / step_mm)))


def trace(policy: Policy, example, start_colrow, start_heading,
          step_mm: float = DEFAULT_STEP_MM, max_steps: Optional[int] = None,
          hand_length_mm: Optional[float] = None, stop_threshold: float = 0.5,
          heading_smoothing: float = 0.0, patch_mm: float = DEFAULT_PATCH_MM,
          out_res: int = DEFAULT_PATCH_RES) -> TraceResult:
    """Run `policy` from a start pose.

    example          : object with channels (H,W,C), mask (H,W), pixel_size
    start_colrow     : starting [col, row]
    start_heading    : [dcol, drow] direction of the first step (need not be unit)
    max_steps        : step budget; if None, `hand_length_mm` must be given and
                       the budget is DEFAULT_BUDGET_FACTOR * hand_length_mm / step_mm
    heading_smoothing: 0 = heading is the direction of the last step; in
                       (0, 1) the new heading is the normalised mix
                       smoothing * old + (1 - smoothing) * last step direction
    """
    if max_steps is None:
        if hand_length_mm is None:
            raise ValueError("give max_steps or hand_length_mm")
        max_steps = default_max_steps(hand_length_mm, step_mm)
    if not 0.0 <= heading_smoothing < 1.0:
        raise ValueError("heading_smoothing must be in [0, 1)")

    step_px = step_mm / example.pixel_size
    pos = np.asarray(start_colrow, dtype=np.float64).copy()
    heading = np.asarray(start_heading, dtype=np.float64)
    heading = heading / np.linalg.norm(heading)

    path = [pos.copy()]
    stop_probs = []
    reason = "max_steps"

    for _ in range(max_steps):
        if not _inside_mask(example.mask, pos):
            reason = "left_mask"
            break
        patch, pmask = sample_patch(example.channels, example.mask, example.pixel_size,
                                    pos, heading, patch_mm, out_res)
        (fwd, right), stop_p = policy(patch, pmask)
        stop_probs.append(float(stop_p))
        if stop_p > stop_threshold:
            reason = "stop"
            break

        right_vec = np.array([-heading[1], heading[0]])
        delta = (fwd * heading + right * right_vec) * step_px
        norm = np.linalg.norm(delta)
        if norm < 1e-9:
            reason = "zero_step"
            break

        direction = delta / norm
        if heading_smoothing > 0.0:
            mixed = heading_smoothing * heading + (1.0 - heading_smoothing) * direction
            m = np.linalg.norm(mixed)
            heading = mixed / m if m > 1e-12 else direction
        else:
            heading = direction
        pos = pos + delta
        path.append(pos.copy())

    return TraceResult(path=np.array(path), stop_reason=reason, stop_probs=stop_probs)


def start_pose_from_trace(trace_colrow: np.ndarray, pixel_size: float, step_mm: float = DEFAULT_STEP_MM):
    """Start point and heading for tracing a hand-drawn line: the first point
    of the line resampled at `step_mm`, heading from it to the second
    resampled point. Returns (start_colrow, heading, resampled_line)."""
    line = resample_polyline(trace_colrow, step_mm, pixel_size)
    if len(line) < 2:
        raise ValueError("trace too short to define a heading")
    heading = line[1] - line[0]
    return line[0], heading / np.linalg.norm(heading), line


def polyline_length_mm(line_colrow: np.ndarray, pixel_size: float) -> float:
    if len(line_colrow) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line_colrow, axis=0), axis=1).sum() * pixel_size)


def points_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Exact distance from each point (N, 2) to a polyline (M, 2), same units
    as the inputs. A one-point polyline is treated as a single point."""
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    a = polyline[:-1][None, :, :]                  # (1, S, 2)
    b = polyline[1:][None, :, :]
    p = points[:, None, :]                         # (N, 1, 2)
    ab = b - a
    denom = (ab ** 2).sum(axis=2)
    t = np.where(denom > 0, ((p - a) * ab).sum(axis=2) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, :, None] * ab
    return np.linalg.norm(p - closest, axis=2).min(axis=1)


COVERAGE_THRESHOLDS_MM = (0.05, 0.1, 0.2)


def trace_metrics(pred_path: np.ndarray, hand_line: np.ndarray, pixel_size: float,
                  step_mm: float = DEFAULT_STEP_MM) -> dict:
    """Closed-loop metrics for one predicted path against one hand line (both
    [col, row] in grid pixels). Distances in mm.

    Both lines are resampled to `step_mm` spacing first, so every metric is
    weighted by arc length rather than by how many points a line happens to
    have. Keys:
      pred_to_hand_mm   mean distance from predicted points to the hand polyline
      hand_to_pred_mm   mean distance from hand points to the predicted polyline
      chamfer_mm        average of the two
      coverage_<t>      fraction of hand points within t mm of the predicted path
      length_ratio      predicted length / hand length
      endpoint_error_mm distance between the two end points
    """
    pred = resample_polyline(pred_path, step_mm, pixel_size) if len(pred_path) > 1 else np.asarray(pred_path, float)
    hand = resample_polyline(hand_line, step_mm, pixel_size)

    d_pred = points_to_polyline_distance(pred, hand) * pixel_size
    d_hand = points_to_polyline_distance(hand, pred) * pixel_size
    out = {
        "pred_to_hand_mm": float(d_pred.mean()),
        "hand_to_pred_mm": float(d_hand.mean()),
    }
    out["chamfer_mm"] = 0.5 * (out["pred_to_hand_mm"] + out["hand_to_pred_mm"])
    for t in COVERAGE_THRESHOLDS_MM:
        out[f"coverage_{t:g}"] = float((d_hand <= t).mean())
    hand_len = polyline_length_mm(hand, pixel_size)
    out["length_ratio"] = polyline_length_mm(pred, pixel_size) / hand_len if hand_len > 0 else float("nan")
    out["endpoint_error_mm"] = float(np.linalg.norm(pred[-1] - hand[-1]) * pixel_size)
    return out


def hand_point_distances(pred_path: np.ndarray, hand_points: np.ndarray, pixel_size: float) -> np.ndarray:
    """Distance (mm) from each of `hand_points` to the predicted path; used to
    split closed-loop distance by segment type."""
    pred = np.asarray(pred_path, dtype=np.float64)
    return points_to_polyline_distance(hand_points, pred) * pixel_size


def trace_hand_line(policy: Policy, example, hand_trace_colrow: np.ndarray,
                    can_stop: bool = True, step_mm: float = DEFAULT_STEP_MM, **trace_kw):
    """Trace from the start of one hand-drawn line and score the result.

    A policy that can stop is given DEFAULT_BUDGET_FACTOR times the hand
    length as a step budget and decides the end itself. A policy that cannot
    stop (can_stop=False) is given exactly the hand length: this is an oracle
    budget and must be stated wherever such results are reported.
    Returns (TraceResult, metrics dict, resampled hand line)."""
    start, heading, hand_line = start_pose_from_trace(hand_trace_colrow, example.pixel_size, step_mm)
    hand_len = polyline_length_mm(hand_line, example.pixel_size)
    if can_stop:
        max_steps = default_max_steps(hand_len, step_mm)
    else:
        max_steps = max(1, int(round(hand_len / step_mm)))
    res = trace(policy, example, start, heading, step_mm=step_mm, max_steps=max_steps, **trace_kw)
    return res, trace_metrics(res.path, hand_line, example.pixel_size, step_mm), hand_line
