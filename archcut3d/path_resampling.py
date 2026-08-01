"""
Resample a raw, unevenly-spaced 3-D polyline (e.g. a mouse-traced path) into
points evenly spaced by arc length. This is the generic prerequisite for the
curved-model workflow: once the user has traced a path, it needs to become a
sequence of `step_size`-spaced points before per-step local tangent-plane
fitting and slicing can run on it.

Mirrors compute_slice_path's stepping convention (slicing.py): steps land at
i/num_steps of the total length for i = 1..num_steps, so the resampled path
never includes the very first raw point, only points from one step onward --
same as the existing straight-line slicing already does.
"""

import numpy as np


def resample_path_by_arc_length(raw_path, step_size):
    """Return (resampled_points, num_steps): resampled_points is
    (num_steps, 3), each row a point on the raw path at the corresponding
    cumulative-arc-length fraction, linearly interpolated between whichever
    two raw points bracket it.
    """
    raw_path = np.asarray(raw_path, dtype=np.float64)
    if raw_path.shape[0] < 2:
        raise ValueError('Need at least 2 points to resample a path.')

    deltas = np.linalg.norm(np.diff(raw_path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = cumulative[-1]

    if total_length <= 0:
        raise ValueError('Path has zero length -- all traced points coincide.')

    num_steps = int(np.ceil(total_length / step_size))
    if num_steps < 1:
        raise ValueError('Path is shorter than the step size -- trace a longer path or use a smaller step size.')

    target_lengths = (np.arange(1, num_steps + 1) / num_steps) * total_length

    resampled = np.empty((num_steps, 3), dtype=np.float64)
    for axis in range(3):
        resampled[:, axis] = np.interp(target_lengths, cumulative, raw_path[:, axis])

    return resampled, num_steps
