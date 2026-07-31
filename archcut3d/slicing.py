"""
Port of getSlices.m -- the core Cut3-D operation: walk a path between two
points on the model and, at regular tolerance-sized steps, extract a
cross-section (y,z profile) of the surface within a width-wide band.

Deliberately excludes the numosc/FFT path-smoothing MATLAB source had: it's
broken there (numosc=0 zeros the whole signal instead of disabling
smoothing) and isn't really used in practice -- confirmed with the user.
Otherwise this replicates the point-cloud-windowing approach faithfully
(not a topological mesh cut), matching MATLAB's numbers rather than
"modernizing" the algorithm.

Two steps, same as the MATLAB source:
  1. compute_slice_path(): the path of minima (or straight interpolation)
     between the two endpoints -- this IS the incision-following centerline.
  2. compute_slices(): per consecutive pair of path points, the actual
     (y, z) cross-section profile.
"""

import numpy as np


def _z_axis_rotation(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _direction_angle(dx, dy):
    """Angle that rotates a direction vector (dx, dy) onto the +X axis,
    matching getSlices.m's `if direct(1)~=0 ... else ...` branch exactly."""
    if dx != 0:
        return -np.arctan(dy / dx)
    return -np.sign(dy) * np.pi / 2


def compute_slice_path(vertices, c1, c2, tolerance, width, tipo=1):
    """Walk from c1 to c2 in `tolerance`-sized steps; at each step, find the
    lowest point (corrected for a tilted wall) within a `width`-wide band
    perpendicular to the path. This traces the bottom of the incision.

    tipo=1 ("minima"): path Y at each step is the actual Y of the point
           where the minimum was found (follows the incision if it wanders).
    tipo=0 ("straight"): path Y is the straight interpolation between the
           two endpoints (ignores any wander).

    Returns (minimav, num_slices): minimav is (num_slices, 3) in the
    model's original coordinate frame.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    direct = c2 - c1
    dist = np.linalg.norm(direct)
    num_slices = int(np.ceil(dist / tolerance))
    if num_slices < 2:
        raise ValueError('Segment shorter than tolerance -- pick a smaller tolerance or farther-apart points.')

    angle = _direction_angle(direct[0], direct[1])
    rz = _z_axis_rotation(angle)

    vn = vertices @ rz.T
    c1n = rz @ c1
    c2n = rz @ c2

    minima = np.zeros((num_slices, 3))
    for i in range(1, num_slices + 1):
        ci = c1n + (c2n - c1n) * i / num_slices
        cj = c1n + (c2n - c1n) * (i - 1) / num_slices

        vx = vn[:, 0]
        lo, hi = (cj[0], ci[0]) if cj[0] <= ci[0] else (ci[0], cj[0])
        band = (vx >= lo) & (vx <= hi)
        pp = vn[band]
        if pp.shape[0] == 0:
            raise ValueError(f'No points found in the x-band for step {i}/{num_slices} -- tolerance too small?')

        vy = pp[:, 1]
        band_y = (vy >= ci[1] - width / 2) & (vy <= ci[1] + width / 2)
        ppf = pp[band_y]
        if ppf.shape[0] == 0:
            raise ValueError(f'No points found in the y-band for step {i}/{num_slices} -- width too small?')
        ppff = ppf[:, 1:3]  # (y, z), matching MATLAB's ppf(:,2:3)

        # Correct for a wall not parallel to the path direction: subtract the
        # straight line between the band's first/last points (in z vs y)
        # before finding the minimum, then read the ORIGINAL z there.
        y0, z0 = ppff[0, 0], ppff[0, 1]
        y1, z1 = ppff[-1, 0], ppff[-1, 1]
        if y1 != y0:
            corrected = ppff[:, 1] - ((z1 - z0) / (y1 - y0)) * (ppff[:, 0] - y0)
        else:
            corrected = ppff[:, 1]
        min_idx = int(np.argmin(corrected))
        minpz = ppff[min_idx, 1]

        row = i - 1
        minima[row, 0] = ci[0]
        minima[row, 1] = ppff[min_idx, 0] * tipo + ci[1] * (1 - tipo)
        minima[row, 2] = minpz

    minimav = minima @ rz  # rotate back (rz is orthogonal: inv(rz) == rz.T, so minima @ rz.T.T == minima @ rz)
    return minimav, num_slices


def compute_slices(vertices, minimav, width):
    """For each consecutive pair of path points, extract that slice's
    (y, z) profile by windowing the original vertex cloud around a local
    rotation that aligns the two path points along the local x-axis.
    Returns a list of (K, 2) arrays, one per slice (len = len(minimav) - 1).
    """
    num_slices = minimav.shape[0] - 1
    slices = []
    for i in range(num_slices):
        dirminima = minimav[i + 1, :2] - minimav[i, :2]
        angle = _direction_angle(dirminima[0], dirminima[1])
        rz = _z_axis_rotation(angle)

        vn = vertices @ rz.T
        min1n = rz @ minimav[i]
        min2n = rz @ minimav[i + 1]

        vx = vn[:, 0]
        lo, hi = (min1n[0], min2n[0]) if min1n[0] <= min2n[0] else (min2n[0], min1n[0])
        band = (vx >= lo) & (vx <= hi)
        pp = vn[band]

        vy = pp[:, 1]
        band_y = (vy >= min1n[1] - width / 2) & (vy <= min1n[1] + width / 2)
        ppf = pp[band_y]
        ppff = ppf[:, 1:3]

        slices.append(ppff)
    return slices
