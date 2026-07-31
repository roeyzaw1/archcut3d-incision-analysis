"""
Normal-alignment auto-positioning.

Port of normal_positioning.m: finds the mesh's dominant area-weighted surface
normal direction (via the eigenvector of the largest eigenvalue of the
area-weighted normal covariance tensor) and returns the rotation that aligns
that direction with +Z, via Rodrigues' rotation formula. Pure numpy, no VTK
or Qt dependency, so it's independently testable.
"""

import numpy as np


def _axis_angle_between(a, b):
    """Axis (unit vector) and angle (degrees) of the rotation that takes
    unit vector `a` onto unit vector `b`. Returned in a form directly usable
    by vtkTransform.RotateWXYZ(angle, *axis) -- the same primitive the
    existing rotate_x/y/z_left/right buttons already use, so composing this
    into self.transform behaves identically to a manual button click."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    if s < 1e-10:
        if c > 0:
            # Already aligned: zero-degree rotation, axis is irrelevant.
            return np.array([0.0, 0.0, 1.0]), 0.0
        # Anti-parallel: 180 degrees around any axis orthogonal to a.
        if abs(a[0]) < abs(a[1]):
            ortho = np.array([0.0, -a[2], a[1]])
        else:
            ortho = np.array([-a[2], 0.0, a[0]])
        ortho = ortho / np.linalg.norm(ortho)
        return ortho, 180.0

    axis = v / s
    angle = np.degrees(np.arctan2(s, c))
    return axis, angle


def compute_normal_alignment_axis_angle(points, faces):
    """Return (axis, angle_degrees) of the rotation that aligns the mesh's
    dominant area-weighted surface-normal direction with +Z.

    points: (N, 3) float array of vertex coordinates.
    faces:  (M, 3) int array of triangle vertex indices.
    """
    p1 = points[faces[:, 0]]
    p2 = points[faces[:, 1]]
    p3 = points[faces[:, 2]]

    cross = np.cross(p2 - p1, p3 - p1)
    cross_norm = np.linalg.norm(cross, axis=1)
    valid = cross_norm > 0

    areas = cross_norm / 2.0
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / cross_norm[valid, None]

    total_area = areas.sum()
    if total_area <= 0:
        return np.array([0.0, 0.0, 1.0]), 0.0

    weighted = normals * areas[:, None]
    tensor = (weighted.T @ normals) / total_area

    eigvals, eigvecs = np.linalg.eigh(tensor)
    dominant_direction = eigvecs[:, np.argmax(eigvals)]

    return _axis_angle_between(dominant_direction, np.array([0.0, 0.0, 1.0]))
