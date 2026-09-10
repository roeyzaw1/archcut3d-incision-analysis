"""
Model positioning math, ported directly from the MATLAB Cut3-D source.

Two independent pieces:
  - compute_initial_orientation(): the automatic positioning applied once
    when a .wrl is loaded (harutot.m's pushbutton_loadfile_Callback: center,
    getNormal.m, repositioningxyz.m, flip Y/Z, de-tilt).
  - compute_normal_alignment_axis_angle(): the on-demand "Align by Normal"
    tool (normal_positioning.m), a different, eigenvector-based normal
    estimate used later in the workflow, not at load time.

Pure numpy, no VTK or Qt dependency, so both are independently testable.
"""

import numpy as np


def compute_initial_orientation(vertices, faces):
    """Port of the positioning block in harutot.m's
    pushbutton_loadfile_Callback, run once right after a .wrl loads:
    center twice, compute an aggregate face normal (getNormal.m), reorient
    via repositioningxyz.m, flip Y and Z, then remove residual X-Z tilt via
    a line fit. Returns a new (N, 3) vertex array; face topology is
    untouched by any of this.
    """
    v = vertices.copy()

    # Center twice, matching the MATLAB source exactly (the second pass is
    # a no-op numerically but that's what the original does).
    v = v - v.mean(axis=0)

    normal = _get_normal(v, faces)
    v = _repositioningxyz(v, normal)

    v[:, 1] = -v[:, 1]
    v[:, 2] = -v[:, 2]

    # Remove residual tilt: fit a line to Z vs X, rotate around Y to level it.
    slope, _intercept = np.polyfit(v[:, 0], v[:, 2], 1)
    anglev = np.arctan(slope)
    ry = np.array([
        [np.cos(anglev), 0.0, np.sin(anglev)],
        [0.0, 1.0, 0.0],
        [-np.sin(anglev), 0.0, np.cos(anglev)],
    ])
    v = v @ ry.T

    return v


def _get_normal(vertices, faces):
    """Port of getNormal.m: sum of raw (unnormalized, so implicitly
    area-weighted) face normal vectors over an evenly-strided subsample of
    at most ~50000 faces, then normalize the sum. Different from (and
    simpler than) the eigenvector-based estimate used by
    compute_normal_alignment_axis_angle."""
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([0.0, 0.0, 0.0])

    step = max(1, -(-n_faces // 50000))  # ceil(n_faces / 50000)
    idx = np.arange(0, n_faces, step)

    p1 = vertices[faces[idx, 0]]
    p2 = vertices[faces[idx, 1]]
    p3 = vertices[faces[idx, 2]]

    nx = ((p2[:, 1] - p1[:, 1]) * (p3[:, 2] - p1[:, 2]) - (p3[:, 1] - p1[:, 1]) * (p2[:, 2] - p1[:, 2])) / 2.0
    ny = ((p2[:, 2] - p1[:, 2]) * (p3[:, 0] - p1[:, 0]) - (p3[:, 2] - p1[:, 2]) * (p2[:, 0] - p1[:, 0])) / 2.0
    nz = ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])) / 2.0

    total = np.array([nx.sum(), ny.sum(), nz.sum()])
    norm = np.linalg.norm(total)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return total / norm


def _repositioningxyz(v, normal):
    """Port of repositioningxyz.m."""
    vx, vy, vz = normal
    alpha = np.arccos(vz / np.sqrt(vz ** 2 + vx ** 2))
    gamma = np.arcsin(vy)

    ry = np.array([
        [np.cos(alpha), 0.0, np.sin(alpha)],
        [0.0, 1.0, 0.0],
        [-np.sin(alpha), 0.0, np.cos(alpha)],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(gamma), -np.sin(gamma)],
        [0.0, np.sin(gamma), np.cos(gamma)],
    ])

    v = v @ ry.T
    v = v @ rx.T
    return v


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


def rodrigues_matrix(axis, angle_rad):
    """3x3 rotation matrix for a rotation of angle_rad around an arbitrary
    unit (or non-unit; normalized here) axis -- Rodrigues' formula, the same
    one used throughout normal_positioning.m."""
    axis = axis / np.linalg.norm(axis)
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


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
