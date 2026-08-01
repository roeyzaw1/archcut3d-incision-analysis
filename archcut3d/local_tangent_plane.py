"""
Local, curvature-weighted tangent-plane fitting -- for the curved/closed
model workflow (experiment/closed-model-workflow branch), where there's no
single global "up" direction for the whole object the way there is for a
flat, single-sided scan.

Reuses the same weighted-normal-covariance-tensor + dominant-eigenvector
math as normal_positioning.m / compute_normal_alignment_axis_angle (see
positioning.py), just re-scoped:
  - restricted to faces near a given point, instead of the whole mesh
  - weighted by curvature-based flatness, instead of by area alone

The idea (from the design discussion): near an incision, the surrounding
"ground" is more trustworthy for defining the local surface plane than the
incision walls themselves, because the incision is exactly the feature
that's *not* part of the surrounding surface. Flatter points get more say.
"""

import numpy as np

from .curvature import flatness_score
from .positioning import dominant_direction_axis_angle, face_normals_and_areas


def compute_local_tangent_plane_axis_angle(point, points, faces, radius, k1, k2, target=None):
    """Fit a local tangent plane near `point`, using faces within `radius`,
    weighted by (face area) x (curvature-based flatness).

    k1, k2: per-vertex principal curvatures for the WHOLE mesh, precomputed
            once via curvature.compute_principal_curvatures() and passed in,
            so this can be called many times (once per path step) without
            recomputing curvature each time.

    Returns (axis, angle_degrees, num_faces_used). num_faces_used is a
    diagnostic -- if it's small, the neighborhood was probably too tight.
    """
    point = np.asarray(point, dtype=np.float64)

    vertex_dist = np.linalg.norm(points - point, axis=1)
    near_vertex = vertex_dist <= radius
    face_near = near_vertex[faces].any(axis=1)
    local_faces = faces[face_near]

    if local_faces.shape[0] == 0:
        raise ValueError(f'No faces found within radius {radius} of the point -- try a larger radius.')

    normals, areas = face_normals_and_areas(points, local_faces)

    flatness = flatness_score(k1, k2)
    face_flatness = flatness[local_faces].mean(axis=1)
    curvature_weight = 1.0 / (1.0 + face_flatness)

    weights = areas * curvature_weight

    axis, angle = dominant_direction_axis_angle(normals, weights, target=target)
    return axis, angle, local_faces.shape[0]
