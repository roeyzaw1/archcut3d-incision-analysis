"""
Per-vertex curvature via PyMeshLab, computed natively rather than shelling
out to the setCurvature binary crest detection uses -- this is standard
differential geometry (quadric fitting), no reason to depend on an external
executable for it.

Only used by the local-tangent-plane fitting for the curved-model workflow
(experiment/closed-model-workflow branch). crest_detection.py's
setCurvature usage is untouched -- swapping that over is a separate
decision, not bundled into this.
"""

import numpy as np
import pymeshlab


def compute_principal_curvatures(vertices, faces):
    """Return (k1, k2) per vertex: the two principal curvatures from quadric
    fitting. NOTE: verified against a sphere, cylinder, and flat plane --
    the magnitudes are correct, but PyMeshLab's "Min Curvature"/"Max
    Curvature" labels do NOT reliably satisfy k1 <= k2 (confirmed on a
    cylinder, where the two calls can come back either order). Treat this
    as an unordered pair, not a (min, max) pair -- flatness_score() below is
    written to not care about the order."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.ascontiguousarray(vertices, dtype=np.float64),
        face_matrix=np.ascontiguousarray(faces, dtype=np.int32),
    ))

    ms.compute_curvature_principal_directions_per_vertex(curvcolormethod='Min Curvature')
    k1 = ms.current_mesh().vertex_scalar_array().copy()

    ms.compute_curvature_principal_directions_per_vertex(curvcolormethod='Max Curvature')
    k2 = ms.current_mesh().vertex_scalar_array().copy()

    return k1, k2


def flatness_score(k1, k2):
    """How far each vertex is from flat: 0 = perfectly flat, larger = more
    curved. Uses whichever principal curvature has the larger magnitude, so
    it catches both dome-like curving (both directions curved) and
    ridge/valley-like curving (only one direction curved, where Gaussian
    curvature alone would read near zero and miss it). Symmetric in
    (k1, k2), so it doesn't matter that they aren't reliably ordered."""
    return np.maximum(np.abs(k1), np.abs(k2))
