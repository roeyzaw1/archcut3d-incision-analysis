import numpy as np

from archcut3d.ml.anchors import build_anchor_table, depth_threshold, label_departures
from tests.test_ml_policies import CENTER_ROW, PX, groove_example


def _example_with_trace(points):
    ex = groove_example()
    ex.traces = [np.asarray(points, float)]
    return ex


def _line(col0, col1, row):
    return [[float(c), float(row)] for c in range(col0, col1)]


def test_hand_line_on_the_groove_has_small_offset_and_positive_depth():
    ex = _example_with_trace(_line(40, 160, CENTER_ROW))
    t = build_anchor_table(ex)
    assert len(t) > 10
    assert np.nanmax(np.abs(t.geo_offset_mm)) < 0.03
    assert np.nanmin(t.geo_depth) > 0.1                      # groove depth is 0.2 mm
    # geometric minimum position lies on the groove centre line
    assert np.nanmax(np.abs(t.geo_pos[:, 1] - CENTER_ROW)) * PX < 0.03


def test_offset_is_signed_right_positive_and_matches_the_displacement():
    row = CENTER_ROW + 0.2 / PX                               # hand line 0.2 mm to the +row side
    t = build_anchor_table(_example_with_trace(_line(40, 160, row)))
    # heading +col => right is +row; the groove is to the LEFT of the hand line => negative offset
    assert abs(np.nanmedian(t.geo_offset_mm) - (-0.2)) < 0.03


def test_departure_labels_split_a_trace_that_leaves_the_groove():
    # first half follows the groove, second half runs 0.25 mm beside it
    pts = _line(40, 100, CENTER_ROW) + _line(100, 160, CENTER_ROW + 0.25 / PX)
    t = build_anchor_table(_example_with_trace(pts))
    thr = label_departures([t])
    cols = t.pos[:, 0]
    early, late = t.departure[cols < 90], t.departure[cols > 115]
    assert early.mean() < 0.1 and late.mean() > 0.9
    assert thr > 0


def test_flat_surface_points_are_departures_by_depth():
    ex = _example_with_trace(_line(40, 160, CENTER_ROW))
    flat = groove_example(depth_mm=0.0)
    flat.traces = [np.asarray(_line(40, 160, 150.0), float)]
    t_groove, t_flat = build_anchor_table(ex), build_anchor_table(flat)
    thr = depth_threshold([t_groove, t_flat], 50.0)           # half of all points are shallower than this
    label_departures([t_groove, t_flat], depth_percentile=50.0)
    assert t_flat.departure.all() and not t_groove.departure.any()
    assert thr > 0


def test_depth_threshold_ignores_band_edge_minima():
    from archcut3d.ml.anchors import AnchorTable
    n = 10
    z = np.zeros((n, 2))
    base = dict(source_file="x", trace=np.zeros(n, int), point=np.arange(n), pos=z, heading=z, next_pos=z,
                geo_pos=z, pixel_size=0.05, half_width_mm=0.5)
    # 5 edge minima (depth 0) and 5 interior dips of depth 0.1..0.5
    t = AnchorTable(geo_offset_mm=np.array([0.5] * 5 + [0.0] * 5),
                    geo_depth=np.array([0.0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5]), **base)
    assert abs(depth_threshold([t], 0.0) - 0.1) < 1e-12 and abs(depth_threshold([t], 100.0) - 0.5) < 1e-12
    thr = label_departures([t], depth_percentile=0.0)
    assert t.departure[:5].all() and not t.departure[5:].any() and thr == 0.1
