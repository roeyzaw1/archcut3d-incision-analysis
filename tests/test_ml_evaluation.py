import numpy as np

from archcut3d.ml.anchors import build_anchor_table, label_departures
from archcut3d.ml.evaluation import evaluate_policy
from archcut3d.ml.policies import GeometricMinimumPolicy, StraightAheadPolicy
from tests.test_ml_policies import CENTER_ROW, PX, groove_example


def _setup(row_offset_mm):
    ex = groove_example()
    row = CENTER_ROW + row_offset_mm / PX
    ex.traces = [np.array([[float(c), row] for c in range(40, 190)])]
    table = build_anchor_table(ex)
    return ex, table


def test_hand_line_on_the_groove_everyone_is_accurate_and_nothing_departs():
    ex, table = _setup(0.0)
    label_departures([table], depth_percentile=1.0)
    assert not table.departure.any()
    for pol, can_stop in ((GeometricMinimumPolicy(), False), (StraightAheadPolicy(), False)):
        seg, traces, paths = evaluate_policy(pol, ex, table, can_stop)
        assert seg["departure"]["n_points"] == 0 and np.isnan(seg["departure"]["geo_to_pred_mm"])
        assert seg["all"]["step_dist_mm"] < 0.03 and seg["all"]["chamfer_mm"] < 0.05
        assert abs(seg["all"]["length_ratio"] - 1.0) < 0.1 and seg["all"]["coverage_0.1"] > 0.95
        assert len(traces) == 1 and len(paths) == 1


def test_hand_line_beside_the_groove_splits_the_policies_on_departure_points():
    ex, table = _setup(0.3)                       # hand line 0.3 mm from the groove: all departures
    label_departures([table], depth_percentile=1.0)
    assert table.departure.all()
    straight, _, _ = evaluate_policy(StraightAheadPolicy(), ex, table, can_stop=False)
    geo, _, _ = evaluate_policy(GeometricMinimumPolicy(), ex, table, can_stop=False)

    # straight-ahead reproduces the (straight) hand line; the geometric tracer goes to the groove
    assert straight["departure"]["cl_hand_to_pred_mm"] < 0.05 and straight["departure"]["frac_closer_to_hand"] == 1.0
    # (the first few points are closer to the hand line: the tracer starts on it and needs a step or two to move over)
    assert geo["departure"]["cl_hand_to_pred_mm"] > 0.2 and geo["departure"]["frac_closer_to_hand"] < 0.2
    assert geo["departure"]["geo_to_pred_mm"] < 0.05 and straight["departure"]["geo_to_pred_mm"] > 0.2
    assert geo["departure"]["step_lateral_abs_mm"] > 0.2 > straight["departure"]["step_lateral_abs_mm"]
    # one-step view, no closed-loop drift: the geometric tracer is by definition at the geometric
    # tracer's step; straight-ahead is at the hand's next point, 0.3 mm from it
    assert geo["departure"]["step_to_geo_mm"] < 1e-9 and geo["departure"]["frac_step_closer_to_hand"] == 0.0
    assert abs(straight["departure"]["step_to_geo_mm"] - 0.3) < 0.05 and straight["departure"]["frac_step_closer_to_hand"] == 1.0
    assert straight["departure"]["n_path_nearby"] == straight["departure"]["n_points"]
    # normal segment is empty here
    assert geo["normal"]["n_points"] == 0 and np.isnan(geo["normal"]["step_dist_mm"])
    assert "max_steps" in geo["all"]["stop_reasons"]


def test_closer_to_hand_is_only_counted_where_the_path_is_nearby():
    # a path 2 mm from both the hand line and the groove says nothing about which it follows
    ex, table = _setup(0.3)
    label_departures([table], depth_percentile=1.0)

    class Veer:                      # steers steadily away from everything
        can_stop = False

        def __call__(self, patch, mask):
            return np.array([1.0, 0.8]), 0.0

    seg, _, _ = evaluate_policy(Veer(), ex, table, can_stop=False)
    assert seg["departure"]["n_path_nearby"] < seg["departure"]["n_points"]


def test_corr_slope_helper():
    from archcut3d.ml.evaluation import _corr_slope
    x = np.linspace(-1, 1, 20)
    r, slope = _corr_slope(0.5 * x + 0.1, x)
    assert abs(r - 1.0) < 1e-9 and abs(slope - 0.5) < 1e-9
    assert np.isnan(_corr_slope(np.zeros(20), x)[0])                 # no variation in the policy
    assert np.isnan(_corr_slope(x[:3], x[:3])[0])                    # too few points


def test_jittered_poses_separate_geometry_seeking_from_straight_on():
    ex, table = _setup(0.3)                       # hand line 0.3 mm beside the groove
    label_departures([table], depth_percentile=1.0)
    geo, _, _ = evaluate_policy(GeometricMinimumPolicy(), ex, table, can_stop=False)
    straight, _, _ = evaluate_policy(StraightAheadPolicy(), ex, table, can_stop=False)
    g, s = geo["departure"], straight["departure"]
    assert g["jit_frac_closer_to_hand"] == 0.0 and g["jit_to_geo_next_mm"] < 1e-9
    assert s["jit_frac_closer_to_hand"] > 0.5      # a displaced pose stepping on stays nearer the hand line
    assert g["jit_pred_to_line_mm"] > 0.15         # the geometric tracer heads for the groove, 0.3 mm from the hand line
    assert s["jit_start_to_line_mm"] > 0.0 and abs(s["jit_start_to_line_mm"] - g["jit_start_to_line_mm"]) < 1e-9
    # both views are identical for every policy: same poses
    again, _, _ = evaluate_policy(StraightAheadPolicy(), ex, table, can_stop=False)
    assert again["departure"]["jit_pred_to_line_mm"] == s["jit_pred_to_line_mm"]
