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
    # normal segment is empty here
    assert geo["normal"]["n_points"] == 0 and np.isnan(geo["normal"]["step_dist_mm"])
    assert "max_steps" in geo["all"]["stop_reasons"]
