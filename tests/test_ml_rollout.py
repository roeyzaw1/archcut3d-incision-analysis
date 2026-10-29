import numpy as np

from archcut3d.ml.rollout import (default_max_steps, points_to_polyline_distance, start_pose_from_trace,
                                  trace, trace_hand_line, trace_metrics)
from tests.fakes import FakeExample, straight_trace

STEP_PX = 6.0   # 0.3 mm step at 0.05 mm/px


def straight_policy(patch, mask):
    return np.array([1.0, 0.0]), 0.0


def test_point_to_polyline_distance():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    pts = np.array([[5.0, 3.0], [12.0, 5.0], [-4.0, 3.0], [10.0, 10.0]])
    d = points_to_polyline_distance(pts, line)
    assert np.allclose(d, [3.0, 2.0, 5.0, 0.0])
    assert np.allclose(points_to_polyline_distance(np.array([[3.0, 4.0]]), np.array([[0.0, 0.0]])), [5.0])


def test_straight_policy_walks_a_straight_line_at_the_step_length():
    ex = FakeExample("A.wrl", [])
    res = trace(straight_policy, ex, [40.0, 100.0], [1.0, 0.0], max_steps=10)
    assert res.stop_reason == "max_steps" and res.n_steps == 10
    assert np.allclose(res.path[:, 1], 100.0)
    assert np.allclose(np.diff(res.path[:, 0]), STEP_PX)


def test_diagonal_heading_is_followed():
    ex = FakeExample("A.wrl", [])
    res = trace(straight_policy, ex, [40.0, 40.0], [1.0, 1.0], max_steps=5)
    d = np.diff(res.path, axis=0)
    assert np.allclose(d[:, 0], d[:, 1]) and np.allclose(np.linalg.norm(d, axis=1), STEP_PX)


def test_stops_when_stop_probability_exceeds_threshold():
    ex = FakeExample("A.wrl", [])
    calls = []

    def policy(patch, mask):
        calls.append(1)
        return np.array([1.0, 0.0]), (1.0 if len(calls) == 4 else 0.0)

    res = trace(policy, ex, [40.0, 100.0], [1.0, 0.0], max_steps=50)
    assert res.stop_reason == "stop" and len(res.path) == 4 and len(res.stop_probs) == 4


def test_stops_when_leaving_the_mask():
    ex = FakeExample("A.wrl", [])
    ex.mask[:, 100:] = False
    res = trace(straight_policy, ex, [60.0, 100.0], [1.0, 0.0], max_steps=50)
    assert res.stop_reason == "left_mask"
    assert res.path[-1, 0] >= 100 and res.path[-2, 0] < 100


def test_heading_smoothing_reduces_turning():
    ex = FakeExample("A.wrl", [])

    def veer(patch, mask):
        return np.array([1.0, 0.5]), 0.0

    def total_turn(res):
        d = np.diff(res.path, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        return abs(ang[-1] - ang[0])

    free = trace(veer, ex, [60.0, 100.0], [1.0, 0.0], max_steps=6, heading_smoothing=0.0)
    smooth = trace(veer, ex, [60.0, 100.0], [1.0, 0.0], max_steps=6, heading_smoothing=0.8)
    assert total_turn(smooth) < total_turn(free)


def test_trace_requires_a_budget():
    import pytest
    ex = FakeExample("A.wrl", [])
    with pytest.raises(ValueError):
        trace(straight_policy, ex, [40.0, 100.0], [1.0, 0.0])


def test_metrics_identical_and_offset_lines():
    px = 0.05
    hand = np.array([[40.0, 100.0 + 0.0], [140.0, 100.0]])
    m = trace_metrics(hand.copy(), hand, px)
    assert m["chamfer_mm"] < 1e-9 and m["coverage_0.05"] == 1.0
    assert abs(m["length_ratio"] - 1.0) < 0.1 and m["endpoint_error_mm"] < 1e-9

    shifted = hand + np.array([0.0, 2.0])          # 2 px = 0.1 mm sideways
    m = trace_metrics(shifted, hand, px)
    assert abs(m["pred_to_hand_mm"] - 0.1) < 1e-9 and abs(m["hand_to_pred_mm"] - 0.1) < 1e-9
    assert m["coverage_0.05"] == 0.0 and m["coverage_0.2"] == 1.0


def test_metrics_short_prediction():
    px = 0.05
    hand = np.array([[40.0, 100.0], [140.0, 100.0]])
    half = np.array([[40.0, 100.0], [90.0, 100.0]])
    m = trace_metrics(half, hand, px)
    assert abs(m["length_ratio"] - 0.5) < 0.1
    assert m["pred_to_hand_mm"] < 1e-9               # what it drew lies on the hand line
    assert m["hand_to_pred_mm"] > 0.5                # but half the hand line is far from it
    assert abs(m["endpoint_error_mm"] - 2.5) < 0.2
    assert 0.4 < m["coverage_0.05"] < 0.6


def test_start_pose_and_budgets():
    px = 0.05
    ex = FakeExample("A.wrl", [straight_trace(100, 40, 140)], pixel_size=px)
    start, heading, line = start_pose_from_trace(ex.traces[0], px)
    assert np.allclose(start, [40.0, 100.0]) and np.allclose(heading, [1.0, 0.0])

    res, m, hand = trace_hand_line(straight_policy, ex, ex.traces[0], can_stop=True)
    assert res.n_steps == default_max_steps(4.95, 0.3) == 25     # trace spans 99 px = 4.95 mm
    res, m, hand = trace_hand_line(straight_policy, ex, ex.traces[0], can_stop=False)
    assert res.n_steps == round(4.95 / 0.3) == 16
    assert m["coverage_0.05"] > 0.9 and abs(m["length_ratio"] - 1.0) < 0.1
