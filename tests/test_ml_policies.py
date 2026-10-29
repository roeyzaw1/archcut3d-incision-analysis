import numpy as np
import torch

from archcut3d.ml.dataset import sample_patch
from archcut3d.ml.model import TracerNet
from archcut3d.ml.policies import GeometricMinimumPolicy, StraightAheadPolicy, TracerPolicy
from archcut3d.ml.rollout import trace
from archcut3d.ml.torch_data import NormStats
from tests.fakes import FakeExample

PX = 0.05          # mm per pixel
CENTER_ROW = 100.3  # groove centre line, deliberately not on a pixel centre


def groove_example(half_width_mm=0.4, depth_mm=0.2, tilt=0.0, size=240):
    """A V-shaped groove along +col at row CENTER_ROW, optionally on a tilted
    surface. Both height and local_relief hold the same profile."""
    rows = np.arange(size)[:, None] * np.ones((1, size))
    d = (rows - CENTER_ROW) * PX                                   # signed lateral distance, mm
    h = -depth_mm * np.clip(1.0 - np.abs(d) / half_width_mm, 0.0, None) + tilt * d
    channels = np.stack([h, h, np.zeros_like(h)], axis=-1)
    return FakeExample("groove.wrl", [], pixel_size=PX, size=size, channels=channels)


def patch_at(ex, col, row, heading=(1.0, 0.0)):
    return sample_patch(ex.channels, ex.mask, ex.pixel_size, (col, row), heading)


def test_straight_ahead_policy():
    step, stop = StraightAheadPolicy()(np.zeros((48, 48, 3)), np.ones((48, 48), bool))
    assert np.allclose(step, [1.0, 0.0]) and stop == 0.0


def test_geometric_minimum_on_a_v_groove():
    ex = groove_example()
    pol = GeometricMinimumPolicy()
    # a pose 0.1 mm to the +row side of the groove, heading +col (right = +row)
    patch, mask = patch_at(ex, 120.0, CENTER_ROW + 0.1 / PX)
    offset = pol.lateral_offset_mm(patch, mask)
    assert abs(offset - (-0.1)) < 0.02                      # groove is 0.1 mm to the left
    step, stop = pol(patch, mask)
    assert step[0] == 1.0 and stop == 0.0 and abs(step[1] - offset / 0.3) < 1e-12

    # on the centre line the minimum is straight ahead
    patch, mask = patch_at(ex, 120.0, CENTER_ROW)
    assert abs(pol.lateral_offset_mm(patch, mask)) < 0.02


def test_tilt_correction_finds_the_groove_not_the_edge():
    ex = groove_example(tilt=0.8)                            # tilt dominates the 0.2 mm depth
    pol = GeometricMinimumPolicy()
    patch, mask = patch_at(ex, 120.0, CENTER_ROW + 0.1 / PX)

    # the raw (uncorrected) minimum of the profile one step ahead is at the edge ...
    per_px = 2.0 / 48
    c = 23.5
    cols = c + np.linspace(-0.5, 0.5, 101) / per_px
    raw = np.array([patch[int(round(c - 0.3 / per_px)), int(round(cc)), 1] for cc in cols])
    assert np.argmin(raw) == 0
    # ... but the tilt-corrected minimum is still the groove
    assert abs(pol.lateral_offset_mm(patch, mask) - (-0.1)) < 0.02


def test_flat_profile_prefers_the_centre_line():
    ex = FakeExample("flat.wrl", [], pixel_size=PX, size=240)
    patch, mask = patch_at(ex, 120.0, 100.0)
    assert abs(GeometricMinimumPolicy().lateral_offset_mm(patch, mask)) < 1e-9


def test_off_mesh_profile_falls_back_to_straight():
    ex = groove_example()
    ex.mask[:] = False
    patch, mask = patch_at(ex, 120.0, CENTER_ROW)
    pol = GeometricMinimumPolicy()
    assert pol.lateral_offset_mm(patch, mask) is None
    assert np.allclose(pol(patch, mask)[0], [1.0, 0.0])


def _lateral_deviation_mm(path):
    return np.abs(path[:, 1] - CENTER_ROW) * PX


def test_rollout_stays_on_a_straight_groove():
    ex = groove_example()
    res = trace(GeometricMinimumPolicy(), ex, [40.0, CENTER_ROW], [1.0, 0.0], max_steps=25)
    assert res.stop_reason == "max_steps"
    assert _lateral_deviation_mm(res.path).max() < 0.05


def test_rollout_converges_onto_the_groove_from_an_offset_start():
    ex = groove_example()
    res = trace(GeometricMinimumPolicy(), ex, [40.0, CENTER_ROW + 0.15 / PX], [1.0, 0.2], max_steps=25)
    dev = _lateral_deviation_mm(res.path)
    assert dev[0] > 0.1 and dev[5:].max() < 0.05


def test_tracer_policy_wraps_the_network_deterministically():
    torch.manual_seed(0)
    net = TracerNet()
    stats = NormStats(height_std=0.3, local_relief_std=0.2, curvature_log_std=0.3)
    pol = TracerPolicy(net, stats)
    patch, mask = patch_at(groove_example(), 120.0, CENTER_ROW)
    (step_a, stop_a), (step_b, stop_b) = pol(patch, mask), pol(patch, mask)
    assert step_a.shape == (2,) and 0.0 <= stop_a <= 1.0
    assert np.array_equal(step_a, step_b) and stop_a == stop_b      # eval mode: no dropout noise
    assert pol.can_stop and not GeometricMinimumPolicy().can_stop
