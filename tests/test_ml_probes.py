import numpy as np
import torch

from archcut3d.ml.anchors import build_anchor_table, label_departures
from archcut3d.ml.dataset import make_dataset
from archcut3d.ml.model import TracerNet
from archcut3d.ml.policies import GeometricMinimumPolicy
from archcut3d.ml.probes import (block_features, fit_probe, participation_ratio, pca_summary,
                                 probe_targets)
from archcut3d.ml.torch_data import NormStats
from tests.test_ml_policies import CENTER_ROW, PX, groove_example


def test_participation_ratio_counts_effective_dimensions():
    assert abs(participation_ratio(np.ones(7)) - 7.0) < 1e-12
    assert abs(participation_ratio(np.array([5.0, 0.0, 0.0])) - 1.0) < 1e-12
    assert 1.0 < participation_ratio(np.array([4.0, 1.0, 1.0])) < 3.0


def test_pca_summary_on_low_rank_and_isotropic_data():
    rng = np.random.default_rng(0)
    low = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 30))          # rank 2 in 30 dims
    s = pca_summary(low)
    assert s["n_components_95"] == 2 and s["participation_ratio"] <= 2.0 + 1e-9
    iso = pca_summary(rng.normal(size=(2000, 20)))
    assert iso["n_components_90"] >= 16 and iso["participation_ratio"] > 18.0
    assert np.all(np.diff(iso["cumulative_variance"]) >= 0) and abs(iso["cumulative_variance"][-1] - 1) < 1e-9


def test_block_features_shapes_and_last_block_matches_the_heads_input():
    torch.manual_seed(0)
    net = TracerNet().eval()
    rng = np.random.default_rng(1)
    patches = rng.normal(size=(5, 48, 48, 3)).astype(np.float32)
    masks = np.ones((5, 48, 48), bool)
    stats = NormStats(1.0, 1.0, 1.0)
    feats = block_features(net, patches, masks, stats)
    assert [f.shape for f in feats] == [(5, 32), (5, 32), (5, 64), (5, 64), (5, 128), (5, 128)]
    from archcut3d.ml.torch_data import normalize_patches
    x = torch.from_numpy(normalize_patches(patches, masks, stats)).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        pooled = net.pool(net.encoder(x)).flatten(1).numpy()
    assert np.allclose(feats[-1], pooled, atol=1e-5)


def _two_model_data(rng, w, kind):
    X = rng.normal(size=(400, 10))
    groups = np.repeat([0, 1], 200)
    if kind == "regression":
        y = X @ w + 0.05 * rng.normal(size=400)
    else:
        y = (X @ w > 0).astype(float)
    return X, y, groups


def test_ridge_probe_reads_out_a_linear_target_and_not_noise():
    rng = np.random.default_rng(0)
    w = rng.normal(size=10)
    X, y, g = _two_model_data(rng, w, "regression")
    Xt, yt, _ = _two_model_data(rng, w, "regression")
    r2, alpha = fit_probe(X, y, g, Xt, yt, "regression")
    assert r2 > 0.99
    r2_noise, _ = fit_probe(X, rng.normal(size=400), g, Xt, rng.normal(size=400), "regression")
    assert r2_noise < 0.05


def test_logistic_probe_and_undefined_cases():
    rng = np.random.default_rng(1)
    w = rng.normal(size=10)
    X, y, g = _two_model_data(rng, w, "classification")
    Xt, yt, _ = _two_model_data(rng, w, "classification")
    auc, c = fit_probe(X, y, g, Xt, yt, "classification")
    assert auc > 0.99
    assert np.isnan(fit_probe(X, y, g, Xt, np.zeros_like(yt), "classification")[0])     # single class in test
    assert np.isnan(fit_probe(X, np.full_like(y, np.nan), g, Xt, yt, "regression")[0])    # target undefined


def test_probe_targets_on_a_synthetic_groove():
    ex = groove_example()
    ex.traces = [np.array([[float(c), CENTER_ROW + 0.1 / PX] for c in range(40, 190)])]
    table = build_anchor_table(ex)
    label_departures([table], offset_mm=0.05, depth_percentile=1.0)       # 0.1 mm off the groove: departures
    ds = make_dataset([ex], n_jitter=2, end_overshoot=0, seed=0)
    idx = np.arange(len(ds))
    t = probe_targets(ds, idx, {ex.source_file: ex}, {ex.source_file: table}, GeometricMinimumPolicy())

    for j, i in enumerate(idx):
        m = ds.meta[i]
        c, h = m["center"], m["heading"]
        if not 50 <= c[0] <= 180:
            continue                      # near the line ends the nearest point is the end point, not the projection
        expected_lat = (CENTER_ROW + 0.1 / PX - c[1]) * h[0] * PX          # nearest hand point is (c_col, row)
        assert abs(t["lateral_offset_of_hand_line"][j] - expected_lat) < 0.02
        d_to_groove = (c[1] - CENTER_ROW) * PX
        assert abs(t["relief_at_centre"][j] - (-0.2 * max(0.0, 1 - abs(d_to_groove) / 0.4))) < 0.02
    clean = ds.kinds == 0
    assert np.all(t["lateral_offset_of_hand_line"][clean] < 1e-9) and np.all(np.abs(t["lateral_offset_of_hand_line"][clean]) < 1e-9)
    assert np.nanmean(t["departure"]) > 0.9 and np.nanmin(t["geometric_depth"]) >= 0.0
