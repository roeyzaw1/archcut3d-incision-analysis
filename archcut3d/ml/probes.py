"""Representation analysis for a trained (or random) TracerNet encoder.

  block_features      global-average-pooled output of each of the 6 conv blocks
  pca_summary         dimensionality of a feature matrix: components needed for
                      80 / 90 / 95 % of the variance, participation ratio,
                      cumulative explained variance
  probe_targets       per-anchor quantities a probe is asked to read out
  fit_probe           ridge (regression) or logistic (classification) probe,
                      regularisation chosen by cross-validation over whole
                      models, features standardised on the training models only
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .dataset import resample_polyline
from .policies import GeometricMinimumPolicy
from .torch_data import normalize_patches

RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_CS = (1e-2, 1e-1, 1.0, 10.0)
TARGETS = {
    "relief_at_centre": "regression",
    "curvature_at_centre": "regression",
    "geometric_depth": "regression",
    "lateral_offset_of_hand_line": "regression",
    "departure": "classification",
}


# ------------------------------------------------------------ features

@torch.no_grad()
def block_features(net, patches: np.ndarray, masks: np.ndarray, stats, keep_channels=(0, 1, 2),
                   device: str = "cpu", batch_size: int = 256):
    """Global-average-pooled output of every encoder block, in eval mode.
    Returns a list of 6 arrays (N, channels_of_block)."""
    net = net.to(device).eval()
    x_all = normalize_patches(patches, masks, stats, keep_channels)
    outs = None
    for i in range(0, len(x_all), batch_size):
        x = torch.from_numpy(x_all[i:i + batch_size]).permute(0, 3, 1, 2).contiguous().to(device)
        feats = []
        z = x
        for block in net.encoder:
            z = block(z)
            feats.append(z.mean(dim=(2, 3)).cpu().numpy())
        outs = [[f] for f in feats] if outs is None else [o + [f] for o, f in zip(outs, feats)]
    return [np.concatenate(o) for o in outs]


# ------------------------------------------------------------ dimensionality

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(sum of eigenvalues)^2 / sum of squared eigenvalues: about the number
    of directions carrying the variance (1 for a single direction, d for d
    equal ones)."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if len(ev) else float("nan")


def pca_summary(features: np.ndarray) -> dict:
    """PCA of the centred feature matrix (N, d)."""
    x = features - features.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    ev = sv ** 2 / max(len(x) - 1, 1)
    total = ev.sum()
    cum = np.cumsum(ev) / total if total > 0 else np.zeros_like(ev)
    return {
        "n_components_80": int(np.searchsorted(cum, 0.80) + 1),
        "n_components_90": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_95": int(np.searchsorted(cum, 0.95) + 1),
        "participation_ratio": participation_ratio(ev),
        "cumulative_variance": cum,
        "n_features": features.shape[1], "n_anchors": len(features),
    }


# ------------------------------------------------------------ targets

def _closest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 1:
        return polyline[0]
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denom = (ab ** 2).sum(axis=1)
    t = np.clip(np.where(denom > 0, ((point - a) * ab).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return closest[np.argmin(np.linalg.norm(closest - point, axis=1))]


def probe_targets(samples, indices, examples: dict, tables: dict, geo: GeometricMinimumPolicy,
                  step_mm: float = 0.3) -> dict:
    """Targets for the anchors `indices` of a TracerSamples. Each value is an
    array aligned with `indices`; NaN where undefined.

      relief_at_centre, curvature_at_centre   channel value at the patch centre
      geometric_depth                         dip depth on the perpendicular through the centre
      lateral_offset_of_hand_line             signed distance (mm, right positive, in the patch frame)
                                              from the patch centre to the nearest point of the hand line
      departure                               departure label (0/1) of the hand point the anchor was cut from
    """
    n = len(indices)
    out = {k: np.full(n, np.nan) for k in TARGETS}
    lines = {}
    dep_lookup = {}
    for name, t in tables.items():
        if t.departure is not None:
            dep_lookup[name] = {(int(tr), int(pt)): bool(d) for tr, pt, d in zip(t.trace, t.point, t.departure)}

    for j, i in enumerate(indices):
        m = samples.meta[i]
        name = m["source_file"]
        ex = examples[name]
        px = ex.pixel_size
        c, h = m["center"], m["heading"]
        for key, ch in (("relief_at_centre", 1), ("curvature_at_centre", 2)):
            out[key][j] = map_coordinates(ex.channels[:, :, ch].astype(np.float64),
                                          [[c[1]], [c[0]]], order=1, mode="nearest")[0]
        res = geo.profile_minimum(samples.patches[i], samples.masks[i], forward_mm=0.0)
        if res is not None:
            out["geometric_depth"][j] = res[1]
        key = (name, m["trace"])
        if key not in lines:
            lines[key] = resample_polyline(ex.traces[m["trace"]], step_mm, px)
        q = _closest_point_on_polyline(c, lines[key])
        right = np.array([-h[1], h[0]])
        out["lateral_offset_of_hand_line"][j] = float((q - c) @ right) * px
        d = dep_lookup.get(name, {}).get((m["trace"], m["point"]))
        if d is not None:
            out["departure"][j] = float(d)
    return out


# ------------------------------------------------------------ probes

def fit_probe(X_train, y_train, groups_train, X_test, y_test, kind: str):
    """Fit a probe on the training anchors and score it on the test anchors.

    kind "regression": ridge; score = R^2. kind "classification": logistic
    regression; score = AUROC. The regularisation strength is chosen by
    cross-validation with whole models held out (`groups_train`); features
    are standardised using the training anchors only. Returns
    (score, chosen_parameter, pearson_r): pearson_r is the correlation of
    prediction and target on the test anchors (regression only; it ignores
    differences in scale and offset between models, which can make R^2
    strongly negative). All NaN if the target is undefined."""
    ok_tr, ok_te = np.isfinite(y_train), np.isfinite(y_test)
    X_train, y_train, groups_train = X_train[ok_tr], y_train[ok_tr], groups_train[ok_tr]
    X_test, y_test = X_test[ok_te], y_test[ok_te]
    nan3 = (float("nan"), float("nan"), float("nan"))
    if len(y_train) < 10 or len(y_test) < 10:
        return nan3
    if kind == "classification" and (len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2):
        return nan3

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    def make(p):
        return Ridge(alpha=p) if kind == "regression" else LogisticRegression(C=p, max_iter=2000)

    def score(model, X, y):
        if kind == "regression":
            return r2_score(y, model.predict(X))
        return roc_auc_score(y, model.decision_function(X))

    params = RIDGE_ALPHAS if kind == "regression" else LOGISTIC_CS
    n_groups = len(np.unique(groups_train))
    best_p, best_s = params[0], -np.inf
    if n_groups >= 2:
        for p in params:
            scores = []
            for tr, va in GroupKFold(n_splits=min(n_groups, 5)).split(Xtr, y_train, groups_train):
                if kind == "classification" and (len(np.unique(y_train[tr])) < 2 or len(np.unique(y_train[va])) < 2):
                    continue
                scores.append(score(make(p).fit(Xtr[tr], y_train[tr]), Xtr[va], y_train[va]))
            if scores and np.mean(scores) > best_s:
                best_p, best_s = p, float(np.mean(scores))
    model = make(best_p).fit(Xtr, y_train)
    r = float("nan")
    if kind == "regression":
        pred = model.predict(Xte)
        if np.std(pred) > 0 and np.std(y_test) > 0:
            r = float(np.corrcoef(pred, y_test)[0, 1])
    return float(score(model, Xte, y_test)), float(best_p), r
