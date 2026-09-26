"""Two-Gaussian depth/FWHM fit for a single slice profile.

Port of the fitting + measurement logic in anal_slices_Callback (harutot.m):
a slice's (x, y) profile is fit to two Gaussian dips sitting on a linear
baseline, and the fit's minimum gives the incision's depth and full-width-
at-half-maximum (FWHM).

    y = -a1*exp(-((x-b1)/c1)^2) - a2*exp(-((x-b2)/c2)^2) + d + e*x

MATLAB fits once and loops asking the user "is it ok? / keep trying?" with
an otherwise-unchanged fit (no re-randomization between attempts). Here
`fit_two_gaussians` instead tries several different starting guesses for
where the two dips are (the most prominent low points first, then a spread
of stepped/random alternatives) and returns the best one that passes basic
sanity checks -- so a "retry" actually changes something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def two_gaussian_model(x, a1, b1, c1, a2, b2, c2, d, e):
    return (-a1 * np.exp(-((x - b1) / c1) ** 2)
           - a2 * np.exp(-((x - b2) / c2) ** 2)
           + d + e * x)


@dataclass
class GaussianFitResult:
    params: tuple          # (a1, b1, c1, a2, b2, c2, d, e)
    yfit: np.ndarray
    mse: float
    sane: bool             # both amplitudes positive, widths reasonable
    depth: float
    fwhm: float
    depth_marker: tuple             # (x_at_min, mfit, baseline_at_min) -- plot as a vertical line
    width_marker: tuple | None      # (x1, x2, y1, y2) -- plot as a horizontal line; None if undefined


def _initial_baseline(x: np.ndarray, y: np.ndarray) -> tuple:
    """Rough (d, e) guess: a straight line through the outer sixths of the
    profile (the shoulders, away from the dip in the middle)."""
    n = len(x)
    edge = max(2, n // 6)
    idx = np.concatenate([np.arange(0, edge), np.arange(n - edge, n)])
    e0, d0 = np.polyfit(x[idx], y[idx], 1)
    return float(d0), float(e0)


def _prominent_minima(x: np.ndarray, y: np.ndarray, d0: float, e0: float) -> tuple:
    """The two x-positions where y dips furthest below the rough baseline,
    away from each other -- used as the first (best-guess) starting pair."""
    depth_below = (d0 + e0 * x) - y
    order = np.argsort(depth_below)[::-1]
    b1 = x[order[0]]
    span = x.max() - x.min()
    for idx in order[1:]:
        if abs(x[idx] - b1) > span * 0.1:
            return float(b1), float(x[idx])
    return float(b1), float(b1 + span * 0.2)   # fallback: single dominant dip


def _candidate_centers(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> list:
    """Starting (b1, b2) pairs to try, in order: the data-driven best guess
    first, then a stepped spread across the profile, then random jitter
    around the best guess -- "stepped and random" retries."""
    d0, e0 = _initial_baseline(x, y)
    x0, x1 = float(x.min()), float(x.max())
    span = x1 - x0

    candidates = [_prominent_minima(x, y, d0, e0)]

    # Stepped pairs, b1 always left-of-centre and b2 its mirror right-of-centre:
    # fracs stays < 0.5 so (b1, b2) never repeats an earlier pair with the two
    # slots swapped (the two Gaussian terms are interchangeable, so e.g.
    # b1=-6,b2=6 and b1=6,b2=-6 would be the exact same starting fit -- a
    # wasted attempt).
    n_stepped = max(0, (n - 1) // 2)
    fracs = np.linspace(0.1, 0.45, max(n_stepped, 1))
    for i in range(n_stepped):
        b1 = x0 + span * fracs[i]
        b2 = x0 + span * (1 - fracs[i])
        if abs(b2 - b1) > span * 0.05:
            candidates.append((b1, b2))

    while len(candidates) < n:
        b1 = x0 + rng.uniform(0.1, 0.9) * span
        b2 = x0 + rng.uniform(0.1, 0.9) * span
        candidates.append((b1, b2))

    return candidates[:n]


def _two_dips_only(x, a1, b1, c1, a2, b2, c2):
    return -a1 * np.exp(-((x - b1) / c1) ** 2) - a2 * np.exp(-((x - b2) / c2) ** 2)


def _fit_from_centers(x: np.ndarray, y: np.ndarray, b1: float, b2: float,
                      d0: float, e0: float):
    """Fit only the two dips (6 free parameters) against the data with the
    shoulder baseline (d0, e0) already subtracted -- the baseline is FIXED,
    not re-optimized alongside the dips.

    Earlier this fit all 8 parameters (dips + baseline) jointly, and the
    baseline (d, e) came out subtly warped by the dip shape -- on a real
    slice this showed up as the depth line's top landing ~0.4mm above the
    actual shoulder level, because the fitted line, extrapolated out to
    the dip's x-position, had drifted from where the real surface is.
    Fixing the baseline first from the shoulders (away from the dip, where
    it's unambiguous) and only fitting the dips against what's left removes
    that failure mode -- depth is always measured against the real
    shoulder line, not a fit compromise.
    """
    span = x.max() - x.min()
    y_range = max(float(y.max() - y.min()), 1e-6)
    baseline = d0 + e0 * x
    resid = y - baseline
    depth_guess = max(float(np.max(-resid)), 1e-6)
    c_guess = max(span / 10.0, 1e-6)

    # Bounds match _is_sane's thresholds (width <= half the profile,
    # amplitude <= 3x the data's own range) so the optimizer can't wander
    # into the "fake the baseline trend with one giant Gaussian" failure
    # mode in the first place, rather than relying only on the sanity
    # filter to catch it afterwards.
    p0 = [depth_guess, b1, c_guess, depth_guess, b2, c_guess]
    bounds_lo = [0, x.min() - span, span * 1e-3, 0, x.min() - span, span * 1e-3]
    bounds_hi = [y_range * 3, x.max() + span, span * 0.5, y_range * 3, x.max() + span, span * 0.5]

    popt6, _ = curve_fit(_two_dips_only, x, resid, p0=p0,
                         bounds=(bounds_lo, bounds_hi), maxfev=20000)
    popt = np.array([popt6[0], popt6[1], popt6[2], popt6[3], popt6[4], popt6[5], d0, e0])
    yfit = two_gaussian_model(x, *popt)
    mse = float(np.mean((yfit - y) ** 2))
    return popt, yfit, mse


def _is_sane(params: tuple, x: np.ndarray, y: np.ndarray, mse: float) -> bool:
    a1, b1, c1, a2, b2, c2, d, e = params
    span = x.max() - x.min()
    y_range = float(y.max() - y.min())
    if a1 <= 0 or a2 <= 0:
        return False
    # width bounds: too thin is a degenerate spike; too wide (more than half
    # the profile) means it's no longer a localized dip -- it's standing in
    # for the overall baseline trend instead, which the linear d+e*x term
    # already covers. Caught a real case: an unbounded-width "Gaussian"
    # (c2 > profile span) faked a slope using amplitude 24 on data with
    # y-range 4.8 -- passed every other check, so the amplitude bound below
    # was added at the same time.
    if c1 < span * 1e-3 or c1 > span * 0.5:
        return False
    if c2 < span * 1e-3 or c2 > span * 0.5:
        return False
    if not (x.min() - span * 0.2 <= b1 <= x.max() + span * 0.2):
        return False
    if not (x.min() - span * 0.2 <= b2 <= x.max() + span * 0.2):
        return False
    # a dip's depth shouldn't dwarf the data's own observed range
    if y_range > 0 and (a1 > y_range * 3 or a2 > y_range * 3):
        return False
    if mse > np.var(y) * 2:      # fit should beat "just predict the mean" by a fair margin
        return False
    return True


def _find_level_crossings(values: np.ndarray, level: float) -> list:
    """Port of soluc.m: indices where `values` crosses `level`, in order.
    Sign-change detection on (values - level)."""
    x_idx = []
    diffs = values - level
    if diffs[0] > 0:
        state = 1
    elif diffs[0] < 0:
        state = -1
    else:
        state = 0
        x_idx.append(0)

    for i in range(len(values)):
        d = diffs[i]
        if state == 1:
            if d < 0:
                x_idx.append(i); state = -1
            elif d == 0:
                x_idx.append(i); state = 0
        elif state == -1:
            if d > 0:
                x_idx.append(i); state = 1
            elif d == 0:
                x_idx.append(i); state = 0
        else:
            if d > 0:
                state = 1
            elif d < 0:
                state = -1
            else:
                x_idx.append(i)
    return x_idx


def compute_depth_and_fwhm(x: np.ndarray, yfit: np.ndarray, d: float, e: float) -> tuple:
    """depth: how far the fitted curve's minimum sits below the linear
    baseline at that same x. FWHM: width of the dip at half that depth.
    Mirrors harutot.m's fallback when the profile only crosses back up to
    half-max on one side (common near the edge of a slice).

    Returns (depth, fwhm, depth_marker, width_marker) -- the marker tuples
    are the same points harutot.m plots as the dashed depth/width lines:
      depth_marker = (x_at_min, mfit, baseline_at_min)         -- vertical line
      width_marker = (x1, x2, y1, y2) or None                  -- horizontal line
    width_marker's y1, y2 are the curve's actual values at x1, x2 (not
    necessarily exactly half_level in the fallback cases), matching
    harutot.m drawing the line at (yfit(X(1))+yfit(X(2)))/2.
    """
    nmfit = int(np.argmin(yfit))
    mfit = float(yfit[nmfit])
    baseline_at_min = d + e * x[nmfit]
    depth = baseline_at_min - mfit
    half_level = mfit + depth / 2
    depth_marker = (float(x[nmfit]), mfit, float(baseline_at_min))

    crossings = _find_level_crossings(yfit, half_level)
    if len(crossings) >= 2:
        i1, i2 = crossings[0], crossings[1]
    elif e < 0 and len(crossings) > 0:
        tail = yfit[crossings[0]:]
        i1, i2 = crossings[0], crossings[0] + int(np.argmax(tail))
    elif e > 0 and len(crossings) > 0:
        head = yfit[:crossings[0] + 1]
        i1, i2 = crossings[0], int(np.argmax(head))
    else:
        i1 = i2 = None

    if i1 is None:
        fwhm = 0.0
        width_marker = None
    else:
        fwhm = abs(float(x[i1] - x[i2]))
        width_marker = (float(x[i1]), float(x[i2]), float(yfit[i1]), float(yfit[i2]))

    return depth, fwhm, depth_marker, width_marker


def fit_two_gaussians(x, y, n_candidates: int = 5, seed=None) -> GaussianFitResult:
    """Try `n_candidates` different starting guesses for the two dip
    centers (best data-driven guess, then stepped, then random), fit each,
    and return the lowest-MSE result that passes the sanity checks -- or,
    if none pass, the lowest-MSE result found (with sane=False) so the
    caller can still show something and let the user decide."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d0, e0 = _initial_baseline(x, y)

    results = []
    for b1, b2 in _candidate_centers(x, y, n_candidates, rng):
        try:
            params, yfit, mse = _fit_from_centers(x, y, b1, b2, d0, e0)
        except RuntimeError:
            continue
        sane = _is_sane(params, x, y, mse)
        results.append((params, yfit, mse, sane))

    if not results:
        raise RuntimeError("two-Gaussian fit did not converge from any starting guess")

    sane_results = [r for r in results if r[3]]
    best = min(sane_results, key=lambda r: r[2]) if sane_results else min(results, key=lambda r: r[2])
    params, yfit, mse, sane = best

    depth, fwhm, depth_marker, width_marker = compute_depth_and_fwhm(x, yfit, params[6], params[7])
    return GaussianFitResult(params=tuple(params), yfit=yfit, mse=mse, sane=sane,
                             depth=depth, fwhm=fwhm,
                             depth_marker=depth_marker, width_marker=width_marker)
