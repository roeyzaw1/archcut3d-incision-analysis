"""
Per-slice analysis: aperture angle (selectAngleLines.m + the final formula in
anal_slices_Callback). The two-Gaussian depth/FWHM fit is a separate, later
step (harder nonlinear optimization, needs its own dedicated verification).
"""

import numpy as np


def nearest_index(xs, ys, click_x, click_y):
    """Snap a click to the nearest actual data point, matching
    selectAngleLines.m's distance-based snap (`[~, Xx1] = min(distances1)`)."""
    d2 = (xs - click_x) ** 2 + (ys - click_y) ** 2
    return int(np.argmin(d2))


def fit_line_through_range(xs, ys, idx_a, idx_b):
    """Fit a line (degree-1 polyfit) through ALL points between two indices
    (inclusive), not just the two clicked points -- matching
    selectAngleLines.m: `xxl = xsort(min(Xx1,Xx2):max(Xx1,Xx2))`.
    xs, ys must already be sorted by x. Returns (slope, intercept)."""
    lo, hi = (idx_a, idx_b) if idx_a <= idx_b else (idx_b, idx_a)
    x_range = xs[lo:hi + 1]
    y_range = ys[lo:hi + 1]
    if len(x_range) < 2:
        raise ValueError('Need at least two distinct points between the clicked pair to fit a line.')
    slope, intercept = np.polyfit(x_range, y_range, 1)
    return float(slope), float(intercept)


def compute_aperture_angle(slope_left, slope_right):
    """Port of the aperture-angle formula in anal_slices_Callback (harutot.m):

        angluo = 180 - atan((Ppr(1)-Ppl(1))/(1+Ppl(1)*Ppr(1))) * 180/pi
        if abs(angluo) > 180: angluo = abs(angluo) - 180

    Sanity check: slopes of 1 and -1 (perpendicular lines) give exactly 90,
    the textbook answer for the tan-difference angle-between-two-lines formula.
    """
    denom = 1.0 + slope_left * slope_right
    ratio = (slope_right - slope_left) / denom if denom != 0 else np.sign(slope_right - slope_left) * np.inf
    angle = 180.0 - np.degrees(np.arctan(ratio))
    if abs(angle) > 180:
        angle = abs(angle) - 180
    return angle
