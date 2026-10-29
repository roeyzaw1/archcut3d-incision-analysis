"""Step policies for rollout.trace.

Every policy is a callable
    policy(patch, mask) -> ((forward, right), stop_prob)
where `patch` is the raw heading-canonicalised (R, R, C) crop from
dataset.sample_patch (channels: height, local_relief, curvature), `mask` the
matching (R, R) bool array, and the step is in units of `step_mm`.

Patch geometry: the patch covers `patch_mm` x `patch_mm`, forward is up (rows
decrease), right is +col, and the centre is at index (R - 1) / 2. So pixel
(i, j) sits (forward, right) = (c - i, j - c) * patch_mm / R millimetres from
the current position, with c = (R - 1) / 2.

  StraightAheadPolicy      always (1, 0), never stops
  GeometricMinimumPolicy   pure-geometry tracer, no learning, never stops
  TracerPolicy             wraps a trained TracerNet
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .dataset import DEFAULT_PATCH_MM, DEFAULT_PATCH_RES, DEFAULT_STEP_MM
from .model import TracerNet
from .torch_data import NormStats, normalize_patches

RELIEF_CHANNEL = 1


class StraightAheadPolicy:
    """Always step one full step straight ahead and never stop."""

    can_stop = False

    def __call__(self, patch, mask):
        return np.array([1.0, 0.0]), 0.0


class GeometricMinimumPolicy:
    """Follow the local-relief minimum, as ArchCut's slice path does.

    From the current pose, look one step ahead. On the line perpendicular to
    the heading through that point, within +/- half_width_mm, take the
    local_relief profile, subtract the straight line through its two end
    values (tilt correction), and move to the position of the minimum
    (refined to sub-sample precision with a parabola through the minimum and
    its neighbours). The step is therefore (1, lateral / step_mm). Never
    stops. If fewer than three profile samples lie on the mesh the step is
    straight ahead.
    """

    can_stop = False

    def __init__(self, step_mm: float = DEFAULT_STEP_MM, half_width_mm: float = 0.5,
                 patch_mm: float = DEFAULT_PATCH_MM, out_res: int = DEFAULT_PATCH_RES,
                 n_samples: int = 101, relief_channel: int = RELIEF_CHANNEL):
        if half_width_mm >= patch_mm / 2 or step_mm >= patch_mm / 2:
            raise ValueError("step and half width must fit inside the patch")
        self.step_mm = step_mm
        self.half_width_mm = half_width_mm
        self.patch_mm = patch_mm
        self.out_res = out_res
        self.n_samples = n_samples
        self.relief_channel = relief_channel

    def lateral_offset_mm(self, patch: np.ndarray, mask: np.ndarray):
        """Lateral position (mm, positive = right) of the tilt-corrected
        minimum on the profile one step ahead; None if the profile is not
        on the mesh."""
        R = self.out_res
        c = (R - 1) / 2.0
        per_px = self.patch_mm / R
        s = np.linspace(-self.half_width_mm, self.half_width_mm, self.n_samples)
        rows = np.full_like(s, c - self.step_mm / per_px)
        cols = c + s / per_px

        relief = map_coordinates(patch[:, :, self.relief_channel].astype(np.float64),
                                 [rows, cols], order=1, mode="nearest")
        valid = map_coordinates(mask.astype(np.float64), [rows, cols], order=1, mode="nearest") > 0.99
        if valid.sum() < 3:
            return None

        idx = np.flatnonzero(valid)
        first, last = idx[0], idx[-1]
        line = np.interp(s, [s[first], s[last]], [relief[first], relief[last]])
        corrected = np.where(valid, relief - line, np.inf)

        # among (near-)ties take the sample closest to the centre line
        near_min = np.flatnonzero(corrected <= corrected.min() + 1e-12)
        k = near_min[np.argmin(np.abs(s[near_min]))]

        offset = s[k]
        if 0 < k < len(s) - 1 and np.isfinite(corrected[k - 1]) and np.isfinite(corrected[k + 1]):
            y0, y1, y2 = corrected[k - 1], corrected[k], corrected[k + 1]
            denom = y0 - 2 * y1 + y2
            if denom > 1e-15:
                offset = s[k] + float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)) * (s[1] - s[0])
        return float(offset)

    def __call__(self, patch, mask):
        offset = self.lateral_offset_mm(patch, mask)
        if offset is None:
            return np.array([1.0, 0.0]), 0.0
        return np.array([1.0, offset / self.step_mm]), 0.0


class TracerPolicy:
    """Wraps a trained TracerNet: normalises the patch with the training
    statistics, runs the network in eval mode, returns step and stop
    probability."""

    can_stop = True

    def __init__(self, net: TracerNet, stats: NormStats, device: str = "cpu", keep_channels=(0, 1, 2)):
        self.net = net.to(device).eval()
        self.stats = stats
        self.device = device
        self.keep_channels = keep_channels

    @torch.no_grad()
    def __call__(self, patch, mask):
        x = normalize_patches(patch[None], mask[None], self.stats, self.keep_channels)   # (1, R, R, 4)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)
        step, stop_logit = self.net(x)
        return step[0].cpu().numpy().astype(np.float64), float(torch.sigmoid(stop_logit[0]).item())
