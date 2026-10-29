"""The tracer model: heading-canonicalized patch -> (step vector, stop prob).

A small conv encoder (~300k parameters, much smaller than ResNet-18, chosen
for a training set of a few thousand samples) feeding two heads: step
regression and stop probability.

Dropout layers exist (spatial dropout in one encoder block, ordinary dropout
in the heads), but evaluation runs the network in eval mode, so dropout is
inactive at inference. Monte-Carlo dropout (keeping dropout on and using the
spread of several stochastic passes as an uncertainty estimate) is planned,
not implemented.

Input:  (B, 4, R, R) -- see torch_data.normalize_patches for the 4 channels
        (height, local_relief, curvature, mask) and what R (patch
        resolution) currently is.
Output: step (B, 2) forward/right in step_mm units, stop_logit (B,) --
        apply sigmoid for a 0..1 probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 4   # height, local_relief, curvature, mask


def _conv_block(in_ch: int, out_ch: int, stride: int, dropout_p: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout2d(dropout_p))
    return nn.Sequential(*layers)


class TracerNet(nn.Module):
    """pool="avg" (default) averages the final feature map over space before
    the heads. pool="flatten" feeds the heads the whole map instead (the
    first head layer grows to (4 * base_width) * (in_res / 8)^2 inputs, the
    hidden widths are unchanged), so the position of features inside the
    patch is kept. coord_channels=True appends two input channels holding the
    normalised row and column coordinates of every pixel (CoordConv). The
    defaults reproduce the original network exactly."""

    def __init__(self, in_channels: int = IN_CHANNELS, base_width: int = 32,
                dropout_p: float = 0.1, head_dropout_p: float = 0.2,
                pool: str = "avg", coord_channels: bool = False, in_res: int = 48):
        super().__init__()
        if pool not in ("avg", "flatten"):
            raise ValueError("pool must be 'avg' or 'flatten'")
        self.coord_channels = coord_channels
        w = base_width
        # 3 stride-2 stages -> spatial dim shrinks by 8x (48 -> 6 at the
        # default patch resolution); channel width doubles each stage.
        self.encoder = nn.Sequential(
            _conv_block(in_channels + (2 if coord_channels else 0), w, stride=1),
            _conv_block(w, w, stride=2),
            _conv_block(w, w * 2, stride=1),
            _conv_block(w * 2, w * 2, stride=2),
            _conv_block(w * 2, w * 4, stride=1, dropout_p=dropout_p),
            _conv_block(w * 4, w * 4, stride=2),
        )
        feat = w * 4
        if pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
            head_in = feat
        else:
            self.pool = nn.Flatten()
            head_in = feat * (in_res // 8) ** 2

        self.step_head = nn.Sequential(
            nn.Linear(head_in, feat // 2), nn.ReLU(inplace=True), nn.Dropout(head_dropout_p),
            nn.Linear(feat // 2, 2),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(head_in, feat // 2), nn.ReLU(inplace=True), nn.Dropout(head_dropout_p),
            nn.Linear(feat // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        if self.coord_channels:
            b, _, h, w = x.shape
            rows = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            cols = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            x = torch.cat([x, rows, cols], dim=1)
        z = self.encoder(x)
        z = self.pool(z).flatten(1)
        step = self.step_head(z)
        stop_logit = self.stop_head(z).squeeze(-1)
        return step, stop_logit

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tracer_loss(step_pred: torch.Tensor, stop_logit_pred: torch.Tensor,
                step_target: torch.Tensor, stop_target: torch.Tensor,
                stop_weight: float = 0.2):
    """smooth-L1 on the step vector + a lightly weighted BCE on the stop
    logit. stop_weight defaults to 0.2 so the step term dominates: a wrong
    step moves the trace off the incision, whereas a missed or spurious stop
    only changes where the trace ends, which can be corrected afterwards by
    trimming or extending the trace."""
    step_loss = F.smooth_l1_loss(step_pred, step_target)
    stop_loss = F.binary_cross_entropy_with_logits(stop_logit_pred, stop_target)
    total = step_loss + stop_weight * stop_loss
    return total, {"step_loss": step_loss.detach(), "stop_loss": stop_loss.detach(), "total": total.detach()}
