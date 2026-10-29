"""PyTorch glue for the tracer dataset: channel normalization + Dataset/
DataLoader + leave-one-model-out split.

Kept separate from dataset.py so label loading and patch sampling stay
usable (and testable) without a torch dependency.

Normalization choices, per channel:
  height        -- subtract the patch's OWN mean (over valid pixels) before
                   scaling. Raw height encodes each patch's absolute
                   position on its model, which varies for reasons that
                   have nothing to do with the incision; only the LOCAL
                   shape matters.
  local_relief  -- scale only, no shift. It's already background-subtracted
                   (rasterize_to_grid); the zero level is meaningful (flat
                   surface) and shouldn't be moved.
  curvature     -- log1p then scale. It's non-negative and heavy-tailed
                   (mostly near 0, occasional sharp spikes on edges/noise);
                   log1p brings the spikes back in range.
A 4th channel (the sampled mask, as 0/1) is appended so the network can
tell "flat surface" from "off the mesh" -- both would otherwise read as 0.
All four are zeroed outside the mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import TracerSamples


@dataclass
class NormStats:
    height_std: float
    local_relief_std: float
    curvature_log_std: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**{k: float(d[k]) for k in ("height_std", "local_relief_std", "curvature_log_std")})


def _per_patch_masked_mean(channel: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """channel, masks: (N, R, R) -> (N,) mean of `channel` over valid pixels
    per patch (0 for a patch with no valid pixels at all)."""
    mask_f = masks.astype(np.float32)
    counts = mask_f.reshape(len(masks), -1).sum(axis=1)
    sums = (channel * mask_f).reshape(len(masks), -1).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).astype(np.float32)


def compute_norm_stats(samples: TracerSamples, eps: float = 1e-6) -> NormStats:
    """Fit NormStats from a training TracerSamples. Only valid (masked)
    pixels count; height is centered per-patch first (see module docstring)."""
    if not samples.masks.any():
        raise ValueError("no valid (masked) pixels to compute stats from")

    height = samples.patches[..., 0]
    means = _per_patch_masked_mean(height, samples.masks)
    centered = height - means[:, None, None]
    height_std = float(centered[samples.masks].std() + eps)

    relief_std = float(samples.patches[..., 1][samples.masks].std() + eps)

    curv_log = np.log1p(np.clip(samples.patches[..., 2], 0, None))
    curv_log_std = float(curv_log[samples.masks].std() + eps)

    return NormStats(height_std=height_std, local_relief_std=relief_std, curvature_log_std=curv_log_std)


def normalize_patches(patches: np.ndarray, masks: np.ndarray, stats: NormStats) -> np.ndarray:
    """(N,R,R,3) + (N,R,R) bool -> (N,R,R,4) float32: [height, local_relief,
    curvature, mask], normalized per the module docstring and zeroed
    everywhere the mask is False."""
    n, r, _, _ = patches.shape
    mask_f = masks.astype(np.float32)

    height = patches[..., 0]
    means = _per_patch_masked_mean(height, masks)
    height_n = (height - means[:, None, None]) / stats.height_std

    relief_n = patches[..., 1] / stats.local_relief_std
    curv_n = np.log1p(np.clip(patches[..., 2], 0, None)) / stats.curvature_log_std

    out = np.stack([height_n, relief_n, curv_n, mask_f], axis=-1).astype(np.float32)
    out *= mask_f[..., None]   # off-mesh -> exactly 0 in every channel (mask channel included)
    return out


class TracerTorchDataset(Dataset):
    """Wraps a TracerSamples (+ NormStats) as a torch Dataset. `indices`
    restricts it to a subset (e.g. one side of a leave-one-model-out split);
    omit for the whole thing."""

    def __init__(self, samples: TracerSamples, stats: NormStats, indices=None):
        self.normed = normalize_patches(samples.patches, samples.masks, stats)   # (N,R,R,4)
        self.steps = samples.steps.astype(np.float32)
        self.stops = samples.stops.astype(np.float32)
        self.indices = np.arange(len(samples)) if indices is None else np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        patch = torch.from_numpy(self.normed[idx]).permute(2, 0, 1).contiguous()   # (4,R,R)
        step = torch.from_numpy(self.steps[idx])                                    # (2,)
        stop = torch.tensor(self.stops[idx])
        return patch, step, stop


def leave_one_group_out_split(samples: TracerSamples, val_group: int):
    """groups == val_group -> validation; everything else -> train. `groups`
    indexes the scanned model a sample came from (see dataset.make_dataset),
    so this holds out one whole model rather than random individual steps
    (which would leak -- adjacent steps on the same line are near-duplicates).
    There is no separate test set here; see nested_split for the protocol
    used by train.train_one_split."""
    train_idx = np.where(samples.groups != val_group)[0]
    val_idx = np.where(samples.groups == val_group)[0]
    return train_idx, val_idx


def group_balanced_sampler(groups: np.ndarray, indices: np.ndarray) -> WeightedRandomSampler:
    """A WeightedRandomSampler over `indices` that gives every group
    (source model) equal expected representation per epoch, regardless of
    how many samples it actually contributed -- counters e.g. one densely-
    labeled model dominating a sparser one."""
    g = groups[indices]
    unique, counts = np.unique(g, return_counts=True)
    count_of = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / count_of[gi] for gi in g], dtype=np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights),
                                 num_samples=len(indices), replacement=True)


def make_dataloaders(samples: TracerSamples, stats: NormStats, val_group: int,
                     batch_size: int = 64, balance_groups: bool = True, num_workers: int = 0):
    """Leave-one-model-out train/val DataLoaders."""
    train_idx, val_idx = leave_one_group_out_split(samples, val_group)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"val_group {val_group} leaves an empty split "
                         f"(train={len(train_idx)}, val={len(val_idx)})")

    train_ds = TracerTorchDataset(samples, stats, train_idx)
    val_ds = TracerTorchDataset(samples, stats, val_idx)

    if balance_groups:
        sampler = group_balanced_sampler(samples.groups, train_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


class NestedSplit(NamedTuple):
    train_idx: np.ndarray
    val_idx: np.ndarray       # early stopping only
    test_idx: np.ndarray      # reported metrics only
    test_group: int
    val_group: int


def nested_split(samples: TracerSamples, test_group: int) -> NestedSplit:
    """Leave-one-model-out with a separate validation model.

    `test_group` is the held-out test model. The validation model, used only
    for early stopping, is the next group id after `test_group` in cyclic
    order (deterministic). All remaining groups are training data. Needs at
    least 3 distinct groups."""
    groups = np.unique(samples.groups).tolist()
    if test_group not in groups:
        raise ValueError(f"test_group {test_group} not in {groups}")
    if len(groups) < 3:
        raise ValueError(f"nested split needs >= 3 groups (train, validation, test); got {len(groups)}")
    i = groups.index(test_group)
    val_group = groups[(i + 1) % len(groups)]
    train_idx = np.where((samples.groups != test_group) & (samples.groups != val_group))[0]
    val_idx = np.where(samples.groups == val_group)[0]
    test_idx = np.where(samples.groups == test_group)[0]
    return NestedSplit(train_idx, val_idx, test_idx, test_group, val_group)


def make_nested_dataloaders(samples: TracerSamples, stats: NormStats, split: NestedSplit,
                            batch_size: int = 64, balance_groups: bool = True, num_workers: int = 0):
    """Train / validation / test DataLoaders for a NestedSplit."""
    train_ds = TracerTorchDataset(samples, stats, split.train_idx)
    val_ds = TracerTorchDataset(samples, stats, split.val_idx)
    test_ds = TracerTorchDataset(samples, stats, split.test_idx)

    if balance_groups:
        sampler = group_balanced_sampler(samples.groups, split.train_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
