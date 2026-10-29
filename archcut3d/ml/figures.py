"""Figures for the experiment write-up (matplotlib, Agg backend)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import resample_polyline


def pick_departure_examples(tables, n: int = 6, per_model: int = 2, min_separation_mm: float = 3.0):
    """Choose the departure points with the largest geometric offsets,
    automatically (no hand picking): greedy by |offset|, at most `per_model`
    per model and at least `min_separation_mm` apart within a model.
    Returns a list of (table_index, row_index)."""
    cands = []
    for ti, t in enumerate(tables):
        if t.departure is None:
            continue
        for k in np.flatnonzero(t.departure & np.isfinite(t.geo_offset_mm)):
            cands.append((abs(t.geo_offset_mm[k]), ti, int(k)))
    cands.sort(key=lambda c: -c[0])
    chosen, count = [], {}
    for _, ti, k in cands:
        t = tables[ti]
        if count.get(ti, 0) >= per_model:
            continue
        if any(ti == tj and np.linalg.norm(t.pos[k] - t.pos[kj]) * t.pixel_size < min_separation_mm
               for tj, kj in chosen):
            continue
        chosen.append((ti, k))
        count[ti] = count.get(ti, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


def departure_example_figure(example, table, k: int, paths: dict, out_path: str, window_mm: float = 3.0):
    """local_relief around hand point k with the hand trace (black) and the
    predicted paths (`paths`: label -> list of [col, row] arrays), plus the
    geometric-minimum positions of the hand points in view."""
    px = example.pixel_size
    half = window_mm / px
    cx, cy = table.pos[k]
    c0, c1 = int(max(0, cx - half)), int(min(example.channels.shape[1], cx + half))
    r0, r1 = int(max(0, cy - half)), int(min(example.channels.shape[0], cy + half))
    img = example.channels[r0:r1, c0:c1, 1].astype(float).copy()
    img[~example.mask[r0:r1, c0:c1]] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = np.nanpercentile(np.abs(img), 98) if np.isfinite(img).any() else 1.0
    ax.imshow(img, cmap="viridis", extent=(c0, c1, r1, r0), vmin=-vmax, vmax=vmax)
    for raw in example.traces:
        line = resample_polyline(raw, 0.3, px)
        ax.plot(line[:, 0], line[:, 1], "-", color="black", lw=2, label="hand trace" if raw is example.traces[0] else None)
    in_view = (table.pos[:, 0] > c0) & (table.pos[:, 0] < c1) & (table.pos[:, 1] > r0) & (table.pos[:, 1] < r1)
    gp = table.geo_pos[in_view]
    ax.plot(gp[:, 0], gp[:, 1], ".", color="white", ms=3, label="geometric minimum")
    colors = ["red", "cyan", "orange", "magenta"]
    for (label, plist), color in zip(paths.items(), colors):
        for i, p in enumerate(plist):
            ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=1.5, label=label if i == 0 else None)
    ax.plot([cx], [cy], "o", mfc="none", mec="yellow", ms=12)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_title(f"{table.source_file}  point {table.point[k]} of trace {table.trace[k]}\n"
                 f"offset {table.geo_offset_mm[k]:+.2f} mm, depth {table.geo_depth[k]:.3f} mm", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
