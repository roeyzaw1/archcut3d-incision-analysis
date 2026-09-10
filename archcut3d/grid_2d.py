"""
Rasterize a positioned mesh onto a regular X/Y grid -- the 2.5-D
representation the ML feature-detection work is built on
(experiment/ml-feature-detection branch).

Assumes the mesh has already been through compute_initial_orientation():
X/Y is the flat plane, Z is height/depth, single-sided surface viewed from
above. Under that assumption the grid is a lossless top-down relief map.

Channels produced (H, W, 3):
  0 - height        : Z at each cell (linearly interpolated between vertices)
  1 - local_relief  : height minus a smoothed version of itself, so the
                      broad shape of the object is subtracted out and a
                      shallow incision isn't drowned by it (the LRM idea
                      from DEM-archaeology work)
  2 - curvature     : per-vertex flatness score (curvature.py), interpolated

Plus:
  mask   (H, W) bool : True where the grid actually sits over the mesh
  origin (x_min, y_min), pixel_size : to map grid cells back to world X/Y

Row/column convention: row 0 is the TOP of the image and corresponds to the
LARGEST Y (so it reads like a normal top-down view, Y up). Column 0 is the
smallest X.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .curvature import compute_principal_curvatures, flatness_score


@dataclass
class MeshGrid:
    channels: np.ndarray   # (H, W, 3) float64: height, local_relief, curvature
    mask: np.ndarray       # (H, W) bool: True where the grid covers the mesh
    origin: tuple          # (x_min, y_min) world coords of column 0 / the smallest-Y row
    pixel_size: float      # world units per cell

    def grid_to_world_xy(self, rows, cols):
        """Map (row, col) indices to world (x, y). Inverse of the
        rasterization -- used later to map a detected centerline in grid
        space back onto the 3D mesh."""
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        n_rows = self.channels.shape[0]
        x = self.origin[0] + cols * self.pixel_size
        y = self.origin[1] + (n_rows - 1 - rows) * self.pixel_size
        return np.stack([x, y], axis=-1)

    def world_xy_to_grid(self, x, y):
        """Map world (x, y) to fractional (col, row) -- the inverse of
        grid_to_world_xy, in the [col, row] ordering the labeling tool uses.
        Points off the grid come back with out-of-range indices, not
        clamped."""
        x = np.asarray(x)
        y = np.asarray(y)
        n_rows = self.channels.shape[0]
        col = (x - self.origin[0]) / self.pixel_size
        row = (n_rows - 1) - (y - self.origin[1]) / self.pixel_size
        return np.stack([col, row], axis=-1)


def _median_vertex_spacing(xy):
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=2)  # k=2: self (0) + nearest other vertex
    return float(np.median(dists[:, 1]))


def rasterize_to_grid(vertices, faces, pixel_size=None, smooth_sigma_px=10.0):
    """vertices: (N, 3) positioned mesh. faces: (M, 3). Returns a MeshGrid.

    pixel_size defaults to the median nearest-neighbour vertex spacing
    (finer loses nothing, coarser starts averaging real detail away).
    smooth_sigma_px sets how much of the broad shape is treated as
    'background' and subtracted for the local_relief channel.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    xy = vertices[:, :2]
    z = vertices[:, 2]

    if pixel_size is None:
        pixel_size = _median_vertex_spacing(xy)

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    n_cols = int(np.ceil((x_max - x_min) / pixel_size)) + 1
    n_rows = int(np.ceil((y_max - y_min) / pixel_size)) + 1

    col_x = x_min + np.arange(n_cols) * pixel_size
    row_y = y_min + np.arange(n_rows) * pixel_size          # ascending Y = ascending row index
    grid_x, grid_y = np.meshgrid(col_x, row_y)              # (n_rows, n_cols)

    height = griddata(xy, z, (grid_x, grid_y), method='linear')  # NaN outside the convex hull

    k1, k2 = compute_principal_curvatures(vertices, faces)
    fscore = flatness_score(k1, k2)
    curvature = griddata(xy, fscore, (grid_x, grid_y), method='linear')

    mask = ~np.isnan(height)

    # NaN-aware Gaussian smoothing (normalized convolution): smooth the
    # masked height and the mask itself, then divide -- so holes and the
    # object edge don't bleed zeros into the smoothed background.
    filled = np.where(mask, height, 0.0)
    weight = mask.astype(np.float64)
    sm_filled = gaussian_filter(filled, smooth_sigma_px)
    sm_weight = gaussian_filter(weight, smooth_sigma_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed_height = sm_filled / sm_weight
    local_relief = np.where(mask, height - smoothed_height, 0.0)

    height = np.where(mask, height, 0.0)
    curvature = np.where(mask & ~np.isnan(curvature), curvature, 0.0)

    channels = np.stack([height, local_relief, curvature], axis=-1)

    # Flip vertically so row 0 = largest Y (top-down view, Y up).
    channels = channels[::-1]
    mask = mask[::-1]

    return MeshGrid(channels=channels, mask=mask, origin=(x_min, y_min), pixel_size=pixel_size)


def save_channel_previews(grid, out_dir):
    """Write each channel + the mask as a PNG for eyeballing. Uses
    matplotlib (already a dependency)."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    names = ['height', 'local_relief', 'curvature']
    for i, name in enumerate(names):
        data = grid.channels[:, :, i].copy()
        data[~grid.mask] = np.nan
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap='viridis')
        plt.colorbar(label=name)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{name}.png'), dpi=100)
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.mask, cmap='gray')
    plt.title('mask (white = mesh present)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mask.png'), dpi=100)
    plt.close()
