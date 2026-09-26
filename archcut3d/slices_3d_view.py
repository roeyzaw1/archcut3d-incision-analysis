"""Combined 3D view of every computed slice at once (visualize_slices.m's
`d3` figure): one 3D scatter with every slice's profile stacked along a
slice-index axis, colored by a red (first slice) -> blue (last slice)
gradient, with a per-slice number label -- lets you see the whole cut in
one view instead of paging through slices one at a time.
"""

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from matplotlib.figure import Figure
from PyQt5 import QtWidgets


def slice_color(index: int, n: int) -> tuple:
    """Red (first slice) -> blue (last slice) gradient, matching
    visualize_slices.m's `[1-(i-1)/(numSlices-1) 0 (i-1)/(numSlices-1)]`."""
    frac = index / (n - 1) if n > 1 else 0.0
    return (1 - frac, 0.0, frac)


class Slices3DViewDialog(QtWidgets.QDialog):
    def __init__(self, slices, tolerance, parent=None):
        super().__init__(parent)
        self.slices = slices
        self.tolerance = tolerance
        self._view_set = False

        self.setWindowTitle("3D Slice Overview")
        self.resize(800, 700)

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas)

        controls = QtWidgets.QHBoxLayout()
        self.labelsCheck = QtWidgets.QCheckBox("Show slice number labels")
        self.labelsCheck.setChecked(True)
        self.labelsCheck.toggled.connect(self.update_plot)
        controls.addWidget(self.labelsCheck)
        controls.addStretch()
        layout.addLayout(controls)

        self.update_plot()

    def set_slices(self, slices, tolerance):
        self.slices = slices
        self.tolerance = tolerance
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        n = len(self.slices)
        show_labels = self.labelsCheck.isChecked()

        # One combined scatter call for every point across every slice,
        # rather than one call per slice: matplotlib's 3D axes depth-sorts
        # per ARTIST (each scatter() call gets a single averaged depth), so
        # with one call per slice, a whole slice can get sorted in front of
        # or behind another it shouldn't -- some slices appearing to hide
        # behind others until you rotate. A single call sorts every point
        # individually instead of every slice as one blob.
        all_x, all_y, all_z, all_colors = [], [], [], []
        for i, profile in enumerate(self.slices):
            if profile.shape[0] == 0:
                continue
            x_rel = profile[:, 0] - profile[:, 0].mean()
            y_pos = np.full(profile.shape[0], -(i + 1) * self.tolerance)
            z = profile[:, 1]
            color = slice_color(i, n)
            all_x.append(x_rel)
            all_y.append(y_pos)
            all_z.append(z)
            all_colors.extend([color] * len(x_rel))

            if show_labels:
                label_x = float(x_rel.min()) - 0.1
                label_y = -(i + 1) * self.tolerance
                label_z = float(z.max() + z.min()) / 2
                inverted = (1 - color[0], 1 - color[1], 1 - color[2])
                self.ax.text(label_x, label_y, label_z, str(i + 1),
                             fontsize=9, fontweight='bold', color=inverted)

        if all_x:
            self.ax.scatter(np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z),
                            s=8, c=all_colors)

        self.ax.set_title("Slice Analysis")
        self.ax.set_xlabel("Relative Position")
        self.ax.set_ylabel("Slice Number (position along path)")
        self.ax.set_zlabel("Height")
        # a view angle chosen to show the whole stack fanned out (rather
        # than matplotlib's default, which tends to view it closer to
        # edge-on) -- still fully rotatable by dragging
        if not self._view_set:
            self.ax.view_init(elev=22, azim=-60)
            self._view_set = True
        self.canvas.draw()
