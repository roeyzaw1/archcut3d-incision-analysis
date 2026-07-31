"""
2-D per-slice profile viewer -- lets you page through each computed slice's
(relative position, height) profile to visually confirm the slicing ran
where expected, and interactively pick the aperture angle for the current
slice (selectAngleLines.m): click two points on each shoulder, fit a line
through every point between them, then report the angle between the two
lines.

The two-Gaussian depth/FWHM fit is a separate, later step.
"""

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from .slice_analysis import nearest_index, fit_line_through_range, compute_aperture_angle


class SliceViewerDialog(QtWidgets.QDialog):
    def __init__(self, slices, parent=None):
        super().__init__(parent)
        self.slices = slices
        self.index = 0
        self.current_x = None
        self.current_y = None

        self._reset_angle_state()

        self.setWindowTitle("Slice Viewer")
        self.resize(600, 560)

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        nav = QtWidgets.QHBoxLayout()
        self.prevButton = QtWidgets.QPushButton("< Previous")
        self.prevButton.clicked.connect(self.show_previous)
        self.sliceLabel = QtWidgets.QLabel()
        self.sliceLabel.setAlignment(QtCore.Qt.AlignHCenter)
        self.nextButton = QtWidgets.QPushButton("Next >")
        self.nextButton.clicked.connect(self.show_next)
        nav.addWidget(self.prevButton)
        nav.addWidget(self.sliceLabel)
        nav.addWidget(self.nextButton)
        layout.addLayout(nav)

        angleRow = QtWidgets.QHBoxLayout()
        self.selectLeftButton = QtWidgets.QPushButton("Select Left Line")
        self.selectLeftButton.setToolTip("Click two points on the left shoulder of the profile")
        self.selectLeftButton.clicked.connect(lambda: self.start_picking('left'))
        self.selectRightButton = QtWidgets.QPushButton("Select Right Line")
        self.selectRightButton.setToolTip("Click two points on the right shoulder of the profile")
        self.selectRightButton.clicked.connect(lambda: self.start_picking('right'))
        self.clearLinesButton = QtWidgets.QPushButton("Clear Lines")
        self.clearLinesButton.clicked.connect(self.clear_lines)
        angleRow.addWidget(self.selectLeftButton)
        angleRow.addWidget(self.selectRightButton)
        angleRow.addWidget(self.clearLinesButton)
        layout.addLayout(angleRow)

        self.angleLabel = QtWidgets.QLabel("Aperture angle: —")
        layout.addWidget(self.angleLabel)

        self.update_plot()

    def _reset_angle_state(self):
        self.picking_side = None       # None, 'left', or 'right'
        self.pending_click_index = None  # first click's snapped index, while picking
        self.left_indices = None       # (idx_a, idx_b) once both clicks landed
        self.right_indices = None
        self.left_line = None          # (slope, intercept)
        self.right_line = None
        if hasattr(self, 'angleLabel'):
            self._update_angle_label()

    def show_previous(self):
        if self.index > 0:
            self.index -= 1
            self._reset_angle_state()
            self.update_plot()

    def show_next(self):
        if self.index < len(self.slices) - 1:
            self.index += 1
            self._reset_angle_state()
            self.update_plot()

    def set_slices(self, slices):
        self.slices = slices
        self.index = 0
        self._reset_angle_state()
        self.update_plot()

    def start_picking(self, side):
        self.picking_side = side
        self.pending_click_index = None
        if side == 'left':
            self.left_indices = None
            self.left_line = None
        else:
            self.right_indices = None
            self.right_line = None
        self.statusHint = f"Click two points for the {side} line."
        self._update_angle_label()
        self.update_plot()

    def clear_lines(self):
        self._reset_angle_state()
        self.update_plot()

    def on_plot_click(self, event):
        if self.picking_side is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = nearest_index(self.current_x, self.current_y, event.xdata, event.ydata)

        if self.pending_click_index is None:
            self.pending_click_index = idx
            self.update_plot()
            return

        idx_a, idx_b = self.pending_click_index, idx
        try:
            slope, intercept = fit_line_through_range(self.current_x, self.current_y, idx_a, idx_b)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Line Fit Failed', str(e))
            self.pending_click_index = None
            self.update_plot()
            return

        if self.picking_side == 'left':
            self.left_indices = (idx_a, idx_b)
            self.left_line = (slope, intercept)
        else:
            self.right_indices = (idx_a, idx_b)
            self.right_line = (slope, intercept)

        self.picking_side = None
        self.pending_click_index = None
        self._update_angle_label()
        self.update_plot()

    def _update_angle_label(self):
        if self.left_line is not None and self.right_line is not None:
            angle = compute_aperture_angle(self.left_line[0], self.right_line[0])
            self.angleLabel.setText(f"Aperture angle: {angle:.1f} deg")
        else:
            self.angleLabel.setText("Aperture angle: —")

    def update_plot(self):
        self.ax.clear()
        profile = self.slices[self.index]
        if profile.shape[0] > 0:
            x_raw = profile[:, 0] - profile[:, 0].mean()
            order = np.argsort(x_raw)
            self.current_x = x_raw[order]
            self.current_y = profile[order, 1]
            self.ax.plot(self.current_x, self.current_y, '.', markersize=3, color='tab:blue')

            for indices, line, color in (
                (self.left_indices, self.left_line, 'tab:red'),
                (self.right_indices, self.right_line, 'tab:green'),
            ):
                if indices is not None and line is not None:
                    lo, hi = min(indices), max(indices)
                    xr = self.current_x[lo:hi + 1]
                    slope, intercept = line
                    self.ax.plot(xr, slope * xr + intercept, '-', linewidth=2, color=color)
                    self.ax.plot(self.current_x[[indices[0], indices[1]]],
                                 self.current_y[[indices[0], indices[1]]],
                                 'o', color=color, markersize=6)

            if self.pending_click_index is not None:
                self.ax.plot(self.current_x[self.pending_click_index], self.current_y[self.pending_click_index],
                             'o', color='orange', markersize=6)
        else:
            self.current_x = np.array([])
            self.current_y = np.array([])
            self.ax.text(0.5, 0.5, 'No points in this slice', ha='center', va='center', transform=self.ax.transAxes)

        self.ax.set_title(f"Slice {self.index + 1} of {len(self.slices)}")
        self.ax.set_xlabel("Relative position (mm)")
        self.ax.set_ylabel("Height (mm)")
        self.ax.grid(True, linestyle=':')
        self.canvas.draw()

        self.sliceLabel.setText(f"{self.index + 1} / {len(self.slices)}")
        self.prevButton.setEnabled(self.index > 0)
        self.nextButton.setEnabled(self.index < len(self.slices) - 1)
