"""
2-D per-slice profile viewer -- lets you page through each computed slice's
(relative position, height) profile to visually confirm the slicing ran
where expected, interactively pick the aperture angle (selectAngleLines.m:
click two points on each shoulder, fit a line through every point between
them, report the angle between the two lines), fit the two-Gaussian
depth/FWHM model (gaussian_fit.py), and export the accumulated results to
Excel -- the batch "Analyze Slices" workflow from anal_slices_Callback in
harutot.m.

Two deliberate simplifications vs the original MATLAB workflow:
  - ONE combined snapshot image per analyzed slice (aperture-angle lines +
    Gaussian fit overlaid on the same live plot) instead of two separate
    figures, since this viewer already shows both on one interactive plot.
  - The free-text "additional comments" field is stored as its own Excel
    column instead of a separate _comments.txt file per slice.
"""

import os
import datetime

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
import openpyxl

from .slice_analysis import nearest_index, fit_line_through_range, compute_aperture_angle
from .gaussian_fit import fit_two_gaussians, two_gaussian_model

RESULTS_HEADER = ['Slice name', 'Slice number', 'Aperture angle [deg]',
                  'af1', 'bf1', 'cf1', 'af2', 'bf2', 'cf2', 'df', 'ef',
                  'Depth [mm]', 'FWHM [mm]',
                  'Keyword 1', 'Keyword 2', 'Keyword 3', 'Comments']


class BatchInfoDialog(QtWidgets.QDialog):
    """The once-per-session keyword/comments prompt (MATLAB's inputdlg for
    'Keyword 1','Keyword 2','Keyword 3','Additional comments')."""

    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Slice Set Description")
        layout = QtWidgets.QFormLayout(self)
        initial = initial or ("", "", "", "")
        self.kw1 = QtWidgets.QLineEdit(initial[0])
        self.kw2 = QtWidgets.QLineEdit(initial[1])
        self.kw3 = QtWidgets.QLineEdit(initial[2])
        self.comments = QtWidgets.QPlainTextEdit(initial[3])
        self.comments.setFixedHeight(80)
        layout.addRow("Keyword 1:", self.kw1)
        layout.addRow("Keyword 2:", self.kw2)
        layout.addRow("Keyword 3:", self.kw3)
        layout.addRow("Additional comments:", self.comments)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        return (self.kw1.text(), self.kw2.text(), self.kw3.text(), self.comments.toPlainText())


class SliceViewerDialog(QtWidgets.QDialog):
    def __init__(self, slices, parent=None, source_model_path=None):
        super().__init__(parent)
        self.slices = slices
        self.index = 0
        self.current_x = None
        self.current_y = None
        self.source_model_path = source_model_path

        self.output_dir = None
        self.session_dir = None
        self.batch_info = None          # (kw1, kw2, kw3, comments) once set
        self.results_rows = []
        self._fit_seed = 0

        self._reset_angle_state()
        self._reset_gauss_state()

        self.setWindowTitle("Slice Viewer")
        self.resize(600, 700)

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

        layout.addWidget(self._hline())

        gaussRow = QtWidgets.QHBoxLayout()
        self.fitGaussianButton = QtWidgets.QPushButton("Fit Two Gaussians")
        self.fitGaussianButton.clicked.connect(self.fit_gaussian)
        self.retryFitButton = QtWidgets.QPushButton("Retry Fit")
        self.retryFitButton.setToolTip("Re-fit from a fresh batch of starting guesses")
        self.retryFitButton.clicked.connect(self.fit_gaussian)
        self.retryFitButton.setEnabled(False)
        self.clearGaussButton = QtWidgets.QPushButton("Clear Fit")
        self.clearGaussButton.clicked.connect(self.clear_gaussian_fit)
        self.clearGaussButton.setEnabled(False)
        gaussRow.addWidget(self.fitGaussianButton)
        gaussRow.addWidget(self.retryFitButton)
        gaussRow.addWidget(self.clearGaussButton)
        layout.addLayout(gaussRow)

        self.gaussLabel = QtWidgets.QLabel("Depth: —   FWHM: —")
        layout.addWidget(self.gaussLabel)

        layout.addWidget(self._hline())

        saveRow = QtWidgets.QHBoxLayout()
        self.saveSliceButton = QtWidgets.QPushButton("Save This Slice's Analysis")
        self.saveSliceButton.setToolTip(
            "Records the aperture angle and/or Gaussian fit (whichever are set) as one row, "
            "and saves a snapshot image")
        self.saveSliceButton.clicked.connect(self.save_slice_analysis)
        self.exportButton = QtWidgets.QPushButton("Export All to Excel...")
        self.exportButton.clicked.connect(self.export_to_excel)
        saveRow.addWidget(self.saveSliceButton)
        saveRow.addWidget(self.exportButton)
        layout.addLayout(saveRow)

        self.resultsLabel = QtWidgets.QLabel("0 slice(s) recorded")
        layout.addWidget(self.resultsLabel)

        self.update_plot()

    @staticmethod
    def _hline():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # --- state reset ---

    def _reset_angle_state(self):
        self.picking_side = None       # None, 'left', or 'right'
        self.pending_click_index = None  # first click's snapped index, while picking
        self.left_indices = None       # (idx_a, idx_b) once both clicks landed
        self.right_indices = None
        self.left_line = None          # (slope, intercept)
        self.right_line = None
        if hasattr(self, 'angleLabel'):
            self._update_angle_label()

    def _reset_gauss_state(self):
        self.gauss_result = None
        if hasattr(self, 'gaussLabel'):
            self._update_gauss_label()
            self.retryFitButton.setEnabled(False)
            self.clearGaussButton.setEnabled(False)

    def current_angle(self):
        if self.left_line is not None and self.right_line is not None:
            return compute_aperture_angle(self.left_line[0], self.right_line[0])
        return None

    # --- navigation ---

    def show_previous(self):
        if self.index > 0:
            self.index -= 1
            self._reset_angle_state()
            self._reset_gauss_state()
            self.update_plot()

    def show_next(self):
        if self.index < len(self.slices) - 1:
            self.index += 1
            self._reset_angle_state()
            self._reset_gauss_state()
            self.update_plot()

    def set_slices(self, slices, source_model_path=None):
        self.slices = slices
        self.index = 0
        self._reset_angle_state()
        self._reset_gauss_state()
        if source_model_path is not None:
            self.source_model_path = source_model_path
        self.update_plot()

    # --- aperture angle picking (unchanged) ---

    def start_picking(self, side):
        self.picking_side = side
        self.pending_click_index = None
        if side == 'left':
            self.left_indices = None
            self.left_line = None
        else:
            self.right_indices = None
            self.right_line = None
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
        angle = self.current_angle()
        self.angleLabel.setText(f"Aperture angle: {angle:.1f} deg" if angle is not None else "Aperture angle: —")

    # --- two-Gaussian fit ---

    def fit_gaussian(self):
        if self.current_x is None or len(self.current_x) < 8:
            QtWidgets.QMessageBox.warning(self, 'Not Enough Points', 'This slice has too few points to fit.')
            return
        self._fit_seed += 1
        try:
            self.gauss_result = fit_two_gaussians(self.current_x, self.current_y, seed=self._fit_seed)
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, 'Fit Failed', str(e))
            return
        self.retryFitButton.setEnabled(True)
        self.clearGaussButton.setEnabled(True)
        self._update_gauss_label()
        self.update_plot()

    def clear_gaussian_fit(self):
        self._reset_gauss_state()
        self.update_plot()

    def _update_gauss_label(self):
        r = self.gauss_result
        if r is None:
            self.gaussLabel.setText("Depth: —   FWHM: —")
        else:
            flag = "" if r.sane else "  (fit looks off -- try Retry)"
            self.gaussLabel.setText(f"Depth: {r.depth:.3f} mm   FWHM: {r.fwhm:.3f} mm{flag}")

    # --- plotting ---

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

            if self.gauss_result is not None:
                r = self.gauss_result
                yfit = two_gaussian_model(self.current_x, *r.params)
                self.ax.plot(self.current_x, yfit, '-', linewidth=2, color='tab:purple',
                             label='two-Gaussian fit')

                # depth: vertical dashed line from the fit's minimum up to the
                # baseline directly above it (harutot.m's depth marker)
                x_min, y_min, y_baseline = r.depth_marker
                self.ax.plot([x_min, x_min], [y_min, y_baseline], '--', color='tab:red', linewidth=1.5)

                # width: horizontal dashed line spanning the two points FWHM
                # was measured between (harutot.m's width marker)
                if r.width_marker is not None:
                    wx1, wx2, wy1, wy2 = r.width_marker
                    self.ax.plot([wx1, wx2], [wy1, wy2], '--', color='tab:red', linewidth=1.5)

                # the numbers themselves, drawn ON the plot -- this is what
                # actually ends up in the saved image (the Qt label below the
                # plot is not part of the saved figure)
                self.ax.annotate(f"Depth = {r.depth:.3f} mm\nFWHM = {r.fwhm:.3f} mm" +
                                 ("" if r.sane else "\n(fit looks off)"),
                                 xy=(x_min, y_min), xytext=(10, 12), textcoords='offset points',
                                 fontsize=9, color='tab:red', va='bottom')

                self.ax.legend(loc='best', fontsize=8)
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

    # --- batch info / output location ---

    def _ensure_batch_info(self) -> bool:
        if self.batch_info is not None:
            return True
        dlg = BatchInfoDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False
        self.batch_info = dlg.values()
        return True

    def _default_session_name(self):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.source_model_path:
            base = os.path.splitext(os.path.basename(self.source_model_path))[0]
            return f"{base}_{stamp}"
        return f"analysis_{stamp}"

    def _ensure_output_dir(self) -> bool:
        if self.session_dir is not None:
            return True
        start_dir = os.path.dirname(self.source_model_path) if self.source_model_path else ''
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Save analysis: select a folder', start_dir)
        if not chosen:
            return False
        self.output_dir = chosen
        self.session_dir = os.path.join(chosen, self._default_session_name())
        os.makedirs(os.path.join(self.session_dir, 'slice_images'), exist_ok=True)
        return True

    # --- save / export ---

    def save_slice_analysis(self):
        angle = self.current_angle()
        if angle is None and self.gauss_result is None:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to Save',
                'Pick an aperture angle and/or fit the two-Gaussian model before saving this slice.')
            return
        if not self._ensure_batch_info():
            return
        if not self._ensure_output_dir():
            return

        default_name = f"Slice {self.index + 1}"
        slice_name, ok = QtWidgets.QInputDialog.getText(
            self, 'Slice Name', 'What do you want to call this slice?', text=default_name)
        if not ok:
            return
        slice_name = slice_name.strip() or default_name

        img_path = os.path.join(self.session_dir, 'slice_images', f"{slice_name}.png")
        self.figure.savefig(img_path, dpi=150)

        kw1, kw2, kw3, comments = self.batch_info
        r = self.gauss_result
        row = {
            'Slice name': slice_name,
            'Slice number': self.index + 1,
            'Aperture angle [deg]': round(angle, 1) if angle is not None else '',
            'af1': r.params[0] if r else '', 'bf1': r.params[1] if r else '', 'cf1': r.params[2] if r else '',
            'af2': r.params[3] if r else '', 'bf2': r.params[4] if r else '', 'cf2': r.params[5] if r else '',
            'df': r.params[6] if r else '', 'ef': r.params[7] if r else '',
            'Depth [mm]': round(r.depth, 3) if r else '',
            'FWHM [mm]': round(r.fwhm, 3) if r else '',
            'Keyword 1': kw1, 'Keyword 2': kw2, 'Keyword 3': kw3, 'Comments': comments,
        }
        self.results_rows.append(row)
        self.resultsLabel.setText(f"{len(self.results_rows)} slice(s) recorded")

    def export_to_excel(self):
        if not self.results_rows:
            QtWidgets.QMessageBox.information(self, 'Nothing to Export', 'No slices have been saved yet.')
            return
        if not self._ensure_output_dir():
            return

        default_path = os.path.join(self.session_dir, 'slices_data.xlsx')
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Analysis Data', default_path, 'Excel Files (*.xlsx)')
        if not fileName:
            return
        if not fileName.lower().endswith('.xlsx'):
            fileName += '.xlsx'

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Slices'
        ws.append(RESULTS_HEADER)
        for row in self.results_rows:
            ws.append([row[col] for col in RESULTS_HEADER])
        wb.save(fileName)

        QtWidgets.QMessageBox.information(
            self, 'Analysis Complete', f'{len(self.results_rows)} slice(s) saved to:\n{fileName}')
