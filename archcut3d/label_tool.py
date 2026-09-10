"""
Incision labeling tool (experiment/ml-feature-detection branch).

Step 2a: view the 2.5D grid for the loaded model, one channel at a time.
Step 2b: trace incision centerlines on it (polylines in grid/pixel coords).
Step 2c: save/load a labeled example -- a self-contained .npz holding the
grid channels, mask, calibration, and the traced centerlines, so it can be
consumed directly for training later.
"""

import os

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import QtWidgets

CHANNEL_NAMES = ['height', 'local_relief', 'curvature']


class LabelToolDialog(QtWidgets.QDialog):
    def __init__(self, grid, source_path, parent=None, initial_traces=None):
        super().__init__(parent)
        self.grid = grid
        self.source_path = source_path or ''
        self.source_name = os.path.basename(self.source_path) if self.source_path else 'model'
        self._cbar = None
        self.channel_index = 1  # local_relief -- usually the clearest for incisions

        # Traces are polylines in grid/pixel coordinates: [col, row] (= imshow
        # data coords). `traces` holds finished curves; `current_trace` is the
        # one being drawn. `initial_traces` seeds it -- e.g. curves traced on
        # the 3D model and projected onto this grid.
        self.traces = [list(map(list, t)) for t in (initial_traces or [])]
        self.current_trace = []
        self._trace_artists = []

        self.setWindowTitle(f"Label Incisions -- {self.source_name}")
        self.resize(760, 820)

        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Channel:"))
        self.channelCombo = QtWidgets.QComboBox()
        self.channelCombo.addItems(CHANNEL_NAMES)
        self.channelCombo.setCurrentIndex(self.channel_index)
        self.channelCombo.currentIndexChanged.connect(self._on_channel_changed)
        controls.addWidget(self.channelCombo)
        controls.addStretch()
        h, w, _ = grid.channels.shape
        controls.addWidget(QtWidgets.QLabel(f"{w}x{h} px  @  {grid.pixel_size:.4f} mm/px"))
        layout.addLayout(controls)

        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Standard matplotlib nav bar: pan, zoom-to-rectangle, home (reset view).
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        data = self._channel_data()
        self._im = self.ax.imshow(data, cmap='viridis')
        self._im.set_clim(np.nanmin(data), np.nanmax(data))
        self.ax.set_title(CHANNEL_NAMES[self.channel_index])
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04)
        self.figure.tight_layout()
        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        trace_controls = QtWidgets.QHBoxLayout()
        trace_controls.addWidget(QtWidgets.QLabel("Left-click: add point   ·   Right-click: finish curve"))
        trace_controls.addStretch()
        finishBtn = QtWidgets.QPushButton("Finish Curve")
        finishBtn.clicked.connect(self._finish_current_trace)
        undoBtn = QtWidgets.QPushButton("Undo Point")
        undoBtn.clicked.connect(self._undo_point)
        delCurveBtn = QtWidgets.QPushButton("Delete Last Curve")
        delCurveBtn.clicked.connect(self._delete_last_trace)
        clearBtn = QtWidgets.QPushButton("Clear All")
        clearBtn.clicked.connect(self._clear_traces)
        for b in (finishBtn, undoBtn, delCurveBtn, clearBtn):
            trace_controls.addWidget(b)
        layout.addLayout(trace_controls)

        io_controls = QtWidgets.QHBoxLayout()
        io_controls.addStretch()
        saveBtn = QtWidgets.QPushButton("Save Labeled Example...")
        saveBtn.clicked.connect(self._save_labeled_example)
        loadBtn = QtWidgets.QPushButton("Load Traces...")
        loadBtn.clicked.connect(self._load_traces)
        io_controls.addWidget(saveBtn)
        io_controls.addWidget(loadBtn)
        layout.addLayout(io_controls)

        self.statusLabel = QtWidgets.QLabel()
        layout.addWidget(self.statusLabel)
        self._redraw_traces()   # draw any seeded initial_traces
        self._update_status()

    # --- grid display ---

    def _channel_data(self):
        data = self.grid.channels[:, :, self.channel_index].astype(float).copy()
        data[~self.grid.mask] = np.nan
        return data

    def _on_channel_changed(self, index):
        self.channel_index = index
        data = self._channel_data()
        self._im.set_data(data)
        self._im.set_clim(np.nanmin(data), np.nanmax(data))
        self._cbar.update_normal(self._im)
        self.ax.set_title(CHANNEL_NAMES[self.channel_index])
        self._redraw_traces()

    # --- tracing ---

    def _on_scroll(self, event):
        """Scroll wheel zooms, centered on the cursor -- smoother than the
        toolbar's zoom-rect for the trace-a-bit / move / trace-a-bit loop."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        scale = (1 / 1.2) if event.button == 'up' else 1.2  # 'up' = zoom in
        x, y = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        self.ax.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.toolbar.mode:            # pan or zoom-rect active -> let the toolbar handle it
            return
        point = [float(event.xdata), float(event.ydata)]
        if event.button == 1:            # left: add a point to the current curve
            self.current_trace.append(point)
            self._redraw_traces()
            self._update_status()
        elif event.button == 3:          # right: finish the current curve
            self._finish_current_trace()

    def _finish_current_trace(self):
        if len(self.current_trace) >= 2:
            self.traces.append(self.current_trace)
            self.current_trace = []
            self._redraw_traces()
            self._update_status()

    def _undo_point(self):
        if self.current_trace:
            self.current_trace.pop()
        elif self.traces:
            self.current_trace = self.traces.pop()
            self.current_trace.pop()
        self._redraw_traces()
        self._update_status()

    def _delete_last_trace(self):
        if self.current_trace:
            self.current_trace = []
        elif self.traces:
            self.traces.pop()
        self._redraw_traces()
        self._update_status()

    def _clear_traces(self):
        self.traces = []
        self.current_trace = []
        self._redraw_traces()
        self._update_status()

    def _redraw_traces(self):
        for artist in self._trace_artists:
            artist.remove()
        self._trace_artists = []

        for trace in self.traces:
            arr = np.array(trace)
            line, = self.ax.plot(arr[:, 0], arr[:, 1], '-', color='red', linewidth=1.5,
                                 marker='.', markersize=4)
            self._trace_artists.append(line)

        if self.current_trace:
            arr = np.array(self.current_trace)
            line, = self.ax.plot(arr[:, 0], arr[:, 1], '--', color='yellow', linewidth=1.5,
                                 marker='o', markersize=5, markerfacecolor='yellow',
                                 markeredgecolor='black')
            self._trace_artists.append(line)

        self.canvas.draw_idle()

    def _update_status(self):
        self.statusLabel.setText(
            f"Curves: {len(self.traces)}   |   current curve: {len(self.current_trace)} point(s)")

    # --- save / load ---

    def _all_finished_traces(self):
        """Finished traces + the in-progress one if it has >= 2 points."""
        traces = list(self.traces)
        if len(self.current_trace) >= 2:
            traces.append(self.current_trace)
        return traces

    def _default_label_path(self):
        if self.source_path:
            base, _ = os.path.splitext(self.source_path)
            return base + '.label.npz'
        return 'model.label.npz'

    def _save_labeled_example(self):
        traces = self._all_finished_traces()
        if not traces:
            QtWidgets.QMessageBox.warning(self, 'Nothing to Save', 'Trace at least one curve first.')
            return

        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Labeled Example', self._default_label_path(), 'Labeled Example (*.label.npz)')
        if not fileName:
            return
        if not fileName.lower().endswith('.npz'):
            fileName += '.label.npz'

        # Ragged trace list -> one (total_points, 2) array + a lengths array,
        # so no pickle is needed.
        trace_lengths = np.array([len(t) for t in traces], dtype=np.int64)
        traces_xy = np.concatenate([np.asarray(t, dtype=np.float64) for t in traces], axis=0)

        np.savez_compressed(
            fileName,
            channels=self.grid.channels.astype(np.float32),
            mask=self.grid.mask,
            channel_names=np.array(CHANNEL_NAMES),
            pixel_size=np.float64(self.grid.pixel_size),
            origin=np.asarray(self.grid.origin, dtype=np.float64),
            source_file=np.array(self.source_name),
            traces_xy=traces_xy,
            trace_lengths=trace_lengths,
        )
        self.statusLabel.setText(f"Saved {len(traces)} curve(s) to {os.path.basename(fileName)}")

    def _load_traces(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Traces', self._default_label_path(),
            'Labeled Example (*.label.npz);;All npz (*.npz)')
        if not fileName:
            return

        try:
            data = np.load(fileName, allow_pickle=False)
            traces_xy = data['traces_xy']
            trace_lengths = data['trace_lengths']
            saved_source = str(data['source_file']) if 'source_file' in data else ''
        except (OSError, KeyError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, 'Load Failed', f'Could not read the file:\n{e}')
            return

        if saved_source and saved_source != self.source_name:
            proceed = QtWidgets.QMessageBox.question(
                self, 'Different Model',
                f'This file was labeled on "{saved_source}", but the tool is open on '
                f'"{self.source_name}". The traces are in pixel coordinates and may not '
                f'line up. Load anyway?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if proceed != QtWidgets.QMessageBox.Yes:
                return

        splits = np.cumsum(trace_lengths)[:-1]
        self.traces = [chunk.tolist() for chunk in np.split(traces_xy, splits)]
        self.current_trace = []
        self._redraw_traces()
        self._update_status()
