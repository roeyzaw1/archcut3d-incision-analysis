"""
Incision labeling tool (experiment/ml-feature-detection branch).

Step 2a: view the 2.5D grid for the loaded model, one channel at a time.
Step 2b: trace incision centerlines on it (polylines in grid/pixel coords).
Save/load of labeled examples (2c) comes next.
"""

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import QtWidgets

CHANNEL_NAMES = ['height', 'local_relief', 'curvature']


class LabelToolDialog(QtWidgets.QDialog):
    def __init__(self, grid, source_name, parent=None):
        super().__init__(parent)
        self.grid = grid
        self._cbar = None
        self.channel_index = 1  # local_relief -- usually the clearest for incisions

        # Traces are polylines in grid/pixel coordinates: [col, row] (= imshow
        # data coords). `traces` holds finished curves; `current_trace` is the
        # one being drawn.
        self.traces = []
        self.current_trace = []
        self._trace_artists = []

        self.setWindowTitle(f"Label Incisions -- {source_name}")
        self.resize(760, 800)

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
        layout.addWidget(self.canvas)

        data = self._channel_data()
        self._im = self.ax.imshow(data, cmap='viridis')
        self._im.set_clim(np.nanmin(data), np.nanmax(data))
        self.ax.set_title(CHANNEL_NAMES[self.channel_index])
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04)
        self.figure.tight_layout()
        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self._on_click)

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

        self.statusLabel = QtWidgets.QLabel()
        layout.addWidget(self.statusLabel)
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

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
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
            # nothing in-progress: reopen the last finished curve to edit it
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
