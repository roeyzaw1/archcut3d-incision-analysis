"""
Incision labeling tool (experiment/ml-feature-detection branch).

Step 2a: view the 2.5D grid for the loaded model, one channel at a time.
Tracing (2b) and save/load of labeled examples (2c) come next.
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

        self.setWindowTitle(f"Label Incisions -- {source_name}")
        self.resize(760, 720)

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

        # Build the image + colorbar once; channel switches only update the data.
        data = self._channel_data()
        self._im = self.ax.imshow(data, cmap='viridis')
        self._im.set_clim(np.nanmin(data), np.nanmax(data))
        self.ax.set_title(CHANNEL_NAMES[self.channel_index])
        self._cbar = self.figure.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04)
        self.figure.tight_layout()
        self.canvas.draw()

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
        self.canvas.draw()
