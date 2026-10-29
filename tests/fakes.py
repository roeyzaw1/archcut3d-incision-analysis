"""Small synthetic stand-ins for LabeledExample, for fast tests."""

import numpy as np


class FakeExample:
    """Provides the attributes make_samples reads from a LabeledExample."""

    def __init__(self, source_file, traces, pixel_size=0.05, size=200, channels=None):
        n = size
        if channels is None:
            channels = np.zeros((n, n, 3), np.float32)
        self.channels = channels.astype(np.float32)
        self.mask = np.ones(self.channels.shape[:2], bool)
        self.pixel_size = float(pixel_size)
        self.traces = [np.asarray(t, float) for t in traces]
        self.source_file = source_file

    @property
    def shape_hw(self):
        return self.channels.shape[:2]


def straight_trace(row, col0=40, col1=140):
    """Horizontal trace along +col at the given row ([col, row] points, 1 px apart)."""
    return [[float(c), float(row)] for c in range(col0, col1)]
