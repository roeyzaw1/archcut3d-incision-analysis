import numpy as np

from archcut3d.grid_2d import MeshGrid
from archcut3d.ml import geo_labels
from archcut3d.ml.dataset import LabeledExample


def _grid(pixel_size=0.05, n=200):
    return MeshGrid(channels=np.zeros((n, n, 3), np.float32), mask=np.ones((n, n), bool),
                    origin=(10.0, 20.0), pixel_size=pixel_size)


def _line_3d(x0, y0, length_mm, n=50):
    t = np.linspace(0, length_mm, n)
    return np.stack([x0 + t, np.full(n, y0), np.zeros(n)], axis=1)


def test_crest_traces_are_projected_filtered_in_mm_and_capped(monkeypatch):
    import archcut3d.crest_lines as cl
    lines = [_line_3d(11.0, 21.0, 2.0), _line_3d(11.0, 22.0, 6.0), _line_3d(11.0, 23.0, 4.0), _line_3d(11.0, 24.0, 8.0)]
    monkeypatch.setattr(cl, "detect_crest_lines", lambda v, f: {})
    monkeypatch.setattr(cl, "crest_lines_as_polylines", lambda result, min_edges, which: lines)
    grid = _grid()
    traces = geo_labels.crest_traces_on_grid(None, None, grid, min_length_mm=3.0, max_traces=2)
    assert len(traces) == 2                                         # 2 mm line dropped, 4 mm line cut by the cap
    lens = [np.linalg.norm(np.diff(t, axis=0), axis=1).sum() * grid.pixel_size for t in traces]
    assert abs(lens[0] - 8.0) < 0.05 and abs(lens[1] - 6.0) < 0.05   # longest first
    # projection: world (11, 24) -> col (11-10)/0.05 = 20, row = (n-1) - (24-20)/0.05 = 119
    assert np.allclose(traces[0][0], [20.0, 199 - 80.0])


def test_points_outside_the_grid_are_dropped(monkeypatch):
    import archcut3d.crest_lines as cl
    line = _line_3d(5.0, 21.0, 12.0)                              # starts 5 mm left of the grid origin
    monkeypatch.setattr(cl, "detect_crest_lines", lambda v, f: {})
    monkeypatch.setattr(cl, "crest_lines_as_polylines", lambda result, min_edges, which: [line])
    (trace,) = geo_labels.crest_traces_on_grid(None, None, _grid(), min_length_mm=3.0)
    assert trace[:, 0].min() >= 0 and len(trace) < 50


def test_crest_example_cache_round_trip(tmp_path):
    ex = LabeledExample(path="x.wrl", channels=np.zeros((20, 30, 3), np.float32), mask=np.ones((20, 30), bool),
                        channel_names=["height", "local_relief", "curvature"], pixel_size=0.07, origin=(1.0, 2.0),
                        source_file="x.wrl", traces=[np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                                                     np.array([[7.0, 8.0], [9.0, 10.0]])])
    path = str(tmp_path / "c.npz")
    geo_labels.save_crest_example(path, ex, {"n": 2})
    back = geo_labels.load_crest_example(path)
    assert back.source_file == "x.wrl" and back.pixel_size == 0.07 and back.origin == (1.0, 2.0)
    assert len(back.traces) == 2 and np.allclose(back.traces[1], ex.traces[1])
