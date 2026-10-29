"""End-to-end smoke test of the experiment runner on synthetic label files."""
import csv
import json
import os

import numpy as np

from archcut3d.ml import experiments, report
from tests.fakes import FakeExample
from tests.test_ml_policies import PX, groove_example


def _write_label_file(path, name, row, tilt):
    ex = groove_example(tilt=tilt)
    cols = np.arange(40, 200, 3, dtype=float)          # hand clicks every 3 px: not machine output
    trace = np.stack([cols, np.full_like(cols, row)], axis=1)
    np.savez_compressed(path, channels=ex.channels.astype(np.float32), mask=ex.mask,
                        channel_names=np.array(["height", "local_relief", "curvature"]),
                        pixel_size=np.float64(PX), origin=np.array([0.0, 0.0]), source_file=np.array(name),
                        traces_xy=trace, trace_lengths=np.array([len(trace)]))


def test_quick_run_writes_all_outputs(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    for name, row, tilt in (("a.wrl", 100.3, 0.0), ("b.wrl", 100.3 + 3.0, 0.2), ("c.wrl", 100.3 - 2.0, -0.2)):
        _write_label_file(str(data / f"{name[0]}.label.npz"), name, row, tilt)
    # one machine-generated file: points 0.5 px apart; must be excluded automatically
    ex = FakeExample("m.wrl", [])
    dense = np.stack([np.arange(40, 120, 0.5), np.full(160, 100.0)], axis=1)
    np.savez_compressed(str(data / "m.label.npz"), channels=ex.channels, mask=ex.mask,
                        channel_names=np.array(["height", "local_relief", "curvature"]), pixel_size=np.float64(PX),
                        origin=np.array([0.0, 0.0]), source_file=np.array("m.wrl"), traces_xy=dense,
                        trace_lengths=np.array([len(dense)]))

    experiments.main(["--data", str(data), "--out", str(out), "--quick", "--device", "cpu",
                      "--experiments", "E0,E0b,E1,E2,E3"])

    for f in ("config.json", "dataset_files.csv", "e0.csv", "e1.csv", "e2.csv", "e3_pca.csv", "e3_probes.csv",
              "e0b_departure_summary.csv", "e0b_departure_points.csv"):
        assert (out / f).exists(), f
    assert list((out / "figures").glob("e0b_example_*.png"))
    assert (out / "figures" / "e3_pca.png").exists()

    cfg = json.load(open(out / "config.json"))
    assert cfg["models"] == ["a.wrl", "b.wrl", "c.wrl"] and cfg["excluded_machine_generated"] == ["m.label.npz"]
    assert cfg["git"]["commit"] and cfg["seeds"] == [0]

    rows = list(csv.DictReader(open(out / "e0.csv")))
    variants = {r["variant"] for r in rows}
    assert {"tracernet", "straight", "geometric"} <= variants
    assert {r["segment"] for r in rows} == {"all", "departure", "normal"}
    e2 = {r["variant"] for r in csv.DictReader(open(out / "e2.csv"))}
    assert e2 == {"only_height", "only_local_relief", "only_curvature"}

    report.main([str(out)])
    text = open(out / "RESULTS.md").read()
    assert "## Dataset" in text and "Departure and normal points" in text and "m.label.npz" in text
