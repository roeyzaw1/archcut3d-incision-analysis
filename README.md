# ArchCut 3-D

Software developed to precisely evaluate the 3-D micromorphological
characteristics of archaeological surface incisions. Analysis is based on a
continuous 3-D slice sequence from a chosen range within the incision path.
Measurements and shapes of individual slices can be extracted. ArchCut 3-D's
interface is designed to provide a computational and mathematical analysis of
engravings to a wide audience.

Built with PyQt5, VTK, and PyMeshLab.

## Project layout

```
archcut3d/              Application package
    app.py               Main window, dialogs, and entry point (main())
    crest_detection.py   Ridge/ravine (crest line) detection, wraps setCurvature
    mesh_smoothing.py    VTK windowed-sinc mesh smoothing, used to pre-condition
                          meshes before crest detection
    resources.py         Shared path helpers (assets, cache dir, setCurvature binary)
assets/
    icons/               Window/dialog icons
    images/              Logo overlay, loading indicator, startup splash
    scale_bars/          Scale bar overlays (2cm, 5cm, 10cm, 20cm, 20mm, 50mm, 100mm, 200mm)
bin/
    setCurvature          setCurvature binary (macOS/Linux)
    setcurvature.exe      setCurvature binary (Windows)
docs/
    Manual.pdf            User manual, opened from the About menu
main.py                  Launcher: `python main.py`
requirements.txt
```

## Running

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

A model file path can be passed as an argument to open it directly:

```bash
python main.py path/to/model.stl
```

## setCurvature

Crest line (ridge/ravine) detection shells out to the `setCurvature` binary in
`bin/`, implementing Yoshizawa, Belyaev & Seidel (2005). The correct binary
for the current OS is picked up automatically by
`archcut3d.resources.get_setcurvature_path()`.

## ML experiments (branch `experiment/ml-feature-detection`)

`archcut3d/ml/` contains an iterative incision tracer (`TracerNet`: a small
CNN that, from a heading-aligned patch of a 2.5D relief grid, predicts the
next step along an incision and a stop probability) and a study of what it
learns: does it follow the local geometric minimum of the surface, or the
hand-drawn line where the two differ?

Inputs are `.label.npz` files written by the Label Incisions tool (grid
channels, calibration and hand-traced lines). Experiments, all
leave-one-scanned-model-out with a separate validation model for early
stopping:

| id | question |
|---|---|
| E0 | TracerNet vs a straight-ahead baseline and a geometric (local-relief minimum) tracer |
| E0b | where the hand line departs from the geometric minimum, and how each tracer behaves there |
| E1 | encoder frozen at random initialisation, heads trained |
| E2 | height / local_relief / curvature as the only input channel |
| E3 | PCA dimensionality and layer-wise linear probes of the encoder (trained vs random) |
| E4 | (optional) flatten instead of average pooling, CoordConv |
| E5 | train on geometry-only crest-line labels generated on the other scans, test on hand traces |

```bash
pip install -r requirements.txt
python -m archcut3d.ml.experiments --data <dir with .label.npz and .wrl> --out results/   # E0-E3, E5
python -m archcut3d.ml.experiments --data <dir> --out results/ --experiments E4          # optional E4
python -m archcut3d.ml.experiments --data <dir> --out results/ --quick                   # smoke test: 1 fold, 1 seed
python -m archcut3d.ml.report results/                                                    # writes results/RESULTS.md
python -m pytest tests                                                                    # fast unit tests
```

Options: `--seeds N` (default 3), `--device auto|cpu|mps`, `--exclude FILE...`,
`--include-machine`, `--scans DIR` (location of the `.wrl` files for E5,
default `--data`). Label files whose points are closer together than 2 pixels
(machine-generated crest-line output rather than hand clicks) are excluded
automatically and listed in `results/dataset_files.csv`.

Outputs in `--out`: `config.json` (arguments, hyperparameters, seeds, git
commit, library versions), one CSV per experiment (one row per test model x
seed x variant x segment), `figures/`, `RESULTS.md`. Trained checkpoints go to
`results/models/` and crest-label caches to `results/cache/` (both git-ignored).
A full run takes roughly an hour on an Apple-silicon laptop. Runs on `mps`
are not bit-reproducible; on `cpu` a seed reproduces a run exactly.
