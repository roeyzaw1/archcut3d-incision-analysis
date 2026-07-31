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
