"""
crest_detection.py

Detects convex (ridges) and concave (ravines) crest lines on a triangular mesh
and overlays them on an ArchCut 3-D VTK renderer.

Wraps the `setCurvature` C++ executable (Yoshizawa, Belyaev & Seidel, 2005) as
a subprocess — same approach as the original MATLAB tool. Crest lines are
rendered as VTK actors that share the main mesh's transform, so they rotate
and translate with the model when the user manipulates it.

Two interactive features beyond plain detection:

  - Show/Hide toggles for ridges and ravines (red and blue, by convention).
  - A live "Min Ridgeness" slider per type — drag right to hide weaker
    crest-line components, isolating the strongest features. Filtering
    operates on whole components (one slider step = one ridgeness threshold,
    applied to every component's per-component ridgeness score).

Optional light pre-smoothing of the detection mesh (off by default) calms the
third-order curvature estimate so crest lines fragment less. See
mesh_smoothing.py. The smoothing applies ONLY to the copy fed to setCurvature;
the displayed mesh and all other tools are unaffected.

Public surface:

    CrestDetector(main_window)
        .add_to_menu(parent_menu)   # Adds "Crest Detection" submenu + filter toolbar.
        .on_file_loaded()           # Call after a new mesh is loaded.
        .clear_crests()             # Removes any current crest overlays.

Expected layout of the bundled executable, relative to the project root:

    archcut3d/
        app.py
    bin/
        setCurvature           # macOS / Linux build (chmod +x required on macOS).
        setcurvature.exe       # Windows build.
"""

import os
import sys
import math
import platform
import subprocess

import vtk
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtWidgets import (
    QDialog, QGridLayout, QLabel, QLineEdit, QMessageBox,
    QToolBar, QSlider, QPushButton, QWidget, QVBoxLayout, QHBoxLayout,
)

from .resources import (
    resource_path as _resource_path,
    get_cache_dir as _get_cache_dir,
    get_setcurvature_path as _get_setcurvature_path,
)


# ---------------------------------------------------------------------------
# Tiny "running…" indicator shown during the (potentially slow) subprocess.
# ---------------------------------------------------------------------------

class _CrestLoadingDialog(QDialog):
    """Modal "running…" dialog shown while setCurvature runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setWindowTitle("Detecting crests")
        self.setFixedSize(260, 90)

        layout = QtWidgets.QVBoxLayout(self)
        label = QLabel("Running crest detection…\nThis can take a few seconds to several minutes.")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)


# ---------------------------------------------------------------------------
# Parameter-input dialog. Mirrors the MATLAB inputdlg defaults exactly.
# ---------------------------------------------------------------------------

class CrestDetectionDialog(QDialog):
    """Collects neighborhood size and the α/β and γ thresholds for detection,
    plus optional light pre-smoothing of the detection mesh."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detect Crests (Ridges & Ravines)")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(_resource_path("assets/icons/frame_icon.png")))
        self.setMinimumWidth(360)

        # Filled in only on accept.
        self.neighborhood_size = None
        self.alpha_beta_threshold = None
        self.gamma_threshold = None
        self.smoothing_iterations = None      # <-- ADDED
        self.smoothing_pass_band = None       # <-- ADDED

        layout = QGridLayout()
        self.setLayout(layout)

        header = QLabel(
            "Detect crest lines on the loaded mesh.\n"
            "Defaults follow Yoshizawa, Belyaev & Seidel (2005)."
        )
        header.setWordWrap(True)
        layout.addWidget(header, 0, 0, 1, 2)

        layout.addWidget(QLabel("Neighborhood size:"), 1, 0)
        self.neighborhood_edit = QLineEdit("1")
        self.neighborhood_edit.setValidator(QIntValidator(1, 50, self))
        self.neighborhood_edit.setToolTip(
            "Integer ≥ 1. Larger values smooth the surface fit but slow detection."
        )
        layout.addWidget(self.neighborhood_edit, 1, 1)

        layout.addWidget(QLabel("α/β threshold (radians):"), 2, 0)
        self.alpha_beta_edit = QLineEdit(f"{math.pi / 3:.4f}")
        self.alpha_beta_edit.setToolTip("Default π/3 ≈ 1.0472. Crest line gap-jumping.")
        layout.addWidget(self.alpha_beta_edit, 2, 1)

        layout.addWidget(QLabel("γ threshold (radians):"), 3, 0)
        self.gamma_edit = QLineEdit(f"{math.pi / 2:.4f}")
        self.gamma_edit.setToolTip("Default π/2 ≈ 1.5708. Crest line gap-jumping.")
        layout.addWidget(self.gamma_edit, 3, 1)

        # <-- ADDED: light pre-smoothing of the detection mesh. Off by default
        # (0 iterations) so behaviour is unchanged unless the user opts in.
        # Smoothing calms the third-order curvature estimate, reducing crest
        # line fragmentation. See mesh_smoothing.py.
        layout.addWidget(QLabel("Smoothing iterations:"), 4, 0)
        self.smoothing_iters_edit = QLineEdit("0")
        self.smoothing_iters_edit.setValidator(QIntValidator(0, 200, self))
        self.smoothing_iters_edit.setToolTip(
            "0 = no smoothing (default). Try 5-20 for light smoothing to reduce\n"
            "crest line fragmentation. Higher values smooth more (and erase fine detail)."
        )
        layout.addWidget(self.smoothing_iters_edit, 4, 1)

        layout.addWidget(QLabel("Smoothing pass-band:"), 5, 0)
        self.smoothing_passband_edit = QLineEdit("0.1")
        self.smoothing_passband_edit.setToolTip(
            "Cutoff for the smoothing low-pass, range (0, 2]. SMALLER = smooth\n"
            "harder; LARGER = keep more detail. Default 0.1. Only used when\n"
            "iterations > 0."
        )
        layout.addWidget(self.smoothing_passband_edit, 5, 1)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 6, 0, 1, 2)   # <-- CHANGED: row 4 -> 6

    def _on_accept(self):
        # Match the MATLAB script: silently fall back to defaults on bad input.
        try:
            n = int(self.neighborhood_edit.text())
        except (TypeError, ValueError):
            n = 5
        if n <= 0:
            n = 5

        try:
            ab = float(self.alpha_beta_edit.text())
        except (TypeError, ValueError):
            ab = math.pi / 3
        if ab <= 0:
            ab = math.pi / 3

        try:
            g = float(self.gamma_edit.text())
        except (TypeError, ValueError):
            g = math.pi / 2
        if g <= 0:
            g = math.pi / 2

        self.neighborhood_size = n
        self.alpha_beta_threshold = ab
        self.gamma_threshold = g

        # <-- ADDED: smoothing params, same silent-fallback style.
        try:
            si = int(self.smoothing_iters_edit.text())
        except (TypeError, ValueError):
            si = 0
        if si < 0:
            si = 0

        try:
            pb = float(self.smoothing_passband_edit.text())
        except (TypeError, ValueError):
            pb = 0.1
        if pb <= 0:
            pb = 0.1

        self.smoothing_iterations = si
        self.smoothing_pass_band = pb

        self.accept()


# Number of slider steps used for the ridgeness filter sliders. QSlider only
# supports integer values, so we map this 0..N range to the actual ridgeness
# range (0..max) of the current data.
_SLIDER_STEPS = 1000


# ---------------------------------------------------------------------------
# Main controller. One instance per MainWindow; owns its actors, menu items,
# and the filter toolbar.
# ---------------------------------------------------------------------------

class CrestDetector:
    """Adds ridges/ravines detection + filtering to an ArchCut 3-D MainWindow.

    Instantiate once with the main window. Attach the menu via `add_to_menu`
    (which also installs the filter toolbar in a hidden state). Notify of
    file loads via `on_file_loaded`.
    """

    # Convention from the geometry-processing literature: convex ridges in red,
    # concave ravines in blue.
    RIDGE_COLOR = (1.0, 0.0, 0.0)
    RAVINE_COLOR = (0.0, 0.0, 1.0)
    LINE_WIDTH = 3.0

    LARGE_MESH_WARNING_THRESHOLD = 100_000   # vertices

    def __init__(self, main_window):
        self.main = main_window

        # VTK actors — None until detection has produced data of that type.
        self.ridges_actor = None
        self.ravines_actor = None

        # Detection results retained for filtering. Each is either None or a dict:
        #     {
        #       'points': [(x, y, z), ...],
        #       'edges_with_comp': [(p1_idx, p2_idx, comp_id), ...],
        #       'component_ridgeness': {comp_id: ridgeness_value},
        #       'max_ridgeness': float,   # used to scale the slider
        #     }
        self.ridges_data = None
        self.ravines_data = None

        # Menu items (created in add_to_menu).
        self._menu_button = None
        self._detect_action = None
        self._toggle_ridges_action = None
        self._toggle_ravines_action = None
        self._clear_action = None

        # Filter toolbar widgets (created in add_to_menu, populated after detection).
        self._filter_toolbar = None
        self._ridges_slider = None
        self._ridges_value_label = None
        self._ridges_section = None  # Container so we can hide it when no ridges.
        self._ravines_slider = None
        self._ravines_value_label = None
        self._ravines_section = None

    # =======================================================================
    # Public API
    # =======================================================================

    def add_to_menu(self, parent_menu):
        """Add the 'Crest Detection' submenu to `parent_menu` (a QMenu).
        Also installs the filter toolbar (hidden) on the main window."""
        self._build_menu(parent_menu)
        self._build_filter_toolbar()

    def on_file_loaded(self):
        """Call after a new mesh is loaded into the main window."""
        if self._menu_button is not None:
            self._menu_button.setEnabled(True)
        self.clear_crests()

    def clear_crests(self):
        """Remove any displayed crest actors, hide the filter toolbar, and
        reset toggle/slider state."""
        if self.ridges_actor is not None:
            self.main.ren.RemoveActor(self.ridges_actor)
            self.ridges_actor = None
        if self.ravines_actor is not None:
            self.main.ren.RemoveActor(self.ravines_actor)
            self.ravines_actor = None

        self.ridges_data = None
        self.ravines_data = None

        if self._toggle_ridges_action is not None:
            self._toggle_ridges_action.setChecked(True)
            self._toggle_ridges_action.setEnabled(False)
        if self._toggle_ravines_action is not None:
            self._toggle_ravines_action.setChecked(True)
            self._toggle_ravines_action.setEnabled(False)
        if self._clear_action is not None:
            self._clear_action.setEnabled(False)

        if self._filter_toolbar is not None:
            self._filter_toolbar.hide()

        try:
            self.main.vtkWidget.GetRenderWindow().Render()
        except Exception:
            pass

    # =======================================================================
    # UI construction
    # =======================================================================

    def _build_menu(self, parent_menu):
        """Create the Crest Detection submenu and attach to `parent_menu`."""
        # Use a real QMenu with title, attached via addMenu(), so the submenu
        # renders correctly when nested in the top-level menu bar. (A
        # QToolButton + QWidgetAction wrapper renders fine in a toolbar but
        # collapses to an empty row inside a menu bar — we hit this bug
        # with the original Analysis submenu items, same fix here.)
        crest_menu = QtWidgets.QMenu("Crest Detection", self.main)

        self._detect_action = QtWidgets.QAction("Detect Crests…", self.main)
        self._detect_action.triggered.connect(self.detect_crests)
        crest_menu.addAction(self._detect_action)

        self._toggle_ridges_action = QtWidgets.QAction("Show Ridges", self.main)
        self._toggle_ridges_action.setCheckable(True)
        self._toggle_ridges_action.setChecked(True)
        self._toggle_ridges_action.setEnabled(False)
        self._toggle_ridges_action.triggered.connect(self._toggle_ridges)
        crest_menu.addAction(self._toggle_ridges_action)

        self._toggle_ravines_action = QtWidgets.QAction("Show Ravines", self.main)
        self._toggle_ravines_action.setCheckable(True)
        self._toggle_ravines_action.setChecked(True)
        self._toggle_ravines_action.setEnabled(False)
        self._toggle_ravines_action.triggered.connect(self._toggle_ravines)
        crest_menu.addAction(self._toggle_ravines_action)

        self._clear_action = QtWidgets.QAction("Clear Crests", self.main)
        self._clear_action.setEnabled(False)
        self._clear_action.triggered.connect(self.clear_crests)
        crest_menu.addAction(self._clear_action)

        parent_menu.addMenu(crest_menu)
        # menuAction() is the action that opens this submenu in its parent.
        # Disabling it greys out the "Crest Detection" entry until a file loads.
        self._menu_button = crest_menu.menuAction()
        self._menu_button.setEnabled(False)

    def _build_filter_toolbar(self):
        """Create the (initially hidden) ridgeness filter toolbar on the right
        side of the main window. Sliders are populated when detection runs."""
        toolbar = QToolBar("Crest Filter", self.main)
        toolbar.setObjectName("CrestFilterToolbar")
        toolbar.setOrientation(Qt.Vertical)
        toolbar.hide()

        # Wrap everything in a single container with our own layout — gives us
        # full control over spacing and lets us hide whole sections at once
        # (e.g. when only ridges were found and not ravines).
        container = QWidget(toolbar)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        title = QLabel("Crest Filter")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        hint = QLabel("Hide weaker components by raising\nthe minimum ridgeness threshold.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        outer.addWidget(hint)

        # Ridges section.
        self._ridges_section, self._ridges_slider, self._ridges_value_label = \
            self._build_filter_section("Ridges (convex)", self.RIDGE_COLOR, self._on_filter_changed)
        outer.addWidget(self._ridges_section)

        # Ravines section.
        self._ravines_section, self._ravines_slider, self._ravines_value_label = \
            self._build_filter_section("Ravines (concave)", self.RAVINE_COLOR, self._on_filter_changed)
        outer.addWidget(self._ravines_section)

        outer.addStretch(1)

        # Reset and Hide buttons at the bottom.
        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Set both thresholds back to 0 (show everything).")
        reset_btn.clicked.connect(self._reset_filter)
        button_row.addWidget(reset_btn)

        hide_btn = QPushButton("Hide Panel")
        hide_btn.setToolTip("Close this panel. Re-open via Analysis → Crest Detection.")
        hide_btn.clicked.connect(toolbar.hide)
        button_row.addWidget(hide_btn)
        outer.addLayout(button_row)

        toolbar.addWidget(container)
        self.main.addToolBar(Qt.RightToolBarArea, toolbar)
        self._filter_toolbar = toolbar

    @staticmethod
    def _build_filter_section(title_text, accent_rgb, on_change):
        """Build a labelled slider section. Returns (container, slider, value_label).
        The container is what's hidden when this section's data is unavailable."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        # Coloured swatch + title.
        header_row = QHBoxLayout()
        swatch = QLabel("  ")
        r, g, b = accent_rgb
        swatch.setStyleSheet(
            f"background-color: rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)});"
            "border: 1px solid #444; min-width: 14px; max-width: 14px;"
            "min-height: 14px; max-height: 14px;"
        )
        header_row.addWidget(swatch)
        header_row.addWidget(QLabel(title_text))
        header_row.addStretch(1)
        v.addLayout(header_row)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(_SLIDER_STEPS)
        slider.setValue(0)
        slider.valueChanged.connect(on_change)
        v.addWidget(slider)

        value_label = QLabel("min ≥ 0.000   (showing all)")
        value_label.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(value_label)

        return container, slider, value_label

    # =======================================================================
    # Detection
    # =======================================================================

    def detect_crests(self):
        """Run the full pipeline: dialog → write input → subprocess → display."""
        current_actor = getattr(self.main, "current_actor", None)
        if current_actor is None:
            QMessageBox.warning(self.main, "No Mesh", "Please load a 3D file first.")
            return

        exe_path = _get_setcurvature_path()
        if exe_path is None:
            exe_name = "setcurvature.exe" if platform.system() == "Windows" else "setCurvature"
            QMessageBox.critical(
                self.main, "Executable Not Found",
                f"Could not find '{exe_name}' in the bin/ folder next to this script.\n\n"
                f"Expected location: bin/{exe_name}\n\n"
                "Place the executable there and try again."
            )
            return

        polydata = current_actor.GetMapper().GetInput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            QMessageBox.warning(
                self.main, "No Mesh Data",
                "The current file does not have polygonal mesh data the crest detector can use.\n\n"
                "If this is a WRL file, click 'Other Functions' first to convert it, then try again."
            )
            return

        n_points = polydata.GetNumberOfPoints()
        if n_points > self.LARGE_MESH_WARNING_THRESHOLD:
            reply = QMessageBox.question(
                self.main, "Large Mesh",
                f"This mesh has {n_points:,} vertices. Crest detection can be slow on meshes "
                "this large (potentially several minutes).\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        dialog = CrestDetectionDialog(self.main)
        if dialog.exec_() != QDialog.Accepted:
            return

        loading = _CrestLoadingDialog(self.main)
        loading.show()
        QtWidgets.QApplication.processEvents()

        try:
            ridges_data, ravines_data = self._run_pipeline(
                polydata, exe_path,
                dialog.neighborhood_size,
                dialog.alpha_beta_threshold,
                dialog.gamma_threshold,
                dialog.smoothing_iterations,   # <-- ADDED
                dialog.smoothing_pass_band,    # <-- ADDED
            )
        except _CrestPipelineError as e:
            loading.close()
            QMessageBox.critical(self.main, "Crest Detection Failed", str(e))
            return
        finally:
            loading.close()

        # Wipe any previous run's actors and data, then build fresh ones.
        self.clear_crests()

        if ridges_data is not None and len(ridges_data["edges_with_comp"]) > 0:
            self.ridges_data = ridges_data
            self.ridges_actor = self._build_crest_actor(ridges_data, self.RIDGE_COLOR)
            self.main.ren.AddActor(self.ridges_actor)
            self._toggle_ridges_action.setEnabled(True)
            self._toggle_ridges_action.setChecked(True)

        if ravines_data is not None and len(ravines_data["edges_with_comp"]) > 0:
            self.ravines_data = ravines_data
            self.ravines_actor = self._build_crest_actor(ravines_data, self.RAVINE_COLOR)
            self.main.ren.AddActor(self.ravines_actor)
            self._toggle_ravines_action.setEnabled(True)
            self._toggle_ravines_action.setChecked(True)

        if self.ridges_actor is None and self.ravines_actor is None:
            QMessageBox.information(
                self.main, "No Crests Found",
                "The detector ran successfully but did not find any ridges or ravines.\n\n"
                "Try a smaller neighborhood size or a different mesh."
            )
            return

        self._clear_action.setEnabled(True)

        # Configure & show the filter toolbar based on what data is available.
        self._configure_filter_toolbar()

        self.main.vtkWidget.GetRenderWindow().Render()

        # Summary popup.
        n_r_edges = len(self.ridges_data["edges_with_comp"]) if self.ridges_data else 0
        n_v_edges = len(self.ravines_data["edges_with_comp"]) if self.ravines_data else 0
        n_r_comp = len(self.ridges_data["component_ridgeness"]) if self.ridges_data else 0
        n_v_comp = len(self.ravines_data["component_ridgeness"]) if self.ravines_data else 0

        # <-- ADDED: surface the smoothing setting in the summary for reproducibility.
        if dialog.smoothing_iterations and dialog.smoothing_iterations > 0:
            smoothing_line = (
                f"Smoothing: {dialog.smoothing_iterations} iterations, "
                f"pass-band {dialog.smoothing_pass_band:g}\n\n"
            )
        else:
            smoothing_line = "Smoothing: off\n\n"

        QMessageBox.information(
            self.main, "Detection Complete",
            f"Ridges (convex): {n_r_edges} edges across {n_r_comp} component(s)\n"
            f"Ravines (concave): {n_v_edges} edges across {n_v_comp} component(s)\n"
            f"{smoothing_line}"
            "Use the Crest Filter panel on the right to hide weaker components."
        )

    # =======================================================================
    # Toggle handlers
    # =======================================================================

    def _toggle_ridges(self):
        if self.ridges_actor is None:
            return
        self.ridges_actor.SetVisibility(self._toggle_ridges_action.isChecked())
        self.main.vtkWidget.GetRenderWindow().Render()

    def _toggle_ravines(self):
        if self.ravines_actor is None:
            return
        self.ravines_actor.SetVisibility(self._toggle_ravines_action.isChecked())
        self.main.vtkWidget.GetRenderWindow().Render()

    # =======================================================================
    # Filter handlers
    # =======================================================================

    def _configure_filter_toolbar(self):
        """After detection: set up sliders from current data, show the toolbar."""
        # Each section is hidden when its corresponding data is missing.
        self._ridges_section.setVisible(self.ridges_data is not None)
        self._ravines_section.setVisible(self.ravines_data is not None)

        # Reset both sliders to "show everything". Block signals while resetting
        # so we don't trigger a needless geometry rebuild on each set.
        for slider in (self._ridges_slider, self._ravines_slider):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)

        # Update the value labels to reflect the new max-ridgeness scale.
        self._update_value_labels()

        # Show the panel only if there's actually something to filter.
        if self.ridges_data is not None or self.ravines_data is not None:
            self._filter_toolbar.show()
        else:
            self._filter_toolbar.hide()

    def _on_filter_changed(self):
        """Slider moved → recompute visible cells for both actors and re-render."""
        if self.ridges_data is not None:
            t = self._slider_to_threshold(self._ridges_slider, self.ridges_data["max_ridgeness"])
            self._rebuild_visible_geometry(self.ridges_actor, self.ridges_data, t)
        if self.ravines_data is not None:
            t = self._slider_to_threshold(self._ravines_slider, self.ravines_data["max_ridgeness"])
            self._rebuild_visible_geometry(self.ravines_actor, self.ravines_data, t)
        self._update_value_labels()
        self.main.vtkWidget.GetRenderWindow().Render()

    def _reset_filter(self):
        """Reset both sliders to 0 (show all components)."""
        for slider in (self._ridges_slider, self._ravines_slider):
            slider.setValue(0)
        # _on_filter_changed runs automatically via valueChanged signals.

    def _update_value_labels(self):
        """Refresh the small "min ≥ X.XXX" labels under each slider."""
        if self.ridges_data is not None:
            t = self._slider_to_threshold(self._ridges_slider, self.ridges_data["max_ridgeness"])
            n_visible = sum(
                1 for r in self.ridges_data["component_ridgeness"].values() if r >= t
            )
            n_total = len(self.ridges_data["component_ridgeness"])
            self._ridges_value_label.setText(
                f"min ≥ {t:.3f}   ({n_visible}/{n_total} components)"
            )

        if self.ravines_data is not None:
            t = self._slider_to_threshold(self._ravines_slider, self.ravines_data["max_ridgeness"])
            n_visible = sum(
                1 for r in self.ravines_data["component_ridgeness"].values() if r >= t
            )
            n_total = len(self.ravines_data["component_ridgeness"])
            self._ravines_value_label.setText(
                f"min ≥ {t:.3f}   ({n_visible}/{n_total} components)"
            )

    @staticmethod
    def _slider_to_threshold(slider, max_ridgeness):
        """Map slider integer position to a ridgeness threshold value."""
        if max_ridgeness <= 0:
            return 0.0
        return (slider.value() / _SLIDER_STEPS) * max_ridgeness

    @staticmethod
    def _rebuild_visible_geometry(actor, data, threshold):
        """Rebuild `actor`'s line cell array to include only edges whose
        component's ridgeness meets the threshold."""
        if actor is None or data is None:
            return

        polydata = actor.GetMapper().GetInput()
        n_pts = polydata.GetNumberOfPoints()
        component_ridgeness = data["component_ridgeness"]

        new_lines = vtk.vtkCellArray()
        for p1, p2, comp_id in data["edges_with_comp"]:
            if 0 <= p1 < n_pts and 0 <= p2 < n_pts:
                if component_ridgeness.get(comp_id, 0.0) >= threshold:
                    new_lines.InsertNextCell(2)
                    new_lines.InsertCellPoint(p1)
                    new_lines.InsertCellPoint(p2)

        polydata.SetLines(new_lines)
        polydata.Modified()

    # =======================================================================
    # Pipeline internals
    # =======================================================================

    def _run_pipeline(self, polydata, exe_path,
                      neighborhood_size, alpha_beta, gamma,
                      smoothing_iterations=0, smoothing_pass_band=0.1):  # <-- ADDED
        """Triangulate → (optional smooth) → write input → run subprocess → parse outputs.
        Returns (ridges_data, ravines_data) — each either a data dict or None."""
        work_dir = _get_cache_dir()
        os.makedirs(work_dir, exist_ok=True)

        # 1. Triangulate so non-triangle meshes work too.
        tri_filter = vtk.vtkTriangleFilter()
        tri_filter.SetInputData(polydata)
        tri_filter.Update()
        tri_polydata = tri_filter.GetOutput()

        # <-- ADDED: optional light smoothing of the detection copy only.
        # The displayed mesh is unaffected (it comes from the VTK reader, not
        # from tri_polydata). Crest points are rendered raw on the smoothed
        # surface (option a); with light smoothing the offset is sub-pixel.
        if smoothing_iterations and smoothing_iterations > 0:
            from mesh_smoothing import smooth_polydata
            tri_polydata = smooth_polydata(
                tri_polydata, smoothing_iterations, smoothing_pass_band, clean=True
            )

        # 2. Write object.txt in setCurvature's PLY2-like format.
        object_txt = os.path.join(work_dir, "object.txt")
        self._write_object_txt(tri_polydata, object_txt, neighborhood_size)

        # 3. Clear any stale outputs from previous runs.
        ridges_txt = os.path.join(work_dir, "ridges.txt")
        ravines_txt = os.path.join(work_dir, "ravines.txt")
        output_txt = os.path.join(work_dir, "output.txt")
        for f in (ridges_txt, ravines_txt, output_txt):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # 4. Run setCurvature. cwd=work_dir keeps its outputs in our cache folder.
        cmd = [exe_path, "object.txt", "output.txt", str(gamma), str(alpha_beta)]
        try:
            result = subprocess.run(
                cmd, cwd=work_dir,
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise _CrestPipelineError(
                "setCurvature did not finish within 10 minutes. "
                "The mesh may be too large or too complex. Try reducing the file size first."
            )
        except OSError as e:
            hint = ""
            if platform.system() == "Darwin":
                hint = (
                    "\n\nOn macOS this often means Gatekeeper is blocking an unsigned "
                    "binary. Right-click the executable in Finder → Open → Open Anyway, "
                    "then try again."
                )
            raise _CrestPipelineError(f"Could not run setCurvature: {e}{hint}")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"exit code {result.returncode}"
            raise _CrestPipelineError(f"setCurvature failed.\n\nDetails:\n{details}")

        if not os.path.exists(ridges_txt) or not os.path.exists(ravines_txt):
            raise _CrestPipelineError(
                "setCurvature finished but did not produce the expected output files."
            )

        ridges_data = self._read_crest_file(ridges_txt)
        ravines_data = self._read_crest_file(ravines_txt)
        return ridges_data, ravines_data

    @staticmethod
    def _write_object_txt(polydata, path, neighborhood_size):
        """Write a triangle mesh in the format setCurvature expects.

        Format (from the tool's manual):
            line 1: number of vertices
            line 2: number of triangles
            line 3: neighborhood size (integer)
            line 4: 1   (with crest line tracing; 0 = without)
            then per vertex:   x, y, z each on its own line
            then per triangle: 3 vertex indices each on its own line

        The MATLAB preprocess_crest writes triangle indices in REVERSED order
        (third, second, first). We mirror that so winding/orientation matches
        what the C tool expects.
        """
        n_points = polydata.GetNumberOfPoints()
        n_cells = polydata.GetNumberOfCells()
        points = polydata.GetPoints()

        with open(path, "w") as fp:
            fp.write(f"{n_points}\n")
            fp.write(f"{n_cells}\n")
            fp.write(f"{int(neighborhood_size)}\n")
            fp.write("1\n")  # with crest line tracing

            for i in range(n_points):
                x, y, z = points.GetPoint(i)
                fp.write(f"{x}\n{y}\n{z}\n")

            for i in range(n_cells):
                cell = polydata.GetCell(i)
                if cell.GetNumberOfPoints() != 3:
                    continue
                ids = cell.GetPointIds()
                a = ids.GetId(0)
                b = ids.GetId(1)
                c = ids.GetId(2)
                fp.write(f"{c}\n{b}\n{a}\n")  # reversed order, matches MATLAB

    @staticmethod
    def _read_crest_file(path):
        """Parse a ridges.txt or ravines.txt file from setCurvature.

        Format:
            line 1: V (number of crest line points)
            line 2: E (number of crest line edges)
            line 3: N (number of connected components)
            next V lines: x y z component_id
            next N lines: ridgeness sphericalness cyclideness
            next E lines: pt1_id pt2_id triangle_id   (-1 = connecting edge)

        Returns a dict:
            {
              'points':              [(x, y, z), ...],
              'edges_with_comp':     [(p1_idx, p2_idx, comp_id), ...],
              'component_ridgeness': {comp_id: ridgeness_value},
              'max_ridgeness':       float,
            }
        Or None if the file is malformed.

        Each edge's component_id is taken from its first endpoint's component_id.
        Edges whose two endpoints disagree on component are skipped (they
        shouldn't exist in a well-formed file, but we're defensive).
        """
        try:
            with open(path, "r") as fp:
                tokens = fp.read().split()
        except OSError:
            return None

        if len(tokens) < 3:
            return {
                "points": [], "edges_with_comp": [],
                "component_ridgeness": {}, "max_ridgeness": 0.0,
            }

        idx = 0
        try:
            n_v = int(tokens[idx]); idx += 1
            n_e = int(tokens[idx]); idx += 1
            n_n = int(tokens[idx]); idx += 1
        except ValueError:
            return None

        # Per-vertex: x, y, z, component_id.
        points = []
        point_components = []
        for _ in range(n_v):
            if idx + 4 > len(tokens):
                break
            try:
                x = float(tokens[idx]); idx += 1
                y = float(tokens[idx]); idx += 1
                z = float(tokens[idx]); idx += 1
                comp = int(tokens[idx]); idx += 1
            except ValueError:
                break
            points.append((x, y, z))
            point_components.append(comp)

        # Per-component: ridgeness, sphericalness, cyclideness (we only keep ridgeness).
        component_ridgeness = {}
        max_ridgeness = 0.0
        for comp_id in range(n_n):
            if idx + 3 > len(tokens):
                break
            try:
                r = float(tokens[idx]); idx += 1
                _s = float(tokens[idx]); idx += 1
                _c = float(tokens[idx]); idx += 1
            except ValueError:
                break
            component_ridgeness[comp_id] = r
            if r > max_ridgeness:
                max_ridgeness = r

        # Per-edge: pt1, pt2, triangle (we only keep pt1, pt2 and derive comp).
        edges_with_comp = []
        for _ in range(n_e):
            if idx + 3 > len(tokens):
                break
            try:
                p1 = int(tokens[idx]); idx += 1
                p2 = int(tokens[idx]); idx += 1
                _tri = int(tokens[idx]); idx += 1
            except ValueError:
                break
            if 0 <= p1 < len(points) and 0 <= p2 < len(points):
                comp1 = point_components[p1]
                comp2 = point_components[p2]
                if comp1 == comp2:
                    edges_with_comp.append((p1, p2, comp1))

        return {
            "points": points,
            "edges_with_comp": edges_with_comp,
            "component_ridgeness": component_ridgeness,
            "max_ridgeness": max_ridgeness,
        }

    def _build_crest_actor(self, data, color):
        """Build a vtkActor of line segments at `data['points']` connected by
        `data['edges_with_comp']`, in the given RGB color, that follows the
        main mesh's transform."""
        points = data["points"]
        edges_with_comp = data["edges_with_comp"]

        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(len(points))
        for i, (x, y, z) in enumerate(points):
            vtk_points.SetPoint(i, x, y, z)

        n_pts = len(points)
        lines = vtk.vtkCellArray()
        for p1, p2, _comp_id in edges_with_comp:
            if 0 <= p1 < n_pts and 0 <= p2 < n_pts:
                lines.InsertNextCell(2)
                lines.InsertCellPoint(p1)
                lines.InsertCellPoint(p2)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(self.LINE_WIDTH)
        actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()

        if hasattr(self.main, "transform") and self.main.transform is not None:
            actor.SetUserTransform(self.main.transform)

        return actor


class _CrestPipelineError(Exception):
    """Raised internally when any stage of the detection pipeline fails."""
    pass