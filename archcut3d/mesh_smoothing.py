"""
mesh_smoothing.py

Light, connectivity-preserving low-pass smoothing of a triangle mesh, used to
calm the third-order curvature estimate that setCurvature computes so that
crest lines fragment less.

The smoother is VTK's windowed-sinc filter (vtkWindowedSincPolyDataFilter).
It is the same *family* as Taubin lambda|mu smoothing: a low-pass filter on
the mesh that attenuates high-frequency surface wiggle (noise, micro-roughness)
while leaving the broad shape in place. It does NOT renumber points or
triangles -- only vertex positions move -- so triangle IDs remain valid if a
future version wants exact triangle-ID + barycentric copy-back.

Two knobs:
  - iterations: more passes = smoother. 0 disables smoothing entirely.
  - pass_band:  the cutoff frequency, in (0, 2]. SMALLER = smooth harder
                (lower cutoff, more high frequencies removed); LARGER =
                preserve more detail. VTK's own docs suggest ~0.1 as a
                reasonable starting point; we default gentler (see caller).

Public surface:
    smooth_polydata(polydata, iterations, pass_band, clean=True) -> vtkPolyData

    MeshSmootherTool(main_window)
        .add_to_menu(parent_menu)   # Adds "Smooth Mesh…" entry.
        .on_file_loaded()           # Call after a new mesh is loaded.

The MeshSmootherTool provides interactive PREVIEW of the smoothed mesh in the
viewer (non-destructive: the original mesh stays in memory and you can toggle
back), plus an optional save-to-file. It reuses smooth_polydata, so the preview
matches exactly what crest detection does to its detection copy.
"""

import os
import tempfile

import vtk
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QDialog, QGridLayout, QLabel, QLineEdit, QMessageBox, QFileDialog,
)

try:
    import pymeshlab
except Exception:
    pymeshlab = None


def smooth_polydata(polydata, iterations, pass_band, clean=True):
    """Return a smoothed copy of `polydata`.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Input triangle mesh (already triangulated by the caller).
    iterations : int
        Number of smoothing iterations. If <= 0, the input is returned
        unchanged (no smoothing, no cleaning) so "off" is truly a no-op.
    pass_band : float
        Windowed-sinc pass band in (0, 2]. Smaller smooths harder.
        Values outside the range are clamped.
    clean : bool
        If True, run vtkCleanPolyData first (merges duplicate/unused points).
        This makes the smoother more robust on messy scans. It can change the
        point count, which is fine for raw-render display (option a).

    Returns
    -------
    vtk.vtkPolyData
        A new smoothed mesh. The input is never modified in place.
    """
    if iterations is None or iterations <= 0:
        # Off: hand back the original untouched. Caller writes it as-is.
        return polydata

    # Clamp pass_band into the filter's valid open interval (0, 2].
    if pass_band is None:
        pass_band = 0.1
    pass_band = max(0.001, min(2.0, float(pass_band)))

    mesh = polydata

    # Optional cleaning: merge coincident vertices and drop unused points so
    # the smoother sees consistent topology. Common on archaeological scans.
    if clean:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(mesh)
        cleaner.Update()
        mesh = cleaner.GetOutput()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(int(iterations))
    smoother.SetPassBand(pass_band)

    # Keep the smoothing honest: don't let the filter pull the surface in odd
    # ways at boundaries or feature edges.
    #  - NormalizeCoordinatesOn improves numerical conditioning (recommended
    #    by VTK for this filter).
    #  - FeatureEdgeSmoothingOff: don't smooth across sharp feature edges
    #    differently; treat the surface uniformly.
    #  - BoundarySmoothingOff: leave open boundaries (holes) where they are
    #    rather than shrinking them.
    smoother.NormalizeCoordinatesOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOff()
    smoother.Update()

    return smoother.GetOutput()


# ---------------------------------------------------------------------------
# Parameter dialog: asks for iterations + pass-band, then the tool runs the
# smoothing once and shows it. Mirrors the crest dialog's simple OK/Cancel
# style (no live preview — just enter values, smooth, then toggle).
# ---------------------------------------------------------------------------

class SmoothParamsDialog(QDialog):
    """Collects smoothing iterations and pass-band. Defaults chosen so the
    effect is clearly visible on a first run (heavier than the gentle preview
    defaults), since the whole point is to SEE the smoothing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Smooth Mesh")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(340)

        # Filled in only on accept.
        self.iterations = None
        self.pass_band = None

        layout = QGridLayout()
        self.setLayout(layout)

        header = QLabel(
            "Windowed-sinc smoothing of the displayed mesh.\n"
            "After it runs, use Analysis -> 'Show Smoothed / Original' to toggle."
        )
        header.setWordWrap(True)
        layout.addWidget(header, 0, 0, 1, 2)

        layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.iter_edit = QLineEdit("20")
        self.iter_edit.setValidator(QIntValidator(1, 500, self))
        self.iter_edit.setToolTip("More iterations sharpen the cutoff. 10-30 is typical.")
        layout.addWidget(self.iter_edit, 1, 1)

        layout.addWidget(QLabel("Pass-band:"), 2, 0)
        # Default 0.05 (clearly visible). Lower = smoother.
        self.pb_edit = QLineEdit("0.05")
        self.pb_edit.setToolTip("Range (0, 2]. SMALLER = smooth harder. Try 0.1, 0.05, 0.02.")
        layout.addWidget(self.pb_edit, 2, 1)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 3, 0, 1, 2)

    def _on_accept(self):
        try:
            it = int(self.iter_edit.text())
        except (TypeError, ValueError):
            it = 20
        if it <= 0:
            it = 20

        try:
            pb = float(self.pb_edit.text())
        except (TypeError, ValueError):
            pb = 0.05
        if pb <= 0:
            pb = 0.05

        self.iterations = it
        self.pass_band = pb
        self.accept()


class MeshSmootherTool:
    """Simple flow: Smooth Mesh... -> ask params -> smooth -> show smoothed.
    A second menu entry toggles between the smoothed result and the original.

    One instance per MainWindow. Wire with add_to_menu(); notify of new files
    with on_file_loaded()."""

    def __init__(self, main_window):
        self.main = main_window
        self._action = None          # "Smooth Mesh..."
        self._toggle_action = None   # "Show Smoothed / Original"

        self._preview_actor = None   # the smoothed actor in the scene
        self._source_polydata = None # original geometry we smoothed from
        self._last_smoothed = None   # smoothed polydata (for save, if wanted)
        self._showing_smoothed = False

    # -- wiring -------------------------------------------------------------

    def add_to_menu(self, parent_menu):
        self._action = QtWidgets.QAction("Smooth Mesh\u2026", self.main)
        self._action.setEnabled(False)
        self._action.triggered.connect(self._run)
        parent_menu.addAction(self._action)

        self._toggle_action = QtWidgets.QAction("Show Original / Smoothed", self.main)
        self._toggle_action.setEnabled(False)
        self._toggle_action.triggered.connect(self._toggle)
        parent_menu.addAction(self._toggle_action)

    def on_file_loaded(self):
        if self._action is not None:
            self._action.setEnabled(True)
        # Clear any previous preview/state for the new file.
        self._teardown_preview()
        self._source_polydata = None
        self._last_smoothed = None
        self._showing_smoothed = False
        if self._toggle_action is not None:
            self._toggle_action.setEnabled(False)

    # -- run ----------------------------------------------------------------

    def _acquire_source(self):
        """Return clean vtkPolyData for the current mesh, doing any needed
        conversion automatically. Returns (polydata, None) or (None, error).

        Strategy:
          1. Try the displayed actor's polydata directly. For STL/PLY/OBJ this
             is always present; for WRL, vtkVRMLImporter usually exposes
             polydata on the actor's mapper too, so this often just works.
          2. Only if that yields nothing usable AND the file is a WRL, fall
             back to converting the WRL to a temporary OBJ via pymeshlab and
             reading that back.
        So in the common case there is no conversion at all."""
        # 1. Try the displayed actor's polydata.
        actor = getattr(self.main, "current_actor", None)
        if actor is not None and actor.GetMapper() is not None:
            pd = actor.GetMapper().GetInput()
            if pd is not None and pd.GetNumberOfPoints() > 0:
                out = vtk.vtkPolyData()
                out.DeepCopy(pd)
                return out, None

        # 2. Fall back: convert a WRL on disk.
        current_file = getattr(self.main, "current_file", None)
        is_wrl = bool(current_file) and current_file.lower().endswith(".wrl")
        if is_wrl:
            if pymeshlab is None:
                return None, ("This is a WRL file and pymeshlab (needed to convert it) "
                              "is not available. Click 'Other Functions' to convert the "
                              "file first, then smooth.")
            try:
                obj_path = os.path.join(tempfile.gettempdir(), "cav_smooth_source.obj")
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(current_file)
                ms.save_current_mesh(obj_path)
                reader = vtk.vtkOBJReader()
                reader.SetFileName(obj_path)
                reader.Update()
                pd = reader.GetOutput()
                if pd is None or pd.GetNumberOfPoints() == 0:
                    return None, "Could not read mesh data after converting the WRL file."
                out = vtk.vtkPolyData()
                out.DeepCopy(pd)
                return out, None
            except Exception as e:
                return None, f"Failed to convert the WRL file for smoothing:\n{e}"

        return None, ("No usable mesh data on the current object. "
                      "If this is a WRL file, click 'Other Functions' first.")

    def _run(self):
        if getattr(self.main, "current_file", None) is None:
            QMessageBox.warning(self.main, "No Mesh", "Please load a 3D file first.")
            return

        dialog = SmoothParamsDialog(self.main)
        if dialog.exec_() != QDialog.Accepted:
            return

        # Acquire a usable source mesh (auto-converts WRL). Show a brief busy
        # cursor since conversion + smoothing can take a moment.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            src, err = self._acquire_source()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if src is None:
            QMessageBox.warning(self.main, "Cannot Smooth", err or "No usable mesh.")
            return

        self._source_polydata = src

        # Smooth.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            smoothed = smooth_polydata(src, dialog.iterations, dialog.pass_band, clean=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._last_smoothed = smoothed

        # Remove any earlier preview before making a new one.
        self._teardown_preview(restore_original=False)

        # Build the smoothed actor.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(smoothed)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Apply the shared transform if one exists (reader-path meshes have it;
        # WRL-imported ones may not — harmless either way).
        if hasattr(self.main, "transform") and self.main.transform is not None:
            actor.SetUserTransform(self.main.transform)
        # Use a neutral light grey so the smoothed mesh never inherits an
        # odd/blue colour from the source actor.
        actor.GetProperty().SetColor(0.85, 0.85, 0.85)
        self._preview_actor = actor
        self.main.ren.AddActor(actor)

        # Show smoothed, hide original. ResetCamera so the result is visible
        # regardless of how the source was positioned (matters for WRL, whose
        # imported actor coordinates may differ from the converted mesh).
        orig = getattr(self.main, "current_actor", None)
        if orig is not None:
            orig.SetVisibility(0)
        self._showing_smoothed = True
        self._toggle_action.setEnabled(True)
        self.main.ren.ResetCamera()
        self.main.vtkWidget.GetRenderWindow().Render()

        n_src = src.GetNumberOfPoints()
        n_out = smoothed.GetNumberOfPoints()
        QMessageBox.information(
            self.main, "Smoothing Done",
            f"Smoothed with {dialog.iterations} iterations, pass-band {dialog.pass_band:g}.\n\n"
            f"Vertices: {n_src} -> {n_out}\n\n"
            "Use Analysis -> 'Show Original / Smoothed' to toggle between them."
        )

    # -- toggle / revert ----------------------------------------------------

    def _toggle(self):
        """Flip between the smoothed preview and the original mesh."""
        orig = getattr(self.main, "current_actor", None)
        if self._preview_actor is None or orig is None:
            return

        if self._showing_smoothed:
            # Switch to original.
            self._preview_actor.SetVisibility(0)
            orig.SetVisibility(1)
            self._showing_smoothed = False
        else:
            # Switch to smoothed.
            self._preview_actor.SetVisibility(1)
            orig.SetVisibility(0)
            self._showing_smoothed = True

        self.main.vtkWidget.GetRenderWindow().Render()

    def _teardown_preview(self, restore_original=True):
        if self._preview_actor is not None:
            self.main.ren.RemoveActor(self._preview_actor)
            self._preview_actor = None
        if restore_original:
            orig = getattr(self.main, "current_actor", None)
            if orig is not None:
                orig.SetVisibility(1)

    # -- optional save ------------------------------------------------------

    def save_smoothed(self):
        """Write the most recent smoothed mesh to a file the user picks."""
        if self._last_smoothed is None:
            QMessageBox.information(self.main, "Nothing to Save",
                                    "Run Smooth Mesh first.")
            return

        default_name = "smoothed.ply"
        try:
            base = os.path.splitext(os.path.basename(self.main.current_file))[0]
            default_name = base + "_smoothed.ply"
        except Exception:
            pass

        fileName, _ = QFileDialog.getSaveFileName(
            self.main, "Save Smoothed Mesh", default_name,
            "PLY Files (*.ply);;STL Files (*.stl);;VTK PolyData (*.vtp)"
        )
        if not fileName:
            return

        ext = os.path.splitext(fileName)[1].lower()
        try:
            if ext == ".stl":
                writer = vtk.vtkSTLWriter()
            elif ext == ".vtp":
                writer = vtk.vtkXMLPolyDataWriter()
            else:
                writer = vtk.vtkPLYWriter()
            writer.SetFileName(fileName)
            writer.SetInputData(self._last_smoothed)
            writer.Write()
        except Exception as e:
            QMessageBox.critical(self.main, "Save Failed", f"Could not save the mesh:\n{e}")
            return

        QMessageBox.information(self.main, "Saved", f"Smoothed mesh written to:\n{fileName}")