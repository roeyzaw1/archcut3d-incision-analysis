"""
ArchCut 3-D

Software developed to precisely evaluate the 3-D micromorphological
characteristics of archaeological surface incisions. Analysis is based on a
continuous 3-D slice sequence from a chosen range within the incision path.
Measurements and shapes of individual slices can be extracted. ArchCut 3-D's
interface is designed to provide a computational and mathematical analysis of
engravings to a wide audience.

Built with PyQt5 and VTK.

This is a from-scratch rebuild of the GUI shell, built in small, individually
testable steps. Current step: open a file, display it, and reposition it via
the rotate buttons -- each click applies a real rotation matrix to the
vertex array (the model itself changes, matching MATLAB's approach exactly),
never a camera/view-only transform. Free mouse-drag rotation of the camera is
also allowed (for casually looking the model over), with a "Main View" button
to snap back to the canonical top-down reference view at any time. Crest
detection and mesh smoothing aren't wired in yet.
"""

import json
import os
import subprocess
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon

from .resources import resource_path
from .wrl_io import read_wrl
from .positioning import compute_initial_orientation, compute_normal_alignment_axis_angle, rodrigues_matrix
from .grid_2d import rasterize_to_grid
from .label_tool import LabelToolDialog
from .crest_lines import detect_crest_lines, crest_lines_as_polylines

APP_NAME = "ArchCut 3-D"


def _rotation_matrix(axis, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 'y':
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # 'z'


class _CameraInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Plain left-drag still orbits the camera. Shift+left-drag slides
    (pans) the view instead -- an easy way to pan on a trackpad, which has
    no physical middle mouse button for VTK's default pan binding. When
    point-select mode is on, a plain left-click picks a point on the model
    instead of starting a rotate.

    Uses AddObserver rather than overriding OnLeftButtonDown/Up: confirmed
    (the hard way) that overriding those virtual methods directly is never
    actually invoked by VTK's real event dispatch (InvokeEvent) in this
    binding -- only a direct Python call to the method "worked", which is
    exactly why the point-picking feature silently did nothing. AddObserver
    is the pattern VTK's own event dispatch actually calls."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.AddObserver('LeftButtonPressEvent', self._on_left_button_press)
        self.AddObserver('LeftButtonReleaseEvent', self._on_left_button_release)
        self.AddObserver('RightButtonPressEvent', self._on_right_button_press)

    def _on_left_button_press(self, obj, event):
        if getattr(self.window, 'trace_mode', False):
            self.window.add_trace_point()
            return
        if getattr(self.window, 'edit_trace_mode', False):
            if self.GetInteractor().GetShiftKey():
                self.window.extend_selected_crest_line()   # Shift = grow the line
            else:
                self.window.select_trace_at_current_position()
            return
        if self.GetInteractor().GetShiftKey():
            self.StartPan()
        else:
            self.OnLeftButtonDown()

    def _on_right_button_press(self, obj, event):
        if getattr(self.window, 'trace_mode', False):
            self.window.finish_current_trace_3d()
            return
        self.OnRightButtonDown()

    def _on_left_button_release(self, obj, event):
        if self.GetState() == vtk.VTKIS_PAN:
            self.EndPan()
        else:
            self.OnLeftButtonUp()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))
        self.resize(990, 600)

        self.frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        layout.addWidget(self.vtkWidget)
        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1.0, 1.0, 1.0)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        # Free camera rotation is allowed for casually looking the model over;
        # "Main View" (below) snaps back to the canonical top-down reference
        # view, which is what actually needs to be aligned before slicing.
        # Keep a persistent Python-side reference -- without it, Python's GC
        # can reclaim the wrapper object even though VTK's C++ side still
        # holds a pointer, silently losing the Shift+drag-pan override.
        self.interactor_style = _CameraInteractorStyle(self)
        self.iren.SetInteractorStyle(self.interactor_style)

        self.current_actor = None
        self.base_vertices = None      # vertices right after load, never mutated
        self.current_vertices = None   # base_vertices rotated by cumulative_rotation
        self.current_faces = None
        self.cumulative_rotation = None  # 3x3 matrix: all button clicks composed together
        self.current_model_path = None
        self.grid_bounds = None
        self.last_dir = ''

        self.label_tool_dialog = None   # experimental: 2.5D grid labeling tool

        # experimental: trace incision centerlines directly on the 3D model
        self.trace_mode = False
        self.traces_3d = []             # finished polylines, each a list of (3,) world points
        self.current_trace_3d = []      # in-progress polyline
        self.trace_3d_actors = []       # VTK actors for the trace visualization, rebuilt on change
        # Edit-Traces mode: click a finished curve to select it (and its
        # nearest point) for trimming / vertex delete / extend.
        self.edit_trace_mode = False
        self.selected_trace_index = None
        self.selected_vertex_index = None

        fileMenu = self.menuBar().addMenu('&File')
        openAction = fileMenu.addAction('Open...')
        openAction.triggered.connect(self.open_file)

        mlMenu = self.menuBar().addMenu('ML (experimental)')
        labelAction = mlMenu.addAction('Label Incisions...')
        labelAction.triggered.connect(self.open_label_tool)

        self._build_rotation_toolbar()
        self._build_trace_dock()

        self.statusBar().addWidget(QtWidgets.QLabel(
            "  Reminder: a correctly positioned model lies flat, aligned with the XY plane, as seen in Main View.  "))

        self.iren.Initialize()

    def _build_rotation_toolbar(self):
        """Position Model toolbar: degrees-per-click field + the 6 MATLAB
        rotate buttons (X up/down, Y left/right, Z clockwise/counter-clockwise),
        using the same button images as harutot.m."""
        self.rotation_toolbar = QtWidgets.QToolBar("Position Model")
        self.addToolBar(self.rotation_toolbar)
        self.rotation_toolbar.setEnabled(False)

        self.rotation_toolbar.addWidget(QtWidgets.QLabel("Degrees per click:"))
        self.angle_input = QtWidgets.QLineEdit("5")
        self.angle_input.setMaximumWidth(50)
        self.rotation_toolbar.addWidget(self.angle_input)
        self.rotation_toolbar.addSeparator()

        rotate_buttons = [
            ('rotate_x_up', 'Rotate Up (X axis)', 'x', -1.0),
            ('rotate_x_down', 'Rotate Down (X axis)', 'x', 1.0),
            ('rotate_y_left', 'Rotate Left (Y axis)', 'y', -1.0),
            ('rotate_y_right', 'Rotate Right (Y axis)', 'y', 1.0),
            ('rotate_z_clockwise', 'Rotate Clockwise (Z axis)', 'z', -1.0),
            ('rotate_z_counterclockwise', 'Rotate Counter-clockwise (Z axis)', 'z', 1.0),
        ]
        for icon_name, tooltip, axis, sign in rotate_buttons:
            button = QtWidgets.QToolButton()
            button.setIcon(QIcon(resource_path(f'assets/buttons/{icon_name}.bmp')))
            button.setIconSize(QSize(40, 40))
            button.setToolTip(tooltip)
            button.setAutoRepeat(True)
            button.clicked.connect(lambda checked=False, a=axis, s=sign: self.rotate_model(a, s))
            self.rotation_toolbar.addWidget(button)

        self.rotation_toolbar.addSeparator()
        alignNormalButton = QtWidgets.QPushButton("Align by Normal")
        alignNormalButton.setToolTip("Automatically rotate the model so its dominant surface faces +Z")
        alignNormalButton.clicked.connect(self.align_by_normal)
        self.rotation_toolbar.addWidget(alignNormalButton)

        mainViewButton = QtWidgets.QPushButton("Main View")
        mainViewButton.setToolTip("Snap the camera back to the top-down reference view (looking straight onto the XY plane)")
        mainViewButton.clicked.connect(self.set_top_view)
        self.rotation_toolbar.addWidget(mainViewButton)

        self.rotation_toolbar.addSeparator()
        savePositionButton = QtWidgets.QPushButton("Save Position")
        savePositionButton.setToolTip("Save the current rotation to a .position.json file next to the model")
        savePositionButton.clicked.connect(self.save_position)
        self.rotation_toolbar.addWidget(savePositionButton)

        loadPositionButton = QtWidgets.QPushButton("Load Position")
        loadPositionButton.setToolTip("Load a previously saved .position.json file and apply its rotation")
        loadPositionButton.clicked.connect(self.load_position)
        self.rotation_toolbar.addWidget(loadPositionButton)

    def _build_trace_dock(self):
        """Crest Lines: incision centerlines, either drawn by hand on the 3D
        model or seeded from Yoshizawa crest detection, then pruned/edited.
        Whatever survives here is what gets projected onto the 2.5D grid and
        saved as training labels."""
        dock = QtWidgets.QDockWidget("Crest Lines (experimental)", self)
        self.trace_dock = dock
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.traceModeButton = QtWidgets.QPushButton("Draw Crest Line")
        self.traceModeButton.setCheckable(True)
        self.traceModeButton.setToolTip("While on, left-click the model to add centerline points; right-click finishes the line")
        self.traceModeButton.toggled.connect(self.toggle_trace_mode)
        layout.addWidget(self.traceModeButton)

        row = QtWidgets.QHBoxLayout()
        finishBtn = QtWidgets.QPushButton("Finish Line")
        finishBtn.clicked.connect(self.finish_current_trace_3d)
        undoBtn = QtWidgets.QPushButton("Undo Point")
        undoBtn.clicked.connect(self.undo_trace_point)
        for b in (finishBtn, undoBtn):
            row.addWidget(b)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        delBtn = QtWidgets.QPushButton("Delete Last Line")
        delBtn.clicked.connect(self.delete_last_trace_3d)
        clearBtn = QtWidgets.QPushButton("Clear All")
        clearBtn.clicked.connect(self.clear_traces_3d)
        for b in (delBtn, clearBtn):
            row2.addWidget(b)
        layout.addLayout(row2)

        self.editTraceButton = QtWidgets.QPushButton("Edit Crest Lines")
        self.editTraceButton.setCheckable(True)
        self.editTraceButton.setToolTip(
            "While on: left-click a line to select it + its nearest point; Shift+click the model to extend "
            "the selected line from the near end; then use the buttons below")
        self.editTraceButton.toggled.connect(self.toggle_edit_trace_mode)
        layout.addWidget(self.editTraceButton)

        editRow = QtWidgets.QHBoxLayout()
        self.deletePointBtn = QtWidgets.QPushButton("Delete Point")
        self.deletePointBtn.setToolTip("Remove the highlighted (cyan) point from the selected line")
        self.deletePointBtn.clicked.connect(self.delete_crest_point)
        self.trimStartBtn = QtWidgets.QPushButton("Trim to Start")
        self.trimStartBtn.setToolTip("Drop every point BEFORE the highlighted point")
        self.trimStartBtn.clicked.connect(self.trim_crest_to_start)
        self.trimEndBtn = QtWidgets.QPushButton("Trim to End")
        self.trimEndBtn.setToolTip("Drop every point AFTER the highlighted point")
        self.trimEndBtn.clicked.connect(self.trim_crest_to_end)
        for b in (self.deletePointBtn, self.trimStartBtn, self.trimEndBtn):
            editRow.addWidget(b)
        layout.addLayout(editRow)

        self.deleteLineBtn = QtWidgets.QPushButton("Delete Selected Line")
        self.deleteLineBtn.setToolTip("Remove the whole selected line -- use this to prune non-incisions")
        self.deleteLineBtn.clicked.connect(self.delete_selected_crest_line)
        layout.addWidget(self.deleteLineBtn)

        crestRow = QtWidgets.QHBoxLayout()
        self.crestWhichCombo = QtWidgets.QComboBox()
        self.crestWhichCombo.addItems(['Both', 'Ridges', 'Ravines'])
        crestRow.addWidget(self.crestWhichCombo)
        crestRow.addWidget(QtWidgets.QLabel("min edges:"))
        self.crestMinEdgesSpin = QtWidgets.QSpinBox()
        self.crestMinEdgesSpin.setRange(2, 500)
        self.crestMinEdgesSpin.setValue(30)
        crestRow.addWidget(self.crestMinEdgesSpin)
        layout.addLayout(crestRow)

        crestBtn = QtWidgets.QPushButton("Detect Crest Lines")
        crestBtn.setToolTip("Run Yoshizawa crest detection and add the crest lines as editable lines to prune/extend")
        crestBtn.clicked.connect(self.detect_crests_as_traces)
        layout.addWidget(crestBtn)

        self.traceStatusLabel = QtWidgets.QLabel()
        layout.addWidget(self.traceStatusLabel)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setEnabled(False)
        self._update_edit_buttons_enabled()
        self._update_trace_status()

    def detect_crests_as_traces(self):
        """Run Yoshizawa crest detection on the current (displayed) mesh and
        add the crest lines as traces_3d for the expert to prune/extend --
        a labeling accelerator. Crest lines come out in the current display
        frame (detection is on current_vertices), matching how the trace
        edit/projection code expects traces_3d."""
        if self.current_vertices is None:
            return
        which = {'Both': 'both', 'Ridges': 'ridges', 'Ravines': 'ravines'}[self.crestWhichCombo.currentText()]
        min_edges = self.crestMinEdgesSpin.value()

        self.statusBar().showMessage("Detecting crest lines (setCurvature)...")
        QtWidgets.QApplication.processEvents()
        try:
            result = detect_crest_lines(self.current_vertices, self.current_faces)
        except FileNotFoundError:
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.warning(
                self, 'setCurvature Not Found',
                "The setCurvature binary is missing from bin/. Can't detect crest lines.")
            return
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.warning(self, 'Crest Detection Failed', str(e))
            return
        self.statusBar().clearMessage()

        polylines = crest_lines_as_polylines(result, min_edges=min_edges, which=which)
        for pl in polylines:
            self.traces_3d.append([np.asarray(p, dtype=np.float64) for p in pl])
        self._redraw_traces_3d()
        self._update_trace_status()
        self.statusBar().showMessage(
            f"Added {len(polylines)} crest line(s) as traces -- prune the ones that aren't incisions.", 5000)

    def toggle_trace_mode(self, checked):
        self.trace_mode = checked
        self.traceModeButton.setText("Draw Crest Line: ON (click the model)" if checked else "Draw Crest Line")
        if checked:
            if self.editTraceButton.isChecked():
                self.editTraceButton.setChecked(False)  # mutually exclusive click modes
            self.statusBar().showMessage("Draw mode ON -- left-click to add points, right-click to finish the line.", 4000)

    def toggle_edit_trace_mode(self, checked):
        self.edit_trace_mode = checked
        self.editTraceButton.setText("Edit Crest Lines: ON (click a line)" if checked else "Edit Crest Lines")
        if checked:
            if self.traceModeButton.isChecked():
                self.traceModeButton.setChecked(False)  # mutually exclusive click modes
            self.statusBar().showMessage(
                "Edit mode ON -- left-click a line to select it; Shift+click the model to extend it.", 5000)
        else:
            self._clear_trace_selection()
            self._redraw_traces_3d()
        self._update_trace_status()

    def _clear_trace_selection(self):
        self.selected_trace_index = None
        self.selected_vertex_index = None

    def _selected_trace(self):
        i = self.selected_trace_index
        if i is None or not (0 <= i < len(self.traces_3d)):
            return None
        return self.traces_3d[i]

    def select_trace_at_current_position(self):
        """Edit mode: pick the model point under the cursor, then hand it to
        _select_trace_near_point (which does the nearest-curve math)."""
        if self.current_vertices is None or not self.traces_3d:
            return
        click_pos = self.iren.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        # In Edit mode you're aiming AT the drawn curves, so the cell picker
        # usually returns a trace line/marker actor (they sit on the surface)
        # rather than the mesh. Accept a hit on the mesh or on any of our own
        # trace actors -- the pick position is on the surface either way.
        picker.InitializePickList()
        picker.AddPickList(self.current_actor)
        for a in self.trace_3d_actors:
            picker.AddPickList(a)
        picker.SetPickFromList(True)
        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
        if picker.GetActor() is None:
            self.statusBar().showMessage("Click didn't land on the model or a curve.", 3000)
            return
        self._select_trace_near_point(np.array(picker.GetPickPosition(), dtype=np.float64))

    def _select_trace_near_point(self, target):
        """Select the finished trace whose nearest vertex is closest to
        `target` (world coords), and remember that vertex as the one trims /
        deletes act on. A target far from every trace clears the selection."""
        if not self.traces_3d:
            return
        best_ti, best_vi, best_dist = None, None, None
        for ti, trace in enumerate(self.traces_3d):
            arr = np.asarray(trace, dtype=np.float64)
            d = np.linalg.norm(arr - target, axis=1)
            vi = int(np.argmin(d))
            if best_dist is None or d[vi] < best_dist:
                best_ti, best_vi, best_dist = ti, vi, float(d[vi])

        max_pick = None
        if self.grid_bounds is not None:
            max_pick = (self.grid_bounds[1] - self.grid_bounds[0]) * 0.03
        if max_pick is not None and best_dist > max_pick:
            self._clear_trace_selection()
            self.statusBar().showMessage("No curve near that click.", 3000)
        else:
            self.selected_trace_index = best_ti
            self.selected_vertex_index = best_vi
            self.statusBar().showMessage(
                f"Selected curve {best_ti + 1}/{len(self.traces_3d)} "
                f"({len(self.traces_3d[best_ti])} pts) -- point {best_vi + 1}.", 4000)
        self._redraw_traces_3d()
        self._update_trace_status()

    def _update_edit_buttons_enabled(self):
        """The per-line edit buttons only do anything with a line selected."""
        has_sel = self._selected_trace() is not None
        for name in ('deletePointBtn', 'trimStartBtn', 'trimEndBtn', 'deleteLineBtn'):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(has_sel)

    def _after_line_edit(self, trace_gone):
        """Shared cleanup after an edit op: fix up the active-point index (or
        drop the selection if the line is gone), then redraw + refresh UI."""
        if trace_gone:
            self._clear_trace_selection()
        else:
            sel = self._selected_trace()
            n = len(sel)
            if self.selected_vertex_index is None or self.selected_vertex_index >= n:
                self.selected_vertex_index = max(0, n - 1)
        self._redraw_traces_3d()
        self._update_trace_status()

    def delete_crest_point(self):
        sel = self._selected_trace()
        if sel is None or self.selected_vertex_index is None:
            return
        del sel[self.selected_vertex_index]
        if len(sel) < 2:
            # a line needs >= 2 points; if this emptied it, drop the whole line
            del self.traces_3d[self.selected_trace_index]
            self.statusBar().showMessage("Point removed -- line had < 2 points left, so it was deleted.", 4000)
            self._after_line_edit(trace_gone=True)
        else:
            self.statusBar().showMessage(f"Point removed -- {len(sel)} left.", 3000)
            self._after_line_edit(trace_gone=False)

    def trim_crest_to_start(self):
        """Keep the highlighted point and everything after it."""
        sel = self._selected_trace()
        if sel is None or self.selected_vertex_index is None:
            return
        cut = self.selected_vertex_index
        if cut == 0:
            return
        if len(sel) - cut < 2:
            self.statusBar().showMessage("Can't trim: fewer than 2 points would remain.", 3000)
            return
        del sel[:cut]
        self.selected_vertex_index = 0
        self.statusBar().showMessage(f"Trimmed {cut} point(s) off the start -- {len(sel)} left.", 3000)
        self._after_line_edit(trace_gone=False)

    def trim_crest_to_end(self):
        """Keep the highlighted point and everything before it."""
        sel = self._selected_trace()
        if sel is None or self.selected_vertex_index is None:
            return
        keep = self.selected_vertex_index + 1
        dropped = len(sel) - keep
        if dropped == 0:
            return
        if keep < 2:
            self.statusBar().showMessage("Can't trim: fewer than 2 points would remain.", 3000)
            return
        del sel[keep:]
        self.selected_vertex_index = len(sel) - 1
        self.statusBar().showMessage(f"Trimmed {dropped} point(s) off the end -- {len(sel)} left.", 3000)
        self._after_line_edit(trace_gone=False)

    def delete_selected_crest_line(self):
        if self._selected_trace() is None:
            return
        del self.traces_3d[self.selected_trace_index]
        self.statusBar().showMessage(f"Line deleted -- {len(self.traces_3d)} left.", 3000)
        self._after_line_edit(trace_gone=True)

    def extend_selected_crest_line(self):
        """Shift+left-click in Edit mode: add the picked model point to the
        end of the selected line nearer the highlighted point, then move the
        highlight onto the new point so repeated Shift+clicks keep growing
        that same end."""
        if self._selected_trace() is None or self.selected_vertex_index is None:
            self.statusBar().showMessage("Select a line first, then Shift+click to extend it.", 3000)
            return
        click_pos = self.iren.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
        if picker.GetActor() is not self.current_actor:
            self.statusBar().showMessage("Nothing added: Shift+click didn't land on the model.", 3000)
            return
        self._extend_selected_crest_line_with_point(
            np.array(picker.GetPickPosition(), dtype=np.float64))

    def _extend_selected_crest_line_with_point(self, new_pt):
        sel = self._selected_trace()
        if sel is None or self.selected_vertex_index is None:
            return
        dist_to_start = self.selected_vertex_index
        dist_to_end = len(sel) - 1 - self.selected_vertex_index
        if dist_to_start <= dist_to_end:
            sel.insert(0, new_pt)
            self.selected_vertex_index = 0
            end = "start"
        else:
            sel.append(new_pt)
            self.selected_vertex_index = len(sel) - 1
            end = "end"
        self.statusBar().showMessage(f"Point added at the {end} -- {len(sel)} points.", 3000)
        self._after_line_edit(trace_gone=False)

    def add_trace_point(self):
        if self.current_vertices is None:
            return
        click_pos = self.iren.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
        if picker.GetActor() is not self.current_actor:
            self.statusBar().showMessage("No point added: click didn't land on the model.", 3000)
            return
        self.current_trace_3d.append(np.array(picker.GetPickPosition(), dtype=np.float64))
        self._redraw_traces_3d()
        self._update_trace_status()

    def finish_current_trace_3d(self):
        if len(self.current_trace_3d) >= 2:
            self.traces_3d.append(self.current_trace_3d)
            self.current_trace_3d = []
            self._redraw_traces_3d()
            self._update_trace_status()

    def undo_trace_point(self):
        if self.current_trace_3d:
            self.current_trace_3d.pop()
        elif self.traces_3d:
            self.current_trace_3d = self.traces_3d.pop()
            self.current_trace_3d.pop()
        self._clear_trace_selection()
        self._redraw_traces_3d()
        self._update_trace_status()

    def delete_last_trace_3d(self):
        if self.current_trace_3d:
            self.current_trace_3d = []
        elif self.traces_3d:
            self.traces_3d.pop()
        self._clear_trace_selection()
        self._redraw_traces_3d()
        self._update_trace_status()

    def clear_traces_3d(self):
        self.traces_3d = []
        self.current_trace_3d = []
        self._clear_trace_selection()
        self._redraw_traces_3d()
        self._update_trace_status()

    def _trace_marker_radius(self):
        if self.grid_bounds is not None:
            return max((self.grid_bounds[1] - self.grid_bounds[0]) * 0.002, 1e-3)
        return 0.05

    def _multiline_actor(self, polylines, color, width):
        """One actor for many polylines: all their points in one vtkPoints,
        one vtkPolyLine cell each. Keeps the actor/draw-call count constant
        no matter how many (or how long) the traces are -- crest detection
        can add dozens of 100+ point lines."""
        vpoints = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        for pl in polylines:
            n = len(pl)
            if n < 2:
                continue
            start = vpoints.GetNumberOfPoints()
            for p in pl:
                vpoints.InsertNextPoint(p)
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                line.GetPointIds().SetId(i, start + i)
            cells.InsertNextCell(line)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vpoints)
        pd.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        return actor

    def _glyph_markers_actor(self, points, color, radius):
        """One actor for all the point markers of a trace, via vtkGlyph3D --
        again constant actor count regardless of point count."""
        vpoints = vtk.vtkPoints()
        for p in points:
            vpoints.InsertNextPoint(p)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vpoints)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(8)
        sphere.SetPhiResolution(8)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(pd)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        return actor

    def _redraw_traces_3d(self):
        for actor in self.trace_3d_actors:
            self.ren.RemoveActor(actor)
        self.trace_3d_actors = []

        # Finished traces: just the lines, one actor for all of them, no
        # per-point markers (there's no interaction that needs them).
        if self.traces_3d:
            self.trace_3d_actors.append(self._multiline_actor(self.traces_3d, (1.0, 0.0, 0.0), 3))

        # In-progress trace: line + markers (you're actively editing it, and
        # it's short), each still just one actor.
        if self.current_trace_3d:
            if len(self.current_trace_3d) >= 2:
                self.trace_3d_actors.append(
                    self._multiline_actor([self.current_trace_3d], (1.0, 0.85, 0.0), 3))
            self.trace_3d_actors.append(
                self._glyph_markers_actor(self.current_trace_3d, (1.0, 0.85, 0.0), self._trace_marker_radius()))

        # Selected trace (Edit mode): draw it highlighted, with a marker on
        # every vertex and a bigger one on the active point, so it's clear
        # what a trim / vertex-delete will act on.
        sel = self._selected_trace()
        if sel is not None:
            self.trace_3d_actors.append(self._multiline_actor([sel], (0.1, 1.0, 0.3), 5))
            self.trace_3d_actors.append(
                self._glyph_markers_actor(sel, (0.1, 1.0, 0.3), self._trace_marker_radius()))
            vi = self.selected_vertex_index
            if vi is not None and 0 <= vi < len(sel):
                self.trace_3d_actors.append(
                    self._glyph_markers_actor([sel[vi]], (0.0, 0.8, 1.0), self._trace_marker_radius() * 2.2))

        for actor in self.trace_3d_actors:
            self.ren.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _update_trace_status(self):
        txt = f"Lines: {len(self.traces_3d)}   |   drawing: {len(self.current_trace_3d)} point(s)"
        sel = self._selected_trace()
        if sel is not None:
            txt += (f"   |   selected line {self.selected_trace_index + 1} "
                    f"({len(sel)} pts), point {(self.selected_vertex_index or 0) + 1}")
        self.traceStatusLabel.setText(txt)
        self._update_edit_buttons_enabled()

    def open_label_tool(self):
        """Experimental: compute the 2.5D grid for the loaded model (from the
        auto-positioned base_vertices, so it's deterministic per file) and
        open the labeling dialog, seeded with any incisions traced on the 3D
        model (projected onto the grid)."""
        if self.base_vertices is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model first.')
            return
        self.statusBar().showMessage("Computing 2.5D grid (curvature can take a moment)...")
        QtWidgets.QApplication.processEvents()
        try:
            grid = rasterize_to_grid(self.base_vertices, self.current_faces)
        except Exception as e:
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.warning(self, 'Grid Failed', f'Could not build the 2.5D grid:\n{e}')
            return
        self.statusBar().clearMessage()

        initial_traces = self._project_3d_traces_to_grid(grid)
        self.label_tool_dialog = LabelToolDialog(
            grid, self.current_model_path, self, initial_traces=initial_traces)
        self.label_tool_dialog.show()
        self.label_tool_dialog.raise_()
        self.label_tool_dialog.activateWindow()

    def _project_3d_traces_to_grid(self, grid):
        """Convert the 3D incision traces (world coords, current display
        frame) into [col, row] polylines on `grid`. The grid is built from
        base_vertices, so undo any manual rotation first: current_vertices =
        base_vertices @ R.T, hence base = current @ R (R = cumulative_rotation,
        orthogonal)."""
        traces = list(self.traces_3d)
        if len(self.current_trace_3d) >= 2:
            traces.append(self.current_trace_3d)
        if not traces:
            return []

        R = self.cumulative_rotation if self.cumulative_rotation is not None else np.eye(3)
        grid_traces = []
        for trace in traces:
            arr = np.asarray(trace, dtype=np.float64)          # (n, 3), current frame
            base = arr @ R                                     # -> base_vertices frame
            colrow = grid.world_xy_to_grid(base[:, 0], base[:, 1])
            grid_traces.append(colrow.tolist())
        return grid_traces

    def open_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select your file', self.last_dir,
            '3D Files (*.wrl *.stl *.ply *.obj)')
        if not fileName:
            return
        self.last_dir = fileName.rsplit('/', 1)[0]

        self.ren.RemoveAllViewProps()
        # RemoveAllViewProps() already dropped the old trace actors from the
        # scene; also clear our own tracking so a new model starts fresh
        # (traces are tied to a specific mesh).
        self.traces_3d = []
        self.current_trace_3d = []
        self.trace_3d_actors = []
        self._clear_trace_selection()
        if getattr(self, 'editTraceButton', None) is not None:
            self.editTraceButton.setChecked(False)
        self._update_trace_status()

        if fileName.lower().endswith('.wrl'):
            # Cut3-D's own load pipeline (harutot.m pushbutton_loadfile_Callback):
            # a dedicated parser (not VTK's generic VRML importer, so vertex/face
            # order matches the MATLAB source exactly) + center/orient/flip/de-tilt.
            vertices, faces = read_wrl(fileName)
            vertices = compute_initial_orientation(vertices, faces)
        else:
            if fileName.lower().endswith('.stl'):
                reader = vtk.vtkSTLReader()
            elif fileName.lower().endswith('.ply'):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            polydata = reader.GetOutput()
            vertices = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
            faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

        # Own the vertex array directly (not a VTK reader/importer pipeline)
        # so the rotate buttons below can mutate it and push the change back
        # into the polydata -- the model itself changes, not a view/camera.
        # base_vertices is the fixed reference (right after auto-positioning);
        # every button click composes into cumulative_rotation and current_vertices
        # is recomputed from base_vertices each time, so floating-point error
        # never accumulates across many clicks.
        self.base_vertices = vertices
        self.cumulative_rotation = np.eye(3)
        self.current_vertices = vertices
        self.current_faces = faces
        self.current_model_path = fileName
        actor = self._make_actor_from_mesh(vertices, faces)
        self.ren.AddActor(actor)
        self.current_actor = actor

        self.set_top_view()
        self.grid_bounds = self._compute_fixed_grid_bounds(vertices)
        self.add_grid_axes(self.grid_bounds)
        self.rotation_toolbar.setEnabled(True)
        self.trace_dock.setEnabled(True)
        self.vtkWidget.GetRenderWindow().Render()

    def rotate_model(self, axis, sign):
        """Apply a real rotation to the vertex array itself (matching
        MATLAB's pushbutton20/21/22/23/27/28_Callback). Composes into
        cumulative_rotation and recomputes current_vertices from the fixed
        base_vertices (rather than repeatedly rotating an already-rotated
        array), so floating-point error never accumulates across clicks.
        The reference grid is NOT recomputed here -- it stays fixed so the
        rotation is visible against a stable frame; only the model moves."""
        if self.current_vertices is None:
            return
        try:
            degrees = float(self.angle_input.text())
        except ValueError:
            degrees = 5.0

        angle_rad = sign * np.radians(degrees)
        rotation = _rotation_matrix(axis, angle_rad)
        self.cumulative_rotation = rotation @ self.cumulative_rotation
        self._apply_cumulative_rotation()

    def align_by_normal(self):
        """Auto-position: rotate the model so its dominant area-weighted
        surface normal points along +Z (normal_positioning.m). Computed
        against the CURRENTLY displayed vertices (whatever rotations have
        already been applied), same as MATLAB's normal_pos_Callback, and
        composed on top of the existing cumulative_rotation."""
        if self.current_vertices is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before aligning it.')
            return

        axis, angle_degrees = compute_normal_alignment_axis_angle(self.current_vertices, self.current_faces)
        if abs(angle_degrees) < 1e-6:
            return  # Already aligned.

        rotation = rodrigues_matrix(axis, np.radians(angle_degrees))
        self.cumulative_rotation = rotation @ self.cumulative_rotation
        self._apply_cumulative_rotation()

    def _apply_cumulative_rotation(self):
        """Recompute current_vertices = base_vertices rotated by
        cumulative_rotation, push into the displayed polydata, and redraw."""
        self.current_vertices = self.base_vertices @ self.cumulative_rotation.T

        polydata = self.current_actor.GetMapper().GetInput()
        polydata.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(self.current_vertices), deep=True))
        polydata.GetPoints().Modified()

        self.vtkWidget.GetRenderWindow().Render()

    def _default_position_file_path(self):
        """Adjacent-file convention: model.wrl -> model.position.json."""
        base, _ext = os.path.splitext(self.current_model_path)
        return base + '.position.json'

    def save_position(self):
        if self.cumulative_rotation is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before saving its position.')
            return
        default_path = self._default_position_file_path()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Position', default_path, 'Position Files (*.position.json)')
        if not fileName:
            return
        if not fileName.lower().endswith('.json'):
            fileName += '.position.json'

        with open(fileName, 'w') as f:
            json.dump({
                'source_file': os.path.basename(self.current_model_path),
                'rotation_matrix': self.cumulative_rotation.tolist(),
            }, f, indent=2)

    def load_position(self):
        if self.base_vertices is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before loading a position.')
            return
        default_path = self._default_position_file_path()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Position', default_path, 'Position Files (*.position.json);;All JSON (*.json)')
        if not fileName:
            return

        try:
            with open(fileName, 'r') as f:
                data = json.load(f)
            rotation = np.array(data['rotation_matrix'], dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError(f'Expected a 3x3 rotation matrix, got shape {rotation.shape}')
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            QtWidgets.QMessageBox.warning(self, 'Load Position Failed', f'Could not read position file:\n{e}')
            return

        self.cumulative_rotation = rotation
        self._apply_cumulative_rotation()

    def _compute_fixed_grid_bounds(self, vertices):
        """A cube centered on the model's centroid, sized to the bounding
        sphere radius so it contains the model at any orientation. Computed
        once at load time and reused for every rotation, so the reference
        grid never resizes/jumps -- only the model visibly moves within it."""
        center = vertices.mean(axis=0)
        radius = float(np.linalg.norm(vertices - center, axis=1).max()) * 1.05
        return (
            center[0] - radius, center[0] + radius,
            center[1] - radius, center[1] + radius,
            center[2] - radius, center[2] + radius,
        )

    def _make_actor_from_mesh(self, vertices, faces):
        """Build a vtkActor directly from numpy vertex/face arrays (as
        opposed to a VTK reader/importer) -- used for .wrl, whose vertices
        get repositioned by compute_initial_orientation() before display."""
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))

        n_faces = faces.shape[0]
        cell_data = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces]).ravel()
        cells = vtk.vtkCellArray()
        cells.SetCells(n_faces, numpy_to_vtkIdTypeArray(cell_data, deep=True))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        return actor

    def set_top_view(self):
        """Camera looking straight down onto the XY plane, Y up on screen --
        the "well-positioned = seen from above" view, fitted to whatever is
        currently loaded."""
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        self.ren.ResetCamera()
        # ResetCamera() only updates VTK's internal camera state; without an
        # explicit Render() the widget doesn't repaint until some other event
        # (e.g. a click into the viewport) happens to trigger one -- which is
        # exactly the "only updates after I click the model" symptom.
        self.vtkWidget.GetRenderWindow().Render()

    def add_grid_axes(self, bounds):
        """Wrap the model in a labeled 3-D grid box (bounds, ticks, gridlines
        on the far walls), like MATLAB's default axes -- so the model sits in
        a fixed spatial reference frame instead of floating in empty space."""
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetBounds(bounds)
        cube_axes.SetCamera(self.ren.GetActiveCamera())
        cube_axes.SetXTitle('X')
        cube_axes.SetYTitle('Y')
        cube_axes.SetZTitle('Z')

        for axis in range(3):
            cube_axes.GetTitleTextProperty(axis).SetColor(0.2, 0.2, 0.2)
            cube_axes.GetLabelTextProperty(axis).SetColor(0.2, 0.2, 0.2)

        cube_axes.GetXAxesLinesProperty().SetColor(0.2, 0.2, 0.2)
        cube_axes.GetYAxesLinesProperty().SetColor(0.2, 0.2, 0.2)
        cube_axes.GetZAxesLinesProperty().SetColor(0.2, 0.2, 0.2)

        cube_axes.DrawXGridlinesOn()
        cube_axes.DrawYGridlinesOn()
        cube_axes.DrawZGridlinesOn()
        cube_axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_FURTHEST)

        self.ren.AddActor(cube_axes)
        self.cube_axes_actor = cube_axes


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
