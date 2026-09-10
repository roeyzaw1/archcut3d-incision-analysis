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
from .slicing import compute_slice_path, compute_slices
from .slice_viewer import SliceViewerDialog
from .grid_2d import rasterize_to_grid
from .label_tool import LabelToolDialog

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

    def _on_left_button_press(self, obj, event):
        if self.window.try_select_marker_at_current_position():
            return  # clicked an existing point marker -> just highlight it in the table
        if getattr(self.window, 'point_select_mode', False):
            self.window.pick_point_at_current_position()
            return
        if self.GetInteractor().GetShiftKey():
            self.StartPan()
        else:
            self.OnLeftButtonDown()

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

        self.point_select_mode = False
        self.selected_points = []       # list of (x, y, z) numpy arrays, in click order
        self.point_marker_actors = []   # matching vtkActor per selected point

        self.point1_row = None          # row index into selected_points, slicing endpoint 1
        self.point2_row = None          # row index into selected_points, slicing endpoint 2
        self.slice_path = None          # (numSlices, 3) path from compute_slice_path
        self.slices = None              # list of (K, 2) arrays from compute_slices
        self.slice_visualization_actors = []
        self.slice_viewer_dialog = None
        self.label_tool_dialog = None   # experimental: 2.5D grid labeling tool

        fileMenu = self.menuBar().addMenu('&File')
        openAction = fileMenu.addAction('Open...')
        openAction.triggered.connect(self.open_file)

        mlMenu = self.menuBar().addMenu('ML (experimental)')
        labelAction = mlMenu.addAction('Label Incisions...')
        labelAction.triggered.connect(self.open_label_tool)

        self._build_rotation_toolbar()
        self._build_points_dock()

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

    def _build_points_dock(self):
        """Select Points panel: click points on the model, numbered in click
        order (harutot.m's selectpoints_button_Callback / uitable2). Slicing
        between two chosen point numbers is a later step -- this is just
        picking and keeping track of points for now."""
        dock = QtWidgets.QDockWidget("Select Points", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.points_dock = dock

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.selectPointsButton = QtWidgets.QPushButton("Select Points")
        self.selectPointsButton.setCheckable(True)
        self.selectPointsButton.setToolTip("While enabled, click on the model to add a numbered point")
        self.selectPointsButton.toggled.connect(self.toggle_point_select_mode)
        layout.addWidget(self.selectPointsButton)

        self.points_table = QtWidgets.QTableWidget(0, 4)
        self.points_table.setHorizontalHeaderLabels(["#", "X", "Y", "Z"])
        self.points_table.horizontalHeader().setStretchLastSection(True)
        self.points_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.points_table.itemSelectionChanged.connect(self.on_points_table_selection_changed)
        layout.addWidget(self.points_table)

        deleteButton = QtWidgets.QPushButton("Delete Selected")
        deleteButton.clicked.connect(self.delete_selected_point)
        layout.addWidget(deleteButton)

        clearButton = QtWidgets.QPushButton("Clear All")
        clearButton.clicked.connect(self.clear_selected_points)
        layout.addWidget(clearButton)

        saveButton = QtWidgets.QPushButton("Save Points")
        saveButton.clicked.connect(self.save_points)
        layout.addWidget(saveButton)

        loadButton = QtWidgets.QPushButton("Load Points")
        loadButton.clicked.connect(self.load_points)
        layout.addWidget(loadButton)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(separator)

        endpointsRow = QtWidgets.QHBoxLayout()
        self.point1Label = QtWidgets.QLabel("Point 1: —")
        self.point2Label = QtWidgets.QLabel("Point 2: —")
        endpointsRow.addWidget(self.point1Label)
        endpointsRow.addWidget(self.point2Label)
        layout.addLayout(endpointsRow)

        setButtonsRow = QtWidgets.QHBoxLayout()
        setPoint1Button = QtWidgets.QPushButton("Set as Point 1")
        setPoint1Button.setToolTip("Use the selected table row as the first slicing endpoint")
        setPoint1Button.clicked.connect(self.set_as_point1)
        setPoint2Button = QtWidgets.QPushButton("Set as Point 2")
        setPoint2Button.setToolTip("Use the selected table row as the second slicing endpoint")
        setPoint2Button.clicked.connect(self.set_as_point2)
        setButtonsRow.addWidget(setPoint1Button)
        setButtonsRow.addWidget(setPoint2Button)
        layout.addLayout(setButtonsRow)

        toleranceRow = QtWidgets.QHBoxLayout()
        toleranceRow.addWidget(QtWidgets.QLabel("Tolerance (mm):"))
        self.toleranceInput = QtWidgets.QLineEdit("1")
        toleranceRow.addWidget(self.toleranceInput)
        toleranceRow.addWidget(QtWidgets.QLabel("Width (mm):"))
        self.widthInput = QtWidgets.QLineEdit("1")
        toleranceRow.addWidget(self.widthInput)
        layout.addLayout(toleranceRow)

        pathModeRow = QtWidgets.QHBoxLayout()
        self.straightPathRadio = QtWidgets.QRadioButton("Straight line")
        self.straightPathRadio.setToolTip("Path stays on a fixed line between the two points; only height follows the surface.")
        self.straightPathRadio.setChecked(True)
        self.minimaPathRadio = QtWidgets.QRadioButton("Follow minima")
        self.minimaPathRadio.setToolTip("Path can wander sideways to track the lowest point in each band -- can zigzag on a noisy real surface.")
        pathModeRow.addWidget(self.straightPathRadio)
        pathModeRow.addWidget(self.minimaPathRadio)
        layout.addLayout(pathModeRow)

        createSlicesButton = QtWidgets.QPushButton("Create Slices")
        createSlicesButton.setToolTip("Slice the model along the path between Point 1 and Point 2")
        createSlicesButton.clicked.connect(self.create_slices)
        layout.addWidget(createSlicesButton)

        self.viewSlicesButton = QtWidgets.QPushButton("View Slices")
        self.viewSlicesButton.setToolTip("Page through each slice's 2-D profile to check the cut")
        self.viewSlicesButton.setEnabled(False)
        self.viewSlicesButton.clicked.connect(self.view_slices)
        layout.addWidget(self.viewSlicesButton)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setEnabled(False)

    def toggle_point_select_mode(self, checked):
        self.point_select_mode = checked
        self.selectPointsButton.setText("Select Points: ON (click the model)" if checked else "Select Points")
        if checked:
            self.statusBar().showMessage("Point-select mode ON -- click on the model to add a point.", 4000)

    def try_select_marker_at_current_position(self):
        """If the current click landed on an existing point marker, select
        its row in the table (so the model <-> list correspondence is
        visible) and report True so the caller doesn't also rotate/add a
        new point on top of it."""
        if not self.point_marker_actors:
            return False
        click_pos = self.iren.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
        hit_actor = picker.GetActor()
        if hit_actor not in self.point_marker_actors:
            return False
        row = self.point_marker_actors.index(hit_actor)
        self.points_table.selectRow(row)
        return True

    def on_points_table_selection_changed(self):
        """Highlight the marker matching the selected table row (blue), and
        reset all others back to red -- the "vice versa" half of model <-> list."""
        row = self.points_table.currentRow()
        for i, marker in enumerate(self.point_marker_actors):
            if i == row:
                marker.GetProperty().SetColor(0.1, 0.4, 1.0)
            else:
                marker.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.vtkWidget.GetRenderWindow().Render()

    def pick_point_at_current_position(self):
        """Click-to-pick: find the actual surface point under the cursor,
        then snap to the nearest mesh VERTEX by X,Y distance only (Z
        ignored) -- matching getClosest.m exactly, which is correct here
        because the model is a single-sided surface viewed from above, so
        (X,Y) alone unambiguously identifies a vertex."""
        if self.current_vertices is None:
            return

        click_pos = self.iren.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
        if picker.GetActor() is not self.current_actor:
            # Silent failure was the whole problem last time -- now it's visible.
            self.statusBar().showMessage(
                "No point added: that click didn't land on the model. Click directly on the model's surface.", 4000)
            return

        world_pos = np.array(picker.GetPickPosition())
        dxdy = self.current_vertices[:, :2] - world_pos[:2]
        idx = int(np.argmin(np.sum(dxdy ** 2, axis=1)))
        closest = self.current_vertices[idx]

        self._add_selected_point(closest)
        self.statusBar().showMessage(f"Point {len(self.selected_points)} added.", 2000)

    def _add_selected_point(self, coord):
        coord = np.asarray(coord, dtype=np.float64)
        self.selected_points.append(coord)
        index = len(self.selected_points)

        radius = 0.0
        if self.grid_bounds is not None:
            radius = (self.grid_bounds[1] - self.grid_bounds[0]) * 0.0015
        radius = max(radius, 1e-3)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(coord)
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        marker = vtk.vtkActor()
        marker.SetMapper(mapper)
        marker.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.ren.AddActor(marker)
        self.point_marker_actors.append(marker)

        row = self.points_table.rowCount()
        self.points_table.insertRow(row)
        self.points_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(index)))
        for col, value in enumerate(coord, start=1):
            self.points_table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{value:.3f}"))

        self.vtkWidget.GetRenderWindow().Render()

    def _renumber_points_table(self):
        for row in range(self.points_table.rowCount()):
            self.points_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))

    def delete_selected_point(self):
        row = self.points_table.currentRow()
        if row < 0:
            return
        self.ren.RemoveActor(self.point_marker_actors.pop(row))
        self.selected_points.pop(row)
        self.points_table.removeRow(row)
        self._renumber_points_table()
        self.vtkWidget.GetRenderWindow().Render()

    def clear_selected_points(self):
        if not self.selected_points:
            return
        reply = QtWidgets.QMessageBox.question(
            self, 'Clear points list', 'Are you sure?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        for marker in self.point_marker_actors:
            self.ren.RemoveActor(marker)
        self.point_marker_actors = []
        self.selected_points = []
        self.points_table.setRowCount(0)
        self._reset_slicing_endpoints()
        self.vtkWidget.GetRenderWindow().Render()

    def _default_points_file_path(self):
        """Adjacent-file convention: model.wrl -> model.points.json."""
        base, _ext = os.path.splitext(self.current_model_path)
        return base + '.points.json'

    def save_points(self):
        if self.current_model_path is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before saving points.')
            return
        default_path = self._default_points_file_path()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Points', default_path, 'Points Files (*.points.json)')
        if not fileName:
            return
        if not fileName.lower().endswith('.json'):
            fileName += '.points.json'

        with open(fileName, 'w') as f:
            json.dump({
                'source_file': os.path.basename(self.current_model_path),
                'points': [p.tolist() for p in self.selected_points],
            }, f, indent=2)

    def load_points(self):
        if self.current_vertices is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before loading points.')
            return
        default_path = self._default_points_file_path()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Points', default_path, 'Points Files (*.points.json);;All JSON (*.json)')
        if not fileName:
            return

        try:
            with open(fileName, 'r') as f:
                data = json.load(f)
            points = [np.array(p, dtype=np.float64) for p in data['points']]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            QtWidgets.QMessageBox.warning(self, 'Load Points Failed', f'Could not read points file:\n{e}')
            return

        self.clear_selected_points_silent()
        for point in points:
            self._add_selected_point(point)

    def clear_selected_points_silent(self):
        """Like clear_selected_points() but without the confirmation dialog
        -- used internally before loading a fresh points list."""
        for marker in self.point_marker_actors:
            self.ren.RemoveActor(marker)
        self.point_marker_actors = []
        self.selected_points = []
        self.points_table.setRowCount(0)
        self._reset_slicing_endpoints()

    def _reset_slicing_endpoints(self):
        self.point1_row = None
        self.point2_row = None
        self.point1Label.setText("Point 1: —")
        self.point2Label.setText("Point 2: —")
        self._clear_slice_visualization()
        self.slice_path = None
        self.slices = None
        self.viewSlicesButton.setEnabled(False)
        if self.slice_viewer_dialog is not None:
            self.slice_viewer_dialog.hide()

    def set_as_point1(self):
        row = self.points_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, 'No Point Selected', 'Select a row in the points table first.')
            return
        self.point1_row = row
        self.point1Label.setText(f"Point 1: #{row + 1}")

    def set_as_point2(self):
        row = self.points_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, 'No Point Selected', 'Select a row in the points table first.')
            return
        self.point2_row = row
        self.point2Label.setText(f"Point 2: #{row + 1}")

    def create_slices(self):
        if self.point1_row is None or self.point2_row is None:
            QtWidgets.QMessageBox.warning(self, 'Endpoints Not Set', 'Set both Point 1 and Point 2 before slicing.')
            return
        if self.point1_row == self.point2_row:
            QtWidgets.QMessageBox.warning(self, 'Invalid Endpoints', 'Point 1 and Point 2 must be different points.')
            return
        try:
            tolerance = float(self.toleranceInput.text())
            width = float(self.widthInput.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Tolerance and width must be numbers.')
            return
        if tolerance <= 0 or width <= 0:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Tolerance and width must be positive.')
            return

        c1 = self.selected_points[self.point1_row]
        c2 = self.selected_points[self.point2_row]
        tipo = 1 if self.minimaPathRadio.isChecked() else 0

        try:
            self.slice_path, num_slices = compute_slice_path(self.current_vertices, c1, c2, tolerance, width, tipo=tipo)
            self.slices = compute_slices(self.current_vertices, self.slice_path, width)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Slicing Failed', str(e))
            return

        self._visualize_slice_path(width)
        self.viewSlicesButton.setEnabled(True)
        if self.slice_viewer_dialog is not None and self.slice_viewer_dialog.isVisible():
            self.slice_viewer_dialog.set_slices(self.slices)
        self.statusBar().showMessage(f"{len(self.slices)} slices created.", 4000)

    def view_slices(self):
        if not self.slices:
            return
        if self.slice_viewer_dialog is None:
            self.slice_viewer_dialog = SliceViewerDialog(self.slices, self)
        else:
            self.slice_viewer_dialog.set_slices(self.slices)
        self.slice_viewer_dialog.show()
        self.slice_viewer_dialog.raise_()
        self.slice_viewer_dialog.activateWindow()

    def open_label_tool(self):
        """Experimental: compute the 2.5D grid for the loaded model (from the
        auto-positioned base_vertices, so it's deterministic per file) and
        open the labeling dialog. Step 2a -- viewing only for now."""
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

        source_name = os.path.basename(self.current_model_path) if self.current_model_path else 'model'
        self.label_tool_dialog = LabelToolDialog(grid, source_name, self)
        self.label_tool_dialog.show()
        self.label_tool_dialog.raise_()
        self.label_tool_dialog.activateWindow()

    def _clear_slice_visualization(self):
        for actor in self.slice_visualization_actors:
            self.ren.RemoveActor(actor)
        self.slice_visualization_actors = []

    def _visualize_slice_path(self, width):
        """Draw the path of slice positions over the model (matching
        MATLAB's plot3 path overlay), plus a perpendicular tick mark at each
        slice showing the width band -- just enough to visually confirm the
        slice ran where expected. The 2D per-slice profile view is a later
        step."""
        self._clear_slice_visualization()

        path = self.slice_path
        points = vtk.vtkPoints()
        for p in path:
            points.InsertNextPoint(p)
        lines = vtk.vtkCellArray()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(path))
        for i in range(len(path)):
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        path_actor = vtk.vtkActor()
        path_actor.SetMapper(mapper)
        path_actor.GetProperty().SetColor(0.0, 0.8, 0.0)
        path_actor.GetProperty().SetLineWidth(3)
        self.ren.AddActor(path_actor)
        self.slice_visualization_actors.append(path_actor)

        # Perpendicular tick marks showing the width band at each slice.
        tick_points = vtk.vtkPoints()
        tick_lines = vtk.vtkCellArray()
        for i in range(len(path) - 1):
            direction = path[i + 1, :2] - path[i, :2]
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            perp = np.array([-direction[1], direction[0]]) / norm
            center = path[i]
            p_start = center + np.array([perp[0], perp[1], 0.0]) * (width / 2)
            p_end = center - np.array([perp[0], perp[1], 0.0]) * (width / 2)
            idx0 = tick_points.InsertNextPoint(p_start)
            idx1 = tick_points.InsertNextPoint(p_end)
            tick_line = vtk.vtkLine()
            tick_line.GetPointIds().SetId(0, idx0)
            tick_line.GetPointIds().SetId(1, idx1)
            tick_lines.InsertNextCell(tick_line)

        tick_polydata = vtk.vtkPolyData()
        tick_polydata.SetPoints(tick_points)
        tick_polydata.SetLines(tick_lines)
        tick_mapper = vtk.vtkPolyDataMapper()
        tick_mapper.SetInputData(tick_polydata)
        tick_actor = vtk.vtkActor()
        tick_actor.SetMapper(tick_mapper)
        tick_actor.GetProperty().SetColor(1.0, 0.6, 0.0)
        self.ren.AddActor(tick_actor)
        self.slice_visualization_actors.append(tick_actor)

        self.vtkWidget.GetRenderWindow().Render()

    def open_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select your file', self.last_dir,
            '3D Files (*.wrl *.stl *.ply *.obj)')
        if not fileName:
            return
        self.last_dir = fileName.rsplit('/', 1)[0]

        self.ren.RemoveAllViewProps()
        # RemoveAllViewProps() already dropped the old point-marker actors/slice
        # visualization from the scene; also clear our own tracking so a new
        # model starts fresh (points/slices are tied to a specific mesh).
        self.point_marker_actors = []
        self.selected_points = []
        self.points_table.setRowCount(0)
        self.slice_path = None
        self.slices = None
        self.slice_visualization_actors = []
        self._reset_slicing_endpoints()

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
        self.points_dock.setEnabled(True)
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
