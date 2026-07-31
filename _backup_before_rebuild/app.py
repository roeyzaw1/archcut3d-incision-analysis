"""
ArchCut 3-D

Software developed to precisely evaluate the 3-D micromorphological
characteristics of archaeological surface incisions. Analysis is based on a
continuous 3-D slice sequence from a chosen range within the incision path.
Measurements and shapes of individual slices can be extracted. ArchCut 3-D's
interface is designed to provide a computational and mathematical analysis of
engravings to a wide audience.

Built with PyQt5, VTK, and PyMeshLab.

Recent changes:
- Added ability to calculate circumference.
- Added the ability to change size of center of mass.
- Cross-platform fixes (Windows + macOS + Linux).
"""

import sys  # needed at module level for resource_path()
import vtk
from vtk import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkTexture
from vtkmodules.vtkIOImage import vtkJPEGReader, vtkPNGReader
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QWidget, QToolBar, QAction, QSizePolicy, QSpacerItem, QPushButton, QLineEdit, QGridLayout, QLabel, QDialog, QVBoxLayout, QComboBox, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator

import numpy as np
import os
import glob
import pymeshlab
import subprocess
import shutil
import csv
import time
import re
import json
import getpass
import math
import platform

from .mesh_smoothing import MeshSmootherTool
from .crest_detection import CrestDetector
from .positioning import compute_normal_alignment_axis_angle
from .resources import resource_path, get_cache_dir, open_with_default_app

APP_NAME = "ArchCut 3-D"
APP_DESCRIPTION = (
    "Software developed to precisely evaluate the 3-D micromorphological "
    "characteristics of archaeological surface incisions. Analysis is based "
    "on a continuous 3-D slice sequence from a chosen range within the "
    "incision path. Measurements and shapes of individual slices can be "
    "extracted. ArchCut 3-D's interface is designed to provide a "
    "computational and mathematical analysis of engravings to a wide "
    "audience."
)

# This class is meant to overlay an image called "loadingMessage.png" which informs the user that the application is loading their file.
class LoadingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(LoadingDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        
        # Set the label.
        self.label = QtWidgets.QLabel(self)
        
        # Set the window title.
        self.setWindowTitle('Loading')
        
        # Set the loading image.
        pixmap = QPixmap(resource_path("assets/images/loading_message.png"))


        self.label.setPixmap(pixmap.scaled(200, 100, Qt.KeepAspectRatio))
        self.label.setAlignment(Qt.AlignCenter)

        # Set the size of the QDialog to 300 by 300 pixels
        self.setFixedSize(200, 100)

# This class handles the batch convert window and its functions.
class BatchConvertWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(BatchConvertWindow, self).__init__(parent)
        self.setMinimumSize(200, 150)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle('Batch Convert')

        self.layout = QtWidgets.QVBoxLayout(self)

        self.selectFolderButton = QtWidgets.QPushButton('Select Folder')
        self.selectFolderButton.clicked.connect(self.select_folder)
        self.layout.addWidget(self.selectFolderButton)

        self.formatLayout = QtWidgets.QHBoxLayout()
        self.formatButtons = QtWidgets.QButtonGroup(self)
        for format in ['wrl', 'obj', 'stl', 'ply']:
            button = QtWidgets.QRadioButton(format.upper())
            self.formatButtons.addButton(button)
            self.formatLayout.addWidget(button)
            if format == 'stl':
                # Automatically select the STL option 
                button.setChecked(True)
        self.layout.addLayout(self.formatLayout)

        self.selectOutputButton = QtWidgets.QPushButton('Select Output')
        self.selectOutputButton.clicked.connect(self.select_output)
        self.selectOutputButton.setEnabled(False)  # Initially disabled
        self.layout.addWidget(self.selectOutputButton)

        self.convertButton = QtWidgets.QPushButton('Convert')
        self.convertButton.clicked.connect(self.convert_multi_files)
        self.convertButton.setEnabled(False)  # Initially disabled
        self.layout.addWidget(self.convertButton)

        self.folderPath = None
        self.outputPath = None

    def select_folder(self):
        self.folderPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        if self.folderPath:
            self.selectOutputButton.setEnabled(True)

    def select_output(self):
        self.outputPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if self.outputPath:
            self.convertButton.setEnabled(True)

    def convert_multi_files(self):
        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.show()
        QtWidgets.QApplication.processEvents()
        if not self.folderPath or not self.outputPath:
            return

        format = self.formatButtons.checkedButton().text().lower()

        for file in os.listdir(self.folderPath):
            if file.lower().endswith(('.wrl', '.obj', '.stl', '.ply')):
                input_file = os.path.join(self.folderPath, file)
                base_name = os.path.splitext(file)[0]
                output_file = os.path.join(self.outputPath, base_name + '.' + format)
                if not convert_file(input_file, output_file):
                    return
                
        self.loading_dialog.hide()

class BoundWindowOptions(QtWidgets.QDialog):
    def __init__(self, main_window, parent=None):
        super(BoundWindowOptions, self).__init__(parent)
        self.main_window = main_window
        
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))

        self.setWindowTitle('Boundary Options')
        self.setMinimumSize(250, 150)

        layout = QGridLayout()
        self.setLayout(layout)

        self.input_field_label = QLineEdit()
        self.font_input = QLineEdit()

        toggleBounderiesButton = QPushButton('Toggle Boundary')
        toggleBounderiesButton.clicked.connect(self.main_window.toggle_bbox)
        layout.addWidget(toggleBounderiesButton, 0, 0, 1, 5)

        rotateLeftButton = QPushButton('Rotate Y Left')
        rotateLeftButton.clicked.connect(self.main_window.rotate_y_left)
        rotateLeftButton.setAutoRepeat(True)
        layout.addWidget(rotateLeftButton, 1, 0)

        rotateRightButton = QPushButton('Rotate Y Right')
        rotateRightButton.clicked.connect(self.main_window.rotate_y_right)
        rotateRightButton.setAutoRepeat(True)
        layout.addWidget(rotateRightButton, 1, 2)

        rotateXLeftButton = QPushButton('Rotate X Left')
        rotateXLeftButton.clicked.connect(self.main_window.rotate_x_left)
        rotateXLeftButton.setAutoRepeat(True)
        layout.addWidget(rotateXLeftButton, 2, 0)

        rotateXRightButton = QPushButton('Rotate X Right')
        rotateXRightButton.clicked.connect(self.main_window.rotate_x_right)
        rotateXRightButton.setAutoRepeat(True)
        layout.addWidget(rotateXRightButton, 2, 2)

        rotateZLeftButton = QPushButton('Rotate Z Left')
        rotateZLeftButton.clicked.connect(self.main_window.rotate_z_left)
        rotateZLeftButton.setAutoRepeat(True)
        layout.addWidget(rotateZLeftButton, 3, 0)

        rotateZRightButton = QPushButton('Rotate Z Right')
        rotateZRightButton.clicked.connect(self.main_window.rotate_z_right)
        rotateZRightButton.setAutoRepeat(True)
        layout.addWidget(rotateZRightButton, 3, 2)

        moveForwardButton = QPushButton('Move Forward')
        moveForwardButton.clicked.connect(self.main_window.move_forward)
        moveForwardButton.setAutoRepeat(True)
        layout.addWidget(moveForwardButton, 6, 2)

        moveBackwardButton = QPushButton('Move Backward')
        moveBackwardButton.clicked.connect(self.main_window.move_backward)
        moveBackwardButton.setAutoRepeat(True)
        layout.addWidget(moveBackwardButton, 6, 0)
        
        upButton = QPushButton('Up')
        upButton.clicked.connect(self.main_window.move_up)
        upButton.setAutoRepeat(True)
        layout.addWidget(upButton, 6, 1)

        downButton = QPushButton('Down')
        downButton.clicked.connect(self.main_window.move_down)
        downButton.setAutoRepeat(True)
        layout.addWidget(downButton, 8, 1)

        leftButton = QPushButton('Left')
        leftButton.setAutoRepeat(True)
        leftButton.clicked.connect(self.main_window.move_left)
        layout.addWidget(leftButton, 7, 0)

        rightButton = QPushButton('Right')
        rightButton.setAutoRepeat(True)
        rightButton.clicked.connect(self.main_window.move_right)
        layout.addWidget(rightButton, 7, 2)

        okLabelButton = QPushButton('Ok')
        okLabelButton.clicked.connect(lambda: self.close())
        layout.addWidget(okLabelButton, 9, 0, 1, 5)


class PositioningToolWindow(QtWidgets.QDialog):
    """Cut Analysis step 1: position the model before slicing. Rotation
    buttons bake a rotation directly into the model's transform (not a
    camera move), same as MATLAB Cut3-D's degang + 6-button positioning."""

    def __init__(self, main_window, parent=None):
        super(PositioningToolWindow, self).__init__(parent)
        self.main_window = main_window

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))

        self.setWindowTitle('Position Model')
        self.setMinimumSize(260, 260)

        layout = QGridLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel('Angle per click (deg):'), 0, 0, 1, 2)
        self.angleInput = QLineEdit('5')
        layout.addWidget(self.angleInput, 0, 2, 1, 1)

        rotateXUpButton = QPushButton('Rotate X Up')
        rotateXUpButton.setAutoRepeat(True)
        rotateXUpButton.clicked.connect(lambda: self.main_window.rotate_x_left(self._angle()))
        layout.addWidget(rotateXUpButton, 1, 0)

        rotateXDownButton = QPushButton('Rotate X Down')
        rotateXDownButton.setAutoRepeat(True)
        rotateXDownButton.clicked.connect(lambda: self.main_window.rotate_x_right(self._angle()))
        layout.addWidget(rotateXDownButton, 1, 2)

        rotateYLeftButton = QPushButton('Rotate Y Left')
        rotateYLeftButton.setAutoRepeat(True)
        rotateYLeftButton.clicked.connect(lambda: self.main_window.rotate_y_left(self._angle()))
        layout.addWidget(rotateYLeftButton, 2, 0)

        rotateYRightButton = QPushButton('Rotate Y Right')
        rotateYRightButton.setAutoRepeat(True)
        rotateYRightButton.clicked.connect(lambda: self.main_window.rotate_y_right(self._angle()))
        layout.addWidget(rotateYRightButton, 2, 2)

        rotateZClockwiseButton = QPushButton('Rotate Z Clockwise')
        rotateZClockwiseButton.setAutoRepeat(True)
        rotateZClockwiseButton.clicked.connect(lambda: self.main_window.rotate_z_right(self._angle()))
        layout.addWidget(rotateZClockwiseButton, 3, 0)

        rotateZCounterButton = QPushButton('Rotate Z Counter-CW')
        rotateZCounterButton.setAutoRepeat(True)
        rotateZCounterButton.clicked.connect(lambda: self.main_window.rotate_z_left(self._angle()))
        layout.addWidget(rotateZCounterButton, 3, 2)

        alignNormalButton = QPushButton('Align by Normal')
        alignNormalButton.setToolTip('Automatically rotate the model so its dominant surface faces +Z')
        alignNormalButton.clicked.connect(self.main_window.align_by_normal_direction)
        layout.addWidget(alignNormalButton, 4, 0, 1, 3)

        savePositionButton = QPushButton('Save Position')
        savePositionButton.clicked.connect(self.main_window.save_position)
        layout.addWidget(savePositionButton, 5, 0, 1, 1)

        loadPositionButton = QPushButton('Load Position')
        loadPositionButton.clicked.connect(self.main_window.load_position)
        layout.addWidget(loadPositionButton, 5, 2, 1, 1)

        okButton = QPushButton('Close')
        okButton.clicked.connect(lambda: self.close())
        layout.addWidget(okButton, 6, 0, 1, 3)

    def _angle(self):
        try:
            return float(self.angleInput.text())
        except ValueError:
            return 5.0


class ScalesWindowOptions(QtWidgets.QDialog):
    def __init__(self, main_window, parent=None):
        super(ScalesWindowOptions, self).__init__(parent)
        self.main_window = main_window

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))

        self.setWindowTitle('Bar Scale')
        self.setMinimumSize(250, 150)

        layout = QGridLayout()
        self.setLayout(layout)

        scalesLabel = QLabel("Scales:")
        layout.addWidget(scalesLabel, 0, 0)

        self.scalesDropdown = QComboBox()
        self.scalesDropdown.addItems(["2cm", "5cm", "10cm", "20cm", "20mm", "50mm", "100mm", "200mm"])
        layout.addWidget(self.scalesDropdown, 0, 1)

        addScaleButton = QPushButton('Add')
        addScaleButton.clicked.connect(self.add_scale)
        layout.addWidget(addScaleButton, 0, 2)

        left_button = QtWidgets.QPushButton('Left')
        left_button.setAutoRepeat(True)
        left_button.clicked.connect(self.main_window.move_scale_left)
        layout.addWidget(left_button, 5, 0)

        right_button = QtWidgets.QPushButton('Right')
        right_button.setAutoRepeat(True)
        right_button.clicked.connect(self.main_window.move_scale_right)
        layout.addWidget(right_button, 5, 2)

        up_button = QtWidgets.QPushButton('Up')
        up_button.setAutoRepeat(True)
        up_button.clicked.connect(self.main_window.move_scale_up)
        layout.addWidget(up_button, 4, 1)

        down_button = QtWidgets.QPushButton('Down')
        down_button.setAutoRepeat(True)
        down_button.clicked.connect(self.main_window.move_scale_down)
        layout.addWidget(down_button, 6, 1)

        removeScaleButton = QPushButton('Remove Scale')
        removeScaleButton.clicked.connect(self.main_window.remove_scale)
        layout.addWidget(removeScaleButton, 8, 0, 1, 5)

        resetScalePositionButton = QPushButton('Reset Position')
        resetScalePositionButton.clicked.connect(self.main_window.reset_image_position)
        layout.addWidget(resetScalePositionButton, 9, 0, 1, 5)

        okScaleButton = QPushButton('Ok') 
        okScaleButton.clicked.connect(lambda: self.close())
        layout.addWidget(okScaleButton, 10, 0, 1, 5)

    def add_scale(self):
        selected_scale = self.scalesDropdown.currentText()
        scale_to_image_map = {
            "2cm": resource_path("assets/scale_bars/2cm.png"),
            "5cm": resource_path("assets/scale_bars/5cm.png"),
            "10cm": resource_path("assets/scale_bars/10cm.png"),
            "20cm": resource_path("assets/scale_bars/20cm.png"),
            "20mm": resource_path("assets/scale_bars/20mm.png"),
            "50mm": resource_path("assets/scale_bars/50mm.png"),
            "100mm": resource_path("assets/scale_bars/100mm.png"),
            "200mm": resource_path("assets/scale_bars/200mm.png")
        }
        if selected_scale in scale_to_image_map:
            self.main_window.open_scale_image(scale_to_image_map[selected_scale])

class LightDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(LightDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.setWindowTitle('Light')
        self.setMinimumSize(250, 150)

        layout = QGridLayout()

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider, 0, 0, 1, 5)

        self.change_light_button = QtWidgets.QPushButton('Add/Reset light')
        self.change_light_button.clicked.connect(self.add_light)
        layout.addWidget(self.change_light_button, 1, 0, 1, 5)

        self.remove_light_button = QtWidgets.QPushButton('Remove light')
        self.remove_light_button.clicked.connect(self.remove_light)
        layout.addWidget(self.remove_light_button, 2, 0, 1, 5)

        self.ok_button = QtWidgets.QPushButton('Ok')
        self.ok_button.clicked.connect(lambda: self.close())
        layout.addWidget(self.ok_button, 3, 0, 1, 5)

        self.left_button = QtWidgets.QPushButton('Left')
        self.left_button.setAutoRepeat(True)
        self.left_button.setEnabled(False)
        self.left_button.clicked.connect(self.move_light_left)
        layout.addWidget(self.left_button, 5, 0)

        self.right_button = QtWidgets.QPushButton('Right')
        self.right_button.setAutoRepeat(True)
        self.right_button.setEnabled(False)
        self.right_button.clicked.connect(self.move_light_right)
        layout.addWidget(self.right_button, 5, 2)

        self.up_button = QtWidgets.QPushButton('Up')
        self.up_button.setAutoRepeat(True)
        self.up_button.setEnabled(False)
        self.up_button.clicked.connect(self.move_light_up)
        layout.addWidget(self.up_button, 4, 1)

        self.down_button = QtWidgets.QPushButton('Down')
        self.down_button.setAutoRepeat(True)
        self.down_button.setEnabled(False)
        self.down_button.clicked.connect(self.move_light_down)
        layout.addWidget(self.down_button, 6, 1)

        self.setLayout(layout)

    def slider_changed(self, value):
        self.parent().light.SetIntensity(value / 100.0)
        self.parent().vtkWidget.GetRenderWindow().Render()

    def move_light_left(self):
        try:
            position = list(self.parent().light.GetPosition())
            position[0] -= 1
            self.parent().light.SetPosition(position)
            self.parent().vtkWidget.GetRenderWindow().Render()
        except:
            return
    
    def move_light_right(self):
        try:
            position = list(self.parent().light.GetPosition())
            position[0] += 1
            self.parent().light.SetPosition(position)
            self.parent().vtkWidget.GetRenderWindow().Render()
        except:
            return

    def move_light_up(self):
        try:
            position = list(self.parent().light.GetPosition())
            position[1] += 1
            self.parent().light.SetPosition(position)
            self.parent().vtkWidget.GetRenderWindow().Render()
        except:
            return

    def move_light_down(self):
        try:
            position = list(self.parent().light.GetPosition())
            position[1] -= 1
            self.parent().light.SetPosition(position)
            self.parent().vtkWidget.GetRenderWindow().Render()
        except:
            return
    
    def add_light(self):
        self.remove_light_button.setEnabled(True)
        self.left_button.setEnabled(True)
        self.right_button.setEnabled(True)
        self.up_button.setEnabled(True)
        self.down_button.setEnabled(True)
        self.slider.setEnabled(True)
        self.parent().ren.RemoveAllLights()
        self.parent().light = vtk.vtkLight()
        self.parent().light.SetIntensity(self.slider.value() / 100.0)
        self.parent().ren.AddLight(self.parent().light)
        self.parent().vtkWidget.GetRenderWindow().Render()
    
    def remove_light(self):
        try:
            self.left_button.setEnabled(False)
            self.right_button.setEnabled(False)
            self.up_button.setEnabled(False)
            self.down_button.setEnabled(False)
            self.slider.setEnabled(False)
            self.slider.setValue(50)
            self.parent().light.SwitchOff()
            self.parent().vtkWidget.GetRenderWindow().Render()
        except:
            return

class ReduceSizeDialog(QDialog):
    def __init__(self, file_name, parent=None):
        super(ReduceSizeDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.setWindowTitle('Reduce file size')
        self.file_name = file_name
        layout = QVBoxLayout()

        self.label = QLabel()
        layout.addWidget(self.label)

        self.size_input = QLineEdit()
        self.size_input.setValidator(QIntValidator())
        layout.addWidget(self.size_input)
        self.size_input.setToolTip('Reduce by a percentage. Example: 50')
        self.size_input.setPlaceholderText('Reduce by percentage')

        self.finalSizeInput = QLineEdit()
        layout.addWidget(self.finalSizeInput)
        self.finalSizeInput.setPlaceholderText('Reduce by final size')
        self.finalSizeInput.setToolTip('Write the final size you want. Example: 5000')        
        self.finalSizeInput.setValidator(QIntValidator())
        self.finalSizeInput.textChanged.connect(self.update_percentage)

        finalSizeEstimate = QLineEdit()
        layout.addWidget(finalSizeEstimate)
        finalSizeEstimate.setPlaceholderText('Estimated final size')
        finalSizeEstimate.setToolTip('This is the estimate of the final file size')
        finalSizeEstimate.setReadOnly(True)

        current_size = os.path.getsize(self.file_name) / 1024

        self.size_input.textChanged.connect(lambda: self.update_estimate(finalSizeEstimate, current_size))

        self.reduce_button = QPushButton('Reduce')
        self.reduce_button.clicked.connect(self.reduce_size)
        layout.addWidget(self.reduce_button)
        self.reduce_button.setEnabled(False)

        self.setLayout(layout)

    def update_percentage(self):
        if self.finalSizeInput.text():
            final_size = float(self.finalSizeInput.text())
            current_size = os.path.getsize(self.file_name) / 1024
            reduction_percentage = 100 * (1 - final_size / current_size)
            self.reduce_button.setEnabled(True)
            if reduction_percentage >= 0:
                self.size_input.setText(str(reduction_percentage))
                self.reduce_button.setEnabled(True)
            else:
                self.size_input.clear()
                self.reduce_button.setEnabled(False)
        else:
            self.size_input.clear()
            self.reduce_button.setEnabled(False)

    def update_estimate(self, finalSizeEstimate, current_size):
        if self.size_input.text():
            reduction_percentage = float(self.size_input.text())
            estimated_final_size = current_size * (1 - reduction_percentage / 100)
            self.reduce_button.setEnabled(True)
            if estimated_final_size >= 0:
                finalSizeEstimate.setText(str(estimated_final_size))
                self.reduce_button.setEnabled(True)
            else:
                finalSizeEstimate.clear()
                self.reduce_button.setEnabled(False)
        else:
            finalSizeEstimate.clear()
            self.reduce_button.setEnabled(False)

    def reduce_size(self):
        import os
        base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        ext = os.path.splitext(self.file_name)[1]
        default_name = base_name + ext

        new_file_name, _ = QFileDialog.getSaveFileName(self, 'Save File', default_name)

        if new_file_name:
            if os.path.splitext(new_file_name)[1] != ext:
                new_file_name = os.path.splitext(new_file_name)[0] + ext
            reduction_percentage = float(self.size_input.text())

            self.loading_dialog = LoadingDialog(self)
            self.loading_dialog.show()
            QtWidgets.QApplication.processEvents()

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(self.file_name)
            current_faces = ms.current_mesh().face_number()
            target_faces = int(current_faces * (1 - reduction_percentage / 100))
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=target_faces)
            ms.save_current_mesh(new_file_name)

            self.loading_dialog.close()

class LabelWindowOptions(QtWidgets.QDialog):
    def __init__(self, main_window, parent=None):
        super(LabelWindowOptions, self).__init__(parent)
        self.main_window = main_window
        
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))

        self.setWindowTitle('Label Options')
        self.setMinimumSize(250, 150)

        layout = QGridLayout()
        self.setLayout(layout)

        self.input_field_label = QLineEdit()
        self.font_input = QLineEdit()
        layout.addWidget(self.input_field_label, 0, 0, 1, 3)
        self.input_field_label.setToolTip('Set the text for the label')

        layout.addWidget(self.font_input, 0, 3)
        self.font_input.setToolTip('Set size of the label. Default is 24')
        self.font_input.setMaximumWidth(40)
        self.font_input.setValidator(QIntValidator())
        self.font_input.setText('24')
        layout.addWidget(self.main_window.dropdown, 1, 0, 1, 5)

        addLabelButton = QPushButton('Add label')
        addLabelButton.clicked.connect(lambda: self.main_window.add_label(self.input_field_label.text()))
        layout.addWidget(addLabelButton, 2, 0, 1, 5)

        removeLabelButton = QPushButton('Remove label')
        removeLabelButton.clicked.connect(self.main_window.remove_label)
        layout.addWidget(removeLabelButton, 3, 0, 1, 5)

        okLabelButton = QPushButton('Ok')
        okLabelButton.clicked.connect(lambda: self.close())
        layout.addWidget(okLabelButton, 4, 0, 1, 5)

        upButton = QPushButton('Up')
        upButton.clicked.connect(self.main_window.move_label_up)
        upButton.setAutoRepeat(True)
        layout.addWidget(upButton, 5, 1)
        
        downButton = QPushButton('Down')
        downButton.clicked.connect(self.main_window.move_label_down)
        downButton.setAutoRepeat(True)
        layout.addWidget(downButton, 7, 1)

        leftButton = QPushButton('Left')
        leftButton.setAutoRepeat(True)
        leftButton.clicked.connect(self.main_window.move_label_left)
        layout.addWidget(leftButton, 6, 0)

        rightButton = QPushButton('Right')
        rightButton.setAutoRepeat(True)
        rightButton.clicked.connect(self.main_window.move_label_right)
        layout.addWidget(rightButton, 6, 2)

def convert_file(input_file, output_file):
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_file)
        if not output_file.lower().endswith('.stl'):
            ms.save_current_mesh(output_file, save_vertex_color=True)
        else:
            ms.save_current_mesh(output_file)
    except Exception as e:
        return False
    return True

def convert_wrl_to_ply(wrl_file, ply_file):
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(wrl_file)
        ms.save_current_mesh(ply_file, save_vertex_color=True)
    except Exception as e:
        return False
    return True
    
def convert_wrl_to_obj(wrl_file, obj_file):
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(wrl_file)
        ms.save_current_mesh(obj_file, save_vertex_color=True)
    except Exception as e:
        return False
    return True

class LineScaleDialog(QDialog):
    def __init__(self, parent=None):
        super(LineScaleDialog, self).__init__(parent)
        self.mainWindow = parent
        self.setWindowTitle("Line Scale")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        
        validator = QIntValidator()
        self.lineScaleColor = 1.0, 0.0, 0.0

        self.sizeEdit = QLineEdit("50")
        self.sizeEdit.setValidator(validator)     
        self.unitCombo = QComboBox()
        self.unitCombo.addItems(["mm", "cm"])
        self.colorCombo = QComboBox()
        self.colorCombo.addItems(["Red", "White", "Black"])
        self.okButton = QPushButton("OK")
        self.cancelButton = QPushButton("Cancel/Remove")

        layout = QVBoxLayout()
        sizeLayout = QHBoxLayout()
        sizeLayout.addWidget(QLabel("Size:"))
        sizeLayout.addWidget(self.sizeEdit)
        sizeLayout.addWidget(self.unitCombo)
        layout.addLayout(sizeLayout)
        layout.addWidget(QLabel("Color:"))
        layout.addWidget(self.colorCombo)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.cancelButton)        
        buttonLayout.addWidget(self.okButton)
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

        self.colorCombo.currentTextChanged.connect(self.updateColor)
        self.okButton.clicked.connect(self.okButtonFunction)
        self.cancelButton.clicked.connect(self.reject)

    def updateColor(self, color):
        if color == "Red":
            self.lineScaleColor = 1.0, 0.0, 0.0
        elif color == "White":
            self.lineScaleColor = 1.0, 1.0, 1.0
        elif color == "Black":
            self.lineScaleColor = 0.0, 0.0, 0.0

    def okButtonFunction (self):
        lineSizeText = self.sizeEdit.text() + " " + self.unitCombo.currentText()
        self.mainWindow.createLineScale(lineSizeText, self.lineScaleColor)
        self.close()

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward)

        # <-- ADDED: Ctrl + left-drag (incl. Ctrl + trackpad drag on Mac) rolls
        # the view clockwise / counter-clockwise. We intercept the mouse events
        # ourselves; when Ctrl isn't held we defer to the normal trackball
        # behaviour by calling the parent On* handlers.
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self._rolling = False
        self._last_roll_angle = 0.0

    def _cursor_angle(self):
        """Angle (degrees) of the cursor about the centre of the render window.
        Used to turn a circular drag motion into a roll amount."""
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        size = interactor.GetRenderWindow().GetSize()
        cx = size[0] / 2.0
        cy = size[1] / 2.0
        return math.degrees(math.atan2(y - cy, x - cx))

    def on_left_button_press(self, obj, event):
        # Latch roll mode if Option/Alt is held at the moment the drag starts.
        # (We use Alt rather than Ctrl because macOS reserves Ctrl+click as a
        # right-click, so a Ctrl+left-drag never reaches us on a Mac.)
        if self.GetInteractor().GetAltKey():
            self._rolling = True
            self._last_roll_angle = self._cursor_angle()
        else:
            self._rolling = False
            super().OnLeftButtonDown()

    def on_mouse_move(self, obj, event):
        if self._rolling:
            angle = self._cursor_angle()
            delta = angle - self._last_roll_angle
            self._last_roll_angle = angle
            camera = self.window.ren.GetActiveCamera()
            camera.Roll(delta)
            camera.OrthogonalizeViewUp()
            self.window.vtkWidget.GetRenderWindow().Render()
        else:
            super().OnMouseMove()

    def on_left_button_release(self, obj, event):
        if self._rolling:
            self._rolling = False
        else:
            super().OnLeftButtonUp()

    def Rotate(self):
        # Positioning must go exclusively through self.transform (the
        # rotate/align buttons), same principle as MATLAB Cut3-D: the object
        # itself is what gets reoriented, never a free camera-drag "view"
        # change. Left-drag orbit is disabled unless the user explicitly
        # turns on Explore Mode (View menu) to casually look the model over.
        if getattr(self.window, 'explore_mode', False):
            super().Rotate()

    def mouse_wheel_forward(self, obj, event):
        super().OnMouseWheelForward()
        if self.window.current_scale_name is not None and hasattr(self.window, 'original_image'):
            self.window.open_scale_image(self.window.current_scale_name)

    def mouse_wheel_backward(self, obj, event):
        super().OnMouseWheelBackward()
        if self.window.current_scale_name is not None and hasattr(self.window, 'original_image'):
            self.window.open_scale_image(self.window.current_scale_name)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))
        self.resize(990, 600)

        self.frame = QtWidgets.QFrame()

        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        style = vtk.vtkInteractorStyleTrackballActor()
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        self.ren.SetBackground(1.0, 1.0, 1.0)
        
        self.iren.AddObserver("KeyPressEvent", self.keypress_event)

        vtk.vtkOutputWindow.SetInstance(vtk.vtkOutputWindow())

        self.exit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.exit_full_screen)
        self.enter_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self)
        self.enter_shortcut.activated.connect(self.enter_full_screen)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        logo_reader = vtk.vtkPNGReader()
        logo_reader.SetFileName(resource_path('assets/images/logo.png'))

        logo_mapper = vtk.vtkImageMapper()
        logo_mapper.SetInputConnection(logo_reader.GetOutputPort())
         
        logo_mapper.SetColorWindow(255)
        logo_mapper.SetColorLevel(127.5)

        self.logo_actor = vtk.vtkActor2D()
        self.logo_actor.SetMapper(logo_mapper)
        
        self.logo_actor.SetWidth(100)
        self.logo_actor.SetHeight(200)

        self.logo_visible = True

        self.ren.AddActor2D(self.logo_actor)
        self.ren.ResetCamera()

        self.setAcceptDrops(True)

        self.lower_threshold = 50
        self.upper_threshold = 150

        self.username = getpass.getuser()

        self.default_style = vtkInteractorStyleTrackballCamera()
        self.custom_style = vtkInteractorStyleTrackballActor()

        self.iren.SetInteractorStyle(self.default_style)
        
        self.is_default_style = True
        
        self.current_scale_name = None

        self.ShowBarScaleMessage = True

        self.vtkWidget.GetRenderWindow().StereoCapableWindowOn()
        self.vtkWidget.GetRenderWindow().SetStereoTypeToRedBlue()

        self.image_counter = 1
        self.axis_index = 1
        self.current_view = 'full'
        self.previous_camera_distance = self.ren.GetActiveCamera().GetDistance()

        self.points_visible = False
        self.wireframe_visible = False
        self.rotate_flag = False
        self.explore_mode = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)

        self.previous_size = self.size() 

        self.last_dir = ''  

        self.setFocusPolicy(QtCore.Qt.StrongFocus) 

        self.label = QtWidgets.QLabel(self)
        self.label.installEventFilter(self)
        self.label_visible = False
        
        self.iren.SetInteractorStyle(CustomInteractorStyle(self))

        self.scale_factor = 1.0
        self.opacity = 1.0
        self.opacity_index = 0

        screen = QtGui.QGuiApplication.primaryScreen()
        self.dpi = screen.logicalDotsPerInch()

        self.image_moved_manually = False

        self.labels = []
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.hide()

        self.menuPart = self.menuBar()
        
        self.toolbar = QToolBar('Toolbar')
        self.addToolBar(self.toolbar)

        self.otherFunctionsToolbar = QToolBar("Other functions")
        self.otherFunctionsToolbar.hide()
        self.addToolBar(Qt.RightToolBarArea, self.otherFunctionsToolbar)

        self.cutToolbar = QToolBar("Cut Toolbar")
        self.cutToolbar.hide()
        self.addToolBar(Qt.RightToolBarArea, self.cutToolbar)

        self.fileMenu = self.menuPart.addMenu('&File')

        openFileAction = QAction('Open File', self)
        openFileAction.triggered.connect(self.open_file)
        self.fileMenu.addAction(openFileAction)

        openFolderAction = QAction('Open Folder', self)
        openFolderAction.triggered.connect(self.open_folder)
        self.fileMenu.addAction(openFolderAction)

        self.aboutCurrentFileAction = QtWidgets.QAction('About File', self)
        self.aboutCurrentFileAction.triggered.connect(self.about_current_file)
        self.aboutCurrentFileAction.setEnabled(False)
        self.fileMenu.addAction(self.aboutCurrentFileAction)        

        self.exportAction = QtWidgets.QAction('Export', self)
        self.exportAction.setEnabled(False)
        self.exportAction.triggered.connect(self.export_file)
        self.fileMenu.addAction(self.exportAction)

        self.reduceSizeAction = QtWidgets.QAction('Reduce Size', self)
        self.reduceSizeAction.setEnabled(False)
        self.reduceSizeAction.triggered.connect(self.open_reduce_size_dialog)
        self.fileMenu.addAction(self.reduceSizeAction)

        self.Toolbarmenu = self.menuPart.addMenu('Toolbar')

        ToolbarTB = QAction('Toggle Toolbar', self)
        ToolbarTB.triggered.connect(self.ToggleTBFunction)
        self.Toolbarmenu.addAction(ToolbarTB)

        self.extraMenu = self.menuPart.addMenu('Extras')

        batchConvertAction = QtWidgets.QAction('Batch Convert', self)
        batchConvertAction.triggered.connect(self.open_batch_convert_window)
        self.extraMenu.addAction(batchConvertAction)

        self.exportCSVAction = QtWidgets.QAction('Export file list as CSV', self)
        self.exportCSVAction.setEnabled(False)
        self.exportCSVAction.triggered.connect(self.export_file_list)
        self.extraMenu.addAction(self.exportCSVAction)

        self.clearCacheButton = QtWidgets.QAction('Clear Cache', self)
        self.clearCacheButton.triggered.connect(self.clearCacheFunction)
        self.extraMenu.addAction(self.clearCacheButton)

        self.aboutMenu = self.menuPart.addMenu('About')

        manualB = QAction('Manual', self)
        manualB.triggered.connect(self.showManual)
        self.aboutMenu.addAction(manualB)

        aboutB = QAction('About', self)
        aboutB.triggered.connect(self.aboutFunction)
        self.aboutMenu.addAction(aboutB)

        self.ListFilesButton = QtWidgets.QPushButton(self)
        self.ListFilesButton.setText('File List')
        self.ListFilesButton.setToolTip("Select a file when a folder is loaded")
        self.ListFilesButton.setEnabled(False)
        self.toolbar.addWidget(self.ListFilesButton)
        
        self.fileMenu = QtWidgets.QMenu(self)
        self.ListFilesButton.setMenu(self.fileMenu)
        self.fileMenu.setStyleSheet("QMenu { menu-scrollable: 1; }")

        self.prevAction = QtWidgets.QPushButton('Previous', self)
        self.prevAction.setToolTip("Go to the previous file")
        self.prevAction.clicked.connect(self.prev_file)
        self.prevAction.setEnabled(False)
        self.toolbar.addWidget(self.prevAction)

        self.nextAction = QtWidgets.QPushButton('Next', self)
        self.nextAction.setToolTip("Go to the next file")
        self.nextAction.clicked.connect(self.next_file)
        self.nextAction.setEnabled(False)
        self.toolbar.addWidget(self.nextAction)

        self.viewMenuButton = QtWidgets.QPushButton(self)
        self.viewMenuButton.setText("View")
        self.viewMenuButton.setToolTip("View options")

        self.viewMenu = QtWidgets.QMenu(self)
        self.viewMenuButton.setMenu(self.viewMenu)
  
        self.resetViewAction = QtWidgets.QAction('Reset View', self)
        self.resetViewAction.triggered.connect(self.reset_view)
        self.resetViewAction.setEnabled(False)
        self.viewMenu.addAction(self.resetViewAction)

        self.exploreModeAction = QtWidgets.QAction('Explore Mode (free rotate)', self)
        self.exploreModeAction.setToolTip('Off by default: the model can only be repositioned via the Cut Analysis rotate buttons. Turn this on to freely orbit the camera for casual viewing.')
        self.exploreModeAction.setCheckable(True)
        self.exploreModeAction.setChecked(False)
        self.exploreModeAction.triggered.connect(self.toggle_explore_mode)
        self.viewMenu.addAction(self.exploreModeAction)

        logo_action = QtWidgets.QAction('Toggle Logo', self)
        logo_action.triggered.connect(self.toggle_logo_visibility)
        self.viewMenu.addAction(logo_action)

        # <-- CHANGED: QtWidgets.qApp is removed in newer PyQt5; use QApplication.instance() instead.
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.fullScreenAction = QtWidgets.QAction('Full Screen', self)
        self.fullScreenAction.triggered.connect(self.enter_full_screen)
        self.fullScreenAction.setEnabled(False)
        self.viewMenu.addAction(self.fullScreenAction)

        self.xrayAction = QtWidgets.QAction('X-Ray', self)
        self.xrayAction.triggered.connect(self.objects_opacity)
        self.xrayAction.setEnabled(False)
        self.viewMenu.addAction(self.xrayAction)

        self.lightAction = QtWidgets.QAction('Light', self)
        self.lightAction.setEnabled(False)
        self.lightAction.triggered.connect(self.open_light_dialog)
        self.viewMenu.addAction(self.lightAction)

        self.labelButton = QtWidgets.QAction('Label Options',self)
        self.labelButton.triggered.connect(self.openLabelDialog)
        self.labelButton.setEnabled(False)
        self.viewMenu.addAction(self.labelButton)

        self.objectColorMenuButton = QtWidgets.QToolButton(self)
        self.objectColorMenuButton.setText("Object's Color")
        self.objectColorMenuButton.setEnabled(False)
        
        self.objectColorMenu = QtWidgets.QMenu(self)
        self.objectColorMenuButton.setMenu(self.objectColorMenu)
        self.objectColorMenuButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        
        objectColorAction = QtWidgets.QWidgetAction(self)
        objectColorAction.setDefaultWidget(self.objectColorMenuButton)
        self.viewMenu.addAction(objectColorAction)

        color_group = QtWidgets.QActionGroup(self)

        colorRedAction = QtWidgets.QAction('Red', self)
        colorRedAction.triggered.connect(self.color_red)
        colorRedAction.setCheckable(True)
        self.objectColorMenu.addAction(colorRedAction)
        color_group.addAction(colorRedAction)

        colorBlackAction = QtWidgets.QAction('Black', self)
        colorBlackAction.triggered.connect(self.color_black)
        colorBlackAction.setCheckable(True)
        self.objectColorMenu.addAction(colorBlackAction)
        color_group.addAction(colorBlackAction)

        colorGrayAction = QtWidgets.QAction('Gray', self)
        colorGrayAction.triggered.connect(self.color_gray)
        colorGrayAction.setCheckable(True)
        self.objectColorMenu.addAction(colorGrayAction)
        color_group.addAction(colorGrayAction)

        colorBlueAction = QtWidgets.QAction('Blue', self)
        colorBlueAction.triggered.connect(self.color_blue)
        colorBlueAction.setCheckable(True)
        self.objectColorMenu.addAction(colorBlueAction)
        color_group.addAction(colorBlueAction)

        colorGoldAction = QtWidgets.QAction('Gold', self)
        colorGoldAction.triggered.connect(self.color_gold)
        colorGoldAction.setCheckable(True)
        self.objectColorMenu.addAction(colorGoldAction)
        color_group.addAction(colorGoldAction)

        colorOrangeAction = QtWidgets.QAction('Orange', self)
        colorOrangeAction.triggered.connect(self.color_orange)
        colorOrangeAction.setCheckable(True)
        self.objectColorMenu.addAction(colorOrangeAction)
        color_group.addAction(colorOrangeAction)

        self.resetColorAction = QtWidgets.QAction('Reset', self)
        self.resetColorAction.triggered.connect(self.reset_color)
        self.resetColorAction.setCheckable(True)
        self.resetColorAction.setChecked(True)
        self.objectColorMenu.addAction(self.resetColorAction)
        color_group.addAction(self.resetColorAction)

        self.autoRotateButton = QtWidgets.QToolButton(self)
        self.autoRotateButton.setText("Auto Rotate")
        self.autoRotateButton.setEnabled(False)

        self.autoRotateMenu = QtWidgets.QMenu(self)
        self.autoRotateButton.setMenu(self.autoRotateMenu)
        self.autoRotateButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        autoRotateAction = QtWidgets.QWidgetAction(self)
        autoRotateAction.setDefaultWidget(self.autoRotateButton)
        self.viewMenu.addAction(autoRotateAction)

        rotation_group = QtWidgets.QActionGroup(self)
        self.rotateAction = QtWidgets.QAction('Rotate', self)
        self.rotateAction.triggered.connect(self.start_rotation)
        self.rotateAction.setCheckable(True)
        rotation_group.addAction(self.rotateAction)

        self.stopAction = QtWidgets.QAction('Stop', self)
        self.stopAction.triggered.connect(self.stop_rotation)
        self.stopAction.setCheckable(True)
        self.stopAction.setChecked(True)
        rotation_group.addAction(self.stopAction)
        self.autoRotateMenu.addAction(self.rotateAction)
        self.autoRotateMenu.addAction(self.stopAction)

        bkg_color_menu = QtWidgets.QMenu(self)
        bkg_color_group = QtWidgets.QActionGroup(self)

        colors = {'Black': (0, 0, 0), 'White': (1, 1, 1), 'Grey': (0.5, 0.5, 0.5), 'Green': (0, 1, 0), 'Blue': (0, 0, 1), 'Red': (1, 0, 0), 'Orange': (1, 0.5, 0)}   
        for color_name, color_value in colors.items():
            color_action = QtWidgets.QAction(color_name, self)
            color_action.setCheckable(True)
            color_action.triggered.connect(lambda checked, color=color_value: self.change_background_color(color))
            bkg_color_menu.addAction(color_action)
            bkg_color_group.addAction(color_action)

            if color_name == 'White':
                color_action.setChecked(True)

        bkg_color_button = QtWidgets.QToolButton(self)
        bkg_color_button.setText('Background Color')
        bkg_color_button.setMenu(bkg_color_menu)
        bkg_color_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        colorAction = QtWidgets.QWidgetAction(self)
        colorAction.setDefaultWidget(bkg_color_button)
        self.viewMenu.addAction(colorAction)

        self.toolbar.addWidget(self.viewMenuButton)    

        self.capture_action = QtWidgets.QPushButton('Capture', self)
        self.capture_action.setToolTip("Capture an image of the current object")
        self.capture_action.clicked.connect(self.capture)
        self.capture_action.setEnabled(False)
        self.toolbar.addWidget(self.capture_action)

        # <-- ADDED: Roll buttons. These spin the view about the screen axis
        # (clockwise / counter-clockwise). They auto-repeat, so holding one
        # down rolls continuously. A guaranteed-working alternative to the
        # Option+drag gesture (handy because macOS reserves Ctrl+click).
        self.rollLeftButton = QtWidgets.QPushButton('\u21BA', self)  # ↺
        self.rollLeftButton.setToolTip("Roll the view counter-clockwise")
        self.rollLeftButton.setAutoRepeat(True)
        self.rollLeftButton.setEnabled(False)
        self.rollLeftButton.clicked.connect(self.roll_view_left)
        self.toolbar.addWidget(self.rollLeftButton)

        self.rollRightButton = QtWidgets.QPushButton('\u21BB', self)  # ↻
        self.rollRightButton.setToolTip("Roll the view clockwise")
        self.rollRightButton.setAutoRepeat(True)
        self.rollRightButton.setEnabled(False)
        self.rollRightButton.clicked.connect(self.roll_view_right)
        self.toolbar.addWidget(self.rollRightButton)

        self.otherFunctionsButton = QtWidgets.QPushButton('Other Functions', self)
        self.otherFunctionsButton.setEnabled(False)
        self.otherFunctionsButton.clicked.connect(self.otherFunctionsToggleFunction)
        self.toolbar.addWidget(self.otherFunctionsButton)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        self.fileTextLabel = QtWidgets.QLabel('File Name:')
        self.toolbar.addWidget(self.fileTextLabel)
        self.fileNameLineEdit = QtWidgets.QLineEdit()
        self.fileNameLineEdit.setReadOnly(True)
        self.fileNameLineEdit.setStyleSheet("background-color: white")
        self.fileNameLineEdit.setMaximumWidth(100)
        self.toolbar.addWidget(self.fileNameLineEdit)

        self.textureMenuButton = QtWidgets.QPushButton(self)
        self.textureMenuButton.setText("Texture")
        self.textureMenuButton.setToolTip("Modify the file's texture")

        self.textureMenu = QtWidgets.QMenu(self)
        self.textureMenuButton.setMenu(self.textureMenu)

        self.textureAction = QtWidgets.QAction('Load Texture', self)
        self.textureAction.triggered.connect(self.apply_texture)
        self.textureAction.setEnabled(False)
        self.textureMenu.addAction(self.textureAction)

        self.reloadTextureAction = QtWidgets.QAction('Reload WRL With Embedded Texture', self)
        self.reloadTextureAction.triggered.connect(self.reload_with_texture)
        self.reloadTextureAction.setEnabled(False)
        self.textureMenu.addAction(self.reloadTextureAction)

        self.removeTextureAction = QtWidgets.QAction('De-Texture PLY', self)
        self.removeTextureAction.triggered.connect(self.remove_texture)
        self.removeTextureAction.setEnabled(False)       
        self.textureMenu.addAction(self.removeTextureAction)

        self.otherFunctionsToolbar.addWidget(self.textureMenuButton)

        self.points_action = QtWidgets.QPushButton('Points', self)
        self.points_action.setToolTip("Show/Hide points")
        self.points_action.clicked.connect(self.toggle_points_visibility)
        self.points_action.setEnabled(False)
        self.otherFunctionsToolbar.addWidget(self.points_action)

        self.wireframe_action = QtWidgets.QPushButton('Mesh', self)
        self.wireframe_action.setToolTip("Show/Hide wireframe")
        self.wireframe_action.clicked.connect(self.toggle_wireframe_visibility)
        self.wireframe_action.setEnabled(False)
        self.otherFunctionsToolbar.addWidget(self.wireframe_action)

        self.ProportionsMenuButton = QtWidgets.QPushButton(self)
        self.ProportionsMenuButton.setText("Proportions")
        self.ProportionsMenuButton.setToolTip("Set dimensions or scales")

        self.ProportionsMenu = QtWidgets.QMenu(self)
        self.ProportionsMenuButton.setMenu(self.ProportionsMenu)

        createLineButton = QtWidgets.QAction('Line Scale', self)
        createLineButton.triggered.connect(self.lineScaleDialogOpen)
        self.ProportionsMenu.addAction(createLineButton)

        self.ScaleButton = QtWidgets.QAction('Bar Scale', self)
        self.ScaleButton.setEnabled(False)
        self.ScaleButton.triggered.connect(self.openScaleDialog)
        self.ProportionsMenu.addAction(self.ScaleButton)

        self.boundaryOptionsAction = QtWidgets.QAction('Boundary Options', self)
        self.boundaryOptionsAction.setEnabled(False)
        self.boundaryOptionsAction.triggered.connect(self.openBoundShiftDialog)
        self.ProportionsMenu.addAction(self.boundaryOptionsAction)

        self.manualMeasureMenuButton = QtWidgets.QToolButton(self)
        self.manualMeasureMenuButton.setText("Manual Measure")

        self.manualMeasureMenu = QtWidgets.QMenu(self)
        self.manualMeasureMenuButton.setMenu(self.manualMeasureMenu)
        self.manualMeasureMenuButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.manualMeasureButton = QtWidgets.QAction("Measure", self)
        self.manualMeasureButton.triggered.connect(self.createManualMeasurment)
        self.manualMeasureMenu.addAction(self.manualMeasureButton)

        self.removeManualMeasureAction = QtWidgets.QAction("Remove", self)
        self.removeManualMeasureAction.triggered.connect(self.removeManualMeasure)
        self.manualMeasureMenu.addAction(self.removeManualMeasureAction)

        self.manualMeasureAction = QtWidgets.QWidgetAction(self)
        self.manualMeasureAction.setDefaultWidget(self.manualMeasureMenuButton)

        self.ProportionsMenu.addAction(self.manualMeasureAction)

        self.DimensionsButton = QtWidgets.QToolButton(self)
        self.DimensionsButton.setText("Dimensions")
        
        self.DimensionsMenu = QtWidgets.QMenu(self)
        self.DimensionsButton.setMenu(self.DimensionsMenu)
        self.DimensionsButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.toggleDimensionsAction = QtWidgets.QAction('Toggle All Axis', self)
        self.toggleDimensionsAction.triggered.connect(self.toggle_dimensions)
        self.toggleDimensionsAction.setCheckable(True)
        self.DimensionsMenu.addAction(self.toggleDimensionsAction)   

        self.toggleXAxisAction = QtWidgets.QAction('Toggle X Axis', self)
        self.toggleXAxisAction.setCheckable(True)
        self.toggleXAxisAction.setChecked(True)
        self.toggleXAxisAction.setEnabled(False)
        self.toggleXAxisAction.triggered.connect(self.toggle_x_axis)
        self.DimensionsMenu.addAction(self.toggleXAxisAction)
        
        self.toggleYAxisAction = QtWidgets.QAction('Toggle Y Axis', self)
        self.toggleYAxisAction.setCheckable(True)
        self.toggleYAxisAction.setChecked(True)
        self.toggleYAxisAction.setEnabled(False)
        self.toggleYAxisAction.triggered.connect(self.toggle_y_axis)
        self.DimensionsMenu.addAction(self.toggleYAxisAction)
        
        self.toggleZAxisAction = QtWidgets.QAction('Toggle Z Axis', self)
        self.toggleZAxisAction.setCheckable(True)
        self.toggleZAxisAction.setChecked(True)
        self.toggleZAxisAction.setEnabled(False)
        self.toggleZAxisAction.triggered.connect(self.toggle_z_axis)    
        self.DimensionsMenu.addAction(self.toggleZAxisAction)

        self.DimensionsAction = QtWidgets.QWidgetAction(self)
        self.DimensionsAction.setEnabled(False) 
        self.DimensionsAction.setDefaultWidget(self.DimensionsButton)
        self.ProportionsMenu.addAction(self.DimensionsAction)
                              
        self.otherFunctionsToolbar.addWidget(self.ProportionsMenuButton) 

        # <-- CHANGED: the Analysis menu used to be a QPushButton on the Other
        # Functions toolbar. It's now a top-level menu in the menu bar (added
        # further down via insertMenu). We keep just the QMenu here.
        self.AnalysisMenu = QtWidgets.QMenu(self)

        # <-- CHANGED: Center of Mass used to be a QToolButton wrapped in a
        # QWidgetAction (works on a toolbar, breaks in a menu bar). Now it's a
        # proper nested QMenu, attached via addMenu().
        self.centerOfMassMenu = QtWidgets.QMenu("Center of Mass", self)
        self.center_of_mass_actor = None

        self.centerOfMassButton = QtWidgets.QAction('Toggle CoM', self)
        self.centerOfMassButton.triggered.connect(self.toggle_center_of_mass)
        self.centerOfMassMenu.addAction(self.centerOfMassButton)

        self.increaseCenterOfMassButton = QtWidgets.QAction('Increase CoM', self)
        self.increaseCenterOfMassButton.triggered.connect(self.increaseCenterOfMass)
        self.centerOfMassMenu.addAction(self.increaseCenterOfMassButton)
        self.increaseCenterOfMassButton.setEnabled(False)

        self.decreaseCenterOfMassButton = QtWidgets.QAction('Decrease CoM', self)
        self.decreaseCenterOfMassButton.triggered.connect(self.decreaseCenterOfMass)
        self.centerOfMassMenu.addAction(self.decreaseCenterOfMassButton)
        self.decreaseCenterOfMassButton.setEnabled(False)

        # menuAction() is the QAction that, when triggered, opens this submenu.
        # Disabling it greys out the submenu entry in its parent menu — so the
        # existing self.centerOfMassMenuButton.setEnabled(...) calls keep working.
        self.AnalysisMenu.addMenu(self.centerOfMassMenu)
        self.centerOfMassMenuButton = self.centerOfMassMenu.menuAction()
        self.centerOfMassMenuButton.setEnabled(False)

        # <-- CHANGED: Volume/Weight, same fix as Center of Mass — proper nested QMenu.
        self.volumeMenu = QtWidgets.QMenu("Volume/Weight", self)

        self.VolumeFindAction = QtWidgets.QAction('Find Volume', self)
        self.VolumeFindAction.triggered.connect(lambda: self.calculate_volume(show_dialog=True))
        self.volumeMenu.addAction(self.VolumeFindAction)

        self.WeightInGoldAction = QtWidgets.QAction('Weight in Gold', self)
        self.WeightInGoldAction.triggered.connect(self.calculate_Gold)
        self.volumeMenu.addAction(self.WeightInGoldAction)
        
        self.WeightInSilverAction = QtWidgets.QAction('Weight in Silver', self)
        self.WeightInSilverAction.triggered.connect(self.calculate_Silver)
        self.volumeMenu.addAction(self.WeightInSilverAction)

        self.WeightInIronAction = QtWidgets.QAction('Weight in Iron', self)
        self.WeightInIronAction.triggered.connect(self.calculate_Iron)
        self.volumeMenu.addAction(self.WeightInIronAction)

        self.WeightInBronzeAction = QtWidgets.QAction('Weight in Bronze', self)
        self.WeightInBronzeAction.triggered.connect(self.calculate_Bronze)
        self.volumeMenu.addAction(self.WeightInBronzeAction)

        self.WeightInClayAction = QtWidgets.QAction('Weight in Clay', self)
        self.WeightInClayAction.triggered.connect(self.calculate_Clay)
        self.volumeMenu.addAction(self.WeightInClayAction)

        self.WeightInObsidianAction = QtWidgets.QAction('Weight in Obsidian', self)
        self.WeightInObsidianAction.triggered.connect(self.calculate_Obsidian)
        self.volumeMenu.addAction(self.WeightInObsidianAction)

        self.WeightInBoneAction = QtWidgets.QAction('Weight in Bone', self)
        self.WeightInBoneAction.triggered.connect(self.calculate_Bone)
        self.volumeMenu.addAction(self.WeightInBoneAction)

        self.WeightInFlintAction = QtWidgets.QAction('Weight in Flint', self)
        self.WeightInFlintAction.triggered.connect(self.calculate_Flint)
        self.volumeMenu.addAction(self.WeightInFlintAction)

        self.WeightInIvoryAction = QtWidgets.QAction('Weight in Ivory', self)
        self.WeightInIvoryAction.triggered.connect(self.calculate_Ivory)
        self.volumeMenu.addAction(self.WeightInIvoryAction)

        # Same pattern as Center of Mass: keep self.VolumeWeightButton as the
        # menu's display action so existing setEnabled() calls keep working.
        self.AnalysisMenu.addMenu(self.volumeMenu)
        self.VolumeWeightButton = self.volumeMenu.menuAction()
        self.VolumeWeightButton.setEnabled(False)

        # <-- ADDED: Crest detection submenu (lives in crest_detection.py).
# <-- ADDED: Crest detection submenu (lives in crest_detection.py).
        self.crest_detector = CrestDetector(self)
        self.crest_detector.add_to_menu(self.AnalysisMenu)
        self.mesh_smoother = MeshSmootherTool(self)
        self.mesh_smoother.add_to_menu(self.AnalysisMenu)


        # <-- ADDED: Mesh smoothing preview tool (lives in mesh_smoothing.py).
        self.mesh_smoother = MeshSmootherTool(self)
        self.mesh_smoother.add_to_menu(self.AnalysisMenu)
        # <-- CHANGED: instead of adding the Analysis menu to the (hidden by default)
        # Other Functions toolbar, expose it as a top-level menu in the menu bar so
        # crest detection and other analysis tools are always one click away.
        # Inserted before "About" so menu order reads: File / Toolbar / Extras / Analysis / About.
        self.AnalysisMenu.setTitle("Analysis")
        self.menuPart.insertMenu(self.aboutMenu.menuAction(), self.AnalysisMenu)

        # Cut Analysis: Cut3-D's position -> select points -> slice -> analyze pipeline.
        # Everything for positioning (rotate buttons, Align by Normal, Save/Load
        # Position) lives in ONE place -- the Position Model panel -- rather than
        # being duplicated between this menu and the panel.
        self.CutAnalysisMenu = QtWidgets.QMenu(self)

        self.positionModelAction = QtWidgets.QAction('Position Model', self)
        self.positionModelAction.triggered.connect(self.openPositioningDialog)
        self.positionModelAction.setEnabled(False)
        self.CutAnalysisMenu.addAction(self.positionModelAction)

        self.CutAnalysisMenu.setTitle("Cut Analysis")
        self.menuPart.insertMenu(self.aboutMenu.menuAction(), self.CutAnalysisMenu)

        cutSectionButton = QtWidgets.QPushButton('Cut Section', self)
        cutSectionButton.clicked.connect(self.toggleCutToolbar)
        self.otherFunctionsToolbar.addWidget(cutSectionButton)      

        self.createCutButton = QtWidgets.QPushButton('Create Cut')
        self.createCutButton.setEnabled(True)
        self.createCutButton.clicked.connect(self.addCutPlane)
        self.cutToolbar.addWidget(self.createCutButton)

        self.cutButton = QtWidgets.QPushButton('Cut')
        self.cutButton.setEnabled(False)
        self.cutButton.clicked.connect(self.cutHere)
        self.cutToolbar.addWidget(self.cutButton)

        self.toggleCutPlaneButton = QtWidgets.QPushButton('Toggle Plane')
        self.toggleCutPlaneButton.setEnabled(False)
        self.toggleCutPlaneButton.clicked.connect(self.toggleCuttingPlane)
        self.cutToolbar.addWidget(self.toggleCutPlaneButton)

        self.switchAxisButton = QtWidgets.QPushButton('Switch Axis')
        self.switchAxisButton.setEnabled(False)
        self.switchAxisButton.clicked.connect(self.switchCutAxis)
        self.cutToolbar.addWidget(self.switchAxisButton)

        self.calculateCircumferenceButton = QtWidgets.QPushButton('Circumference')
        self.calculateCircumferenceButton.setEnabled(False)
        self.calculateCircumferenceButton.clicked.connect(self.calculateCircumference)
        self.cutToolbar.addWidget(self.calculateCircumferenceButton)
        self.calculateCircumferenceButton.setToolTip("Available after a cut is created")

        self.toggleCutObjectButton = QtWidgets.QPushButton('Toggle Object')
        self.toggleCutObjectButton.setEnabled(False)
        self.toggleCutObjectButton.clicked.connect(self.toggleCutObjectVisibility)
        self.cutToolbar.addWidget(self.toggleCutObjectButton)

        self.resetObjectAfterCutButton = QtWidgets.QPushButton('Reset')
        self.resetObjectAfterCutButton.clicked.connect(self.resetCutToDefault)
        self.cutToolbar.addWidget(self.resetObjectAfterCutButton)

        self.show()

        self.iren.Initialize()

    def open_file(self, fileName=None, delete_after_use=False, wrlRefreshName=False, newFileLoad=True, plyWOTexture=True, dropToExe=False):
        if not fileName:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select your file', self.last_dir, '3D Files (*.stl *.wrl *.ply *.obj)')
            if fileName != '':
                self.fileInfo = fileName
        
        self.WrlPlyName = fileName
        
        if dropToExe:
            self.fileInfo = fileName

        # <-- CHANGED: cross-platform cache directory.
        dir_path = get_cache_dir()

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not fileName:
            return

        if self.otherFunctionsToolbar.isVisible and newFileLoad==True:
            self.otherFunctionsToolbar.hide()
        
        self.resetCutToDefault()
        self.removeCutObject(True)
        
        if fileName:

            self.last_dir = os.path.dirname(fileName)

            self.loading_dialog = LoadingDialog(self)
            self.loading_dialog.show()
            QtWidgets.QApplication.processEvents()

            baseName = os.path.basename(self.fileInfo)
            self.fileNameLineEdit.setText(baseName)

            width = self.fileNameLineEdit.fontMetrics().boundingRect(baseName).width()
            self.fileNameLineEdit.setMaximumWidth(width + 10)

            self.current_fileName = os.path.basename(self.fileInfo)

            self.current_file = self.fileInfo

        self.ren.RemoveAllViewProps()

        if hasattr(self, 'lineScaleActor') and hasattr(self, 'lineScaleTextActor'):
            self.ren.RemoveActor2D(self.lineScaleActor)
            self.ren.RemoveActor2D(self.lineScaleTextActor)
            del self.lineScaleActor
            del self.lineScaleTextActor
            
            self.vtkWidget.GetRenderWindow().Render()

        self.logo_visible = True
        self.ren.AddActor2D(self.logo_actor)

        reader = None

        if fileName.endswith('.stl'):
            reader = vtk.vtkSTLReader()
            self.reloadTextureAction.setEnabled(False)  
            self.objectColorMenuButton.setEnabled(True)
            self.removeTextureAction.setEnabled(False)
            self.textureAction.setEnabled(False)

        elif fileName.endswith('.ply'):
            reader = vtk.vtkPLYReader()
            self.reloadTextureAction.setEnabled(False)  
            self.objectColorMenuButton.setEnabled(False)
            
            if plyWOTexture == True:
                self.removeTextureAction.setEnabled(True)                      
            self.textureAction.setEnabled(False)
        
        elif fileName.endswith('.obj'):
            self.reloadTextureAction.setEnabled(False)

            if wrlRefreshName == True:
                self.fileInfo = fileName
            
            reader = vtk.vtkOBJReader()
            self.objectColorMenuButton.setEnabled(True)
            self.removeTextureAction.setEnabled(False)                              
            self.textureAction.setEnabled(True)

        elif fileName.endswith('.wrl'):

            self.fileInfo = fileName
    
            importer = vtk.vtkVRMLImporter()
            importer.SetFileName(fileName)
            importer.SetRenderWindow(self.vtkWidget.GetRenderWindow())
            self.objectColorMenuButton.setEnabled(True)
            
            importer.Update()

            self.capture_action.setEnabled(True)            
            self.rollLeftButton.setEnabled(True)   # <-- ADDED
            self.rollRightButton.setEnabled(True)  # <-- ADDED
            self.autoRotateButton.setEnabled(True)
            self.fullScreenAction.setEnabled(True)
            self.resetViewAction.setEnabled(True)
            self.labelButton.setEnabled(True)
            self.stopAction.setEnabled(False)
            self.ListFilesButton.setEnabled(False)
            self.prevAction.setEnabled(False)
            self.nextAction.setEnabled(False)
            self.xrayAction.setEnabled(True)
            self.exportAction.setEnabled(True)
            self.lightAction.setEnabled(True)
            self.otherFunctionsButton.setEnabled(True)
            self.exportCSVAction.setEnabled(False)
            self.aboutCurrentFileAction.setEnabled(True)
            self.reduceSizeAction.setEnabled(True)
            self.mesh_smoother.on_file_loaded()
            self.positionModelAction.setEnabled(True)

            self.stop_rotation()

            self.stopAction.setChecked(True)

            # WRL files can import as more than one actor (VRML shapes/materials);
            # share one transform across all of them so positioning moves the
            # whole model together, same contract as the reader branch below.
            self.transform = vtk.vtkTransform()

            actors = importer.GetRenderer().GetActors()
            actors.InitTraversal()
            for i in range(actors.GetNumberOfItems()):
                actor = actors.GetNextActor()
                actor.SetUserTransform(self.transform)

                if hasattr(self, 'current_actor'):
                    self.ren.RemoveActor(self.current_actor)

                self.ren.AddActor(actor)
                self.current_actor = actor
                self.actor = actor

                self.current_actor.GetProperty().SetColor(0.8, 0.8, 0.8)

                self.loading_dialog.close()

                self.ren.ResetCamera()

                self.vtkWidget.GetRenderWindow().Render()

        if reader is not None:

            reader.SetFileName(fileName)

            self.capture_action.setEnabled(True)            
            self.rollLeftButton.setEnabled(True)   # <-- ADDED
            self.rollRightButton.setEnabled(True)  # <-- ADDED
            self.points_action.setEnabled(True)  
            self.wireframe_action.setEnabled(True)
            self.autoRotateButton.setEnabled(True)
            self.DimensionsAction.setEnabled(True)
            self.centerOfMassMenuButton.setEnabled(True)
            self.decreaseCenterOfMassButton.setEnabled(False)
            self.increaseCenterOfMassButton.setEnabled(False)
            self.VolumeWeightButton.setEnabled(True)
            # <-- ADDED: let the crest detector know a new mesh is loaded so it
            # enables its menu and clears any leftover crests from the prior file.
            self.crest_detector.on_file_loaded()
            self.mesh_smoother.on_file_loaded()
            self.fullScreenAction.setEnabled(True)
            self.resetViewAction.setEnabled(True)
            self.ScaleButton.setEnabled(True)
            self.labelButton.setEnabled(True)
            self.stopAction.setEnabled(False)
            self.ListFilesButton.setEnabled(False)
            self.prevAction.setEnabled(False)
            self.nextAction.setEnabled(False)
            self.xrayAction.setEnabled(True) 
            self.exportAction.setEnabled(True)
            self.lightAction.setEnabled(True)
            self.otherFunctionsButton.setEnabled(True)
            self.exportCSVAction.setEnabled(False)   
            self.aboutCurrentFileAction.setEnabled(True)       
            self.reduceSizeAction.setEnabled(True)   
            self.boundaryOptionsAction.setEnabled(True)
            self.mesh_smoother.on_file_loaded()
            self.positionModelAction.setEnabled(True)

            self.resetColorAction.setChecked(True)

            self.toggleDimensionsAction.setChecked(False)

            self.toggleXAxisAction.setChecked(True)
            self.toggleYAxisAction.setChecked(True)
            self.toggleZAxisAction.setChecked(True)
            self.toggleXAxisAction.setEnabled(False)
            self.toggleYAxisAction.setEnabled(False)
            self.toggleZAxisAction.setEnabled(False)

            self.remove_scale()

            self.stopAction.setChecked(True)

            self.stop_rotation()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            
            if hasattr(self, 'current_actor'):
                self.ren.RemoveActor(self.current_actor)

            self.ren.AddActor(self.actor)
            self.current_actor = self.actor

            points_filter = vtk.vtkVertexGlyphFilter()
            points_filter.SetInputConnection(reader.GetOutputPort())
        
            points_mapper = vtk.vtkPolyDataMapper()
            points_mapper.SetInputConnection(points_filter.GetOutputPort())
        
            self.points_actor = vtk.vtkActor()
            self.points_actor.SetMapper(points_mapper)

            self.add_bounding_box(self.actor)

            self.points_actor.GetProperty().SetColor(0, 0, 0)
            self.points_actor.GetProperty().SetPointSize(2)

            wireframe_filter = vtk.vtkDataSetSurfaceFilter()
            wireframe_filter.SetInputConnection(reader.GetOutputPort())
         
            wireframe_mapper = vtk.vtkPolyDataMapper()
            wireframe_mapper.SetInputConnection(wireframe_filter.GetOutputPort())
         
            self.wireframe_actor = vtk.vtkActor()
            self.wireframe_actor.SetMapper(wireframe_mapper)
         
            self.wireframe_actor.GetProperty().SetColor(0, 0, 0)
            self.wireframe_actor.GetProperty().SetLineWidth(1)
            self.wireframe_actor.GetProperty().SetRepresentationToWireframe()

            self.opacity_index = 0
            
            self.loading_dialog.close()
            
            self.vtkWidget.GetRenderWindow().Render()
            self.ren.ResetCamera()

            self.outline = vtk.vtkOutlineFilter()
            self.outline.SetInputConnection(reader.GetOutputPort())

            outlineMapper = vtk.vtkPolyDataMapper()
            outlineMapper.SetInputConnection(self.outline.GetOutputPort())

            self.outlineActor = vtk.vtkActor()
            self.outlineActor.SetMapper(outlineMapper)
            self.outlineActor.GetProperty().SetColor(1,0,0)

            self.transform = vtk.vtkTransform()

            self.actor.SetUserTransform(self.transform)

        if delete_after_use:
            os.remove(fileName)


    def capture(self):
        screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.vtkWidget.winId())

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(None,"Save Capture",f"Capture {self.image_counter}","PNG Files (*.png);;JPG Files (*.jpg);;TIFF Files (*.tif)", options=options)

        if fileName:
            fileFormat = 'PNG'
            if '.jpg' in fileName:
                fileFormat = 'JPG'
            elif '.tif' in fileName:
                fileFormat = 'TIFF'

            screenshot.save(fileName, fileFormat)

            self.image_counter += 1


    def change_background_color(self, color):
      self.ren.SetBackground(color)
      self.vtkWidget.GetRenderWindow().Render()


    def toggle_explore_mode(self, checked):
        self.explore_mode = checked

    def toggle_logo_visibility(self):
        self.logo_visible = not self.logo_visible
        if self.logo_visible:
            self.ren.AddActor2D(self.logo_actor)
        else:
            self.ren.RemoveActor2D(self.logo_actor)

        self.vtkWidget.GetRenderWindow().Render()


    def objects_opacity(self):
        opacities = [0.5, 1.0]

        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetOpacity(opacities[self.opacity_index])
            self.opacity_index = (self.opacity_index + 1) % len(opacities)
            self.vtkWidget.GetRenderWindow().Render()

    def inverse_color(self, color):
        return tuple(1 - c for c in color)

    def toggle_points_visibility(self):
        if hasattr(self, 'cut_actor') or hasattr(self, 'clip_actor'):
            self.resetCutToDefault()

        self.points_visible = not self.points_visible

        if self.points_visible:
            self.ren.AddActor(self.points_actor)
            self.ren.RemoveActor(self.actor)
            self.current_actor.GetProperty().SetOpacity(0.0)

            bg_color = self.ren.GetBackground()
            inv_color = self.inverse_color(bg_color)
            self.points_actor.GetProperty().SetColor(inv_color)
        
        else:
            self.ren.RemoveActor(self.points_actor)
            self.ren.AddActor(self.actor)
            self.current_actor.GetProperty().SetOpacity(1.0)
        
        self.vtkWidget.GetRenderWindow().Render()

    def toggle_wireframe_visibility(self):
        if hasattr(self, 'cut_actor') or hasattr(self, 'clip_actor'):
            self.resetCutToDefault()

        self.wireframe_visible = not self.wireframe_visible
        
        if self.wireframe_visible:
            self.ren.AddActor(self.wireframe_actor)
            self.ren.RemoveActor(self.actor)
            self.current_actor.GetProperty().SetOpacity(0.0)
            
            bg_color = self.ren.GetBackground()
            inv_color = self.inverse_color(bg_color)
            self.wireframe_actor.GetProperty().SetColor(inv_color)
        
        else:
            self.ren.RemoveActor(self.wireframe_actor)
            if not self.points_visible:
                self.ren.AddActor(self.actor)
                self.current_actor.GetProperty().SetOpacity(1.0)

        self.vtkWidget.GetRenderWindow().Render()

    def keypress_event(self, obj, event):
        key = obj.GetKeySym()
        camera = self.ren.GetActiveCamera()
        ctrl_key = obj.GetControlKey()
        
        if ctrl_key:
            if key == "Left":
                camera.Roll(1)
            elif key == "Right":
                camera.Roll(-1)
        
        else: 
            if key == "Up":
                camera.Elevation(1)
            elif key == "Down":
                camera.Elevation(-1)
            elif key == "Left":
                camera.Azimuth(1)
            elif key == "Right":
                camera.Azimuth(-1)
        camera.OrthogonalizeViewUp()

        self.vtkWidget.GetRenderWindow().Render()
    
    def start_rotation(self):
        self.rotate_flag = True
        self.stopAction.setEnabled(True)
        self.timer.start(10)

    def rotate(self):
        if self.rotate_flag:
            self.ren.GetActiveCamera().Azimuth(0.3)
            self.vtkWidget.GetRenderWindow().Render()

    def stop_rotation(self):
        self.rotate_flag = False
        self.stopAction.setEnabled(False)

    def add_bounding_box(self, actor, BckgrChk=False):
        bounding_box = vtk.vtkCubeAxesActor()

        bounding_box.SetBounds(actor.GetBounds())

        bounds = actor.GetBounds()

        min_corner = [bounds[0], bounds[2], bounds[4]]

        bounding_box.SetXTitle("mm")
        bounding_box.SetYTitle("mm")
        bounding_box.SetZTitle("mm")

        bounding_box.SetFlyModeToOuterEdges()
        bounding_box.SetXAxisRange(0, bounds[1]-min_corner[0])
        bounding_box.SetYAxisRange(0, bounds[3]-min_corner[1])
        bounding_box.SetZAxisRange(0, bounds[5]-min_corner[2])
        
        bg_color = self.ren.GetBackground()
        inv_color = self.inverse_color(bg_color)

        bounding_box.GetTitleTextProperty(0).SetColor(inv_color)
        bounding_box.GetLabelTextProperty(0).SetColor(inv_color)
        bounding_box.GetTitleTextProperty(1).SetColor(inv_color)
        bounding_box.GetLabelTextProperty(1).SetColor(inv_color)
        bounding_box.GetTitleTextProperty(2).SetColor(inv_color)
        bounding_box.GetLabelTextProperty(2).SetColor(inv_color)
        bounding_box.GetXAxesLinesProperty().SetColor(inv_color)
        bounding_box.GetYAxesLinesProperty().SetColor(inv_color)
        bounding_box.GetZAxesLinesProperty().SetColor(inv_color)

        bounding_box.SetCamera(self.ren.GetActiveCamera())
        
        bounding_box.SetVisibility(False)

        self.ren.AddActor(bounding_box)
        self.bounding_box = bounding_box
    
    def toggle_dimensions(self):
        if hasattr(self, 'bounding_box'):
            visibility = not self.bounding_box.GetVisibility()
            self.bounding_box.SetVisibility(visibility)
            self.toggleXAxisAction.setEnabled(visibility)
            self.toggleYAxisAction.setEnabled(visibility)
            self.toggleZAxisAction.setEnabled(visibility)
            self.vtkWidget.GetRenderWindow().Render()

    def toggle_x_axis(self):
        if hasattr(self, 'bounding_box'):
            visibility = not self.bounding_box.GetXAxisVisibility()
            self.bounding_box.SetXAxisVisibility(visibility)
            self.bounding_box.SetXAxisLabelVisibility(visibility)
            self.vtkWidget.GetRenderWindow().Render()

    def toggle_y_axis(self):
        if hasattr(self, 'bounding_box'):
            visibility = not self.bounding_box.GetYAxisVisibility()
            self.bounding_box.SetYAxisVisibility(visibility)
            self.bounding_box.SetYAxisLabelVisibility(visibility)
            self.vtkWidget.GetRenderWindow().Render()

    def toggle_z_axis(self):
        if hasattr(self, 'bounding_box'):
            visibility = not self.bounding_box.GetZAxisVisibility()
            self.bounding_box.SetZAxisVisibility(visibility)
            self.bounding_box.SetZAxisLabelVisibility(visibility)
            self.vtkWidget.GetRenderWindow().Render() 

    def toggle_center_of_mass(self):
        if not hasattr(self, 'current_actor'):
            return

        if self.center_of_mass_actor:
            self.ren.RemoveActor(self.center_of_mass_actor)
            self.center_of_mass_actor = None

            self.current_actor.GetProperty().SetOpacity(1.0)

            self.increaseCenterOfMassButton.setEnabled(False)
            self.decreaseCenterOfMassButton.setEnabled(False)
        else:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.transform)
            transformFilter.SetInputData(self.current_actor.GetMapper().GetInput())
            transformFilter.Update()

            centerOfMassFilter = vtk.vtkCenterOfMass()
            centerOfMassFilter.SetInputData(transformFilter.GetOutput())
            centerOfMassFilter.SetUseScalarsAsWeights(False)
            centerOfMassFilter.Update()

            center = centerOfMassFilter.GetCenter()

            self.sphereSource = vtk.vtkSphereSource()
            self.sphereSource.SetCenter(center)

            self.coMDefaultSize = 0.5
            self.sphereSource.SetRadius(self.coMDefaultSize)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.sphereSource.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)

            self.ren.AddActor(actor)
            self.center_of_mass_actor = actor

            self.current_actor.GetProperty().SetOpacity(0.5)

            self.increaseCenterOfMassButton.setEnabled(True)
            self.decreaseCenterOfMassButton.setEnabled(True)

        self.vtkWidget.GetRenderWindow().Render()
    

    def increaseCenterOfMass (self):
        self.coMDefaultSize += 0.1
        print(self.coMDefaultSize)
        self.sphereSource.SetRadius(self.coMDefaultSize)
        self.vtkWidget.GetRenderWindow().Render()
        
    def decreaseCenterOfMass (self):
        self.coMDefaultSize -= 0.1
        print(self.coMDefaultSize)
        self.sphereSource.SetRadius(self.coMDefaultSize)
        self.vtkWidget.GetRenderWindow().Render()


    def calculate_volume(self, show_dialog=True):
        polydata = self.current_actor.GetMapper().GetInput()

        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(polydata)
        mass_properties.Update()

        if show_dialog:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))          
            msg.setText("The object's volume is {} mm³".format(mass_properties.GetVolume()))
            msg.setWindowTitle("Volume Information")
            msg.exec_()
        return mass_properties

    def calculate_Gold(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        GoldDensity = 19.3 / 1000
        WeightGold = volume * GoldDensity 
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in gold is {} grams".format(WeightGold))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightGold
    
    def calculate_Silver(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        SilverDensity = 10.49 / 1000
        WeightSilver = volume * SilverDensity
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))
        msg.setText("The object's estimated weight in silver is {} grams".format(WeightSilver))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightSilver
    
    def calculate_Iron(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        ironDensity = 7.874 / 1000
        WeightIron = volume * ironDensity
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))
        msg.setText("The object's estimated weight in iron is {} grams".format(WeightIron))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightIron
    
    def calculate_Bronze(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        bronzeDensity = 8.73 / 1000
        WeightBronze = volume * bronzeDensity 
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in bronze is {} grams".format(WeightBronze))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightBronze
    
    def calculate_Clay(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        clayDensity = 1.6 / 1000
        WeightClay = volume * clayDensity   
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in clay is {} grams".format(WeightClay))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightClay   
    
    def calculate_Obsidian(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        ObsidianDensity = 2.4 / 1000
        WeightObsidian = volume * ObsidianDensity
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in obsidian is {} grams".format(WeightObsidian))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightObsidian

    def calculate_Bone(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        BoneDensity = 1.5 / 1000
        WeightBone = volume * BoneDensity     
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in bone is {} grams".format(WeightBone))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightBone
    
    def calculate_Flint(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        FlintDensity = 2.71 / 1000
        WeightFlint = volume * FlintDensity 
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in flint is {} grams".format(WeightFlint))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightFlint
    
    def calculate_Ivory(self):
        volume = self.calculate_volume(show_dialog=False).GetVolume()
        IvoryDensity = 1.85 / 1000
        WeightIvory = volume * IvoryDensity       
        msg = QMessageBox()      
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png'))) 
        msg.setText("The object's estimated weight in ivory is {} grams".format(WeightIvory))
        msg.setWindowTitle("Weight Information")
        msg.exec_()
        return WeightIvory   

    def apply_texture(self):
        textureFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Texture File', '', 'Image Files (*.jpg *.png)')

        if textureFileName:
            if textureFileName.lower().endswith('.jpg'):
                reader = vtkJPEGReader()
            elif textureFileName.lower().endswith('.png'):
                reader = vtkPNGReader()                  
            else:
                return
            reader.SetFileName(textureFileName)
            reader.Update()

            if reader.GetOutput().GetNumberOfCells() == 0:
                return
            texture = vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
            texture.Update()

            if texture.GetInput().GetNumberOfCells() == 0:
                return
            if hasattr(self, 'current_actor'):
                self.current_actor.SetTexture(texture)
                self.current_actor.Modified()
                self.vtkWidget.GetRenderWindow().Render()
    
    def drag_texture(self, textureFileName):

        if textureFileName:
            if textureFileName.lower().endswith('.jpg'):
                reader = vtkJPEGReader()
            elif textureFileName.lower().endswith('.png'):
                reader = vtkPNGReader()    
            else:
                return
            reader.SetFileName(textureFileName)
            reader.Update()

            if reader.GetOutput().GetNumberOfCells() == 0:
                return
            texture = vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
            texture.Update()

            if texture.GetInput().GetNumberOfCells() == 0:
                return
            if hasattr(self, 'current_actor'):
                self.current_actor.SetTexture(texture)
                self.current_actor.Modified()
                self.vtkWidget.GetRenderWindow().Render()    

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and path.endswith(('.stl', '.obj', '.ply', '.wrl')):
                self.fileInfo = path
                self.open_file(path)
                self.vtkWidget.GetRenderWindow().Render()
            elif path.endswith(('.jpg', '.png')):
                self.drag_texture(path)
            elif os.path.isdir(path):
                self.load_folder(path)
                self.exportCSVAction.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, 'Invalid File', 'The file is not a 3D file or a texture map file. If you are trying to add a texture map, make sure its in PNG or JPG format.')


    def aboutFunction(self):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))
            msg.setText(f"{APP_NAME}\n\n{APP_DESCRIPTION}")
            msg.setWindowTitle(f"About {APP_NAME}")
            msg.exec_()

    def showManual(self):
        try:
            pdf_path = resource_path("docs/Manual.pdf")
            # <-- CHANGED: cross-platform "open with default app" instead of os.startfile.
            open_with_default_app(pdf_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Unable to Open the Manual', 'It seems that the manual is not accessible by the application. Please access it by going to the installation folder instead.')
            return 

    def reset_view(self):
        if hasattr(self, 'current_actor'):
            self.ren.ResetCamera()
            self.iren.Render()

    # <-- ADDED: roll the camera about the screen axis (used by the toolbar
    # roll buttons). Change the 2.0 to roll in bigger/smaller steps.
    def roll_view_left(self):
        camera = self.ren.GetActiveCamera()
        camera.Roll(2.0)
        camera.OrthogonalizeViewUp()
        self.vtkWidget.GetRenderWindow().Render()

    def roll_view_right(self):
        camera = self.ren.GetActiveCamera()
        camera.Roll(-2.0)
        camera.OrthogonalizeViewUp()
        self.vtkWidget.GetRenderWindow().Render()


    def color_red(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(1,0,0)
            self.iren.Render()
    
    def color_black(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(0,0,0)
            self.iren.Render()

    def color_gray(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(0.5,0.5,0.5)
            self.iren.Render()

    def color_blue(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(0,0,1)
            self.iren.Render()

    def color_gold(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(1.0,0.843,0.0)
            self.iren.Render()

    def color_orange(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(1.0,0.647,0.0)
            self.iren.Render()

    def reset_color(self):
        if hasattr(self, 'current_actor'):
            self.current_actor.GetProperty().SetColor(1,1,1)
            self.iren.Render()


    def exit_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
            self.resize(self.previous_size)  
            self.toolbar.show()
            self.menuPart.show()
        if self.otherFunctionsToolbar.isHidden:
            self.otherFunctionsToolbar.hide()
    
    
    def enter_full_screen(self):
        if not self.isFullScreen():         
            self.previous_size = self.size()
            self.showFullScreen()
            self.toolbar.hide()
            self.menuPart.hide()
            self.otherFunctionsToolbar.hide() 

            self.cutToolbar.hide()

            if not hasattr(self, 'original_image'):
                self.label.hide()
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information) 
            msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))           
            msg.setText("Press escape to exit fullscreen mode")
            msg.setWindowTitle("Escape to exit")
            msg.exec_()


    def remove_texture(self):
        if hasattr(self, 'current_actor'):
            fileName = self.current_actor.GetMapper().GetInputConnection(0, 0).GetProducer().GetFileName()
            if fileName.endswith('.ply'):
                delete_after_use = True

                self.loading_dialog = LoadingDialog(self)
                self.loading_dialog.show()
                QtWidgets.QApplication.processEvents()

                # <-- CHANGED: cross-platform cache directory.
                dir_path = get_cache_dir()
                os.makedirs(dir_path, exist_ok=True)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                obj_file = os.path.join(dir_path, os.path.basename(fileName[:-4] + '.obj'))

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(fileName)
                ms.save_current_mesh(obj_file)

                reader = vtk.vtkOBJReader()
                reader.SetFileName(obj_file)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                self.current_actor.SetMapper(mapper)
                self.objectColorMenuButton.setEnabled(True)
                self.ren.ResetCamera()
                self.removeTextureAction.setEnabled(False)   

                if delete_after_use:
                    os.remove(obj_file)

                self.loading_dialog.close()


    def open_scale_image(self, image_name):
        

        self.current_scale_name = image_name

        baseImageName = os.path.basename(image_name)

        scale, unit = re.findall(r"(\d+)(\w+)", baseImageName)[0]

        scale = int(scale)

        if unit.lower() == 'cm':
            scale *= 10

        imageScaleRealWidth = scale

        bounds = self.current_actor.GetBounds()
        size = self.vtkWidget.GetRenderWindow().GetSize()
        camera = self.ren.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        distance = ((position[0]-focal_point[0])**2 + (position[1]-focal_point[1])**2 + (position[2]-focal_point[2])**2)**0.5
        height = 2*distance*math.tan(math.radians(camera.GetViewAngle()/2))
        pixel_size = ((bounds[1]-bounds[0])/height*size[1], (bounds[3]-bounds[2])/height*size[1])

        imageScaleWidth = imageScaleRealWidth / (bounds[1] - bounds[0]) * pixel_size[0]

        imageScaleWidth = int(imageScaleWidth)

        image_path = image_name
        self.original_full_size_image = QtGui.QImage(image_path)
        self.original_image = QtGui.QImage(image_path)

        if self.label.pixmap():
            current_size = self.label.pixmap().size()
        else:
            current_size = self.original_image.size()

        original_size = self.original_image.size()
        new_height = int(original_size.height() * imageScaleWidth / original_size.width())

        pixmap = QtGui.QPixmap.fromImage(self.original_image).scaled(imageScaleWidth, new_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        self.label.show()
        self.label.setPixmap(pixmap)
        self.label.setGeometry((self.frame.width() - pixmap.width()) // 2, self.frame.height() - pixmap.height(), pixmap.width(), pixmap.height())


    def move_scale_up(self):
        self.label.move(self.label.x(), self.label.y() - 5)
        self.image_moved_manually = True

    def move_scale_down(self):
        self.label.move(self.label.x(), self.label.y() + 5)
        self.image_moved_manually = True

    def move_scale_left(self):
        self.label.move(self.label.x() - 5, self.label.y())
        self.image_moved_manually = True

    def move_scale_right(self):
        self.label.move(self.label.x() + 5, self.label.y())
        self.image_moved_manually = True

    def reset_image_position(self):
        if not hasattr(self, 'original_image'):
            return

        if self.label.isVisible():
            pixmap = self.label.pixmap()
            self.label.move((self.frame.width() - pixmap.width()) // 2, self.frame.height() - pixmap.height())
            self.image_moved_manually = False

    def remove_scale(self):
        if hasattr(self, 'original_image'):
            self.label.clear()
            self.label.hide()
            del self.original_image

        if hasattr(self, 'original_full_size_image'):
            del self.original_full_size_image

    def openScaleDialog(self):
        if self.ShowBarScaleMessage == True:
            reply = QMessageBox().information(self, 'Notice', 'Bar scale works only with pinch to zoom and with the scroll wheel.', QMessageBox.Ok | QMessageBox.Cancel)
            if reply == QMessageBox.Ok:
                self.ShowBarScaleMessage = False
                self.scale_window_options = ScalesWindowOptions(self)
                self.scale_window_options.show()
        elif self.ShowBarScaleMessage == False:
            self.scale_window_options = ScalesWindowOptions(self)
            self.scale_window_options.show()           


    def openLabelDialog (self):
        filename_without_extension = os.path.splitext(self.current_fileName)[0]
        self.dropdown.show()
        self.label_window_options = LabelWindowOptions(self)
        self.label_window_options.show()
        self.label_window_options.input_field_label.setText(filename_without_extension)

    def add_label(self, text):

        label = QtWidgets.QLabel(self.vtkWidget)
        label.setAlignment(QtCore.Qt.AlignCenter)
        fontSizeText = self.label_window_options.font_input.text()
        fontSize = int(fontSizeText) if fontSizeText else 24
        label.setStyleSheet(f"background-color: White; font-size: {fontSize}px;")
        label.setText(text)
        label.adjustSize()
        x = (self.vtkWidget.width() - label.width()) // 2
        y = label.y() + 20
        label.move(x, y)
        label.show()

        self.labels.append(label)
        self.dropdown.addItem(text)

        self.dropdown.setCurrentIndex(self.dropdown.count() - 1)
        
        self.label_window_options.input_field_label.setText('')

    def remove_label(self):
        if self.labels:
            label = self.labels[self.dropdown.currentIndex()]
            self.labels.remove(label)
            self.dropdown.removeItem(self.dropdown.currentIndex())
            label.deleteLater()


    def move_label_up(self):
        try:
            label = self.labels[self.dropdown.currentIndex()]
            x = label.x()
            y = max(label.y() - 10, 0)
            label.move(x, y)
        except:
            return

    def move_label_down(self):
        try:        
            label = self.labels[self.dropdown.currentIndex()]
            x = label.x()
            y = min(label.y() + 10, self.vtkWidget.height() - label.height())
            label.move(x, y)
        except:
            return
    
    def move_label_left(self):
        try:
            label = self.labels[self.dropdown.currentIndex()]
            x = max(label.x() - 10, 0)
            y = label.y()
            label.move(x, y)
        except:
            return

    
    def move_label_right(self):
        try:
            label = self.labels[self.dropdown.currentIndex()]
            x = max(label.x() + 10, 0)
            y = label.y()
            label.move(x, y)
        except:
            return
    

    def open_folder(self):
        folderName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder')

        if folderName:
            self.files = []
            for ext in ['*.stl', '*.wrl', '*.ply', '*.obj']:
                self.files.extend(glob.glob(os.path.join(folderName, ext)))

            if not self.files:
                QtWidgets.QMessageBox.warning(self, 'No 3D Files', 'No 3D files found in the selected folder.')
                return

            self.prevAction.setEnabled(True)
            self.nextAction.setEnabled(True)
            
            self.exportCSVAction.setEnabled(True)

            self.current_file_index = 0
            self.fileInfo = self.files[self.current_file_index]
            self.open_file(self.fileInfo)

            self.fileMenu.clear()

            self.fileActions = []
            for i, file in enumerate(self.files):
                action = QtWidgets.QAction(os.path.basename(file), self)
                action.setCheckable(True)
                action.triggered.connect(lambda checked, i=i: self.select_file(i))
                self.fileMenu.addAction(action)
                self.fileActions.append(action)

            self.fileActions[self.current_file_index].setChecked(True)

            self.loading_dialog.close()
            self.update_buttons()

    def load_folder(self, folderName=None):
        if folderName is None:
            folderName = QtWidgets.QFileDialog.getExistingDirectory(self)

        if folderName:

            self.files = []
            for ext in ['*.stl', '*.wrl', '*.ply', '*.obj']:
                self.files.extend(glob.glob(os.path.join(folderName, ext)))

            if not self.files:
                QtWidgets.QMessageBox.warning(self, 'No 3D Files', 'No 3D files found in the selected folder.')
                return

            self.prevAction.setEnabled(True)
            self.nextAction.setEnabled(True)
            self.exportCSVAction.setEnabled(True)

            self.current_file_index = 0


            self.fileInfo = self.files[self.current_file_index]
            self.open_file(self.fileInfo)


            self.fileMenu.clear()

            self.fileActions = []
            for i, file in enumerate(self.files):
                action = QtWidgets.QAction(os.path.basename(file), self)
                action.setCheckable(True)
                action.triggered.connect(lambda checked, i=i: self.select_file(i))
                self.fileMenu.addAction(action)
                self.fileActions.append(action)

            self.fileActions[self.current_file_index].setChecked(True)

            self.loading_dialog.close()

            self.update_buttons()

            self.vtkWidget.GetRenderWindow().Render()

    def select_file(self, index):
        self.current_file_index = index
        self.fileInfo = self.files[self.current_file_index]
        self.open_file(self.fileInfo)
        
        for i, action in enumerate(self.fileActions):
            action.setChecked(i == self.current_file_index)        
        self.update_buttons()

    def next_file(self):
        if self.current_file_index < len(self.files) - 1:
            self.current_file_index += 1
            self.fileInfo = self.files[self.current_file_index]
            self.open_file(self.fileInfo)
            
            for action in self.fileActions:
                action.setChecked(False)
            self.fileActions[self.current_file_index].setChecked(True)    
            
            self.exportCSVAction.setEnabled(True)

            self.update_buttons()
            self.vtkWidget.GetRenderWindow().Render()

    def prev_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.fileInfo = self.files[self.current_file_index]
            self.open_file(self.fileInfo)
                
            for action in self.fileActions:
                action.setChecked(False)
            self.fileActions[self.current_file_index].setChecked(True)

            self.exportCSVAction.setEnabled(True)

            self.update_buttons()
            self.vtkWidget.GetRenderWindow().Render()
    
    def update_buttons(self):
        self.nextAction.setEnabled(self.current_file_index < len(self.files) - 1)
        self.prevAction.setEnabled(self.current_file_index > 0)
        self.ListFilesButton.setEnabled(True)
        self.exportCSVAction.setEnabled(True)

    def ToggleTBFunction (self):
        if self.toolbar.isVisible():
            self.toolbar.hide()
        elif self.toolbar.isHidden():
            self.toolbar.show()


    def open_batch_convert_window(self):
        self.batchConvertWindow = BatchConvertWindow(self)
        self.batchConvertWindow.show()

    def open_light_dialog(self):
        self.light_dialog = LightDialog(self)
        self.light_dialog.show()

    def otherFunctionsToggleFunction (self):

        if self.current_file.endswith('.wrl') and self.otherFunctionsToolbar.isHidden():
            self.tempName = self.current_file

            self.wrlPrepForAnalysis()
            self.reloadTextureAction.setEnabled(True) 

            baseName = os.path.basename(self.tempName)
            self.fileNameLineEdit.setText(baseName)
            width = self.fileNameLineEdit.fontMetrics().boundingRect(baseName).width()
            self.fileNameLineEdit.setMaximumWidth(width + 10)
            self.current_fileName = os.path.basename(self.tempName)
            self.current_file = self.tempName

            self.otherFunctionsToolbar.show()
                        


        elif self.current_file.endswith(('.ply', '.obj', '.stl')) and self.otherFunctionsToolbar.isHidden():
            self.otherFunctionsToolbar.show()

        elif self.otherFunctionsToolbar.isVisible:
            self.otherFunctionsToolbar.hide()

    def wrlPrepForAnalysis(self):
        if self.WrlPlyName is None or not self.current_file.endswith('.wrl'):
            return
        
        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.show()
        QtWidgets.QApplication.processEvents()

        self.wrlToPlyName = self.current_file
        
        # <-- CHANGED: cross-platform cache directory.
        dir_path = get_cache_dir()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        obj_file = os.path.join(dir_path, os.path.basename(self.current_file[:-4] + '.obj'))
        mtl_file = os.path.join(dir_path, os.path.basename(self.current_file[:-4] + '.obj.mtl'))

        if not convert_wrl_to_obj(self.current_file, obj_file):
            return
        self.current_file = obj_file

        self.loading_dialog.close() 
        self.open_file(self.current_file, True, False)
        


    def reload_with_texture(self):
        if self.WrlPlyName is None or not self.current_file.endswith('.wrl'):
            return

        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.show()
        QtWidgets.QApplication.processEvents()

        self.wrlToPlyName = self.current_file

        # <-- CHANGED: cross-platform cache directory.
        dir_path = get_cache_dir()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ply_file = os.path.join(dir_path, os.path.basename(self.current_file[:-4] + '.ply'))

        if not convert_wrl_to_ply(self.current_file, ply_file):
            return
        self.current_file = ply_file

        self.loading_dialog.close() 
        self.open_file(self.current_file, True, False, False, False)


    def export_file_list(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('CSV Options')
        dialog.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        dialog.resize(250, 200)

        layout = QtWidgets.QVBoxLayout(dialog)

        idCheckbox = QtWidgets.QCheckBox('Include ID')
        idCheckbox.setToolTip('Add an ID column that gives the first item an ID of 1 and it increments by 1 for the following files')
        locusCheckbox = QtWidgets.QCheckBox('Include Locus number')
        locusCheckbox.setToolTip('The option relies on the file name being in this format "L123_B123" or "B123_L123"')
        basketCheckbox = QtWidgets.QCheckBox('Include Basket Number')
        basketCheckbox.setToolTip('The option relies on the file name being in this format "L123_B123" or "B123_L123"')        
        extensionCheckbox = QtWidgets.QCheckBox('Include file extension')
        extensionCheckbox.setToolTip('Add a column for the type of file')
        sizeCheckbox = QtWidgets.QCheckBox('Include file size (KB)')
        sizeCheckbox.setToolTip('Add a column with the size of the file in Kilobytes')
        locationCheckbox = QtWidgets.QCheckBox('Include file location')
        locationCheckbox.setToolTip('Add a column with the file path of the file on your computer')
        modifiedCheckbox = QtWidgets.QCheckBox('Include last modified')
        modifiedCheckbox.setToolTip('Add a column with the last time the file was modified (most popular)')
        createdCheckbox = QtWidgets.QCheckBox('Include created on')
        createdCheckbox.setToolTip('Add a column with when the file was created (works if the file was made on this computer)')


        layout.addWidget(idCheckbox)
        layout.addWidget(locusCheckbox)
        layout.addWidget(basketCheckbox)
        layout.addWidget(extensionCheckbox)
        layout.addWidget(sizeCheckbox)
        layout.addWidget(locationCheckbox)
        layout.addWidget(modifiedCheckbox)
        layout.addWidget(createdCheckbox)


        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        layout.addWidget(buttonBox)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')
            if fileName:
                with open(fileName, 'w', newline='') as file:
                    writer = csv.writer(file)
                    titles = []
                    if idCheckbox.isChecked():
                        titles.append('ID')
                    titles.append('Files')
                    if locusCheckbox.isChecked():
                        titles.append('Locus')
                    if basketCheckbox.isChecked():
                        titles.append('Basket')
                    if extensionCheckbox.isChecked():
                        titles.append('File Extensions')
                    if sizeCheckbox.isChecked():
                        titles.append('File Size (KB)')
                    if locationCheckbox.isChecked():
                        titles.append('Path to File')
                    if modifiedCheckbox.isChecked():
                        titles.append('Last Modified')
                    if createdCheckbox.isChecked():
                        titles.append('Created On')
                    writer.writerow(titles)
                    for i, f in enumerate(self.files, start=1):
                        name, ext = os.path.splitext(os.path.basename(f))
                        row = []
                        if idCheckbox.isChecked():
                            row.append(i)
                        row.append(name)

                        
                        if locusCheckbox.isChecked():
                            locus = re.search('L(\d+)', name, re.IGNORECASE)
                            row.append(locus.group(1) if locus else '')
                        if basketCheckbox.isChecked():
                            basket = re.search('B(\d+)', name, re.IGNORECASE)
                            row.append(basket.group(1) if basket else '')


                        if extensionCheckbox.isChecked():
                            row.append(ext[1:])
                        if sizeCheckbox.isChecked():
                            row.append(os.path.getsize(f) / 1024)
                        if locationCheckbox.isChecked():
                            row.append(os.path.dirname(f))
                        if modifiedCheckbox.isChecked():
                            row.append(time.ctime(os.path.getmtime(f)))
                        if createdCheckbox.isChecked():
                            row.append(time.ctime(os.path.getctime(f)))
                        writer.writerow(row)

    def about_current_file(self):
        if self.current_fileName is not None:
            fileInfo = QtCore.QFileInfo(self.fileInfo)

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle('About Current File')
            dialog.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)


            layout = QtWidgets.QVBoxLayout(dialog)

            layout.addWidget(QtWidgets.QLabel(f'File Name: {fileInfo.fileName()}'))
            layout.addWidget(QtWidgets.QLabel(f'File Type: {fileInfo.suffix()}'))
            layout.addWidget(QtWidgets.QLabel(f'File Size: {fileInfo.size() / 1024:.2f} KB'))
            layout.addWidget(QtWidgets.QLabel(f'Last Modified: {fileInfo.lastModified().toString()}'))
            layout.addWidget(QtWidgets.QLabel(f'Created: {time.ctime(os.path.getctime(self.fileInfo))}'))
            pathLabel = QtWidgets.QLabel(f'Path: <a href="file:///{fileInfo.path()}">{fileInfo.path()}</a>')
            pathLabel.setOpenExternalLinks(True)
            layout.addWidget(pathLabel)

            buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
            buttonBox.accepted.connect(dialog.accept)
            layout.addWidget(buttonBox)

            dialog.exec_()

    def open_reduce_size_dialog(self):
        dialog = ReduceSizeDialog(self.fileInfo, self)
        dialog.label.setText(f'Current file size: {self.get_current_file_size()} KB')
        dialog.exec_()

    
    def get_current_file_size(self):
        if hasattr(self, 'current_actor'):
            return os.path.getsize(self.fileInfo) / 1024
        else:
            return 0
        
    def toggle_bbox(self):
        if self.outlineActor in self.ren.GetActors():
            self.ren.RemoveActor(self.outlineActor)
        else:
            self.ren.AddActor(self.outlineActor)
        self.vtkWidget.GetRenderWindow().Render()

    def openBoundShiftDialog (self):
        self.resetCutToDefault()
        self.label_window_options = BoundWindowOptions(self)
        self.label_window_options.show()

    def openPositioningDialog(self):
        self.resetCutToDefault()
        self.positioning_tool_window = PositioningToolWindow(self)
        self.positioning_tool_window.show()


    def rotate_x_left(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateX(-angle)
        self.transform.Translate(-center[0], -center[1], -center[2])
        self.vtkWidget.GetRenderWindow().Render()

    def rotate_x_right(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateX(angle)
        self.transform.Translate(-center[0], -center[1], -center[2])
        self.vtkWidget.GetRenderWindow().Render()

    def rotate_z_left(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateZ(-angle)
        self.transform.Translate(-center[0], -center[1], -center[2])
        self.vtkWidget.GetRenderWindow().Render()

    def rotate_z_right(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateZ(angle)
        self.transform.Translate(-center[0], -center[1], -center[2])
        self.vtkWidget.GetRenderWindow().Render()

    def rotate_y_left(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateY(-angle)
        self.transform.Translate(-center[0], -center[1], -center[2])

        self.vtkWidget.GetRenderWindow().Render()

    def rotate_y_right(self, angle=0.5):
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()

        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateY(angle)
        self.transform.Translate(-center[0], -center[1], -center[2])

        self.vtkWidget.GetRenderWindow().Render()

    def move_left(self):
        self.transform.Translate(-0.5, 0, 0)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def move_right(self):
        self.transform.Translate(0.5, 0, 0)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def move_up(self):
        self.transform.Translate(0, 0.5, 0)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def move_down(self):
        self.transform.Translate(0, -0.5, 0)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def move_forward(self):
        self.transform.Translate(0, 0, 0.5)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def move_backward(self):
        self.transform.Translate(0, 0, -0.5)
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.vtkWidget.GetRenderWindow().Render()

    def align_by_normal_direction(self):
        """Auto-position: rotate the model so its dominant area-weighted
        surface normal points along +Z. Port of normal_positioning.m."""
        if not hasattr(self, 'current_actor') or self.current_actor is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before aligning it.')
            return

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(self.transform)
        transform_filter.SetInputConnection(self.current_actor.GetMapper().GetInputConnection(0, 0))
        transform_filter.Update()
        polydata = transform_filter.GetOutput()

        polys = polydata.GetPolys().GetData()
        if polys is None or polys.GetNumberOfTuples() == 0:
            QtWidgets.QMessageBox.warning(self, 'Alignment Failed', 'The current mesh has no triangle faces to align by.')
            return

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        faces = vtk_to_numpy(polys).reshape(-1, 4)[:, 1:4]

        axis, angle = compute_normal_alignment_axis_angle(points, faces)
        if abs(angle) < 1e-6:
            return  # Already aligned.

        # Same Translate/Rotate/Translate-around-center pattern as
        # rotate_x_left etc., using the same RotateWXYZ primitive (RotateX/
        # Y/Z are just special cases of it) instead of hand-building a
        # matrix, so this composes into self.transform identically to a
        # manual button click.
        center = self.current_actor.GetCenter()
        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.Translate(center[0], center[1], center[2])
        self.transform.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        self.transform.Translate(-center[0], -center[1], -center[2])
        self.vtkWidget.GetRenderWindow().Render()

    def save_position(self):
        """Save the model's current cumulative transform (self.transform) as
        a 4x4 matrix in JSON. Not compatible with old MATLAB .pos files."""
        if not hasattr(self, 'current_actor') or self.current_actor is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before saving its position.')
            return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Position', self.last_dir, 'ArchCut 3-D Position (*.acpos)')
        if not fileName:
            return
        if not fileName.lower().endswith('.acpos'):
            fileName += '.acpos'

        matrix = self.transform.GetMatrix()
        rows = [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
        with open(fileName, 'w') as f:
            json.dump({'transform_4x4': rows}, f, indent=2)

    def load_position(self):
        """Load a position saved by save_position(), replacing the model's
        current orientation entirely (absolute, not relative)."""
        if not hasattr(self, 'current_actor') or self.current_actor is None:
            QtWidgets.QMessageBox.warning(self, 'No Model Loaded', 'Load a model before loading a position.')
            return
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Position', self.last_dir, 'ArchCut 3-D Position (*.acpos)')
        if not fileName:
            return

        try:
            with open(fileName, 'r') as f:
                data = json.load(f)
            rows = data['transform_4x4']
            matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    matrix.SetElement(i, j, rows[i][j])
        except (OSError, ValueError, KeyError, IndexError) as e:
            QtWidgets.QMessageBox.warning(self, 'Load Position Failed', f'Could not read position file:\n{e}')
            return

        self.hide_center_of_mass()
        self.resetCutToDefault()
        self.transform.SetMatrix(matrix)
        self.vtkWidget.GetRenderWindow().Render()

    def hide_center_of_mass(self):
        if self.center_of_mass_actor:
            self.ren.RemoveActor(self.center_of_mass_actor)
            self.center_of_mass_actor = None

            self.current_actor.GetProperty().SetOpacity(1.0)



    def update_center_of_mass(self):
        if not hasattr(self, 'current_actor'):
            return

        if self.center_of_mass_actor:
            self.ren.RemoveActor(self.center_of_mass_actor)

        center_of_mass_world = self.transform.TransformPoint(self.center_of_mass_local)

        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center_of_mass_world)
        sphereSource.SetRadius(1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)

        self.ren.AddActor(actor)
        self.center_of_mass_actor = actor

        self.current_actor.GetProperty().SetOpacity(0.5)

    def export_file(self):
        if not hasattr(self, 'current_file'):
            return
        
        options = QtWidgets.QFileDialog.Options()
        base_name = os.path.basename(self.current_file)
        base_name = os.path.splitext(base_name)[0]
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", base_name,"WRL Files (*.wrl);;STL Files (*.stl);;PLY Files (*.ply);;OBJ Files (*.obj)", options=options)
        if fileName:
            self.loading_dialog = LoadingDialog(self)
            self.loading_dialog.show()
            QtWidgets.QApplication.processEvents()
            if not convert_file(self.current_file, fileName):
                return
            self.loading_dialog.hide()



    def createLineScale(self, lineSize, lineScaleColor):
        
        self.lineSource = vtk.vtkLineSource()

        if 'cm' in lineSize:
            self.realLineSize = float(lineSize.replace(' cm', '')) * 10
        else:
            self.realLineSize = float(lineSize.replace(' mm', ''))

        self.mapper = vtk.vtkPolyDataMapper2D()
        self.mapper.SetInputConnection(self.lineSource.GetOutputPort())

        self.lineScaleActor = vtk.vtkActor2D()
        self.lineScaleActor.SetMapper(self.mapper)
        self.lineScaleActor.GetProperty().SetLineWidth(6)
        self.lineScaleActor.GetProperty().SetColor(lineScaleColor)

        self.ren.AddActor2D(self.lineScaleActor)

        self.ren.GetActiveCamera().AddObserver("ModifiedEvent", self.updateLineScale)


        self.lineScaleTextActor = vtk.vtkTextActor()
        self.lineScaleTextActor.SetInput(lineSize)
        self.lineScaleTextActor.GetTextProperty().SetFontSize(24)
        self.lineScaleTextActor.GetTextProperty().SetColor(lineScaleColor)
        self.lineScaleTextActor.GetTextProperty().SetJustificationToCentered()

        self.ren.AddActor2D(self.lineScaleTextActor)

        self.updateLineScale()

        self.vtkWidget.GetRenderWindow().Render()

    def updateLineScale(self, obj=None, event=None):

        if not hasattr(self, 'lineScaleActor') or not hasattr(self, 'lineScaleTextActor'):
            return
        current_camera_distance = self.ren.GetActiveCamera().GetDistance()

        if abs(current_camera_distance - self.previous_camera_distance) > 0.01:

            bounds = self.current_actor.GetBounds()
            size = self.vtkWidget.GetRenderWindow().GetSize()
            camera = self.ren.GetActiveCamera()
            position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
            distance = ((position[0]-focal_point[0])**2 + (position[1]-focal_point[1])**2 + (position[2]-focal_point[2])**2)**0.5
            height = 2*distance*math.tan(math.radians(camera.GetViewAngle()/2))
            pixel_size = ((bounds[1]-bounds[0])/height*size[1], (bounds[3]-bounds[2])/height*size[1])

            lineSize = self.realLineSize / (bounds[1] - bounds[0]) * pixel_size[0]

            self.center = [size[0] / 2, size[1] / 2]

            p0 = [self.center[0] - lineSize / 2, 40, 0.0]
            p1 = [self.center[0] + lineSize / 2, 40, 0.0]
            self.lineSource.SetPoint1(p0)
            self.lineSource.SetPoint2(p1)

            if hasattr(self, 'lineScaleTextActor'):
                self.lineScaleTextActor.SetDisplayPosition(int(self.center[0]), 50)

            self.lineSource.Update()
            self.vtkWidget.GetRenderWindow().Render()
            
        self.previous_camera_distance = current_camera_distance

    def lineScaleDialogOpen (self):

        if hasattr(self, 'lineScaleActor') and hasattr(self, 'lineScaleTextActor'):
            self.ren.RemoveActor2D(self.lineScaleActor)
            self.ren.RemoveActor2D(self.lineScaleTextActor)
            del self.lineScaleActor
            del self.lineScaleTextActor

            self.vtkWidget.GetRenderWindow().Render()

        launchLineScaleDialog = LineScaleDialog(self)
        launchLineScaleDialog.show()

    def clearCacheFunction(self):
        # <-- CHANGED: cross-platform cache directory.
        dir_path = get_cache_dir()
        if not os.path.exists(dir_path):
            QMessageBox.critical(self, "Error", "The cache folder was not found.")
            return
        reply = QMessageBox.question(self, 'Message',
                     "Would you like to clear the cache folder?", QMessageBox.Yes | 
                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", "The cache file cannot be deleted. Please clear it manually from your cache folder.")
                    return

    def toggleCutToolbar(self):
        if self.cutToolbar.isHidden():
            self.cutToolbar.show()        
        elif self.cutToolbar.isVisible():
            self.cutToolbar.hide()


    def addCutPlane(self):

        self.cutButton.setEnabled(True)
        self.toggleCutPlaneButton.setEnabled(True)
        self.createCutButton.setEnabled(False)
        self.switchAxisButton.setEnabled(True)

        self.planeWidget = vtk.vtkImplicitPlaneWidget2()
        self.planeWidget.SetInteractor(self.iren)
        rep = vtk.vtkImplicitPlaneRepresentation()
        rep.SetPlaceFactor(1.25)
        rep.DrawOutlineOff()

        if hasattr(self, 'current_actor'):
            bounds = self.current_actor.GetBounds()
            center = [(bounds[1] + bounds[0]) / 2, (bounds[3] + bounds[2]) / 2, (bounds[5] + bounds[4]) / 2]
            rep.PlaceWidget(bounds)
            rep.SetOrigin(center)

        self.planeWidget.SetRepresentation(rep)
        self.planeWidget.On()
        plane = vtk.vtkPlane()
        rep.GetPlane(plane)
        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetClipFunction(plane)
        self.clipper.SetInputConnection(self.current_actor.GetMapper().GetInputConnection(0, 0))
        self.clipper.Update()
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(plane)
        self.cutter.SetInputConnection(self.current_actor.GetMapper().GetInputConnection(0, 0))
        self.cutter.Update()

        self.clip_actor = vtk.vtkActor()
        self.clip_actor.SetMapper(vtk.vtkPolyDataMapper())
        self.clip_actor.GetMapper().SetInputConnection(self.clipper.GetOutputPort())

        self.cut_actor = vtk.vtkActor()
        self.cut_actor.SetMapper(vtk.vtkPolyDataMapper())
        self.cut_actor.GetProperty().SetLineWidth(4)
        self.cut_actor.GetMapper().SetInputConnection(self.cutter.GetOutputPort())

        self.vtkWidget.GetRenderWindow().Render()

    def updateCut(self, obj, event):

        rep = self.planeWidget.GetRepresentation()
        plane = vtk.vtkPlane()
        rep.GetPlane(plane)
        self.clipper.SetClipFunction(plane)
        self.cutter.SetCutFunction(plane)
        self.clip_actor.GetMapper().SetInputConnection(self.clipper.GetOutputPort())
        self.cut_actor.GetMapper().SetInputConnection(self.cutter.GetOutputPort())
        self.vtkWidget.GetRenderWindow().Render()

    def RemoveCut(self):
        if hasattr(self, 'planeWidget'):
            self.planeWidget.Off()
            self.planeWidget = None
        if hasattr(self, 'cut_actor'):
            self.ren.RemoveActor(self.cut_actor)
            self.cut_actor = None
        if hasattr(self, 'current_actor'):
            self.current_actor.SetVisibility(1)
        self.vtkWidget.GetRenderWindow().Render()

    def toggleCuttingPlane(self):
        if not hasattr(self, 'planeWidget'):
            return
        if not hasattr(self, 'isPlaneVisible'):
            self.isPlaneVisible = True
        if self.isPlaneVisible:
            self.planeWidget.Off()
        else:
            self.planeWidget.On()
        self.isPlaneVisible = not self.isPlaneVisible
        self.vtkWidget.GetRenderWindow().Render()

    def switchCutAxis(self):
        axes = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        axis = axes[self.axis_index]
        if hasattr(self, 'planeWidget'):
            self.planeWidget.GetRepresentation().SetNormal(axis)
        self.axis_index = (self.axis_index + 1) % len(axes)
        self.updateCut(self.planeWidget, None)
        self.vtkWidget.GetRenderWindow().Render()

    def toggleCutObjectVisibility(self):
        if hasattr(self, 'clip_actor'):
            self.clip_actor.SetVisibility(not self.clip_actor.GetVisibility())
        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def cutHere(self):
        if hasattr(self, 'planeWidget'):
            
            self.toggleCutObjectButton.setEnabled(True)
            rep = self.planeWidget.GetRepresentation()
            plane = vtk.vtkPlane()
            rep.GetPlane(plane)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.transform)
            transformFilter.SetInputConnection(self.current_actor.GetMapper().GetInputConnection(0, 0))
            transformFilter.Update()

            self.clipper.SetClipFunction(plane)
            self.cutter.SetCutFunction(plane)

            self.clipper.SetInputConnection(transformFilter.GetOutputPort())
            self.cutter.SetInputConnection(transformFilter.GetOutputPort())

            cutMapper = vtk.vtkPolyDataMapper()
            cutMapper.SetInputConnection(self.cutter.GetOutputPort())
            cutActor = vtk.vtkActor()
            cutActor.SetMapper(cutMapper)
            cutActor.GetProperty().SetColor(1,0,0)
            clipMapper = vtk.vtkPolyDataMapper()
            clipMapper.SetInputConnection(self.clipper.GetOutputPort())
            clipActor = vtk.vtkActor()
            clipActor.SetMapper(clipMapper)
            self.calculateCircumferenceButton.setEnabled(True)
            if hasattr(self, 'clip_actor'):
                self.ren.RemoveActor(self.clip_actor)
            self.ren.AddActor(clipActor)
            self.clip_actor = clipActor
            if hasattr(self, 'cut_actor'):
                self.ren.RemoveActor(self.cut_actor)
            if hasattr(self, 'current_actor'):
                self.current_actor.SetVisibility(0)
            self.ren.AddActor(cutActor)
            self.cut_actor = cutActor
            if self.current_view == 'full':
                self.current_actor.SetVisibility(0)
                if hasattr(self, 'cut_actor'):
                    self.cut_actor.SetVisibility(1)
            if hasattr(self, 'planeWidget'):
                self.current_view = 'cut'
            self.vtkWidget.GetRenderWindow().Render()        
            self.ren.ResetCamera()

    def calculateCircumference(self):
        if not hasattr(self, 'cutter'):
            print("No cut has been made yet.")
            return

        cut_output = self.cutter.GetOutput()

        points = cut_output.GetPoints()

        circumference = 0.0

        if points.GetNumberOfPoints() > 1:
            for i in range(points.GetNumberOfPoints()):
                p1 = points.GetPoint(i)
                p2 = points.GetPoint((i + 1) % points.GetNumberOfPoints())

                distance = vtk.vtkMath.Distance2BetweenPoints(p1, p2) ** 0.5

                circumference += distance

        self.calculateCircumferenceButton.setEnabled(False)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon(resource_path('assets/icons/frame_icon.png')))          
        msg.setText("The circumference of the cut object is {} mm".format(circumference))
        msg.setWindowTitle("Circumference")
        msg.exec_()

    def removeCutObject(self, hideToolbar=False):
        if hasattr(self, 'planeWidget'):
            if self.planeWidget is not None:
                self.planeWidget.Off()
                self.planeWidget = None
        if hasattr(self, 'cut_actor'):
            self.ren.RemoveActor(self.cut_actor)
            self.cut_actor = None
        if hasattr(self, 'clip_actor'):
            self.ren.RemoveActor(self.clip_actor)
            self.clip_actor = None
        if hasattr(self, 'current_actor'):
            self.ren.RemoveActor(self.current_actor)
            self.current_actor = None

        if hideToolbar == True:
            self.cutToolbar.hide()

        self.vtkWidget.GetRenderWindow().Render()

    def resetCutToDefault(self):
        if hasattr(self, 'cut_actor'):
            self.ren.RemoveActor(self.cut_actor)
            self.cut_actor = None

        if hasattr(self, 'clip_actor'):
            self.ren.RemoveActor(self.clip_actor)
            self.clip_actor = None

        if hasattr(self, 'planeWidget'):
            if self.planeWidget is not None:
                self.planeWidget.Off()
                self.planeWidget = None

        if hasattr(self, 'current_actor'):
            self.current_actor.SetVisibility(1)


        self.cutButton.setEnabled(False)
        self.toggleCutPlaneButton.setEnabled(False)
        self.createCutButton.setEnabled(True)
        self.switchAxisButton.setEnabled(False)
        self.calculateCircumferenceButton.setEnabled(False)
        self.toggleCutObjectButton.setEnabled(False)
        self.vtkWidget.GetRenderWindow().Render()

    def createManualMeasurment (self):
        handle = vtk.vtkPointHandleRepresentation3D()
        rep = vtk.vtkDistanceRepresentation3D()
        rep.SetHandleRepresentation(handle)

        self.current_actor.GetProperty().SetOpacity(0.5)

        self.distanceWidget = vtk.vtkDistanceWidget()
        self.distanceWidget.SetInteractor(self.iren)
        self.distanceWidget.SetRepresentation(rep)
        self.distanceWidget.EnabledOn()
        self.distanceWidget.ProcessEventsOn()

        background_color = self.ren.GetBackground()

        invertColor = tuple(1 - component for component in background_color)
        
        manualMeasureColor = invertColor

        handle.GetProperty().SetColor(manualMeasureColor)
        
        rep.GetLineProperty().SetColor(manualMeasureColor)

        rep.GetLabelProperty().SetColor(manualMeasureColor)

    def removeManualMeasure (self):
        if hasattr(self, 'distanceWidget'):
            self.distanceWidget.SetEnabled(False)
            self.vtkWidget.GetRenderWindow().Render()
            
            self.current_actor.GetProperty().SetOpacity(1)
            
            del self.distanceWidget

            self.vtkWidget.GetRenderWindow().Render()


def main():
    app = QtWidgets.QApplication(sys.argv)

    splash_pix = QPixmap(resource_path('assets/images/splash.jpg'))
    splash = QtWidgets.QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.showMessage(APP_NAME, Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
    splash.show()
    app.processEvents()

    window = MainWindow()
    # Ensure the window comes to the front (especially helpful on macOS).
    window.raise_()
    window.activateWindow()
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        window.open_file(file_path, False, False, True, True, True)

    splash.finish(window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()