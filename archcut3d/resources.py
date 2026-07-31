"""
Shared filesystem helpers for ArchCut 3-D.

Centralizes resource/path resolution so the main window, crest detection,
and any future modules agree on where assets, the setCurvature binary, and
the per-user cache directory live. Resolution works both when running from
source and when frozen into a PyInstaller bundle.
"""

import os
import platform
import subprocess
import sys


def _project_root():
    """Root of the ArchCut 3-D project: the folder containing archcut3d/,
    assets/, bin/, and docs/."""
    try:
        # PyInstaller extracts bundled data next to _MEIPASS.
        return sys._MEIPASS
    except AttributeError:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(relative_path):
    """Resolve a path (e.g. 'assets/icons/frame_icon.png') relative to the
    project root, in dev and inside a PyInstaller bundle alike."""
    return os.path.join(_project_root(), relative_path)


def get_cache_dir():
    """Per-OS scratch directory for ArchCut 3-D's temporary working files."""
    system = platform.system()
    if system == "Windows":
        base = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif system == "Darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        return os.path.join(os.path.expanduser("~"), ".archcut3d_cache")
    return os.path.join(base, "ArchCut 3-D Cache")


def get_setcurvature_path():
    """Locate the platform-appropriate setCurvature binary in bin/.
    Returns an absolute path, or None if it isn't present."""
    exe_name = "setcurvature.exe" if platform.system() == "Windows" else "setCurvature"
    candidate = os.path.join(_project_root(), "bin", exe_name)
    return candidate if os.path.exists(candidate) else None


def open_with_default_app(path):
    """Open a file with the OS's default application for its type."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])
