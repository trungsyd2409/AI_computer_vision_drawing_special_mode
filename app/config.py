"""Central place for constants and the live, user-tweakable settings.

Everything the settings window can change lives in :class:`Settings`. The
object is shared between threads; each field is a plain Python value, so a
single assignment is atomic and no lock is needed for reads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

APP_TITLE = "Spacetime Canvas"

# Logical canvas size. All gameplay maths (brush size, vortex radius, spin
# detection) happens in these units, independent of the real window size.
CANVAS_W = 800
CANVAS_H = 600
CANVAS_ASPECT = CANVAS_W / CANVAS_H

# Resolution of the simulated fabric height field (one cell = 4x4 canvas px).
FABRIC_W = 200
FABRIC_H = 150

MAX_VORTICES = 6
MAX_PARTICLES = 3000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
SHADER_DIR = Path(__file__).resolve().parent / "shaders"

IS_WINDOWS = sys.platform.startswith("win")

# Choices offered in the settings window.
RESOLUTIONS = [(640, 480), (960, 540), (1280, 720)]
FPS_CHOICES = [15, 30, 60]
BACKENDS = ["Auto", "DirectShow", "Media Foundation"] if IS_WINDOWS else ["Auto", "V4L2"]


@dataclass
class Settings:
    # --- Camera (driver level, changing these reopens the device) ---------
    camera_index: int = 0
    resolution: tuple[int, int] = (640, 480)
    target_fps: int = 60
    backend: str = "DirectShow" if IS_WINDOWS else "Auto"
    use_mjpg: bool = True

    # --- Camera (driver level, applied live) ------------------------------
    auto_exposure: bool = True
    exposure: float = -6.0  # log2 seconds on DirectShow, e.g. -6 = 1/64 s

    # --- Image grading (done on the GPU, always works) --------------------
    mirror: bool = True
    brightness: float = 0.0   # -0.5 .. 0.5
    contrast: float = 1.0     # 0.5 .. 1.8
    saturation: float = 0.8   # 0.0 .. 1.6
    camera_dim: float = 0.72  # how much of the camera image shows through

    # --- Effect --------------------------------------------------------------
    brush_radius: float = 34.0     # canvas px
    dent_depth: float = 1.0        # 0.3 .. 1.6
    recovery_seconds: float = 3.0  # time for a groove to spring back
    vortex_decay: float = 2.6      # seconds for a vortex to lose ~63% of spin
    grid_opacity: float = 1.0      # 0 .. 1.5
    show_skeleton: bool = True

    # --- Gesture ---------------------------------------------------------------
    pinch_on: float = 0.42   # thumb-pinky distance / palm size to engage
    pinch_off: float = 0.60  # ... and to release (hysteresis)
