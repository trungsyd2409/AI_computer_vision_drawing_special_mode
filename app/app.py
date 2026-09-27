"""Window, main loop and keyboard/mouse handling.

Thread layout
    camera thread   -> grabs webcam frames
    tracker thread  -> runs MediaPipe on the newest frame
    main thread     -> simulation, rendering (GLFW + ModernGL) and the Tk
                       settings window
"""

from __future__ import annotations

import sys
import time

import glfw
import moderngl
import numpy as np

from .camera import CameraStream
from .config import APP_TITLE, CANVAS_H, CANVAS_W, IS_WINDOWS, Settings
from .hand_tracker import HandTracker
from .hud import Hud, HudInfo
from .renderer import Renderer
from .scene import Scene
from .settings_window import SettingsWindow

MIN_FPS = 15


def _enable_dpi_awareness() -> None:
    """Keep the Tk window sharp on Windows displays scaled above 100%."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class App:
    def __init__(self):
        self.settings = Settings()
        self.camera = CameraStream(self.settings)
        self.tracker = HandTracker(self.camera, self.settings)
        self.scene = Scene(self.settings)
        self.hud = Hud()
        self.window = None
        self.renderer: Renderer | None = None
        self.settings_window: SettingsWindow | None = None
        self._grid_backup = self.settings.grid_opacity

    # ---------------------------------------------------------------- setup
    def _create_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialised")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)   # required on macOS
        glfw.window_hint(glfw.SAMPLES, 0)
        self.window = glfw.create_window(CANVAS_W, CANVAS_H, APP_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not open an OpenGL 3.3 window")
        glfw.set_window_aspect_ratio(self.window, 4, 3)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._on_key)

    def _on_key(self, _win, key, _scancode, action, _mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_X and self.settings_window is not None:
            self.settings_window.toggle()
        elif key == glfw.KEY_C:
            self.scene.clear()
        elif key == glfw.KEY_H:
            self.settings.show_skeleton = not self.settings.show_skeleton
        elif key == glfw.KEY_G:
            if self.settings.grid_opacity > 0:
                self._grid_backup = self.settings.grid_opacity
                self.settings.grid_opacity = 0.0
            else:
                self.settings.grid_opacity = self._grid_backup or 1.0
        elif key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(self.window, True)

    def _mouse(self) -> tuple[np.ndarray, bool]:
        """Mouse position in canvas pixels and whether the left button is held."""
        x, y = glfw.get_cursor_pos(self.window)
        w, h = glfw.get_window_size(self.window)
        pos = np.array([x / max(w, 1) * CANVAS_W, y / max(h, 1) * CANVAS_H], dtype=np.float64)
        down = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        return pos, down

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        _enable_dpi_awareness()
        self._create_window()
        ctx = moderngl.create_context()
        self.renderer = Renderer(ctx)
        self.settings_window = SettingsWindow(self.settings, self.camera, on_clear=self.scene.clear)

        self.camera.start()
        self.tracker.start()

        t0 = time.perf_counter()
        last = t0
        frames, fps, fps_clock = 0, 0, t0
        last_seq = -1
        cam_size = (640, 480)
        hands = []
        cam_ok_since = None

        try:
            while not glfw.window_should_close(self.window):
                now = time.perf_counter()
                dt = min(now - last, 0.1)
                last = now

                glfw.poll_events()
                self.settings_window.pump()

                # --- Newest camera frame + hands ------------------------------
                result = self.tracker.latest()
                if result is not None and result.seq != last_seq:
                    last_seq = result.seq
                    self.renderer.upload_camera(result.frame_rgb)
                    cam_size = (result.frame_rgb.shape[1], result.frame_rgb.shape[0])
                    hands = result.hands

                # --- Simulation ---------------------------------------------------
                space = glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS
                state = self.scene.update(
                    dt, now - t0, hands, cam_size,
                    mouse=self._mouse(), key_press=space,
                    tracking_ok=self.tracker.available, camera_ok=self.camera.is_open,
                )
                self.renderer.upload_height(self.scene.fabric.height)

                # --- HUD (a few times per second is plenty) ----------------------
                frames += 1
                if now - fps_clock >= 0.25:
                    fps = round(frames / (now - fps_clock))
                    frames, fps_clock = 0, now
                fb_w, fb_h = glfw.get_framebuffer_size(self.window)
                if fb_w == 0 or fb_h == 0:     # minimised
                    glfw.swap_buffers(self.window)
                    continue

                if self.camera.is_open:
                    cam_ok_since = cam_ok_since or now
                else:
                    cam_ok_since = None
                info = HudInfo(
                    fps=fps,
                    cam_fps=round(self.camera.fps),
                    track_fps=round(self.tracker.fps),
                    status=self.scene.status,
                    status_live=self.scene.pen_down,
                    warning=self._warning(now, cam_ok_since),
                )
                img = self.hud.render(info, (fb_w, fb_h))
                if img is not None:
                    self.renderer.upload_hud(img)

                ctx.screen.use()
                self.renderer.render(state, self.settings, (fb_w, fb_h))
                glfw.swap_buffers(self.window)
        finally:
            self.tracker.stop()
            self.camera.stop()
            self.settings_window.destroy()
            glfw.terminate()

    def _warning(self, now: float, cam_ok_since: float | None) -> str:
        # Give the camera a moment to settle before judging its frame rate.
        if cam_ok_since is None or now - cam_ok_since < 3.0:
            return ""
        if self.camera.fps < MIN_FPS:
            return "Camera below 15 FPS  ·  press X and lower the resolution or turn off auto exposure"
        if self.tracker.available and self.tracker.fps < MIN_FPS:
            return "Hand tracking below 15 FPS  ·  press X and choose 640 x 480"
        return ""


def main() -> int:
    try:
        App().run()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
