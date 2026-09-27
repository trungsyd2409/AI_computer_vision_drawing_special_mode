"""Webcam capture running on its own thread.

Reading from a webcam blocks until the next frame arrives. Doing that on the
render thread would tie the animation to the camera clock, so a background
thread keeps grabbing frames and only the most recent one is kept.

Driver changes (resolution, fps, backend...) are requested from the UI thread
and executed by the capture thread, because OpenCV's ``VideoCapture`` is not
safe to touch from two threads at once.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .config import Settings

_BACKEND_FLAGS = {
    "Auto": cv2.CAP_ANY,
    "DirectShow": cv2.CAP_DSHOW,
    "Media Foundation": cv2.CAP_MSMF,
    "V4L2": cv2.CAP_V4L2,
}


class CameraStream:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._reopen_requested = True
        self._live_props_dirty = False

        # Public, read-only status for the HUD and settings window.
        self.fps = 0.0
        self.actual_size = (0, 0)
        self.reported_fps = 0.0
        self.is_open = False
        self.error = ""

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release()

    def request_reopen(self) -> None:
        """Ask the capture thread to reopen the device with new settings."""
        self._reopen_requested = True

    def request_live_update(self) -> None:
        """Ask the capture thread to push exposure settings to the driver."""
        self._live_props_dirty = True

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Return ``(sequence_number, frame)`` of the newest frame (BGR)."""
        with self._lock:
            return self._seq, self._frame

    # ------------------------------------------------------------ internals
    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.is_open = False

    def _open(self) -> None:
        self._release()
        s = self.settings
        flag = _BACKEND_FLAGS.get(s.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(s.camera_index, flag)
        if not cap.isOpened():
            self.error = f"Cannot open camera {s.camera_index} ({s.backend})"
            cap.release()
            return

        # FOURCC must be set before the size: many webcams only offer
        # 720p at 60 fps when streaming compressed MJPG.
        if s.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, s.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self.is_open = True
        self.error = ""
        self.actual_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._apply_live_props()

    def _apply_live_props(self) -> None:
        if self._cap is None:
            return
        s = self.settings
        # The meaning of these values is driver specific. 0.75 / 0.25 is the
        # convention understood by DirectShow and V4L2 in OpenCV.
        if s.auto_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, float(s.exposure))

    def _run(self) -> None:
        last_t = time.perf_counter()
        while self._running:
            if self._reopen_requested:
                self._reopen_requested = False
                self._open()
                self.fps = 0.0
                last_t = time.perf_counter()

            if self._live_props_dirty:
                self._live_props_dirty = False
                self._apply_live_props()

            if self._cap is None:
                time.sleep(0.25)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.error = "Camera stopped sending frames"
                time.sleep(0.05)
                continue

            if self.settings.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                # Exponential moving average keeps the number readable.
                self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1

            self.actual_size = (frame.shape[1], frame.shape[0])
            with self._lock:
                self._frame = frame
                self._seq += 1
