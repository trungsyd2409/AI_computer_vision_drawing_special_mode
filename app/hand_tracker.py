"""Hand landmark detection with MediaPipe, on a background thread.

The tracker always works on the newest camera frame and publishes the frame
together with the hands found in it, so the image on screen and the finger
positions drawn on top of it come from exactly the same moment.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraStream
from .config import MODEL_PATH, MODEL_URL, Settings

# Landmark indices used across the project (MediaPipe hand topology).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Frames wider than this are shrunk before detection. The landmarks come back
# normalised, so accuracy barely changes while latency drops a lot.
_DETECT_MAX_WIDTH = 640


@dataclass
class Hand:
    side: str                 # "Left" or "Right", from the user's point of view
    score: float
    points: np.ndarray        # (21, 2) normalised image coordinates, x right, y down


@dataclass
class TrackResult:
    seq: int
    frame_rgb: np.ndarray
    hands: list[Hand] = field(default_factory=list)


def ensure_model() -> None:
    """Download the hand landmarker bundle the first time the app runs."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {MODEL_PATH} ...")
    tmp = MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(MODEL_PATH)


class HandTracker:
    def __init__(self, camera: CameraStream, settings: Settings):
        self.camera = camera
        self.settings = settings
        self._lock = threading.Lock()
        self._result: TrackResult | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._landmarker = None
        self._mp = None

        self.fps = 0.0
        self.available = False
        self.error = ""

    def start(self) -> None:
        try:
            self._mp, self._landmarker = self._create_landmarker()
            self.available = True
        except Exception as exc:  # missing package, no network for the model...
            self.error = f"Hand tracking disabled: {exc}"
            print(self.error)
        self._running = True
        self._thread = threading.Thread(target=self._run, name="tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._landmarker is not None:
            self._landmarker.close()

    def latest(self) -> TrackResult | None:
        with self._lock:
            return self._result

    # ------------------------------------------------------------ internals
    @staticmethod
    def _create_landmarker():
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp, vision.HandLandmarker.create_from_options(options)

    def _detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[Hand]:
        mp = self._mp
        h, w = rgb.shape[:2]
        if w > _DETECT_MAX_WIDTH:
            scale = _DETECT_MAX_WIDTH / w
            small = cv2.resize(rgb, (_DETECT_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        out = self._landmarker.detect_for_video(image, timestamp_ms)

        hands: list[Hand] = []
        for landmarks, handedness in zip(out.hand_landmarks, out.handedness):
            pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
            cat = handedness[0]
            side = cat.category_name
            # MediaPipe labels hands assuming a mirrored (selfie) image. When
            # the user turns mirroring off, the labels have to be swapped.
            if not self.settings.mirror:
                side = "Left" if side == "Right" else "Right"
            hands.append(Hand(side=side, score=float(cat.score), points=pts))
        return hands

    def _run(self) -> None:
        last_seq = -1
        last_ts = 0
        last_t = time.perf_counter()
        while self._running:
            seq, frame = self.camera.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands: list[Hand] = []
            if self._landmarker is not None:
                # Timestamps must strictly increase in VIDEO mode.
                ts = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts
                try:
                    hands = self._detect(rgb, ts)
                except Exception as exc:
                    self.error = str(exc)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else self.fps * 0.9 + (1 / dt) * 0.1

            with self._lock:
                self._result = TrackResult(seq=seq, frame_rgb=rgb, hands=hands)
