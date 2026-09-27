"""Game logic: turns tracked hands into pen strokes, vortices and particles.

This is the glue between input (hands, mouse, keyboard) and the effects
(fabric, vortices, particles). It produces a :class:`FrameState` that the
renderer draws. It has no window or GPU dependency.
"""

from __future__ import annotations

import numpy as np

from .config import CANVAS_H, CANVAS_W, Settings
from .fabric import Fabric
from .gestures import OneEuroFilter, PinchDetector, SpinDetector
from .hand_tracker import (HAND_CONNECTIONS, INDEX_TIP, MIDDLE_MCP, PINKY_TIP,
                           THUMB_TIP, WRIST, Hand)
from .particles import ParticleSystem
from .renderer import FrameState, camera_crop_scale
from .vortex import VortexField

_RIGHT_RGBA = (0.35, 0.90, 1.00, 0.40)
_LEFT_RGBA = (0.70, 0.45, 1.00, 0.40)


class Scene:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fabric = Fabric()
        self.vortices = VortexField()
        self.particles = ParticleSystem()
        self.pen_filter = OneEuroFilter()
        self.pinch = PinchDetector(settings.pinch_on, settings.pinch_off)
        self.spin = SpinDetector()

        self.pen_pos: np.ndarray | None = None
        self.pen_down = False
        self.spinning = False
        self._prev_pen: np.ndarray | None = None
        self._last_pen_for_vel: np.ndarray | None = None
        self.status = "SHOW YOUR HANDS"

    def clear(self) -> None:
        self.fabric.clear()
        self.vortices.clear()
        self.particles.clear()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def to_canvas(points: np.ndarray, cam_size: tuple[int, int]) -> np.ndarray:
        """Normalised camera coordinates -> canvas pixels, honouring the crop."""
        sx, sy = camera_crop_scale(*cam_size)
        uv = 0.5 + (points - 0.5) / np.array([sx, sy], dtype=np.float32)
        return uv * np.array([CANVAS_W, CANVAS_H], dtype=np.float32)

    @staticmethod
    def _assign_hands(hands: list[Hand]) -> tuple[Hand | None, Hand | None]:
        """Return (pen_hand, clutch_hand) = (right, left)."""
        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            if a.side == b.side:
                # The classifier sometimes labels both hands the same.
                # In a mirrored view the user's right hand is on the right.
                a, b = sorted((a, b), key=lambda hd: float(hd.points[:, 0].mean()))
                return b, a
            return (a, b) if a.side == "Right" else (b, a)
        if len(hands) == 1:
            return (hands[0], None) if hands[0].side == "Right" else (None, hands[0])
        return None, None

    # ----------------------------------------------------------------- update
    def update(self, dt: float, t: float, hands: list[Hand] | None, cam_size: tuple[int, int],
               mouse: tuple[np.ndarray, bool] | None = None, key_press: bool = False,
               tracking_ok: bool = True, camera_ok: bool = True) -> FrameState:
        s = self.settings
        self.fabric.set_recovery(s.recovery_seconds)
        self.vortices.decay_seconds = s.vortex_decay
        self.pinch.on_ratio, self.pinch.off_ratio = s.pinch_on, s.pinch_off

        state = FrameState(time=t)
        pen_hand, clutch_hand = self._assign_hands(hands or [])

        # --- Left hand: the "clutch" that presses the pen into the fabric ------
        clutch_active = False
        if clutch_hand is not None:
            pts = self.to_canvas(clutch_hand.points, cam_size)
            clutch_active = self.pinch.update(pts[THUMB_TIP], pts[PINKY_TIP], pts[WRIST], pts[MIDDLE_MCP])
            mid = (pts[THUMB_TIP] + pts[PINKY_TIP]) * 0.5
            state.clutch = (float(mid[0]), float(mid[1]))
            state.clutch_closeness = self.pinch.closeness
            state.clutch_on = clutch_active
        else:
            self.pinch.release()

        # --- Right index fingertip: the pen --------------------------------------
        raw_pen: np.ndarray | None = None
        if pen_hand is not None:
            raw_pen = self.to_canvas(pen_hand.points, cam_size)[INDEX_TIP]
        want_down = pen_hand is not None and (clutch_active or key_press)

        # The mouse works as a pen too (handy without a camera).
        if mouse is not None and mouse[1]:
            raw_pen, want_down = mouse[0], True

        if raw_pen is None:
            self.pen_filter.reset()
            self.pen_pos = None
        else:
            self.pen_pos = self.pen_filter(raw_pen, dt)

        pen_vel = np.zeros(2)
        if self.pen_pos is not None and self._last_pen_for_vel is not None and dt > 0:
            pen_vel = (self.pen_pos - self._last_pen_for_vel) / dt
        self._last_pen_for_vel = None if self.pen_pos is None else self.pen_pos.copy()

        self.pen_down = bool(want_down and self.pen_pos is not None)
        self.spinning = False
        if self.pen_down:
            start = self._prev_pen if self._prev_pen is not None else self.pen_pos
            self.fabric.press(start, self.pen_pos, s.brush_radius, s.dent_depth)
            self._prev_pen = self.pen_pos.copy()
            event = self.spin.update(t, self.pen_pos)
            if event is not None:
                self.vortices.feed(event, dt)
                self.spinning = True
            self.particles.emit_pen(self.pen_pos.astype(np.float32), pen_vel.astype(np.float32), dt)
        else:
            self._prev_pen = None
            self.spin.reset()

        # --- Simulation ----------------------------------------------------------
        self.fabric.step(dt)
        self.vortices.update(dt)
        self.particles.emit_vortices(self.vortices.vortices, dt)
        self.particles.update(self.vortices.vortices, dt)

        # --- Output for the renderer ------------------------------------------------
        state.vortices = self.vortices.vortices
        state.pen = None if self.pen_pos is None else (float(self.pen_pos[0]), float(self.pen_pos[1]))
        state.pen_down = self.pen_down
        state.particles = self.particles.vertex_data()
        state.skeleton, state.joints = self._skeleton(hands or [], cam_size)

        self.status = self._status(pen_hand, clutch_hand, tracking_ok, camera_ok)
        return state

    def _skeleton(self, hands: list[Hand], cam_size) -> tuple[np.ndarray, np.ndarray]:
        lines, joints = [], []
        for hand in hands:
            pts = self.to_canvas(hand.points, cam_size)
            rgba = _RIGHT_RGBA if hand.side == "Right" else _LEFT_RGBA
            hue = 0.0 if hand.side == "Right" else 0.5
            for a, b in HAND_CONNECTIONS:
                lines.append((*pts[a], *rgba))
                lines.append((*pts[b], *rgba))
            for p in pts:
                joints.append((p[0], p[1], 1.3, 0.7, hue))
        return (np.array(lines, dtype=np.float32).reshape(-1, 6),
                np.array(joints, dtype=np.float32).reshape(-1, 5))

    def _status(self, pen_hand, clutch_hand, tracking_ok: bool, camera_ok: bool) -> str:
        if self.pen_down:
            return "VORTEX" if self.spinning else "PRESSING"
        if not camera_ok:
            return "NO CAMERA  ·  DRAG MOUSE TO PRESS"
        if not tracking_ok:
            return "HAND TRACKING OFF  ·  DRAG MOUSE TO PRESS"
        if pen_hand is not None and clutch_hand is not None:
            return "READY  ·  TOUCH LEFT THUMB TO PINKY"
        if pen_hand is not None:
            return "SHOW YOUR LEFT HAND TO PRESS"
        if clutch_hand is not None:
            return "SHOW YOUR RIGHT INDEX FINGER"
        return "SHOW YOUR HANDS"
