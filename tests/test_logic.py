"""Tests for the parts that do not need a camera or a GPU.

Run with:  python -m pytest -q
"""

import math

import numpy as np

from app.fabric import Fabric
from app.gestures import OneEuroFilter, PinchDetector, SpinDetector
from app.particles import ParticleSystem
from app.config import Settings
from app.scene import Scene
from app.vortex import Vortex, VortexField


def _circle(detector, cx, cy, r, turns_per_sec, seconds, fps=60, clockwise=True):
    event = None
    sign = 1 if clockwise else -1
    for i in range(int(seconds * fps)):
        t = i / fps
        a = sign * 2 * math.pi * turns_per_sec * t
        event = detector.update(t, np.array([cx + r * math.cos(a), cy + r * math.sin(a)])) or event
    return event


def test_spin_detects_clockwise_and_counter_clockwise():
    cw = _circle(SpinDetector(), 400, 300, 40, 1.5, 1.0, clockwise=True)
    ccw = _circle(SpinDetector(), 400, 300, 40, 1.5, 1.0, clockwise=False)
    assert cw is not None and cw.turns_per_sec > 1.0
    assert ccw is not None and ccw.turns_per_sec < -1.0
    assert np.allclose(cw.center, [400, 300], atol=6)
    assert abs(cw.radius - 40) < 6


def test_spin_ignores_straight_lines_and_zigzags():
    det = SpinDetector()
    hit = None
    for i in range(60):
        t = i / 60
        x = 200 + 300 * t
        y = 300 + (20 if i % 10 < 5 else -20)
        hit = det.update(t, np.array([x, y])) or hit
    assert hit is None


def test_pinch_hysteresis():
    p = PinchDetector(on_ratio=0.4, off_ratio=0.6)
    wrist, mcp = np.array([0.0, 0.0]), np.array([0.0, -100.0])   # palm length 100
    thumb = np.array([0.0, -50.0])
    assert not p.update(thumb, thumb + [120, 0], wrist, mcp)   # 1.2 palms apart
    assert p.update(thumb, thumb + [30, 0], wrist, mcp)        # touching
    assert p.update(thumb, thumb + [50, 0], wrist, mcp)        # between thresholds: stays on
    assert not p.update(thumb, thumb + [70, 0], wrist, mcp)    # past release


def test_one_euro_reduces_jitter():
    f = OneEuroFilter()
    rng = np.random.default_rng(0)
    raw = np.array([400.0, 300.0]) + rng.normal(0, 3, (120, 2))
    out = np.array([f(p, 1 / 60) for p in raw])
    assert out[30:].std(axis=0).max() < raw[30:].std(axis=0).max() * 0.6


def test_fabric_groove_springs_back():
    fab = Fabric()
    fab.set_recovery(3.0)
    fab.press(np.array([200, 300]), np.array([600, 300]), 34, 1.0)
    assert fab.height.min() < -0.95
    for _ in range(60 * 4):
        fab.step(1 / 60)
    assert fab.height.min() > -0.05
    assert fab.height.max() < 0.05      # no big overshoot above the rest level


def test_vortex_follows_direction_then_fades():
    field = VortexField()
    field.decay_seconds = 1.0
    det = SpinDetector()
    for i in range(90):
        t = i / 60
        a = 2 * math.pi * 1.5 * t
        ev = det.update(t, np.array([400 + 40 * math.cos(a), 300 + 40 * math.sin(a)]))
        if ev:
            field.feed(ev, 1 / 60)
        field.update(1 / 60)
    assert len(field.vortices) == 1 and field.vortices[0].spin > 0.5
    for _ in range(60 * 6):
        field.update(1 / 60)
    assert not field.vortices


def test_particles_spiral_into_vortex():
    field = VortexField()
    field.vortices.append(Vortex(pos=np.array([400.0, 300.0]), radius=100.0, spin=2.0))
    ps = ParticleSystem(500)
    for _ in range(30):
        ps.emit_vortices(field.vortices, 1 / 60)
        ps.update(field.vortices, 1 / 60)
    alive = ps.life > 0
    r0 = np.linalg.norm(ps.pos[alive] - [400, 300], axis=1).mean()
    for _ in range(40):
        ps.update(field.vortices, 1 / 60)
    alive2 = ps.life > 0
    r1 = np.linalg.norm(ps.pos[alive2] - [400, 300], axis=1).mean()
    assert alive.sum() > 20 and r1 < r0


def test_camera_to_canvas_mapping_handles_widescreen_crop():
    # 16:9 camera on a 4:3 canvas: the sides are cropped, the centre stays put.
    pts = np.array([[0.5, 0.5], [0.5 - 0.375, 0.5]], dtype=np.float32)
    out = Scene.to_canvas(pts, (1280, 720))
    assert np.allclose(out[0], [400, 300])
    assert np.allclose(out[1], [0, 300], atol=1e-3)


def test_scene_press_with_left_pinch():
    from app.hand_tracker import Hand
    scene = Scene(Settings())
    right = np.tile([0.6, 0.5], (21, 1)).astype(np.float32)
    left = np.tile([0.2, 0.7], (21, 1)).astype(np.float32)
    left[9] = [0.2, 0.6]      # palm length 0.1
    left[4] = [0.18, 0.66]
    left[20] = [0.2, 0.66]    # thumb touching pinky
    hands = [Hand("Right", 1.0, right), Hand("Left", 1.0, left)]
    scene.update(1 / 60, 0.0, hands, (640, 480))
    assert scene.pen_down
    assert scene.fabric.height.min() < -0.5
