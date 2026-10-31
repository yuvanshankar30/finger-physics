"""OpenCV drawing: video, balls, simulated hands, and the HUD."""

import colorsys

import cv2
import numpy as np

from .handsim import DRAW_BONES, FINGER_NAMES, JOINTS, N, TIPS
from .world import CAPSULE_RADIUS

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
BG = (13, 9, 7)
TEXT = (246, 237, 230)
MUTED = (168, 150, 138)
HAND_COLORS = [(240, 200, 70), (40, 170, 255)]  # BGR: cyan, amber

HELP = (
    "pinch or close hand to grab | [P]hysics [R]predict [L]imits [W]raw [V]ideo [H]ud [B]alls [C]lear  "
    "spring [ ]  lead - =  smooth 9 0  [Q]uit"
)


def hsv_bgr(h_deg, s=0.85, v=0.95):
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def _panel(img, x, y, w, h):
    """Darken a rectangle in place as a translucent panel background."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.3).astype(np.uint8)


def _text(img, s, org, color=TEXT, scale=0.45, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, AA)


def draw_scene(frame, size, hands, world, params, T, hand_timeout):
    W, H = size
    if params.show_video and frame is not None:
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.convertScaleAbs(img, alpha=0.55)
    else:
        img = np.empty((H, W, 3), np.uint8)
        img[:] = BG

    for i in range(world.n):
        c = _pt(world.pos[i])
        rr = int(world.r[i])
        cv2.circle(img, c, rr, tuple(int(v) for v in world.color[i]), -1, AA)
        if world.grabbed[i] >= 0:
            cv2.circle(img, c, rr, (255, 255, 255), 2, AA)

    for hi, h in enumerate(hands):
        stale = T - h.meas_time
        if stale > hand_timeout:
            continue
        base = np.array(HAND_COLORS[hi % 2], float)
        fade = 1.0 if stale < 0.12 else max(0.0, 1 - (stale - 0.12) / (hand_timeout - 0.12))
        x = h.x
        if params.show_raw:
            for i in range(N):
                cv2.circle(img, _pt(h.raw[i]), 3, (230, 230, 230), -1, AA)
        cr = CAPSULE_RADIUS * h.scale
        thick = max(2, int(2 * cr))
        for a, b in DRAW_BONES:  # capsule body (matches collision radius)
            cv2.line(img, _pt(x[a]), _pt(x[b]), tuple(base * 0.45 * fade), thick, AA)
        for a, b in DRAW_BONES:  # bright core
            cv2.line(img, _pt(x[a]), _pt(x[b]), tuple(base * fade), max(2, int(cr * 0.35)), AA)

        ang = h.angles()
        flex = {j[3]: ang[n] for n, j in enumerate(JOINTS)}
        for i in range(N):
            rad = max(3, int(cr * (0.7 if i in TIPS else 0.55)))
            if i in flex:  # green (straight) → red (fully bent)
                col = hsv_bgr(max(0.0, 130 - max(0.0, flex[i]) * 1.2), v=0.95 * fade)
            else:
                col = tuple(int(255 * fade) for _ in range(3))
            cv2.circle(img, _pt(x[i]), rad, col, -1, AA)
        if h.pinching or h.gripping:
            cv2.circle(img, _pt(h.hold_point), int(cr * (1.4 if not h.gripping else 2.4)), (255, 255, 255), 2, AA)
    return img


def draw_hud(img, stats, hands, params):
    if not params.show_hud:
        return img
    H, W = img.shape[:2]

    # Stats, top-left.
    rows = [
        (f"{stats['cam_fps']:.0f}", "cam fps"),
        (f"{stats['infer_ms']:.1f}", "infer ms"),
        (f"{stats['latency_ms']:.0f}", "cam->pose ms"),
        (f"{stats['fps']:.0f}", "render fps"),
        (f"{stats['physics_hz']:.0f}", "physics Hz"),
        (stats["backend"], "backend"),
    ]
    _panel(img, 10, 10, 330, 62)
    for i, (val, lab) in enumerate(rows):
        cx, cy = 20 + (i % 3) * 108, 32 + (i // 3) * 28
        _text(img, val, (cx, cy), TEXT, 0.55, 1)
        _text(img, lab, (cx + 8 + 12 * len(val), cy), MUTED, 0.38)

    # Settings line.
    on = lambda b: "on" if b else "off"  # noqa: E731
    s = (
        f"physics {on(params.physics)}  predict {on(params.prediction)}  limits {on(params.joint_limits)}"
        f"   spring {params.stiffness_hz:.0f}Hz  lead {params.lead_ms:.0f}ms  smooth {params.min_cutoff:.1f}Hz"
    )
    _panel(img, 10, 78, 8 + int(7.2 * len(s)), 24)
    _text(img, s, (18, 95), MUTED, 0.4)

    # Joint angles, top-right.
    pw = 290
    px = W - pw - 10
    y = 10
    if not hands:
        _panel(img, px, y, pw, 30)
        _text(img, "Show a hand to the camera.", (px + 10, y + 20), MUTED)
    for hi, h in enumerate(hands):
        ph = 32 + 22 * 5
        _panel(img, px, y, pw, ph)
        title = f"{h.label or 'Hand'} hand"
        _text(img, title, (px + 10, y + 20), HAND_COLORS[hi % 2], 0.5)
        if not h.flex_sign:
            _text(img, "curl fingers to calibrate", (px + 120, y + 20), MUTED, 0.38)
        ang = h.angles()
        for f, name in enumerate(FINGER_NAMES):
            ry = y + 44 + 22 * f
            _text(img, name, (px + 10, ry), MUTED, 0.4)
            joints = [(n, j) for n, j in enumerate(JOINTS) if j[0] == f]
            col0 = 1 if f == 0 else 0  # align thumb's 2 joints under PIP/DIP
            for k, (n, j) in enumerate(joints):
                cx = px + 70 + (k + col0) * 72
                _text(img, j[1], (cx, ry), MUTED, 0.32)
                _text(img, f"{ang[n]:4.0f}", (cx + 26, ry), TEXT, 0.42)
                frac = float(np.clip(ang[n] / 110, 0, 1))
                cv2.line(img, (cx, ry + 5), (cx + int(64 * frac), ry + 5), hsv_bgr(130 - 130 * frac), 2)
        y += ph + 8

    # Help, bottom.
    _panel(img, 0, H - 26, W, 26)
    _text(img, HELP, (10, H - 9), MUTED, 0.4)
    return img
