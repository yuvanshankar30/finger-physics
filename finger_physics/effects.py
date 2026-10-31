"""Hand-to-hand contact detection and the touch animation (sparks, shockwave
rings, glow with bloom)."""

import cv2
import numpy as np

from .handsim import DRAW_BONES
from .world import CAPSULE_RADIUS

MAX_PARTICLES = 900
SPARK_GRAVITY = 500.0
SPARK_DRAG = 2.5  # 1/s
TOUCH_COOLDOWN = 0.25  # s between "new touch" bursts for the same hand pair
WHITE = np.array([255.0, 255.0, 255.0])


def _sample_points(h):
    """Joints plus bone midpoints (2D) and their velocities."""
    a, b = DRAW_BONES[:, 0], DRAW_BONES[:, 1]
    pts = np.vstack([h.x[:, :2], (h.x[a, :2] + h.x[b, :2]) / 2])
    vel = np.vstack([h.v[:, :2], (h.v[a, :2] + h.v[b, :2]) / 2])
    return pts, vel


def _closest_on_bones(pts, h):
    """Distance from each point to hand h's nearest bone, the closest point on
    it, and that point's velocity."""
    A, B = h.x[DRAW_BONES[:, 0], :2], h.x[DRAW_BONES[:, 1], :2]
    VA, VB = h.v[DRAW_BONES[:, 0], :2], h.v[DRAW_BONES[:, 1], :2]
    AB = B - A
    t = np.einsum("ksd,sd->ks", pts[:, None] - A[None], AB) / np.maximum(np.sum(AB * AB, 1), 1e-9)
    t = np.clip(t, 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(pts[:, None] - Q, axis=2)
    s = np.argmin(d, axis=1)
    k = np.arange(len(pts))
    tt = t[k, s][:, None]
    return d[k, s], Q[k, s], VA[s] * (1 - tt) + VB[s] * tt


def hand_contacts(h1, h2):
    """Contact points between two hands' capsules, and relative speeds there."""
    R = CAPSULE_RADIUS * (h1.scale + h2.scale)
    out_p, out_s = [], []
    for ha, hb in ((h1, h2), (h2, h1)):
        pts, vel = _sample_points(ha)
        d, q, vq = _closest_on_bones(pts, hb)
        hit = d < R
        if hit.any():
            out_p.append((pts[hit] + q[hit]) / 2)
            out_s.append(np.linalg.norm(vel[hit] - vq[hit], axis=1))
    if not out_p:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(out_p), np.concatenate(out_s)


class Effects:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.life = np.zeros(0)
        self.max_life = np.zeros(0)
        self.color = np.zeros((0, 3))
        self.size = np.zeros(0)
        self.rings = []  # [x, y, age, duration, max_radius, color]
        self.glows = []  # (point, radius, color) for this frame
        self.touching = {}  # (id1, id2) -> bool
        self.last_burst = {}
        self.touches = 0  # total touch events (for tests / stats)

    def emit(self, at, n, speed, colors, spread=np.pi * 2, direction=0.0, size=(2, 4.5), life=(0.35, 0.9)):
        n = int(min(n, MAX_PARTICLES - len(self.life)))
        if n <= 0:
            return
        ang = direction + self.rng.uniform(-spread / 2, spread / 2, n)
        spd = self.rng.uniform(speed[0], speed[1], n)
        self.pos = np.vstack([self.pos, np.repeat(np.asarray(at, float)[None], n, 0)])
        self.vel = np.vstack([self.vel, np.c_[np.cos(ang), np.sin(ang)] * spd[:, None]])
        lf = self.rng.uniform(life[0], life[1], n)
        self.life = np.append(self.life, lf)
        self.max_life = np.append(self.max_life, lf)
        pick = self.rng.integers(0, len(colors), n)
        self.color = np.vstack([self.color, np.asarray(colors, float)[pick]])
        self.size = np.append(self.size, self.rng.uniform(size[0], size[1], n))

    def update(self, dt, hands, T, hand_colors):
        self.glows = []
        live_pairs = set()
        for i in range(len(hands)):
            for j in range(i + 1, len(hands)):
                h1, h2 = hands[i], hands[j]
                key = (min(h1.id, h2.id), max(h1.id, h2.id))
                live_pairs.add(key)
                pts, speed = hand_contacts(h1, h2)
                touching = len(pts) > 0
                colors = [hand_colors[i % 2], hand_colors[j % 2], WHITE]
                if touching:
                    center = pts.mean(axis=0)
                    rel = float(speed.max())
                    scale = (h1.scale + h2.scale) / 2
                    new = not self.touching.get(key) and T - self.last_burst.get(key, -1) > TOUCH_COOLDOWN
                    if new:  # the moment of touch: burst + shockwave
                        self.touches += 1
                        self.last_burst[key] = T
                        power = np.clip(rel / 800, 0.4, 2.0)
                        self.emit(center, 70 * power, (250, 900 * power), colors)
                        self.rings.append([*center, 0.0, 0.55, scale * (1.2 + power), colors[0]])
                        self.rings.append([*center, -0.08, 0.5, scale * 0.8 * (1 + power), colors[1]])
                    # Sustained contact: sparks scale with how fast the hands rub.
                    rate = np.clip(rel / 150, 0.5, 10)
                    k = self.rng.integers(0, len(pts), int(rate))
                    for p in pts[k]:
                        self.emit(p, 1, (60, 220 + rel * 0.3), colors, life=(0.2, 0.5), size=(1.5, 3))
                    for p in pts[:: max(1, len(pts) // 6)]:
                        self.glows.append((p, scale * 0.22, np.mean(colors[:2], axis=0) * 0.6))
                    self.glows.append((center, scale * 0.3, WHITE * 0.4))
                self.touching[key] = touching
        for key in list(self.touching):
            if key not in live_pairs:
                del self.touching[key]

        # Integrate particles.
        if len(self.life):
            self.vel[:, 1] += SPARK_GRAVITY * dt
            self.vel *= np.exp(-SPARK_DRAG * dt)
            self.pos += self.vel * dt
            self.life -= dt
            alive = self.life > 0
            for name in ("pos", "vel", "life", "max_life", "color", "size"):
                setattr(self, name, getattr(self, name)[alive])
        for r in self.rings:
            r[2] += dt
        self.rings = [r for r in self.rings if r[2] < r[3]]

    def draw(self, img):
        if not (len(self.life) or self.rings or self.glows):
            return img
        H, W = img.shape[:2]
        layer = np.zeros_like(img)
        for p, rad, col in self.glows:
            cv2.circle(layer, (int(p[0]), int(p[1])), max(2, int(rad)), tuple(float(c) * 0.5 for c in col), -1, cv2.LINE_AA)
        for x, y, age, dur, rmax, col in self.rings:
            if age < 0:
                continue
            f = age / dur
            rad = int(rmax * (1 - (1 - f) ** 3))  # ease-out expansion
            thick = max(1, int(8 * (1 - f)))
            cv2.circle(layer, (int(x), int(y)), max(1, rad), tuple(float(c) * (1 - f) for c in col), thick, cv2.LINE_AA)
        if len(self.life):
            fade = (self.life / self.max_life)[:, None]
            cols = self.color * fade
            for p, c, s in zip(self.pos.astype(int), cols, self.size):
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(layer, (int(p[0]), int(p[1])), int(max(1, s)), tuple(float(v) for v in c), -1, cv2.LINE_AA)
        # Cheap bloom: blur a quarter-size copy and add it back.
        small = cv2.resize(layer, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.add(img, layer)
        return cv2.addWeighted(img, 1.0, bloom, 1.2, 0)
