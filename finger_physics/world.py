"""2D ball sandbox the simulated hands can push, flick, and pinch-grab.

Hand bones are kinematic capsules: effectively infinite mass, and they pass
their own velocity to whatever they hit.
"""

import colorsys

import numpy as np

from .handsim import DRAW_BONES

GRAVITY = 1800.0  # px/s²
RESTITUTION = 0.45
WALL_RESTITUTION = 0.5
AIR_DRAG = 0.15  # 1/s
MAX_BALLS = 60
MAX_SPEED = 5000.0
CAPSULE_RADIUS = 0.09  # × hand scale

PINCH_ON = 0.32  # pinch distance / hand scale
PINCH_OFF = 0.45


class World:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.r = np.zeros(0)
        self.color = np.zeros((0, 3), np.uint8)  # BGR
        self.grabbed = np.zeros(0, int)  # hand id, or -1

    @property
    def n(self):
        return len(self.r)

    def add(self, pos, vel, r):
        """Add one ball (mostly for tests)."""
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, vel])
        self.r = np.append(self.r, r)
        self.color = np.vstack([self.color, [[90, 200, 255]]]).astype(np.uint8)
        self.grabbed = np.append(self.grabbed, -1)

    def spawn(self, count, width):
        k = max(0, min(count, MAX_BALLS - self.n))
        if not k:
            return
        r = self.rng.uniform(16, 34, k)
        x = r + self.rng.uniform(0, 1, k) * (width - 2 * r)
        y = -r - self.rng.uniform(0, 300, k)
        colors = []
        for h in self.rng.uniform(0, 1, k):
            rr, gg, bb = colorsys.hsv_to_rgb(h, 0.7, 1.0)
            colors.append([bb * 255, gg * 255, rr * 255])
        self.pos = np.vstack([self.pos, np.c_[x, y]])
        self.vel = np.vstack([self.vel, np.c_[self.rng.uniform(-100, 100, k), np.zeros(k)]])
        self.r = np.append(self.r, r)
        self.color = np.vstack([self.color, np.array(colors, np.uint8)])
        self.grabbed = np.append(self.grabbed, -np.ones(k, int))

    def clear(self):
        self.__init__()

    def release(self, hand):
        self.grabbed[self.grabbed == hand.id] = -1

    def update_grabs(self, hands):
        """Pinch thumb+index (with hysteresis) to grab the nearest ball."""
        for h in hands:
            d = h.pinch_distance()
            was = h.pinching
            h.pinching = d < PINCH_OFF if was else d < PINCH_ON
            h.pinch_point = (h.x[4, :2] + h.x[8, :2]) / 2
            holding = bool(np.any(self.grabbed == h.id))
            if h.pinching and not was and not holding and self.n:
                gap = np.linalg.norm(self.pos - h.pinch_point, axis=1) - self.r
                gap[self.grabbed >= 0] = np.inf
                i = int(np.argmin(gap))
                if gap[i] < 0.6 * h.scale:
                    self.grabbed[i] = h.id
            if not h.pinching and holding:
                self.release(h)

    def step(self, dt, hands, W, H):
        if not self.n:
            return
        pos, vel, r = self.pos, self.vel, self.r
        free = self.grabbed < 0

        vel[free, 1] += GRAVITY * dt
        vel[free] *= np.exp(-AIR_DRAG * dt)
        w = 2 * np.pi * 12
        for h in hands:
            m = self.grabbed == h.id
            if m.any():  # stiff critically damped spring to the pinch point
                vel[m] += (w * w * (h.pinch_point - pos[m]) - 2 * w * vel[m]) * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel *= np.minimum(1.0, MAX_SPEED / np.maximum(speed, 1e-9))
        pos += vel * dt

        self._collide_balls(free)
        for h in hands:
            self._collide_hand(h)

        # Walls; the top is open so spawned balls can fall in.
        lo = pos[:, 0] < r
        pos[lo, 0] = r[lo]
        vel[lo & (vel[:, 0] < 0), 0] *= -WALL_RESTITUTION
        hi = pos[:, 0] > W - r
        pos[hi, 0] = W - r[hi]
        vel[hi & (vel[:, 0] > 0), 0] *= -WALL_RESTITUTION
        fl = pos[:, 1] > H - r
        pos[fl, 1] = H - r[fl]
        vel[fl & (vel[:, 1] > 0), 1] *= -WALL_RESTITUTION
        vel[fl, 0] *= 0.995

    def _collide_balls(self, free):
        n = self.n
        if n < 2:
            return
        pos, vel, r = self.pos, self.vel, self.r
        d = pos[None, :, :] - pos[:, None, :]  # d[i, j] = p_j - p_i
        dist = np.linalg.norm(d, axis=2)
        R = r[:, None] + r[None, :]
        inv_m = np.where(free, 1.0 / (r * r), 0.0)  # mass ∝ area; held = kinematic
        wsum = inv_m[:, None] + inv_m[None, :]
        hit = (dist < R) & (dist > 1e-6) & (wsum > 0) & np.triu(np.ones((n, n), bool), 1)
        I, J = np.nonzero(hit)
        if not len(I):
            return
        nrm = d[I, J] / dist[I, J][:, None]
        ws = wsum[I, J]
        pen = (R[I, J] - dist[I, J]) / ws
        corr = nrm * pen[:, None]
        np.add.at(pos, I, -corr * inv_m[I][:, None])
        np.add.at(pos, J, corr * inv_m[J][:, None])
        rel = np.sum((vel[J] - vel[I]) * nrm, axis=1)
        imp = np.where(rel < 0, -(1 + RESTITUTION) * rel / ws, 0.0)[:, None] * nrm
        np.add.at(vel, I, -imp * inv_m[I][:, None])
        np.add.at(vel, J, imp * inv_m[J][:, None])

    def _collide_hand(self, h):
        mask = self.grabbed != h.id
        if not mask.any():
            return
        idx = np.nonzero(mask)[0]
        P = self.pos[idx]
        A = h.x[DRAW_BONES[:, 0], :2]
        B = h.x[DRAW_BONES[:, 1], :2]
        VA = h.v[DRAW_BONES[:, 0], :2]
        VB = h.v[DRAW_BONES[:, 1], :2]
        AB = B - A
        ab2 = np.maximum(np.sum(AB * AB, axis=1), 1e-9)
        t = np.clip(np.einsum("nsk,sk->ns", P[:, None, :] - A[None], AB) / ab2, 0, 1)
        Q = A[None] + t[..., None] * AB[None]  # closest points (n, S, 2)
        D = P[:, None, :] - Q
        dist = np.linalg.norm(D, axis=2)
        R = self.r[idx][:, None] + CAPSULE_RADIUS * h.scale
        pen = R - dist
        s = np.argmax(pen, axis=1)  # deepest contact per ball
        k = np.arange(len(idx))
        hit = pen[k, s] > 0
        if not hit.any():
            return
        k, s = k[hit], s[hit]
        bi = idx[hit]
        d, dd = D[k, s], dist[k, s]
        perp = np.c_[-AB[s, 1], AB[s, 0]]
        nrm = np.where((dd > 1e-6)[:, None], d / np.maximum(dd, 1e-9)[:, None], normalize2(perp))
        self.pos[bi] = Q[k, s] + nrm * R[k, 0][:, None]
        tt = t[k, s][:, None]
        vq = VA[s] * (1 - tt) + VB[s] * tt  # bone velocity at contact
        rel = self.vel[bi] - vq
        rn = np.sum(rel * nrm, axis=1)
        v = self.vel[bi] - np.where(rn < 0, (1 + RESTITUTION) * rn, 0.0)[:, None] * nrm
        tan = np.c_[-nrm[:, 1], nrm[:, 0]]
        rt = np.sum((v - vq) * tan, axis=1)
        self.vel[bi] = v - 0.08 * rt[:, None] * tan  # a little friction


def normalize2(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
