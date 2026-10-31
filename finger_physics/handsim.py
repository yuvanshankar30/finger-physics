"""Physically simulated hand skeleton.

Each of the 21 MediaPipe landmarks is a particle. Particles are pulled toward
the (filtered + latency-predicted) measurement by a critically damped spring
with velocity feed-forward. Position-based-dynamics constraints then enforce:

* consistent bone lengths, learned per user as ratios of palm size, and
* anatomical joint limits (no impossible hyperextension or over-bending).

Everything is vectorized with NumPy so the solver can run at 240 Hz in Python.
"""

import numpy as np

from .filters import OneEuro, lp_alpha

N = 21
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIPS = [4, 8, 12, 16, 20]
BASES = np.array([5, 9, 13, 17])
DEG = np.pi / 180.0

CHAIN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
PALM = [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
BRACE = [(0, 9), (0, 13), (5, 17), (1, 5)]

DRAW_BONES = np.array(CHAIN + PALM)  # drawn and used as collision capsules
CONSTRAINTS = np.array(CHAIN + PALM + BRACE)  # all distance constraints
CA, CB = CONSTRAINTS[:, 0], CONSTRAINTS[:, 1]
M = len(CONSTRAINTS)

# Palm bones are nearly rigid: their mean length is the hand's "scale".
SCALE_PAIRS = np.array([(0, 5), (0, 9), (0, 13), (0, 17), (5, 17)])

# Heavy wrist/knuckles so constraints bend fingers rather than the palm.
INV_MASS = np.ones(N)
INV_MASS[0] = 0.15
INV_MASS[[1, 5, 9, 13, 17]] = 0.4

# Jacobi solver scatter matrices: (N, M) @ (M, 3) accumulates per-particle
# corrections, averaged over how many constraints touch each particle.
_counts = np.bincount(CONSTRAINTS.ravel(), minlength=N).astype(float)
_GA = np.zeros((N, M))
_GA[CA, np.arange(M)] = 1.0
_GB = np.zeros((N, M))
_GB[CB, np.arange(M)] = 1.0
_GA /= _counts[:, None]
_GB /= _counts[:, None]
OMEGA = 1.5  # Jacobi over-relaxation

# Reported joints: (finger, name, a, b, c) with the angle measured at b.
JOINTS = [(0, "MCP", 1, 2, 3), (0, "IP", 2, 3, 4)]
for _f, _b in enumerate(BASES):
    _b = int(_b)
    JOINTS += [
        (_f + 1, "MCP", 0, _b, _b + 1),
        (_f + 1, "PIP", _b, _b + 1, _b + 2),
        (_f + 1, "DIP", _b + 1, _b + 2, _b + 3),
    ]
J_A = np.array([j[2] for j in JOINTS])
J_B = np.array([j[3] for j in JOINTS])
J_C = np.array([j[4] for j in JOINTS])
J_LIMITED = np.array([j[0] != 0 for j in JOINTS])


class _Level:
    """One joint level (MCP, PIP or DIP) across the four fingers."""

    def __init__(self, off, flex_deg, ext_deg):
        self.a = np.zeros(4, int) if off == 0 else BASES + off - 1
        self.b = BASES + off
        self.c = BASES + off + 1
        self.desc = BASES[:, None] + np.arange(off + 1, 4)[None, :]  # distal points
        self.flex = flex_deg * DEG
        self.ext = ext_deg * DEG


LEVELS = [_Level(0, 100, 45), _Level(1, 120, 12), _Level(2, 95, 25)]
PIP = LEVELS[1]


def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def signed_angle(u, w, k):
    """Signed angle from u to w about unit axis k, measured in the plane ⟂ k."""
    up = u - (u @ k)[..., None] * k
    wp = w - (w @ k)[..., None] * k
    return np.arctan2(np.cross(up, wp) @ k, np.sum(up * wp, axis=-1))


def rotate(v, k, th):
    """Rodrigues rotation of vectors v about unit axis k by angles th."""
    th = np.asarray(th, dtype=float)
    c = np.cos(th)[..., None]
    s = np.sin(th)[..., None]
    return v * c + np.cross(k, v) * s + k * (v @ k)[..., None] * (1 - c)


class HandSim:
    _next_id = 1

    def __init__(self, params):
        self.p = params
        z = lambda: np.zeros((N, 3))  # noqa: E731
        self.x, self.v = z(), z()  # simulated state
        self.raw, self.f, self.fv = z(), z(), z()  # raw / filtered / velocity
        self.target, self.tv = z(), z()
        self.filter = OneEuro(params)
        self.ratio = np.zeros(M)
        self.calib = 0
        self.scale = 1.0
        self.initialized = False
        self.meas_time = 0.0
        self.prev_meas_time = None
        self.flex_evidence = 0.0
        self.flex_sign = 0
        self.label = ""
        self.pinching = False
        self.gripping = False
        self.cupping = False
        self.pinch_point = np.zeros(2)
        self.hold_point = np.zeros(2)
        self.id = HandSim._next_id
        HandSim._next_id += 1

    # ------------------------------------------------------------ measurement
    def ingest(self, pts, t, label=""):
        """Feed one detection: pts (21, 3) in screen pixels, t capture time (s)."""
        pts = np.asarray(pts, dtype=float).reshape(N, 3)
        if label != self.label:
            self.label = label
            self.flex_evidence = 0.0
            self.flex_sign = 0
        self.raw = pts.copy()

        snap = not self.initialized or np.linalg.norm(pts[0] - self.f[0]) > 4 * self.scale
        if snap:
            self.filter.reset()
            self.prev_meas_time = None

        prev = self.f
        f = self.filter(pts, t)
        if self.prev_meas_time is None:
            self.fv = np.zeros((N, 3))
        else:
            dt = max(1e-3, t - self.prev_meas_time)
            a = lp_alpha(self.p.velocity_cutoff, dt)
            self.fv = a * (f - prev) / dt + (1 - a) * self.fv
        self.f = f
        self.prev_meas_time = t
        self.meas_time = t

        palm = np.linalg.norm(f[SCALE_PAIRS[:, 1]] - f[SCALE_PAIRS[:, 0]], axis=1)
        self.scale = max(8.0, float(palm.mean()))

        r = np.linalg.norm(f[CB] - f[CA], axis=1) / self.scale
        if self.calib == 0:
            self.ratio = r
        else:
            alpha = 0.15 if self.calib < 30 else 0.01
            ok = np.ones(M, bool)
            if self.calib > 20:  # reject outliers once calibrated
                ok = np.abs(r - self.ratio) / np.maximum(self.ratio, 1e-6) <= 0.35
            self.ratio[ok] += alpha * (r[ok] - self.ratio[ok])
        self.calib += 1

        self._learn_flex_sign()

        if snap:
            self.x = f.copy()
            self.v = np.zeros((N, 3))
            self.initialized = True

    @staticmethod
    def lateral_axis(p):
        """Knuckle axis, index MCP → pinky MCP: fingers flex about this."""
        return normalize(p[17] - p[5])

    def _learn_flex_sign(self):
        """Which way about the knuckle axis is "flexion" depends on handedness
        and mirroring. Learn it: PIP joints barely hyperextend, so any large
        PIP bend is almost certainly flexion."""
        f = self.f
        s = signed_angle(f[PIP.b] - f[PIP.a], f[PIP.c] - f[PIP.b], self.lateral_axis(f))
        big = np.abs(s) > 35 * DEG
        self.flex_evidence = float(np.clip(self.flex_evidence + s[big].sum(), -20, 20))
        if abs(self.flex_evidence) > 3:
            self.flex_sign = int(np.sign(self.flex_evidence))

    # ------------------------------------------------------------ simulation
    def update_targets(self, T):
        """Targets for render time T: extrapolate across pipeline latency."""
        h = 0.0
        if self.p.prediction:
            h = float(np.clip(T - self.meas_time + self.p.lead_ms / 1000.0, 0.0, 0.1))
        stale = T - self.meas_time
        vel_scale = max(0.0, 1 - (stale - 0.12) / 0.1) if stale > 0.12 else 1.0
        shift = self.fv * h
        mag = np.linalg.norm(shift, axis=1, keepdims=True)
        max_shift = 0.8 * self.scale
        shift *= np.minimum(1.0, max_shift / np.maximum(mag, 1e-9))
        self.target = self.f + shift * vel_scale
        self.tv = self.fv * vel_scale

    def step(self, dt):
        p = self.p
        if not p.physics:
            self.x = self.target.copy()
            self.v = self.tv.copy()
            return
        # Critically damped spring with velocity feed-forward: a constantly
        # moving target is tracked with zero steady-state lag.
        w = 2 * np.pi * p.stiffness_hz
        acc = w * w * (self.target - self.x) + 2 * w * p.damping * (self.tv - self.v)
        v = self.v + acc * dt
        q = self.x + v * dt
        if self.calib >= 3:
            for _ in range(p.iterations):
                q = self._solve_bones(q)
                if p.joint_limits and self.flex_sign:
                    q = self._solve_limits(q)
        self.v = (q - self.x) / dt
        self.x = q

    def _solve_bones(self, q):
        d = q[CB] - q[CA]
        length = np.linalg.norm(d, axis=1)
        rest = self.ratio * self.scale
        wa, wb = INV_MASS[CA], INV_MASS[CB]
        s = OMEGA * self.p.bone_stiffness * (length - rest) / np.maximum(length, 1e-6) / (wa + wb)
        return q + _GA @ (d * (s * wa)[:, None]) - _GB @ (d * (s * wb)[:, None])

    def _solve_limits(self, q):
        q = q.copy()
        L = self.lateral_axis(q)
        sgn = self.flex_sign
        for lv in LEVELS:
            s = signed_angle(q[lv.b] - q[lv.a], q[lv.c] - q[lv.b], L) * sgn
            delta = (np.clip(s, -lv.ext, lv.flex) - s) * sgn * 0.5
            if not delta.any():
                continue
            pivot = q[lv.b][:, None, :]
            rel = q[lv.desc] - pivot
            q[lv.desc] = pivot + rotate(rel, L, np.broadcast_to(delta[:, None], rel.shape[:2]))
        return q

    # ------------------------------------------------------------ readouts
    def angles(self):
        """Flexion angle (degrees) of every joint in JOINTS."""
        x = self.x
        u = x[J_B] - x[J_A]
        w = x[J_C] - x[J_B]
        cos = np.sum(u * w, axis=1) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-9
        )
        out = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if self.flex_sign:
            signed = signed_angle(u, w, self.lateral_axis(x)) * self.flex_sign
            out[J_LIMITED] = np.degrees(signed[J_LIMITED])
        return out

    def grip_ratio(self):
        """Mean fingertip distance from the palm centre, in hand-scale units.
        About 1.2 with the hand open, under 0.8 when it is closed."""
        c = self.x[[0, 5, 9, 13, 17]].mean(axis=0)
        return float(np.linalg.norm(self.x[[8, 12, 16, 20]] - c, axis=1).mean() / self.scale)

    def grip_point(self):
        """Where a gripped ball sits: palm centre, shifted toward the knuckles."""
        c = self.x[[0, 5, 9, 13, 17], :2].mean(axis=0)
        return 0.4 * c + 0.6 * self.x[[5, 9, 13, 17], :2].mean(axis=0)

    def pinch_distance(self):
        """Thumb-tip to index-tip distance in units of hand scale."""
        return float(np.linalg.norm(self.x[4] - self.x[8]) / self.scale)
