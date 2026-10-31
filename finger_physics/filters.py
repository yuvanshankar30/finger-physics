"""Vectorized One Euro filter (Casiez et al., CHI 2012).

An adaptive low-pass: strong smoothing when a point is still (removes jitter),
weak smoothing when it moves fast (removes lag). Operates element-wise on whole
NumPy arrays, so all 63 landmark coordinates are filtered in one call.
"""

import numpy as np


def lp_alpha(cutoff_hz, dt):
    """Smoothing factor of a first-order low-pass with the given cutoff."""
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff_hz, dtype=float))
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, params):
        """`params` needs min_cutoff, beta, d_cutoff; read live on every call."""
        self.p = params
        self.reset()

    def reset(self):
        self.x = None
        self.dx = None
        self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.t is None:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            self.t = t
            return self.x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = lp_alpha(self.p.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        cutoff = self.p.min_cutoff + self.p.beta * np.abs(self.dx)
        a = lp_alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()
