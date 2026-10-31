"""Synthetic hand motion for the headless self-test."""

import numpy as np

from .handsim import N, rotate


def synthetic_hand(t, size):
    """An open hand that sweeps sideways while its fingers curl and open."""
    W, H = size
    s = H / 480
    p = np.zeros((N, 3))
    p[1:5] = [[-30, -20, 0], [-55, -45, 0], [-75, -70, 0], [-90, -90, 0]]
    for f, dx in enumerate([-30, -10, 10, 30]):
        b = 5 + 4 * f
        p[b : b + 4] = [[dx, -90, 0], [dx, -130, 0], [dx, -155, 0], [dx, -175, 0]]
    bend = np.radians(45 + 40 * np.sin(t * 2.0))
    k = np.array([1.0, 0, 0])
    for b in [5, 9, 13, 17]:
        for j in (b + 1, b + 2):
            p[j + 1 : b + 4] = p[j] + rotate(p[j + 1 : b + 4] - p[j], k, bend)
    p *= 1.6 * s
    return p + [W * (0.5 + 0.25 * np.sin(t * 0.9)), H * 0.8, 0]
