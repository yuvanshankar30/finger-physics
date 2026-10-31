"""Live-tunable settings shared by the filter, physics, and renderer."""

from dataclasses import dataclass


@dataclass
class Params:
    # Feature toggles
    physics: bool = True
    prediction: bool = True
    joint_limits: bool = True
    show_raw: bool = False
    show_video: bool = True
    show_hud: bool = True

    # Physics skeleton
    stiffness_hz: float = 28.0  # spring natural frequency toward the measurement
    damping: float = 1.0  # damping ratio (1 = critical)
    iterations: int = 6  # constraint solver iterations per substep
    bone_stiffness: float = 0.6  # 0..1, how hard bone lengths are enforced

    # Latency prediction
    lead_ms: float = 12.0  # extra extrapolation on top of measured latency
    velocity_cutoff: float = 15.0  # Hz, low-pass on the velocity estimate

    # One Euro filter
    min_cutoff: float = 1.6
    beta: float = 0.06
    d_cutoff: float = 4.0  # fast speed estimate = filter opens up quickly
