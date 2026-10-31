import numpy as np

from finger_physics.filters import OneEuro
from finger_physics.handsim import CONSTRAINTS, JOINTS, N, HandSim, rotate, signed_angle
from finger_physics.params import Params
from finger_physics.world import World


def open_hand(ox=300.0, oy=400.0):
    """Flat open hand in pixels, fingers pointing up (-y)."""
    p = np.zeros((N, 3))
    p[0] = [0, 0, 0]
    p[1:5] = [[-30, -20, 0], [-55, -45, 0], [-75, -70, 0], [-90, -90, 0]]
    for f, dx in enumerate([-30, -10, 10, 30]):
        b = 5 + 4 * f
        p[b : b + 4] = [[dx, -90, 0], [dx, -130, 0], [dx, -155, 0], [dx, -175, 0]]
    return p + [ox, oy, 0]


def curl(p, deg):
    """Bend every finger's PIP and DIP by `deg` about the knuckle (x) axis."""
    out = p.copy()
    k = np.array([1.0, 0, 0])
    for b in [5, 9, 13, 17]:
        for j in (b + 1, b + 2):
            out[j + 1 : b + 4] = out[j] + rotate(out[j + 1 : b + 4] - out[j], k, np.radians(deg))
    return out


def joint(finger, name):
    return next(i for i, j in enumerate(JOINTS) if j[0] == finger and j[1] == name)


def test_one_euro_suppresses_jitter():
    f = OneEuro(Params(min_cutoff=1, beta=0.01))
    ys = [f(100 + (2 if i % 2 else -2), i / 60) for i in range(300)]
    assert max(abs(y - 100) for y in ys[60:]) < 0.5


def test_one_euro_keeps_up_with_fast_motion():
    f = OneEuro(Params(min_cutoff=1, beta=0.05))
    for i in range(120):
        y = f(i * 20.0, i / 60)  # 1200 px/s
    assert 119 * 20 - y < 30


def test_signed_angle_matches_rotate():
    k = np.array([0, 0, 1.0])
    v = rotate(np.array([1.0, 0, 0]), k, 0.7)
    assert abs(signed_angle(np.array([1.0, 0, 0]), v, k) - 0.7) < 1e-9


def test_converges_on_static_hand():
    h = HandSim(Params(lead_ms=0))
    hand = open_hand()
    for i in range(60):
        h.ingest(hand, i / 30, "Right")
    h.update_targets(2.0)
    for _ in range(480):
        h.step(1 / 240)
    assert np.abs(h.x - hand).max() < 1


def test_bone_constraint_rejects_length_glitch():
    h = HandSim(Params(lead_ms=0))
    hand = open_hand()
    for i in range(60):
        h.ingest(hand, i / 30, "Right")
    glitch = hand.copy()
    glitch[8, 1] -= 60  # fingertip suddenly stretched out
    h.ingest(glitch, 61 / 30, "Right")
    h.update_targets(61 / 30)
    for _ in range(8):
        h.step(1 / 240)
    j = next(i for i, (a, b) in enumerate(CONSTRAINTS) if a == 7 and b == 8)
    rest = h.ratio[j] * h.scale
    assert abs(np.linalg.norm(h.x[8] - h.x[7]) - rest) / rest < 0.5


def test_learns_flexion_and_reports_curl():
    h = HandSim(Params(lead_ms=0))
    t = 0.0
    for hand in [open_hand()] * 30 + [curl(open_hand(), 70)] * 30:
        t += 1 / 30
        h.ingest(hand, t, "Right")
    assert h.flex_sign != 0
    h.update_targets(t)
    for _ in range(480):
        h.step(1 / 240)
    assert abs(h.angles()[joint(2, "PIP")] - 70) < 8


def test_joint_limits_block_hyperextension():
    h = HandSim(Params(lead_ms=0))
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        h.ingest(curl(open_hand(), 70), t, "Right")
    for _ in range(4):
        t += 1 / 30
        h.ingest(curl(open_hand(), -60), t, "Right")  # bent backwards
    h.update_targets(t)
    for _ in range(240):
        h.step(1 / 240)
    assert h.angles()[joint(2, "PIP")] > -25


def test_moving_hand_knocks_ball():
    w = World(seed=0)
    w.add([330, 250], [0, 0], 20)
    h = HandSim(Params(physics=False, lead_ms=0))
    t = 0.0
    for i in range(30):
        t += 1 / 30
        h.ingest(open_hand(150 + i * 12, 400), t, "Right")
        h.update_targets(t)
        for _ in range(8):
            h.step(1 / 240)
            w.step(1 / 240, [h], 2000, 270)  # floor keeps the ball in the hand's path
    assert w.pos[0, 0] > 360 or abs(w.vel[0, 0]) > 50


def test_pinch_grabs_and_releases_ball():
    w = World(seed=0)
    hand = open_hand()
    pinch = hand.copy()
    pinch[4] = pinch[8] = (hand[4] + hand[8]) / 2
    w.add(pinch[4, :2], [0, 0], 20)
    h = HandSim(Params(physics=False))
    h.ingest(pinch, 0.0)
    h.update_targets(0.0)
    h.step(1 / 240)
    w.update_grabs([h])
    assert w.grabbed[0] == h.id
    h.ingest(hand, 0.03)
    h.update_targets(0.03)
    h.step(1 / 240)
    w.update_grabs([h])
    assert w.grabbed[0] == -1


def test_balls_settle_on_floor():
    w = World(seed=1)
    w.spawn(20, 800)
    for _ in range(240 * 4):
        w.step(1 / 240, [], 800, 600)
    assert np.all(w.pos[:, 1] <= 600 - w.r + 1e-6)
    assert np.abs(w.vel).max() < 200


def _settled(points, params=None):
    h = HandSim(params or Params(physics=False))
    h.ingest(points, 0.0)
    h.update_targets(0.0)
    h.step(1 / 240)
    return h


def test_closing_hand_grips_ball_and_opening_throws_it():
    w = World(seed=0)
    open_h = open_hand()
    fist = curl(open_hand(), 100)
    # Close the hand fully at the MCPs too.
    k = np.array([1.0, 0, 0])
    for b in [5, 9, 13, 17]:
        fist[b + 1 : b + 4] = fist[b] + rotate(fist[b + 1 : b + 4] - fist[b], k, np.radians(60))
    h = _settled(open_h)
    w.add(h.grip_point(), [0, 0], 22)
    w.update_grabs([h])
    assert h.grip_ratio() > 1.1 and w.grabbed[0] == -1  # open hand: no grab
    h = _settled(fist)
    h.id = 7
    w.update_grabs([h])
    assert h.gripping and w.grabbed[0] == 7
    h.ingest(open_h, 0.05)
    h.update_targets(0.05)
    h.step(1 / 240)
    w.update_grabs([h])
    assert w.grabbed[0] == -1


def test_touching_hands_trigger_effects():
    from finger_physics.effects import Effects, hand_contacts

    a = _settled(open_hand(300, 400))
    far = _settled(open_hand(700, 400))
    assert len(hand_contacts(a, far)[0]) == 0
    near = _settled(open_hand(355, 400))  # fingers overlap
    pts, _ = hand_contacts(a, near)
    assert len(pts) > 0
    fx = Effects(seed=0)
    fx.update(1 / 60, [a, near], 1.0, [(255, 200, 70), (40, 170, 255)])
    assert fx.touches == 1 and len(fx.life) > 20 and fx.rings
    fx.update(1 / 60, [a, near], 1.02, [(255, 200, 70), (40, 170, 255)])
    assert fx.touches == 1  # sustained contact is not a new touch
    img = fx.draw(np.zeros((480, 640, 3), np.uint8))
    assert img.max() > 0
