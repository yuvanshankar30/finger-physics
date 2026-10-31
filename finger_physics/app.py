"""Main loop: ingest detections, run physics at 240 Hz, render, handle keys."""

import time

import cv2
import numpy as np

from .camera import Camera, SyntheticSource
from .effects import Effects
from .handsim import HandSim
from .params import Params
from .render import HAND_COLORS, draw_hud, draw_scene
from .tracker import Tracker
from .world import World

WINDOW = "Finger Physics"
SUBSTEP = 1 / 240
MAX_SUBSTEPS = 24
HAND_TIMEOUT = 0.3  # s without a detection before a hand is dropped


class App:
    def __init__(self, args):
        self.args = args
        self.params = Params()
        self.world = World()
        self.effects = Effects()
        self.last_log = time.perf_counter()
        self.hands = []
        self.det_seq = 0
        self.fps = 0.0

        if args.selftest:
            self.source = SyntheticSource(args.width, args.height, args.fps)
        else:
            src = args.video if args.video else args.camera
            self.source = Camera(src, args.width, args.height, args.fps)
        self.source.start()

        deadline = time.perf_counter() + 6
        while self.source.latest()[0] is None:
            if self.source.error or time.perf_counter() > deadline:
                raise SystemExit(
                    self.source.error
                    or "No frames from the camera. On macOS, allow camera access for your "
                    "terminal app in System Settings → Privacy & Security → Camera."
                )
            time.sleep(0.02)
        frame = self.source.latest()[0]
        fh, fw = frame.shape[:2]
        self.size = (int(fw * args.scale), int(fh * args.scale))

        if isinstance(self.source, Camera) and not self.source.is_file:
            cap = self.source.cap
            print(
                f"camera: {fw}x{fh}, driver reports {cap.get(cv2.CAP_PROP_FPS):.0f} fps "
                f"(requested {args.width}x{args.height} @ {args.fps})",
                flush=True,
            )
        self.tracker = Tracker(self.source, args.hands, use_gpu=not args.cpu)
        self.tracker.start()

    # ------------------------------------------------------------ tracking
    def to_screen(self, pts):
        """Normalized landmarks (already mirrored) → display pixels."""
        W, H = self.size
        return np.c_[pts[:, 0] * W, pts[:, 1] * H, pts[:, 2] * W]

    def ingest(self, det):
        dets = [{"pts": self.to_screen(p), "label": lab, "used": False} for p, lab in det.hands]
        # Greedy nearest-wrist matching keeps each physical hand on one sim.
        pairs = sorted(
            (np.linalg.norm(d["pts"][0] - h.f[0]), hi, di)
            for hi, h in enumerate(self.hands)
            for di, d in enumerate(dets)
        )
        matched = set()
        for dist, hi, di in pairs:
            if hi in matched or dets[di]["used"] or dist > 6 * self.hands[hi].scale:
                continue
            matched.add(hi)
            dets[di]["used"] = True
            self.hands[hi].ingest(dets[di]["pts"], det.capture_time, dets[di]["label"])
        for d in dets:
            if not d["used"] and len(self.hands) < self.args.hands:
                h = HandSim(self.params)
                h.ingest(d["pts"], det.capture_time, d["label"])
                self.hands.append(h)

    def drop_stale(self, T):
        for h in [h for h in self.hands if T - h.meas_time > HAND_TIMEOUT]:
            self.world.release(h)
            self.hands.remove(h)

    # ------------------------------------------------------------ loop
    def tick(self, T, dt, acc):
        got = self.tracker.poll(self.det_seq)
        if got:
            self.det_seq, det = got
            self.ingest(det)
        self.drop_stale(T)
        for h in self.hands:
            h.update_targets(T)
        self.world.update_grabs(self.hands)

        acc += dt
        steps = 0
        while acc >= SUBSTEP and steps < MAX_SUBSTEPS:
            for h in self.hands:
                h.step(SUBSTEP)
            self.world.step(SUBSTEP, self.hands, *self.size)
            acc -= SUBSTEP
            steps += 1
        if steps == MAX_SUBSTEPS:
            acc = 0.0
        self.effects.update(dt, self.hands, T, HAND_COLORS)

        frame = self.source.latest()[0]  # newest frame, not the detected one
        img = draw_scene(frame, self.size, self.hands, self.world, self.params, T, HAND_TIMEOUT)
        img = self.effects.draw(img)
        stats = {
            "cam_fps": self.source.fps,
            "infer_ms": self.tracker.infer_ms,
            "latency_ms": self.tracker.latency_ms,
            "fps": self.fps,
            "physics_hz": 1 / SUBSTEP,
            "backend": self.tracker.backend,
        }
        if not self.args.selftest and T - self.last_log > 10:
            self.last_log = T
            print(
                f"cam {stats['cam_fps']:.0f} fps | infer {stats['infer_ms']:.1f} ms ({stats['backend']}) | "
                f"cam->pose {stats['latency_ms']:.0f} ms | render {stats['fps']:.0f} fps | "
                f"hands {len(self.hands)} | touches {self.effects.touches}",
                flush=True,
            )
        return draw_hud(img, stats, self.hands, self.params), acc

    def handle_key(self, key):
        p = self.params
        c = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or c == "q":
            return False
        toggles = {"p": "physics", "r": "prediction", "l": "joint_limits",
                   "w": "show_raw", "v": "show_video", "h": "show_hud"}
        if c in toggles:
            setattr(p, toggles[c], not getattr(p, toggles[c]))
        elif c == "b":
            self.world.spawn(10, self.size[0])
        elif c == "c":
            self.world.clear()
        elif c == "[":
            p.stiffness_hz = max(2.0, p.stiffness_hz - 2)
        elif c == "]":
            p.stiffness_hz = min(60.0, p.stiffness_hz + 2)
        elif c == "-":
            p.lead_ms = max(0.0, p.lead_ms - 2)
        elif c == "=":
            p.lead_ms = min(80.0, p.lead_ms + 2)
        elif c == "9":
            p.min_cutoff = max(0.2, round(p.min_cutoff - 0.2, 2))
        elif c == "0":
            p.min_cutoff = min(8.0, round(p.min_cutoff + 0.2, 2))
        return True

    def run(self):
        self.world.spawn(12, self.size[0])
        if self.args.selftest:
            return self.run_selftest()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        min_dt = 1 / self.args.max_fps
        last = time.perf_counter()
        acc = 0.0
        try:
            while True:
                now = time.perf_counter()
                dt = min(0.1, now - last)
                last = now
                if dt > 0:
                    self.fps += 0.05 * (1 / dt - self.fps)
                img, acc = self.tick(now, dt, acc)
                cv2.imshow(WINDOW, img)
                if not self.handle_key(cv2.waitKey(1)):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                # Cap the render rate so we don't starve the detection thread.
                spare = min_dt - (time.perf_counter() - now)
                if spare > 0:
                    time.sleep(spare)
        finally:
            self.close()

    def run_selftest(self):
        """Headless: run the real model on blank frames, plus a synthetic hand
        fed straight into the physics, then save a screenshot."""
        from .selftest import synthetic_pair

        t_end = time.perf_counter() + self.args.selftest
        last = time.perf_counter()
        acc = 0.0
        frames = 0
        next_fake = 0.0
        img = None
        render_ms = []
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            dt = min(0.1, now - last)
            last = now
            if dt > 0:
                self.fps += 0.05 * (1 / dt - self.fps)
            if now >= next_fake:  # 30 Hz fake "detections"
                next_fake = now + 1 / 30
                while len(self.hands) < 2:
                    self.hands.append(HandSim(self.params))
                for h, pts, lab in zip(self.hands, synthetic_pair(now, self.size), ("Left", "Right")):
                    h.ingest(pts, now, lab)
            t0 = time.perf_counter()
            img, acc = self.tick(now, dt, acc)
            render_ms.append((time.perf_counter() - t0) * 1000)
            frames += 1
            time.sleep(max(0.0, 1 / 120 - (time.perf_counter() - now)))
        out = self.args.screenshot
        if out and img is not None:
            cv2.imwrite(out, img)
        h = self.hands[0] if self.hands else None
        print(
            f"selftest ok: {frames} frames, render {self.fps:.0f} fps, "
            f"tick {np.mean(render_ms):.2f} ms avg / {np.percentile(render_ms, 95):.2f} ms p95, "
            f"model infer {self.tracker.infer_ms:.1f} ms on {self.tracker.backend}, "
            f"detections {self.det_seq}, flex_sign {h.flex_sign if h else None}, "
            f"balls {self.world.n}, touches {self.effects.touches}" + (f", screenshot {out}" if out else "")
        )
        self.close()

    def close(self):
        self.tracker.stop()
        self.source.stop()
        cv2.destroyAllWindows()

