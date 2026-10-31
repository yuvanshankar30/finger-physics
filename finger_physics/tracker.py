"""MediaPipe hand-landmark detection on a dedicated thread."""

import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "finger-physics" / "hand_landmarker.task"


def ensure_model(path=MODEL_PATH):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {path} …")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not download the hand model ({e}).\n"
            f"Download it manually from {MODEL_URL}\nand save it as {path}"
        ) from e
    tmp.rename(path)
    return path


@dataclass
class Detection:
    hands: list = field(default_factory=list)  # [(pts (21,3) normalized, label)]
    capture_time: float = 0.0
    done_time: float = 0.0
    infer_ms: float = 0.0


class Tracker(threading.Thread):
    def __init__(self, source, num_hands=2, use_gpu=True, model_path=None):
        super().__init__(daemon=True)
        self.source = source
        self.running = True
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.infer_ms = 0.0
        self.latency_ms = 0.0
        self.backend = "?"
        self.model = ensure_model(model_path or MODEL_PATH)
        self.num_hands = num_hands
        self.landmarker = self._create(self.model, num_hands, use_gpu)

    def _create(self, model, num_hands, use_gpu):
        def opts(delegate):
            return vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model), delegate=delegate),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        if use_gpu:
            try:
                lm = vision.HandLandmarker.create_from_options(opts(BaseOptions.Delegate.GPU))
                self.backend = "GPU"
                return lm
            except Exception as e:  # noqa: BLE001
                print(f"GPU delegate unavailable ({e}); using CPU.")
        self.backend = "CPU"
        return vision.HandLandmarker.create_from_options(opts(BaseOptions.Delegate.CPU))

    def run(self):
        last_id = -1
        last_ts = -1
        while self.running:
            frame, t_cap, fid = self.source.wait_newer(last_id)
            if frame is None or fid == last_id:
                continue
            last_id = fid
            if self.backend == "GPU":  # GPU delegate only accepts 4-channel input
                image = mp.Image(image_format=mp.ImageFormat.SRGBA,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            else:
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(last_ts + 1, int(t_cap * 1000))
            last_ts = ts
            t0 = time.perf_counter()
            try:
                res = self.landmarker.detect_for_video(image, ts)
            except Exception as e:  # noqa: BLE001
                if self.backend != "GPU":
                    raise
                print(f"GPU inference failed ({e}); switching to CPU.")
                self.landmarker.close()
                self.landmarker = self._create(self.model, self.num_hands, use_gpu=False)
                continue
            t1 = time.perf_counter()
            hands = []
            for i, lms in enumerate(res.hand_landmarks or []):
                pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
                cats = res.handedness[i] if i < len(res.handedness) else []
                # Frames are mirrored before detection, matching MediaPipe's
                # selfie-view assumption, so labels are the user's real hands.
                hands.append((pts, cats[0].category_name if cats else ""))
            det = Detection(hands, t_cap, t1, (t1 - t0) * 1000)
            a = 0.1
            self.infer_ms += a * (det.infer_ms - self.infer_ms)
            self.latency_ms += a * ((t1 - t_cap) * 1000 - self.latency_ms)
            with self.lock:
                self.latest = det
                self.seq += 1

    def poll(self, last_seq):
        """Return (seq, detection) if a newer detection exists, else None."""
        with self.lock:
            if self.seq != last_seq and self.latest is not None:
                return self.seq, self.latest
        return None

    def stop(self):
        self.running = False
        self.join(timeout=1)
        self.landmarker.close()
