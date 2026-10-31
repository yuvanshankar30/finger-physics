"""Threaded frame sources that always expose only the newest frame.

Reading the camera on its own thread means detection and rendering never wait
on capture, and stale frames are dropped instead of queuing up (queued frames
are pure latency).
"""

import sys
import threading
import time

import cv2
import numpy as np


class FrameSource(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cond = threading.Condition()
        self.frame = None
        self.capture_time = 0.0
        self.frame_id = -1
        self.running = True
        self.fps = 0.0
        self.error = None

    def _publish(self, frame, t):
        with self.cond:
            if self.frame_id >= 0 and t > self.capture_time:
                inst = 1.0 / (t - self.capture_time)
                self.fps = inst if not self.fps else self.fps + 0.05 * (inst - self.fps)
            self.frame = frame
            self.capture_time = t
            self.frame_id += 1
            self.cond.notify_all()

    def latest(self):
        with self.cond:
            return self.frame, self.capture_time, self.frame_id

    def wait_newer(self, last_id, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id or not self.running, timeout)
            return self.frame, self.capture_time, self.frame_id

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()


class Camera(FrameSource):
    """Webcam (int index) or video file (path). Frames are mirrored."""

    def __init__(self, source=0, width=640, height=480, fps=60, mirror=True):
        super().__init__()
        self.is_file = isinstance(source, str)
        if not self.is_file and sys.platform == "darwin":
            self.cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.is_file:
            # Small frames + high fps = less inference time and less latency.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.file_dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.is_file else 0
        self.mirror = mirror

    def run(self):
        if not self.cap.isOpened():
            self.error = "Could not open the camera/video source."
            self.stop()
            return
        next_t = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                if self.is_file:  # loop the clip
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.005)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self._publish(frame, t)
            if self.is_file:  # pace file playback at its native fps
                next_t += self.file_dt
                time.sleep(max(0.0, next_t - time.perf_counter()))
        self.cap.release()


class SyntheticSource(FrameSource):
    """Blank frames at a fixed rate, for headless self-tests."""

    def __init__(self, width=640, height=480, fps=60):
        super().__init__()
        self.size = (height, width, 3)
        self.dt = 1.0 / fps

    def run(self):
        img = np.full(self.size, 30, np.uint8)
        while self.running:
            self._publish(img, time.perf_counter())
            time.sleep(self.dt)
