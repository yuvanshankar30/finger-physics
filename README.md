# Finger Physics

A Python desktop app that tracks finger joints from your webcam in real time. A physically simulated hand skeleton keeps the motion smooth, holds bone lengths steady, stops fingers from bending in impossible ways, and predicts ahead to hide camera and inference latency.

- Tracks 21 landmarks per hand, for up to 2 hands, using MediaPipe Hand Landmarker (GPU delegate when available)
- Shows live flexion angles for every MCP / PIP / DIP joint (thumb: MCP / IP)
- 240 Hz physics skeleton, vectorized with NumPy, with bone-length and anatomical joint-limit constraints
- Ball sandbox: push, flick, and pinch-grab balls with your fingers
- Capture, inference, and rendering each run on their own thread, so none of them waits on the others

## Install and run

Requires Python 3.10–3.13.

```sh
git clone https://github.com/yuvanshankar30/finger-physics.git
cd finger-physics
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
finger-physics            # or: python -m finger_physics
```

The first run downloads the hand model (~7.5 MB) into `~/.cache/finger-physics/`. On macOS, allow camera access for your terminal app under **System Settings → Privacy & Security → Camera**.

Useful flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--camera N` | `0` | camera index |
| `--video PATH` | — | use a video file instead of the camera |
| `--width/--height` | `640×480` | capture size (smaller = faster inference) |
| `--fps` | `60` | requested camera frame rate |
| `--scale` | `1.5` | window size relative to capture size |
| `--hands` | `2` | max hands to track |
| `--max-fps` | `120` | render rate cap |
| `--cpu` | off | force CPU inference |
| `--selftest SECONDS` | — | headless smoke test (no camera): real model + synthetic hand |

## Controls

| Key | Action |
| --- | --- |
| `P` | physics skeleton on/off (off = filtered + predicted points only) |
| `R` | latency prediction on/off |
| `L` | anatomical joint limits on/off |
| `W` | show raw (unfiltered) landmarks |
| `V` | show/hide video |
| `H` | show/hide HUD |
| `B` / `C` | spawn balls / clear balls |
| `[` `]` | spring stiffness − / + |
| `-` `=` | prediction lead − / + |
| `9` `0` | One Euro smoothing (min cutoff) − / + |
| `Q` / `Esc` | quit |

Pinch thumb and index together to grab a ball, and open them to throw it.

## How the pipeline cuts lag

```
camera thread ──(newest frame only)──► tracker thread: MediaPipe (GPU)
      │                                        │ landmarks + captureTime
      │                                        ▼
      │                           One Euro filter → velocity estimate
      │                                        │
      └──► newest frame ──► main loop: predict to *now* + lead
                                   │
                         240 Hz spring + PBD skeleton ──► OpenCV render
```

1. **No queued frames.** The camera thread keeps only the newest frame (`CAP_PROP_BUFFERSIZE=1`, 640×480 at 60 fps). The tracker thread always runs on the latest frame and drops stale ones, and the renderer always shows the newest frame rather than the one that was detected.
2. **Adaptive smoothing.** A vectorized [One Euro filter](https://gery.casiez.net/1euro/) on all 63 coordinates removes jitter when the hand is still and barely smooths when it moves fast.
3. **Latency prediction.** Each detection keeps its frame's capture timestamp. At render time, every joint is extrapolated by `(now − captureTime) + lead` using a filtered velocity estimate. The extrapolation is clamped and fades out if detections stop.
4. **Physics skeleton.** Each joint is a particle pulled toward its predicted target by a critically damped spring with *velocity feed-forward*, so a moving target is tracked with no steady-state lag. Position-based-dynamics constraints (a Jacobi solver written as small matrix products) then:
   - hold **bone lengths** fixed. Lengths are learned per user as ratios of palm size, which stays stable as the hand moves toward or away from the camera, so one bad frame can't stretch a finger;
   - enforce **joint limits** for each finger's MCP / PIP / DIP, about the knuckle axis. The flexion direction is learned from your first finger curl.
5. **Decoupled rates.** Physics runs at a fixed 240 Hz, and rendering runs at up to 120 fps. Joints keep moving smoothly between camera frames.

On an Apple-silicon Mac, `--selftest` measures about 8 ms of GPU inference per frame and about 4 ms per render/physics tick.

## Project layout

```
finger_physics/
  __main__.py   CLI entry point
  app.py        main loop: detection ingest, hand matching, physics substeps, keys
  camera.py     threaded camera / video / synthetic frame sources
  tracker.py    MediaPipe HandLandmarker thread + model download
  filters.py    vectorized One Euro filter
  handsim.py    physics skeleton: springs, bone + joint-limit constraints, angles
  world.py      ball sandbox with kinematic finger capsules and pinch grabbing
  render.py     OpenCV drawing + HUD
  selftest.py   synthetic hand motion for --selftest
tests/          pytest unit tests for filters, physics, and world
```

```sh
pip install -e ".[dev]" && pytest
```

## License

MIT
