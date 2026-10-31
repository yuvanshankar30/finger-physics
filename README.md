# Finger Physics

Real-time finger-joint tracking from a webcam. A physically simulated hand skeleton keeps the motion smooth, holds bone lengths steady, and stops fingers from bending in impossible ways. It also predicts ahead to hide camera and inference latency.

**Live demo:** https://yuvanshankar30.github.io/finger-physics/ (allow camera access)

- Tracks 21 landmarks per hand, for up to 2 hands (MediaPipe Hand Landmarker on the GPU)
- Shows live flexion angles for every MCP / PIP / DIP joint (thumb: MCP / IP)
- 240 Hz physics skeleton with bone-length and anatomical joint-limit constraints
- Ball sandbox: push, flick, and pinch-grab balls with your fingers
- Runs entirely in the browser with no build step. No video leaves your device.

## Run locally

```sh
npm start          # serves on http://localhost:5180
# or: python3 -m http.server 5180
```

The camera needs a secure context, so use `localhost` or HTTPS. Chrome or Edge works best because they support `requestVideoFrameCallback` and WebGL for the GPU delegate.

```sh
npm test           # physics + filter unit tests (Node 18+)
```

## How the pipeline cuts lag

```
camera frame ──► HandLandmarker (GPU) ──► One Euro filter ──► velocity estimate
     │                                                          │
     └── captureTime ─────────────► predict to *now* + lead ◄───┘
                                              │
                         240 Hz spring + PBD skeleton  ──► render (display rate)
```

1. **Low input latency.** The camera runs at 640×480 and asks for 60 fps. Detection starts in `requestVideoFrameCallback`, so each new frame is processed once, as soon as it arrives, without polling. The video itself is shown through a CSS-mirrored `<video>` element that the browser composites directly.
2. **Adaptive smoothing.** A [One Euro filter](https://gery.casiez.net/1euro/) on every coordinate removes jitter when the hand is still and barely smooths when it moves fast.
3. **Latency prediction.** Each detection keeps its camera `captureTime`. At render time, every joint is extrapolated by `(now − captureTime) + lead` using a filtered velocity estimate. The extrapolation is clamped and fades out if detections stop.
4. **Physics skeleton.** Each joint is a particle pulled toward its predicted target by a critically damped spring with *velocity feed-forward*, so a moving target is tracked with no steady-state lag. Position-based-dynamics constraints then:
   - hold **bone lengths** fixed. Lengths are learned per user as ratios of palm size, which stays stable as the hand moves toward or away from the camera, so one bad frame can't stretch a finger;
   - enforce **joint limits** for each finger's MCP / PIP / DIP, about the knuckle axis. The flexion direction is learned from your first finger curl, so it works for either hand.
5. **Decoupled rates.** Physics runs at a fixed 240 Hz and rendering runs at the display rate (60/120 Hz+). Joints keep moving smoothly between 30–60 Hz camera frames.

## Controls

| Control | Effect |
| --- | --- |
| Physics | Turn the spring/constraint skeleton on or off (off = filtered + predicted points only) |
| Prediction | Turn latency extrapolation on or off |
| Joint limits | Turn anatomical angle clamping on or off |
| Raw points | Show the unfiltered MediaPipe landmarks to compare against |
| Spring | Spring natural frequency (higher = snappier, lower = floatier) |
| Lead | Extra prediction on top of the measured pipeline latency (to cover display latency) |
| Smoothing / Speed β | One Euro `minCutoff` / `beta` |

Keys: `B` spawns balls, `C` clears, `H` hides the panels, `P` toggles physics. Pinch thumb and index to grab a ball.

## Project layout

```
index.html, style.css   UI
src/main.js             camera, detection loop, hand matching, render loop, HUD
src/filters.js          One Euro + low-pass filters
src/handsim.js          physics skeleton: springs, bone + joint-limit constraints, angles
src/world.js            ball sandbox with kinematic finger capsules and pinch grabbing
test/                   node:test unit tests
```

## License

MIT
