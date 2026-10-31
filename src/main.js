import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { HandSim, DRAW_BONES, JOINTS, FINGER_NAMES, TIPS, N } from './handsim.js';
import { World } from './world.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const SUBSTEP = 1 / 240; // physics runs at 240 Hz regardless of display rate
const MAX_HANDS = 2;
const HAND_TIMEOUT = 0.3; // s without detection before a hand is dropped

const params = {
  physics: true,
  prediction: true,
  jointLimits: true,
  showRaw: false,
  showVideo: true,
  stiffnessHz: 18,
  damping: 1.0,
  iterations: 4,
  boneStiffness: 0.6,
  leadMs: 12,
  velocityCutoff: 8,
  filter: { minCutoff: 1.6, beta: 0.01, dCutoff: 1.0 },
};

// ------------------------------------------------------------------ DOM
const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
const overlay = $('overlay');
const statusEl = $('status');
const startBtn = $('start');
const anglesEl = $('angles');
const stats = { camFps: $('s-cam'), infer: $('s-infer'), lat: $('s-lat'), fps: $('s-fps'), hz: $('s-hz') };

let W = 0, H = 0, dpr = 1;
function resize() {
  dpr = Math.min(2, window.devicePixelRatio || 1);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
}
window.addEventListener('resize', resize);
resize();

// ------------------------------------------------------------------ controls
function bindCheckbox(id, key) {
  const el = $(id);
  el.checked = params[key];
  el.addEventListener('change', () => {
    params[key] = el.checked;
    if (key === 'showVideo') video.classList.toggle('hidden', !el.checked);
  });
}
function bindRange(id, get, set, fmt) {
  const el = $(id);
  const out = $(id + '-v');
  el.value = get();
  const show = () => (out.textContent = fmt(Number(el.value)));
  el.addEventListener('input', () => {
    set(Number(el.value));
    show();
  });
  show();
}
bindCheckbox('physics', 'physics');
bindCheckbox('prediction', 'prediction');
bindCheckbox('jointLimits', 'jointLimits');
bindCheckbox('showRaw', 'showRaw');
bindCheckbox('showVideo', 'showVideo');
bindRange('stiffness', () => params.stiffnessHz, (v) => (params.stiffnessHz = v), (v) => `${v} Hz`);
bindRange('lead', () => params.leadMs, (v) => (params.leadMs = v), (v) => `${v} ms`);
bindRange('smoothing', () => params.filter.minCutoff, (v) => (params.filter.minCutoff = v), (v) => `${v.toFixed(1)} Hz`);
bindRange('beta', () => params.filter.beta, (v) => (params.filter.beta = v), (v) => v.toFixed(3));

const world = new World();
$('spawn').addEventListener('click', () => world.spawn(10, W));
$('clear').addEventListener('click', () => world.clear());
$('toggle-panel').addEventListener('click', () => document.body.classList.toggle('panels-hidden'));
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const k = e.key.toLowerCase();
  if (k === 'b') world.spawn(10, W);
  else if (k === 'c') world.clear();
  else if (k === 'h') document.body.classList.toggle('panels-hidden');
  else if (k === 'p') $('physics').click();
});

// ------------------------------------------------------------------ tracking
let landmarker = null;
const hands = [];
const ema = { camDt: 0, infer: 0, lat: 0, fps: 0 };
const smooth = (key, v, a = 0.1) => (ema[key] = ema[key] ? ema[key] + a * (v - ema[key]) : v);

async function loadModel() {
  statusEl.textContent = 'Loading hand model…';
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: MAX_HANDS,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, opts('GPU'));
    $('s-backend').textContent = 'GPU';
  } catch (err) {
    console.warn('GPU delegate unavailable, falling back to CPU', err);
    landmarker = await HandLandmarker.createFromOptions(fileset, opts('CPU'));
    $('s-backend').textContent = 'CPU';
  }
}

async function startCamera() {
  statusEl.textContent = 'Starting camera…';
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    // Low resolution + high frame rate = less inference time and less latency.
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
  });
  video.srcObject = stream;
  await video.play();
}

/**
 * Map normalized landmarks into screen pixels, matching the mirrored
 * `object-fit: cover` video underneath the canvas.
 */
function toScreen(lm, out) {
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  const s = Math.max(W / vw, H / vh);
  const dw = vw * s, dh = vh * s;
  const ox = (W - dw) / 2, oy = (H - dh) / 2;
  for (let i = 0; i < N; i++) {
    out[3 * i] = ox + (1 - lm[i].x) * dw;
    out[3 * i + 1] = oy + lm[i].y * dh;
    out[3 * i + 2] = lm[i].z * dw;
  }
  return out;
}

let lastVideoTs = -1;
let lastCapture = null;
function processFrame(now, meta) {
  const t0 = performance.now();
  // captureTime shares performance.now()'s clock; fall back if absent/odd.
  let capture = meta && meta.captureTime ? meta.captureTime : t0;
  if (capture > t0 || t0 - capture > 500) capture = t0;
  if (lastCapture !== null) smooth('camDt', capture - lastCapture);
  lastCapture = capture;

  const ts = Math.max(lastVideoTs + 1, t0);
  lastVideoTs = ts;
  const res = landmarker.detectForVideo(video, ts);
  const t1 = performance.now();
  smooth('infer', t1 - t0);
  smooth('lat', t1 - capture);
  handleResult(res, capture / 1000);
}

function handleResult(res, tCap) {
  const lms = res.landmarks || [];
  const hd = res.handedness || res.handednesses || [];
  const dets = lms.map((lm, i) => {
    const raw = hd[i] && hd[i][0] ? hd[i][0].categoryName : '';
    // MediaPipe assumes a pre-mirrored selfie image; we feed it unmirrored.
    const label = raw === 'Left' ? 'Right' : raw === 'Right' ? 'Left' : raw;
    return { pts: toScreen(lm, new Float32Array(N * 3)), label, used: false };
  });

  // Greedy nearest-wrist matching keeps each physical hand on its own sim.
  const pairs = [];
  hands.forEach((h, hi) =>
    dets.forEach((d, di) => pairs.push({ hi, di, dist: HandSim.distBetween(d.pts, h.f, 0) })),
  );
  pairs.sort((a, b) => a.dist - b.dist);
  const matched = new Set();
  for (const p of pairs) {
    if (matched.has(p.hi) || dets[p.di].used) continue;
    if (p.dist > 6 * hands[p.hi].scale) continue;
    matched.add(p.hi);
    dets[p.di].used = true;
    hands[p.hi].ingest(dets[p.di].pts, tCap, dets[p.di].label);
  }
  for (const d of dets) {
    if (d.used || hands.length >= MAX_HANDS) continue;
    const h = new HandSim(params);
    h.ingest(d.pts, tCap, d.label);
    hands.push(h);
  }
}

function dropStaleHands(T) {
  for (let i = hands.length - 1; i >= 0; i--) {
    if (T - hands[i].measTime > HAND_TIMEOUT) {
      world.release(hands[i]);
      hands.splice(i, 1);
    }
  }
}

function startDetectionLoop() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    // Fires exactly once per new camera frame: no polling, no duplicate work.
    const onFrame = (now, meta) => {
      processFrame(now, meta);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    let lastTime = -1;
    const poll = () => {
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        processFrame(performance.now(), null);
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }
}

// ------------------------------------------------------------------ render loop
let lastFrame = null;
let acc = 0;
let lastHud = 0;
const angleBuf = new Float32Array(JOINTS.length);
const HAND_HUES = [190, 32];

function frame(now) {
  requestAnimationFrame(frame);
  const T = now / 1000;
  const dt = lastFrame === null ? 1 / 60 : Math.min(0.1, T - lastFrame);
  lastFrame = T;
  smooth('fps', dt > 0 ? 1 / dt : 60, 0.05);

  dropStaleHands(T);
  for (const h of hands) h.updateTargets(T);
  world.updateGrabs(hands);

  acc += dt;
  let steps = 0;
  while (acc >= SUBSTEP && steps < 24) {
    for (const h of hands) h.step(SUBSTEP);
    world.step(SUBSTEP, hands, W, H);
    acc -= SUBSTEP;
    steps++;
  }
  if (steps === 24) acc = 0;

  draw(T);
  if (now - lastHud > 100) {
    lastHud = now;
    updateHud();
  }
}

function draw(T) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  for (const b of world.balls) {
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${b.hue} 80% 60% / 0.92)`;
    ctx.fill();
    if (b.grabbedBy !== null) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
  }

  hands.forEach((h, hi) => {
    const alpha = Math.max(0, Math.min(1, 1 - (T - h.measTime - 0.12) / (HAND_TIMEOUT - 0.12)));
    const hue = HAND_HUES[hi % HAND_HUES.length];
    const x = h.x;
    ctx.globalAlpha = alpha;

    if (params.showRaw) {
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      for (let i = 0; i < N; i++) {
        ctx.beginPath();
        ctx.arc(h.raw[3 * i], h.raw[3 * i + 1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Bones as capsules matching the collision radius.
    const cr = h.scale * 0.09;
    ctx.lineCap = 'round';
    ctx.lineWidth = cr * 2;
    ctx.strokeStyle = `hsl(${hue} 90% 60% / 0.35)`;
    ctx.beginPath();
    for (const [i, j] of DRAW_BONES) {
      ctx.moveTo(x[3 * i], x[3 * i + 1]);
      ctx.lineTo(x[3 * j], x[3 * j + 1]);
    }
    ctx.stroke();
    ctx.lineWidth = Math.max(2, cr * 0.35);
    ctx.strokeStyle = `hsl(${hue} 95% 70%)`;
    ctx.stroke();

    // Joints colored by flexion: green (straight) → red (fully bent).
    h.angles(angleBuf);
    const flexAt = new Map();
    JOINTS.forEach((j, n) => flexAt.set(j.b, angleBuf[n]));
    for (let i = 0; i < N; i++) {
      const f = flexAt.get(i);
      const r = TIPS.includes(i) ? cr * 0.7 : cr * 0.55;
      ctx.beginPath();
      ctx.arc(x[3 * i], x[3 * i + 1], Math.max(3, r), 0, Math.PI * 2);
      ctx.fillStyle = f === undefined ? '#e8f4ff' : `hsl(${Math.max(0, 130 - Math.max(0, f) * 1.2)} 90% 55%)`;
      ctx.fill();
    }

    if (h.pinching) {
      ctx.beginPath();
      ctx.arc(h.pinchPoint[0], h.pinchPoint[1], cr * 1.4, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  });
}

function updateHud() {
  stats.camFps.textContent = ema.camDt ? (1000 / ema.camDt).toFixed(0) : '–';
  stats.infer.textContent = ema.infer.toFixed(1);
  stats.lat.textContent = ema.lat.toFixed(0);
  stats.fps.textContent = ema.fps.toFixed(0);
  stats.hz.textContent = (1 / SUBSTEP).toFixed(0);

  if (!hands.length) {
    anglesEl.innerHTML = '<p class="muted">Show a hand to the camera.</p>';
    return;
  }
  let html = '';
  hands.forEach((h, hi) => {
    h.angles(angleBuf);
    const calibrated = h.flexSign !== 0;
    html += `<div class="hand"><h3 style="--hue:${HAND_HUES[hi % 2]}">${h.label || 'Hand'} hand`;
    html += calibrated ? '' : ' <span class="muted">· curl fingers once to calibrate</span>';
    html += '</h3><table><tbody>';
    FINGER_NAMES.forEach((name, f) => {
      html += `<tr><th>${name}</th>`;
      const js = JOINTS.map((j, n) => [j, n]).filter(([j]) => j.finger === f);
      if (f === 0) html += '<td></td>';
      for (const [j, n] of js) {
        const a = angleBuf[n];
        const pct = Math.max(0, Math.min(100, (a / 110) * 100));
        html += `<td><span class="jn">${j.name}</span><span class="jv">${a.toFixed(0)}°</span><i style="width:${pct}%"></i></td>`;
      }
      html += '</tr>';
    });
    html += '</tbody></table></div>';
  });
  anglesEl.innerHTML = html;
}

// ------------------------------------------------------------------ boot
const modelReady = loadModel().catch((err) => {
  statusEl.textContent = 'Failed to load hand model: ' + err.message;
  throw err;
});

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  try {
    await startCamera();
    await modelReady;
    overlay.classList.add('hidden');
    world.spawn(12, W);
    startDetectionLoop();
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      err.name === 'NotAllowedError' ? 'Camera permission was denied.' : 'Error: ' + (err.message || err);
    startBtn.disabled = false;
  }
});
modelReady.then(() => {
  if (!startBtn.disabled) statusEl.textContent = 'Ready.';
});
