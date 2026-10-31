import { test } from 'node:test';
import assert from 'node:assert/strict';
import { OneEuro } from '../src/filters.js';
import { HandSim, CONSTRAINTS, JOINTS, N, signedAngle, rotate } from '../src/handsim.js';
import { World } from '../src/world.js';

const params = () => ({
  physics: true,
  prediction: true,
  jointLimits: true,
  stiffnessHz: 18,
  damping: 1,
  iterations: 4,
  boneStiffness: 0.6,
  leadMs: 0,
  velocityCutoff: 8,
  filter: { minCutoff: 1.6, beta: 0.01, dCutoff: 1 },
});

// A flat open hand in pixels: palm toward camera, fingers pointing up (-y).
function openHand(ox = 300, oy = 400, s = 1) {
  const p = new Float32Array(N * 3);
  const set = (i, x, y) => { p[3 * i] = ox + x * s; p[3 * i + 1] = oy + y * s; p[3 * i + 2] = 0; };
  set(0, 0, 0);
  set(1, -30, -20); set(2, -55, -45); set(3, -75, -70); set(4, -90, -90);
  [-30, -10, 10, 30].forEach((dx, f) => {
    const b = 5 + 4 * f;
    set(b, dx, -90);
    set(b + 1, dx, -130);
    set(b + 2, dx, -155);
    set(b + 3, dx, -175);
  });
  return p;
}

/** Curl every finger's PIP/DIP by `deg` about the knuckle axis (x-axis). */
function curlHand(p, deg) {
  const out = new Float32Array(p);
  const axis = [1, 0, 0];
  for (const base of [5, 9, 13, 17]) {
    for (const joint of [base + 1, base + 2]) {
      for (let q = joint + 1; q <= base + 3; q++) {
        const v = [out[3 * q] - out[3 * joint], out[3 * q + 1] - out[3 * joint + 1], out[3 * q + 2] - out[3 * joint + 2]];
        rotate(v, axis, (deg * Math.PI) / 180);
        out[3 * q] = out[3 * joint] + v[0];
        out[3 * q + 1] = out[3 * joint + 1] + v[1];
        out[3 * q + 2] = out[3 * joint + 2] + v[2];
      }
    }
  }
  return out;
}

test('One Euro filter suppresses jitter on a still signal', () => {
  const f = new OneEuro({ minCutoff: 1, beta: 0.01, dCutoff: 1 });
  let maxDev = 0;
  for (let i = 0; i < 300; i++) {
    const y = f.filter(100 + (i % 2 ? 2 : -2), i / 60);
    if (i > 60) maxDev = Math.max(maxDev, Math.abs(y - 100));
  }
  assert.ok(maxDev < 0.5, `deviation ${maxDev}`);
});

test('One Euro filter keeps up with fast motion', () => {
  const f = new OneEuro({ minCutoff: 1, beta: 0.05, dCutoff: 1 });
  let y = 0;
  for (let i = 0; i < 120; i++) y = f.filter(i * 20, i / 60); // 1200 px/s
  assert.ok(119 * 20 - y < 30, `lag ${119 * 20 - y}px`);
});

test('signedAngle and rotate agree', () => {
  const k = [0, 0, 1];
  const v = rotate([1, 0, 0], k, 0.7);
  assert.ok(Math.abs(signedAngle([1, 0, 0], v, k) - 0.7) < 1e-9);
});

test('sim converges onto a static hand', () => {
  const h = new HandSim(params());
  const hand = openHand();
  for (let i = 0; i < 60; i++) h.ingest(hand, i / 30, 'Right');
  h.updateTargets(2);
  for (let i = 0; i < 480; i++) h.step(1 / 240);
  for (let i = 0; i < N * 3; i++) assert.ok(Math.abs(h.x[i] - hand[i]) < 1, `coord ${i}`);
});

test('bone constraints reject a single-frame length glitch', () => {
  const h = new HandSim(params());
  const hand = openHand();
  for (let i = 0; i < 60; i++) h.ingest(hand, i / 30, 'Right');
  const glitch = new Float32Array(hand);
  glitch[3 * 8 + 1] -= 60; // fingertip stretched way out
  h.ingest(glitch, 61 / 30, 'Right');
  h.updateTargets(61 / 30);
  for (let i = 0; i < 8; i++) h.step(1 / 240);
  const j = CONSTRAINTS.findIndex(([a, b]) => a === 7 && b === 8);
  const len = HandSim.dist(h.x, 7, 8);
  const rest = h.ratio[j] * h.scale;
  assert.ok(Math.abs(len - rest) / rest < 0.5, `len ${len} rest ${rest}`);
});

test('learns flexion direction and reports curl angles', () => {
  const h = new HandSim(params());
  const open = openHand();
  const fist = curlHand(open, 70);
  let t = 0;
  for (let i = 0; i < 30; i++) h.ingest(open, (t += 1 / 30), 'Right');
  for (let i = 0; i < 30; i++) h.ingest(fist, (t += 1 / 30), 'Right');
  assert.notEqual(h.flexSign, 0);
  h.updateTargets(t);
  for (let i = 0; i < 480; i++) h.step(1 / 240);
  const a = h.angles();
  const pip = JOINTS.findIndex((j) => j.finger === 2 && j.name === 'PIP');
  assert.ok(Math.abs(a[pip] - 70) < 8, `PIP ${a[pip]}`);
});

test('joint limits block impossible hyperextension', () => {
  const h = new HandSim(params());
  const open = openHand();
  let t = 0;
  for (let i = 0; i < 30; i++) h.ingest(curlHand(open, 70), (t += 1 / 30), 'Right');
  const bad = curlHand(open, -60); // fingers bent backwards
  for (let i = 0; i < 4; i++) h.ingest(bad, (t += 1 / 30), 'Right');
  h.updateTargets(t);
  for (let i = 0; i < 240; i++) h.step(1 / 240);
  const a = h.angles();
  const pip = JOINTS.findIndex((j) => j.finger === 2 && j.name === 'PIP');
  assert.ok(a[pip] > -25, `PIP hyperextended to ${a[pip]}`);
});

test('a moving hand knocks a ball', () => {
  const w = new World();
  w.balls.push({ x: 330, y: 250, vx: 0, vy: 0, r: 20, hue: 0, grabbedBy: null });
  const h = new HandSim({ ...params(), physics: false });
  let t = 0;
  for (let i = 0; i < 30; i++) {
    h.ingest(openHand(150 + i * 12, 400), (t += 1 / 30), 'Right');
    h.updateTargets(t);
    for (let k = 0; k < 8; k++) {
      h.step(1 / 240);
      w.step(1 / 240, [h], 2000, 270); // floor keeps the ball in the hand's path
    }
  }
  assert.ok(w.balls[0].x > 360 || Math.abs(w.balls[0].vx) > 50, `ball x ${w.balls[0].x}`);
});
