// Physically simulated hand skeleton.
//
// Each of the 21 MediaPipe landmarks is a particle. Particles are pulled toward
// the (filtered + latency-predicted) measurement by a critically damped spring
// with velocity feed-forward, then position-based-dynamics constraints enforce:
//   * consistent bone lengths (learned per user as ratios of palm size), and
//   * anatomical joint limits (no impossible finger hyperextension / overbend).
// The result is a skeleton that is smooth, never "rubber-bands", and keeps
// moving at display rate between camera frames.

import { OneEuro, LowPass, lpAlpha } from './filters.js';

export const N = 21;
export const FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
export const TIPS = [4, 8, 12, 16, 20];

const CHAIN = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];
const PALM = [[0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
const BRACE = [[0, 9], [0, 13], [5, 17], [1, 5]];

/** Bones drawn on screen and used as collision capsules. */
export const DRAW_BONES = [...CHAIN, ...PALM];
/** All distance constraints (drawn bones plus palm cross-braces). */
export const CONSTRAINTS = [...CHAIN, ...PALM, ...BRACE];

// Palm bones are effectively rigid, so their mean length is a depth-invariant
// "hand scale" that every other bone length is expressed relative to.
const SCALE_PAIRS = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]];
const SCALE_IDX = SCALE_PAIRS.map(([a, b]) =>
  CONSTRAINTS.findIndex(([c, d]) => c === a && d === b),
);

const DEG = Math.PI / 180;

/** Joints we report angles for. flex/ext are limits in radians. */
export const JOINTS = [];
JOINTS.push({ finger: 0, name: 'MCP', a: 1, b: 2, c: 3, tip: 4, limited: false });
JOINTS.push({ finger: 0, name: 'IP', a: 2, b: 3, c: 4, tip: 4, limited: false });
[5, 9, 13, 17].forEach((base, f) => {
  const tip = base + 3;
  JOINTS.push({ finger: f + 1, name: 'MCP', a: 0, b: base, c: base + 1, tip, limited: true, flex: 100 * DEG, ext: 45 * DEG });
  JOINTS.push({ finger: f + 1, name: 'PIP', a: base, b: base + 1, c: base + 2, tip, limited: true, flex: 120 * DEG, ext: 12 * DEG });
  JOINTS.push({ finger: f + 1, name: 'DIP', a: base + 1, b: base + 2, c: base + 3, tip, limited: true, flex: 95 * DEG, ext: 25 * DEG });
});
const PIP_JOINTS = JOINTS.filter((j) => j.name === 'PIP');

// Wrist and knuckles are heavy so constraints bend fingers, not the palm.
const INV_MASS = new Float32Array(N).fill(1);
INV_MASS[0] = 0.15;
for (const i of [1, 5, 9, 13, 17]) INV_MASS[i] = 0.4;

// ---------------------------------------------------------------- vector math
// Points live in flat Float32Arrays: point i is [3i, 3i+1, 3i+2].

function sub(arr, i, j, out) {
  out[0] = arr[3 * i] - arr[3 * j];
  out[1] = arr[3 * i + 1] - arr[3 * j + 1];
  out[2] = arr[3 * i + 2] - arr[3 * j + 2];
  return out;
}
const len3 = (v) => Math.hypot(v[0], v[1], v[2]);
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function cross3(a, b, out) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}
function normalize(v) {
  const l = len3(v) || 1;
  v[0] /= l; v[1] /= l; v[2] /= l;
  return v;
}
/** Remove the component of v along unit axis k (in place). */
function projectOut(v, k) {
  const d = dot3(v, k);
  v[0] -= d * k[0]; v[1] -= d * k[1]; v[2] -= d * k[2];
  return v;
}

/** Signed angle from u to w about unit axis k, measured in the plane ⟂ k. */
export function signedAngle(u, w, k) {
  const up = projectOut([u[0], u[1], u[2]], k);
  const wp = projectOut([w[0], w[1], w[2]], k);
  const c = cross3(up, wp, [0, 0, 0]);
  return Math.atan2(dot3(c, k), dot3(up, wp));
}

/** Rodrigues rotation of v about unit axis k by angle th (in place). */
export function rotate(v, k, th) {
  const c = Math.cos(th), s = Math.sin(th);
  const kxv = cross3(k, v, [0, 0, 0]);
  const kdv = dot3(k, v) * (1 - c);
  const x = v[0] * c + kxv[0] * s + k[0] * kdv;
  const y = v[1] * c + kxv[1] * s + k[1] * kdv;
  const z = v[2] * c + kxv[2] * s + k[2] * kdv;
  v[0] = x; v[1] = y; v[2] = z;
  return v;
}

const tmpU = [0, 0, 0], tmpW = [0, 0, 0], tmpL = [0, 0, 0], tmpQ = [0, 0, 0];

// ---------------------------------------------------------------- simulation

export class HandSim {
  /**
   * @param params live settings object (see DEFAULT_PARAMS in main.js)
   */
  constructor(params) {
    this.params = params;
    this.x = new Float32Array(N * 3); // simulated positions
    this.v = new Float32Array(N * 3); // simulated velocities
    this.p = new Float32Array(N * 3); // scratch: predicted positions
    this.raw = new Float32Array(N * 3); // last raw measurement
    this.f = new Float32Array(N * 3); // last filtered measurement
    this.fv = new Float32Array(N * 3); // measurement velocity estimate
    this.target = new Float32Array(N * 3);
    this.tv = new Float32Array(N * 3);
    this.filters = Array.from({ length: N * 3 }, () => new OneEuro(params.filter));
    this.velLP = Array.from({ length: N * 3 }, () => new LowPass());
    this.ratio = new Float32Array(CONSTRAINTS.length);
    this.calib = 0;
    this.scale = 1;
    this.initialized = false;
    this.measTime = 0;
    this.prevMeasTime = null;
    this.flexEvidence = 0;
    this.flexSign = 0;
    this.label = '';
    this.pinching = false;
    this.grab = null;
    this.id = HandSim.nextId++;
  }

  /** Bone length between points i and j of a flat array. */
  static dist(arr, i, j) {
    return len3(sub(arr, i, j, tmpQ));
  }

  /**
   * Feed one detection.
   * @param pts Float32Array(63) landmark positions in screen pixels
   * @param t   capture timestamp in seconds
   * @param label handedness label
   */
  ingest(pts, t, label) {
    if (label !== this.label) {
      this.label = label;
      this.flexEvidence = 0;
      this.flexSign = 0;
    }
    this.raw.set(pts);

    // Re-acquired far away (or first frame): reset filters and snap.
    const jump = this.initialized ? HandSim.distBetween(pts, this.f, 0) : Infinity;
    const snap = !this.initialized || jump > 4 * this.scale;
    if (snap) {
      this.filters.forEach((fl) => fl.reset());
      this.velLP.forEach((lp) => lp.reset());
      this.prevMeasTime = null;
    }

    const dt = this.prevMeasTime === null ? 0 : Math.max(1e-3, t - this.prevMeasTime);
    const aVel = dt > 0 ? lpAlpha(this.params.velocityCutoff, dt) : 1;
    for (let i = 0; i < N * 3; i++) {
      const prev = this.f[i];
      const y = this.filters[i].filter(pts[i], t);
      this.f[i] = y;
      this.fv[i] = dt > 0 ? this.velLP[i].filter((y - prev) / dt, aVel) : this.velLP[i].filter(0, 1);
    }
    this.prevMeasTime = t;
    this.measTime = t;

    // Hand scale = mean palm bone length (rigid, so depth-invariant-ish).
    let s = 0;
    for (const [a, b] of SCALE_PAIRS) s += HandSim.dist(this.f, a, b);
    this.scale = Math.max(8, s / SCALE_PAIRS.length);

    // Learn each bone's length as a ratio of hand scale (robust running mean).
    for (let j = 0; j < CONSTRAINTS.length; j++) {
      const [a, b] = CONSTRAINTS[j];
      const r = HandSim.dist(this.f, a, b) / this.scale;
      if (this.calib === 0) this.ratio[j] = r;
      else {
        const dev = Math.abs(r - this.ratio[j]) / (this.ratio[j] || 1);
        if (this.calib > 20 && dev > 0.35) continue; // reject outliers
        const alpha = this.calib < 30 ? 0.15 : 0.01;
        this.ratio[j] += alpha * (r - this.ratio[j]);
      }
    }
    for (const j of SCALE_IDX) this.ratio[j] = Math.max(this.ratio[j], 1e-3);
    this.calib++;

    this.learnFlexSign();

    if (snap) {
      this.x.set(this.f);
      this.v.fill(0);
      this.initialized = true;
    }
  }

  static distBetween(a, b, i) {
    return Math.hypot(a[3 * i] - b[3 * i], a[3 * i + 1] - b[3 * i + 1], a[3 * i + 2] - b[3 * i + 2]);
  }

  /** Lateral (abduction) axis across the knuckles: index MCP → pinky MCP. */
  lateralAxis(arr, out) {
    return normalize(sub(arr, 17, 5, out));
  }

  /**
   * Which rotation direction about the knuckle axis is "flexion" depends on
   * handedness and mirroring. Rather than trusting labels, learn it: PIP
   * joints barely hyperextend, so any large PIP bend is almost surely flexion.
   */
  learnFlexSign() {
    const L = this.lateralAxis(this.f, tmpL);
    for (const j of PIP_JOINTS) {
      sub(this.f, j.b, j.a, tmpU);
      sub(this.f, j.c, j.b, tmpW);
      const s = signedAngle(tmpU, tmpW, L);
      if (Math.abs(s) > 35 * DEG) this.flexEvidence += Math.sign(s) * Math.abs(s);
    }
    this.flexEvidence = Math.max(-20, Math.min(20, this.flexEvidence));
    if (Math.abs(this.flexEvidence) > 3) this.flexSign = Math.sign(this.flexEvidence);
  }

  /** Compute targets for render time T (seconds), with latency prediction. */
  updateTargets(T) {
    const P = this.params;
    let h = 0;
    if (P.prediction) h = Math.min(0.1, Math.max(0, T - this.measTime + P.leadMs / 1000));
    // Fade velocity if detections stop arriving so we don't fly off.
    const stale = T - this.measTime;
    const velScale = stale > 0.12 ? Math.max(0, 1 - (stale - 0.12) / 0.1) : 1;
    const maxShift = 0.8 * this.scale;
    for (let i = 0; i < N; i++) {
      let dx = this.fv[3 * i] * h, dy = this.fv[3 * i + 1] * h, dz = this.fv[3 * i + 2] * h;
      const m = Math.hypot(dx, dy, dz);
      if (m > maxShift) {
        const k = maxShift / m;
        dx *= k; dy *= k; dz *= k;
      }
      this.target[3 * i] = this.f[3 * i] + dx * velScale;
      this.target[3 * i + 1] = this.f[3 * i + 1] + dy * velScale;
      this.target[3 * i + 2] = this.f[3 * i + 2] + dz * velScale;
      this.tv[3 * i] = this.fv[3 * i] * velScale;
      this.tv[3 * i + 1] = this.fv[3 * i + 1] * velScale;
      this.tv[3 * i + 2] = this.fv[3 * i + 2] * velScale;
    }
  }

  /** Advance the simulation by dt seconds. */
  step(dt) {
    const P = this.params;
    const { x, v, p, target, tv } = this;
    if (!P.physics) {
      x.set(target);
      v.set(tv);
      return;
    }
    // Critically damped spring toward target, with velocity feed-forward so a
    // constantly moving target is tracked with zero steady-state lag.
    const w = 2 * Math.PI * P.stiffnessHz;
    const k = w * w;
    const c = 2 * w * P.damping;
    for (let i = 0; i < N * 3; i++) {
      const a = k * (target[i] - x[i]) + c * (tv[i] - v[i]);
      v[i] += a * dt;
      p[i] = x[i] + v[i] * dt;
    }

    if (this.calib >= 3) {
      for (let it = 0; it < P.iterations; it++) {
        this.solveBones(p);
        if (P.jointLimits && this.flexSign !== 0) this.solveLimits(p);
      }
    }

    for (let i = 0; i < N * 3; i++) {
      v[i] = (p[i] - x[i]) / dt;
      x[i] = p[i];
    }
  }

  solveBones(p) {
    const stiff = this.params.boneStiffness;
    for (let j = 0; j < CONSTRAINTS.length; j++) {
      const [a, b] = CONSTRAINTS[j];
      const rest = this.ratio[j] * this.scale;
      const dx = p[3 * b] - p[3 * a], dy = p[3 * b + 1] - p[3 * a + 1], dz = p[3 * b + 2] - p[3 * a + 2];
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-6) continue;
      const wa = INV_MASS[a], wb = INV_MASS[b];
      const s = (stiff * (len - rest)) / len / (wa + wb);
      p[3 * a] += dx * s * wa; p[3 * a + 1] += dy * s * wa; p[3 * a + 2] += dz * s * wa;
      p[3 * b] -= dx * s * wb; p[3 * b + 1] -= dy * s * wb; p[3 * b + 2] -= dz * s * wb;
    }
  }

  /** Clamp each finger joint's flexion into its anatomical range. */
  solveLimits(p) {
    const L = this.lateralAxis(p, tmpL);
    const sgn = this.flexSign;
    for (const j of JOINTS) {
      if (!j.limited) continue;
      sub(p, j.b, j.a, tmpU);
      sub(p, j.c, j.b, tmpW);
      const s = signedAngle(tmpU, tmpW, L) * sgn;
      const clamped = Math.min(j.flex, Math.max(-j.ext, s));
      if (clamped === s) continue;
      const delta = (clamped - s) * sgn * 0.5;
      // Rotate the child and every point distal to it about the joint.
      const bx = p[3 * j.b], by = p[3 * j.b + 1], bz = p[3 * j.b + 2];
      for (let q = j.c; q <= j.tip; q++) {
        tmpQ[0] = p[3 * q] - bx; tmpQ[1] = p[3 * q + 1] - by; tmpQ[2] = p[3 * q + 2] - bz;
        rotate(tmpQ, L, delta);
        p[3 * q] = bx + tmpQ[0]; p[3 * q + 1] = by + tmpQ[1]; p[3 * q + 2] = bz + tmpQ[2];
      }
    }
  }

  /** Flexion angle of every joint in JOINTS, degrees (signed when known). */
  angles(out = new Float32Array(JOINTS.length)) {
    const L = this.lateralAxis(this.x, tmpL);
    JOINTS.forEach((j, n) => {
      sub(this.x, j.b, j.a, tmpU);
      sub(this.x, j.c, j.b, tmpW);
      if (j.limited && this.flexSign !== 0) {
        out[n] = (signedAngle(tmpU, tmpW, L) * this.flexSign) / DEG;
      } else {
        const c = dot3(tmpU, tmpW) / ((len3(tmpU) * len3(tmpW)) || 1);
        out[n] = Math.acos(Math.max(-1, Math.min(1, c))) / DEG;
      }
    });
    return out;
  }

  /** Thumb-tip to index-tip distance in units of hand scale. */
  pinchDistance() {
    return HandSim.dist(this.x, 4, 8) / this.scale;
  }
}
HandSim.nextId = 1;
