// 2D rigid-ball sandbox the simulated hands can push, flick and pinch-grab.
// Hand bones act as kinematic capsules: they have effectively infinite mass and
// transfer their own velocity to whatever they hit.

import { DRAW_BONES } from './handsim.js';

const GRAVITY = 1800; // px/s²
const RESTITUTION = 0.45;
const WALL_RESTITUTION = 0.5;
const AIR_DRAG = 0.15; // 1/s
const MAX_BALLS = 60;

export const PINCH_ON = 0.32; // pinch distance / hand scale
export const PINCH_OFF = 0.45;

export class World {
  constructor() {
    this.balls = [];
  }

  spawn(count, width) {
    for (let i = 0; i < count && this.balls.length < MAX_BALLS; i++) {
      const r = 16 + Math.random() * 18;
      this.balls.push({
        x: r + Math.random() * (width - 2 * r),
        y: -r - Math.random() * 300,
        vx: (Math.random() - 0.5) * 200,
        vy: 0,
        r,
        hue: Math.floor(Math.random() * 360),
        grabbedBy: null,
      });
    }
  }

  clear() {
    this.balls.length = 0;
  }

  /** Update pinch state and grabs for each hand (call once per frame). */
  updateGrabs(hands) {
    for (const h of hands) {
      const d = h.pinchDistance();
      const wasPinching = h.pinching;
      h.pinching = wasPinching ? d < PINCH_OFF : d < PINCH_ON;
      const mx = (h.x[12] + h.x[24]) / 2, my = (h.x[13] + h.x[25]) / 2;
      h.pinchPoint = [mx, my];
      if (h.pinching && !wasPinching && !h.grab) {
        let best = null, bestD = Infinity;
        for (const b of this.balls) {
          if (b.grabbedBy) continue;
          const dd = Math.hypot(b.x - mx, b.y - my) - b.r;
          if (dd < 0.6 * h.scale && dd < bestD) { best = b; bestD = dd; }
        }
        if (best) { best.grabbedBy = h.id; h.grab = best; }
      }
      if (!h.pinching && h.grab) this.release(h);
    }
  }

  release(hand) {
    if (hand.grab) hand.grab.grabbedBy = null;
    hand.grab = null;
  }

  step(dt, hands, W, H) {
    const balls = this.balls;
    const drag = Math.exp(-AIR_DRAG * dt);

    for (const b of balls) {
      if (b.grabbedBy !== null) {
        const h = hands.find((hh) => hh.id === b.grabbedBy);
        if (!h) { b.grabbedBy = null; continue; }
        // Stiff critically damped spring to the pinch point.
        const w = 2 * Math.PI * 12, k = w * w, c = 2 * w;
        b.vx += (k * (h.pinchPoint[0] - b.x) - c * b.vx) * dt;
        b.vy += (k * (h.pinchPoint[1] - b.y) - c * b.vy) * dt;
      } else {
        b.vy += GRAVITY * dt;
        b.vx *= drag; b.vy *= drag;
      }
      b.x += b.vx * dt;
      b.y += b.vy * dt;
    }

    // Ball–ball.
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i], b = balls[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const R = a.r + b.r;
        const d2 = dx * dx + dy * dy;
        if (d2 >= R * R || d2 < 1e-9) continue;
        const d = Math.sqrt(d2), nx = dx / d, ny = dy / d;
        const ma = a.r * a.r, mb = b.r * b.r; // mass ∝ area
        const wa = a.grabbedBy !== null ? 0 : 1 / ma, wb = b.grabbedBy !== null ? 0 : 1 / mb;
        if (wa + wb === 0) continue;
        const pen = R - d;
        a.x -= nx * pen * (wa / (wa + wb)); a.y -= ny * pen * (wa / (wa + wb));
        b.x += nx * pen * (wb / (wa + wb)); b.y += ny * pen * (wb / (wa + wb));
        const rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (rel < 0) {
          const jImp = (-(1 + RESTITUTION) * rel) / (wa + wb);
          a.vx -= jImp * wa * nx; a.vy -= jImp * wa * ny;
          b.vx += jImp * wb * nx; b.vy += jImp * wb * ny;
        }
      }
    }

    // Hand capsules (kinematic).
    for (const h of hands) {
      const cr = h.scale * 0.09;
      const x = h.x, v = h.v;
      for (const b of balls) {
        if (b.grabbedBy === h.id) continue;
        for (const [i, j] of DRAW_BONES) {
          const ax = x[3 * i], ay = x[3 * i + 1], bx = x[3 * j], by = x[3 * j + 1];
          const abx = bx - ax, aby = by - ay;
          const ab2 = abx * abx + aby * aby || 1e-9;
          let t = ((b.x - ax) * abx + (b.y - ay) * aby) / ab2;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const qx = ax + abx * t, qy = ay + aby * t;
          let dx = b.x - qx, dy = b.y - qy;
          const R = b.r + cr;
          const d2 = dx * dx + dy * dy;
          if (d2 >= R * R) continue;
          let d = Math.sqrt(d2);
          if (d < 1e-6) { dx = -aby; dy = abx; d = Math.hypot(dx, dy) || 1; }
          const nx = dx / d, ny = dy / d;
          b.x = qx + nx * R;
          b.y = qy + ny * R;
          // Velocity of the bone at the contact point.
          const vqx = v[3 * i] * (1 - t) + v[3 * j] * t;
          const vqy = v[3 * i + 1] * (1 - t) + v[3 * j + 1] * t;
          const rvx = b.vx - vqx, rvy = b.vy - vqy;
          const rn = rvx * nx + rvy * ny;
          if (rn < 0) {
            b.vx -= (1 + RESTITUTION) * rn * nx;
            b.vy -= (1 + RESTITUTION) * rn * ny;
          }
          // A little friction so fingers can roll / carry balls.
          const tx = -ny, ty = nx;
          const rt = (b.vx - vqx) * tx + (b.vy - vqy) * ty;
          b.vx -= rt * 0.08 * tx;
          b.vy -= rt * 0.08 * ty;
        }
      }
    }

    // Walls (top is open so spawned balls can fall in).
    for (const b of balls) {
      if (b.x < b.r) { b.x = b.r; if (b.vx < 0) b.vx = -b.vx * WALL_RESTITUTION; }
      if (b.x > W - b.r) { b.x = W - b.r; if (b.vx > 0) b.vx = -b.vx * WALL_RESTITUTION; }
      if (b.y > H - b.r) {
        b.y = H - b.r;
        if (b.vy > 0) b.vy = -b.vy * WALL_RESTITUTION;
        b.vx *= 0.995;
      }
    }
  }
}
