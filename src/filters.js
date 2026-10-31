// Signal filters used on raw landmark coordinates.
//
// The One Euro filter (Casiez et al., CHI 2012) is an adaptive low-pass:
// heavy smoothing when a point is still (kills jitter), light smoothing when
// it moves fast (kills lag). That trade-off is exactly what hand tracking needs.

/** Smoothing factor for a first-order low-pass with the given cutoff (Hz). */
export function lpAlpha(cutoffHz, dt) {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dt);
}

export class LowPass {
  constructor() {
    this.y = null;
  }
  filter(x, alpha) {
    this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y;
    return this.y;
  }
  reset() {
    this.y = null;
  }
}

/**
 * One Euro filter. `params` is a shared, live object
 * ({ minCutoff, beta, dCutoff }) so UI sliders apply instantly.
 */
export class OneEuro {
  constructor(params) {
    this.p = params;
    this.x = new LowPass();
    this.dx = new LowPass();
    this.t = null;
  }

  /** @param x sample  @param t timestamp in seconds */
  filter(x, t) {
    if (this.t === null) {
      this.t = t;
      this.dx.filter(0, 1);
      return this.x.filter(x, 1);
    }
    let dt = t - this.t;
    if (dt <= 1e-6) dt = 1e-3;
    this.t = t;
    const d = (x - this.x.y) / dt;
    const edx = this.dx.filter(d, lpAlpha(this.p.dCutoff, dt));
    const cutoff = this.p.minCutoff + this.p.beta * Math.abs(edx);
    return this.x.filter(x, lpAlpha(cutoff, dt));
  }

  reset() {
    this.x.reset();
    this.dx.reset();
    this.t = null;
  }
}
