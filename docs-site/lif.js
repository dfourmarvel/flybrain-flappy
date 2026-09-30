// Browser port of src/flybrain/lif.py (Phase 2 step P2, docs/PLAN.md). Single fly, no dependencies.
//
// Physics identical to lif.py `_run_kernel`:
//   per step: (1) delayed synaptic input arrives from the ring buffer into g,
//             (2) Poisson kick of w_syn*f_poi into g (applied even while refractory),
//             (3) refractory neurons freeze v and g and count down; others run the exact
//                 linear integrator (not Euler),
//             (4) v > v_th -> reset v, zero g, start refractory, latch the spike;
//   then a second pass propagates this step's spikes into the ring-buffer slot just read,
//   so they arrive exactly D = t_dly/dt steps later. The slot is keyed on the ABSOLUTE step,
//   so chunked run() calls equal one long call.
//
// Deliberate differences from lif.py (statistical, not spike-for-spike, equivalence: PLAN L11):
//   - state is Float64Array (Python uses float32);
//   - RNG is xoshiro128** seeded via splitmix32, one sequential stream (Python uses a
//     counter-based splitmix64 hash);
//   - Poisson draws for all driven neurons happen before the per-neuron arrival/integrate loop
//     within a step. Arrival and kick are both additions to the same neuron's own g, so this is
//     the same maths as lif.py's per-neuron (1)-then-(2) order.

function splitmix32Stream(seed) {
  let a = seed | 0;
  return function next() {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return t >>> 0;
  };
}

/** Seeded uniform [0, 1) generator (xoshiro128** seeded via splitmix32) for code outside the
 * hot loop, e.g. level generation. Never Math.random. */
export function makeUniform(seed) {
  const sm = splitmix32Stream(seed >>> 0);
  let s0 = sm() | 0, s1 = sm() | 0, s2 = sm() | 0, s3 = sm() | 0;
  if ((s0 | s1 | s2 | s3) === 0) s0 = 1;
  return function uniform() {
    let r = Math.imul(s1, 5);
    r = (r << 7) | (r >>> 25);
    r = Math.imul(r, 9) >>> 0;
    const t = s1 << 9;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = (s3 << 11) | (s3 >>> 21);
    return r * 2.3283064365386963e-10;
  };
}

// Refractory countdown given to silenced neurons each run() call; far above any call length.
const SILENCED_RF = 0x3fffffff;

function wholeSteps(periodMs, dt) {
  const n = periodMs / dt;
  const r = Math.round(n);
  if (Math.abs(n - r) > 1e-9) throw new Error(`${periodMs} ms is not a whole number of ${dt} ms steps`);
  return r;
}

export class LiveBrain {
  /**
   * @param {object} model - parsed model.json
   * @param {{indptr: Int32Array, indices: Uint16Array, weights: Int16Array}} csr
   * @param {{seed?: number, iExt?: Float64Array|null}} [opts]
   */
  constructor(model, csr, { seed = 0, iExt = null } = {}) {
    const c = model.constants;
    const N = model.n_neurons;
    this.N = N;
    this.dt = c.dt;
    this.v0 = c.v_0;
    this.vRst = c.v_rst;
    this.vTh = c.v_th;
    this.D = wholeSteps(c.t_dly, c.dt);
    this.R = wholeSteps(c.t_rfc, c.dt);
    this.eG = Math.exp(-c.dt / c.tau);
    this.eM = Math.exp(-c.dt / c.t_mbr);
    this.gCoef = c.tau / (c.tau - c.t_mbr);
    this.poissonStep = c.w_syn * c.f_poi;

    if (csr.indptr.length !== N + 1) throw new Error("indptr length != n_neurons + 1");
    const nnz = csr.indices.length;
    this.indptr = csr.indptr;
    this.indices = csr.indices;
    // Per-edge weight = signed synapse count x w_syn, computed once.
    const w = new Float64Array(nnz);
    for (let p = 0; p < nnz; p++) w[p] = csr.weights[p] * c.w_syn;
    this.weights = w;

    // inhibitionOff: a pruned CSR with every negative-weight edge removed. Measured faster than
    // a zeroed-weights copy or a per-edge sign branch (fewer edges to walk); see live.test.mjs T7.
    const ipE = new Int32Array(N + 1);
    let cnt = 0;
    for (let p = 0; p < nnz; p++) if (csr.weights[p] > 0) cnt++;
    const ixE = new Uint16Array(cnt);
    const wE = new Float64Array(cnt);
    let q = 0;
    for (let i = 0; i < N; i++) {
      ipE[i] = q;
      for (let p = csr.indptr[i]; p < csr.indptr[i + 1]; p++) {
        if (csr.weights[p] > 0) {
          ixE[q] = csr.indices[p];
          wE[q] = w[p];
          q++;
        }
      }
    }
    ipE[N] = q;
    this.indptrExc = ipE;
    this.indicesExc = ixE;
    this.weightsExc = wE;

    this.iExt = iExt ? Float64Array.from(iExt) : new Float64Array(N);
    if (this.iExt.length !== N) throw new Error("iExt length != n_neurons");

    this.v = new Float64Array(N);
    this.g = new Float64Array(N);
    this.rf = new Int32Array(N);
    this.buf = new Float64Array(this.D * N);
    this._spikeList = new Int32Array(N);
    this._driven = new Int32Array(N);
    this._probs = new Float64Array(N);
    this._counts = new Int32Array(N);
    this.silenced = null;
    this._sil = new Uint8Array(N); // 1 = silenced (copy of `silenced`)
    this._silList = new Int32Array(N);
    this._nSil = 0;
    this.inhibitionOff = false;
    this.reset(seed);
  }

  /** v = v_0, g = 0, refractory = 0, delay buffer empty, step = 0; reseed the RNG. */
  reset(seed = 0) {
    this.v.fill(this.v0);
    this.g.fill(0);
    this.rf.fill(0);
    this.buf.fill(0);
    this.stepGlobal = 0;
    const sm = splitmix32Stream(seed >>> 0);
    let s0 = sm(), s1 = sm(), s2 = sm(), s3 = sm();
    if ((s0 | s1 | s2 | s3) === 0) s0 = 1; // xoshiro must not start from all-zero state
    this._s0 = s0 | 0;
    this._s1 = s1 | 0;
    this._s2 = s2 | 0;
    this._s3 = s3 | 0;
  }

  /** @param {{silenced?: Uint8Array|null, inhibitionOff?: boolean}} lesions */
  setLesions({ silenced = null, inhibitionOff = false } = {}) {
    if (silenced && silenced.length !== this.N) throw new Error("silenced length != n_neurons");
    this.silenced = silenced;
    this.inhibitionOff = !!inhibitionOff;
    // Neurons leaving or entering the silenced set restart from rest.
    let n = 0;
    for (let i = 0; i < this.N; i++) {
      const s = silenced && silenced[i] ? 1 : 0;
      if (s !== this._sil[i]) {
        this.v[i] = this.v0;
        this.g[i] = 0;
        this.rf[i] = 0;
      }
      this._sil[i] = s;
      if (s) this._silList[n++] = i;
    }
    this._nSil = n;
  }

  /**
   * Advance nSteps with per-neuron Poisson rates (Hz). Returns spike counts for this window.
   * The returned Int32Array is reused by the next call: copy it if you need to keep it.
   * @param {number} nSteps
   * @param {Float32Array} rates - length N
   * @returns {Int32Array}
   */
  run(nSteps, rates) {
    const N = this.N;
    if (rates.length !== N) throw new Error("rates length != n_neurons");
    if (!(nSteps >= 0 && nSteps < SILENCED_RF)) throw new Error("nSteps out of range");
    const v = this.v, g = this.g, rf = this.rf, buf = this.buf, iExt = this.iExt;
    const counts = this._counts;
    counts.fill(0);
    const spikeList = this._spikeList;
    const sil = this._sil;
    const ip = this.inhibitionOff ? this.indptrExc : this.indptr;
    const ix = this.inhibitionOff ? this.indicesExc : this.indices;
    const w = this.inhibitionOff ? this.weightsExc : this.weights;
    const D = this.D, R = this.R;
    const v0 = this.v0, vRst = this.vRst, vTh = this.vTh;
    const eG = this.eG, eM = this.eM, gCoef = this.gCoef, kick = this.poissonStep;
    const decayDiff = gCoef * (eG - eM);

    // Neurons with rate > 0 (the only ones that draw from the RNG) and their per-step
    // probability rate*dt/1000, fixed for this call.
    const driven = this._driven;
    let nDriven = 0;
    for (let i = 0; i < N; i++) {
      if (rates[i] > 0 && sil[i] === 0) driven[nDriven++] = i;
    }
    const probs = this._probs;
    const pScale = this.dt / 1000;
    for (let k = 0; k < nDriven; k++) probs[k] = rates[driven[k]] * pScale;

    // Silenced neurons ride the refractory branch for the whole call (v and g frozen, never
    // integrate, never spike), which keeps a lesion check out of the per-neuron hot loop.
    const silList = this._silList, nSil = this._nSil;
    for (let k = 0; k < nSil; k++) {
      const i = silList[k];
      v[i] = v0;
      g[i] = 0;
      rf[i] = SILENCED_RF;
    }

    let s0 = this._s0, s1 = this._s1, s2 = this._s2, s3 = this._s3;
    let step = this.stepGlobal;

    for (let n = 0; n < nSteps; n++, step++) {
      const slot = step % D;
      const off = slot * N;

      // (2) Poisson drive, applied even to refractory neurons (xoshiro128** inline).
      for (let k = 0; k < nDriven; k++) {
        let r = Math.imul(s1, 5);
        r = (r << 7) | (r >>> 25);
        r = Math.imul(r, 9) >>> 0;
        const t = s1 << 9;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = (s3 << 11) | (s3 >>> 21);
        if (r * 2.3283064365386963e-10 < probs[k]) g[driven[k]] += kick;
      }

      // pass 1: arrival, refractory or integrate, threshold
      let nSpk = 0;
      for (let i = 0; i < N; i++) {
        // (1) delayed synaptic input arrives
        const gi = g[i] + buf[off + i];
        buf[off + i] = 0;
        // (3) refractory: v and g frozen (arrivals and kicks still accumulate into g)
        if (rf[i] > 0) {
          g[i] = gi;
          rf[i]--;
          continue;
        }
        const ie = iExt[i];
        const vNew = v0 + ie + (v[i] - v0 - ie) * eM + gi * decayDiff;
        // (4) spike -> reset, refractory countdown, propagation deferred to pass 2
        if (vNew > vTh) {
          v[i] = vRst;
          g[i] = 0;
          rf[i] = R;
          counts[i]++;
          spikeList[nSpk++] = i;
        } else {
          v[i] = vNew;
          g[i] = gi * eG;
        }
      }

      // pass 2: this step's spikes land in the slot just read -> arrive exactly D steps later
      for (let s = 0; s < nSpk; s++) {
        const i = spikeList[s];
        const end = ip[i + 1];
        for (let p = ip[i]; p < end; p++) buf[off + ix[p]] += w[p];
      }
    }

    for (let k = 0; k < nSil; k++) {
      const i = silList[k];
      g[i] = 0; // drop input that accumulated while silenced
      rf[i] = 0;
    }

    this._s0 = s0;
    this._s1 = s1;
    this._s2 = s2;
    this._s3 = s3;
    this.stepGlobal = step;
    return counts;
  }
}
