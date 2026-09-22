"""Step 3 -- leaky integrate-and-fire (LIF) simulator.

Reimplements the Shiu, Sterne, Spiller et al. 2024 (Nature 634:210-219) LIF model used by
`philshiu/Drosophila_brain_model/model.py` (MIT licence). All neuron constants below are copied
from that file, cited here per docs/PLAN.md Step 3 -- see docs/DATA.md for the parameter table.

Equations (exact linear integration over dt, Brian2's `linear` method -- NOT Euler):
    dv/dt = (v_0 - v + g + I_ext) / t_mbr
    dg/dt = -g / tau
    e_g = exp(-dt/tau); e_m = exp(-dt/t_mbr)
    v_new = v_0 + I_ext + (v - v_0 - I_ext) * e_m + g * (tau / (tau - t_mbr)) * (e_g - e_m)
    g_new = g * e_g

Per step: (1) delayed synaptic input arrives from the ring buffer, (2) Poisson drive is applied
(gated on rate > 0 so the RNG is only drawn for genuinely driven neurons -- rate is 0 everywhere
except input seeds in practice), (3) refractory neurons freeze v/g and just count down, otherwise
the exact integrator runs, (4) a spike resets v/g, starts the refractory countdown and writes
`sign * synapse_count * w_syn` into the ring buffer slot for every postsynaptic neuron in that
row -- read back exactly D = round(t_dly/dt) steps later.

RNG: a counter-based splitmix64 hash keyed on (per-candidate seed, global step, neuron index), so
batched and single-candidate runs are bit-identical (each candidate's stream depends only on its
own seed, never on which other candidates share the call).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import List
from scipy import sparse

DERIVED_DIR = Path(__file__).resolve().parents[2] / "data" / "derived"

# --- Model constants -- philshiu/Drosophila_brain_model/model.py (MIT), docs/PLAN.md Step 3 ---
V_0 = -52.0        # mV, resting potential
V_RST = -52.0      # mV, reset potential after a spike
V_TH = -45.0       # mV, spike threshold (v > v_th)
T_MBR = 20.0       # ms, membrane time constant
TAU = 5.0          # ms, synaptic conductance decay time constant
T_RFC = 2.2        # ms, refractory period (v and g frozen while refractory)
T_DLY = 1.8        # ms, synaptic delay
W_SYN = 0.275      # mV, voltage step per synapse (per unit signed synapse count)
F_POI = 250.0      # Hz-equivalent weight factor for the Poisson input drive
DT_DEFAULT = 0.2   # ms -- deviation from the paper's 0.1 ms, justified by the dt-check test
                    # (docs/PLAN.md Step 3 "Amended 2026-09-22"); t_dly/t_rfc stay whole steps.

POISSON_STEP = W_SYN * F_POI  # mV added to g on a Poisson event (= 68.75 mV)


def _steps(period_ms: float, dt: float) -> int:
    """Convert a period in ms to a whole number of dt-steps, asserting it divides exactly."""
    n = period_ms / dt
    r = round(n)
    assert abs(n - r) < 1e-9, f"{period_ms} ms is not a whole number of {dt} ms steps (got {n})"
    return int(r)


def load_network() -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Load the frozen sub-circuit written by Step 2: (W, neurons).

    W is scipy CSR float32, W[pre, post] = sign * synapse_count (w_syn not yet applied).
    neurons is the Step 2 table (idx, bodyId, type, somaSide, consensus_nt, sign, role);
    row order matches W's row/col order exactly (input seeds, then output seeds, then
    interneurons, each block sorted by bodyId -- see docs/DATA.md).
    """
    W = sparse.load_npz(DERIVED_DIR / "subcircuit.npz")
    neurons = pd.read_parquet(DERIVED_DIR / "neurons.parquet")
    return W, neurons


# --- RNG: counter-based splitmix64, see module docstring ---


@njit(cache=True)
def _hash64(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x


@njit(cache=True)
def _uniform(seed_b, step_global, i):
    # Hash the seed and the neuron index separately before combining: a plain seed ^ i lets
    # seed s at neuron i collide with seed s' at neuron i^s^s', so nearby seeds shared draws.
    key = _hash64(np.uint64(step_global) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i))
    x = _hash64(_hash64(seed_b) ^ key)
    return np.float64(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)  # 2**53


@njit(cache=True)
def _run_kernel(
    v, g, rf, buf, spike_count,
    ip, ix, dat, rate, seed,
    step_global0, n_steps, dt,
    v_0, v_th, e_g, e_m, g_coef, poisson_step,
    i_ext, D, R, record_mask, spiked,
):
    """The one simulation kernel: advances (v, g, rf, buf) by n_steps and accumulates
    spike_count in place. Also records (step, candidate, neuron) spike events for any
    neuron flagged in record_mask -- used by run() (all-False mask) and run_record().

    Each step runs in two passes over (b, i) so that every neuron reads the *start-of-step*
    delay-buffer slot before any of this step's own spikes write into that same slot -- a
    spike's propagation must land D steps in the future, never leak into the same step for
    a higher-indexed postsynaptic neuron processed later in the same pass."""
    B, N = v.shape
    rec_step = List.empty_list(types.int64)
    rec_b = List.empty_list(types.int64)
    rec_n = List.empty_list(types.int64)
    for k in range(n_steps):
        step_global = step_global0 + k
        # slot must be keyed on the ABSOLUTE step count, not the local k -- otherwise the
        # ring buffer's phase resets every call and chunked run() calls (Step 5: one call
        # per 25 ms game frame) desync from a single continuous run of the same length.
        slot = step_global % D
        # pass 1: arrival + Poisson drive + integrate/refractory -> latch spikes, no propagation yet
        for b in range(B):
            sb = seed[b]
            for i in range(N):
                # (1) delayed synaptic input arrives
                gi = g[b, i] + buf[slot, b, i]
                buf[slot, b, i] = 0.0
                g[b, i] = gi

                # (2) Poisson drive -- gated on rate>0 so the RNG is skipped for silent neurons
                r = rate[b, i]
                if r > 0.0:
                    u = _uniform(sb, step_global, i)
                    if u < r * dt / 1000.0:
                        g[b, i] += poisson_step

                # (3) refractory: v/g frozen, else exact linear integration
                if rf[b, i] > 0:
                    rf[b, i] -= 1
                    continue
                vi = v[b, i]
                gi2 = g[b, i]
                ie = i_ext[i]
                v_new = v_0 + ie + (vi - v_0 - ie) * e_m + gi2 * g_coef * (e_g - e_m)
                g_new = gi2 * e_g
                v[b, i] = v_new
                g[b, i] = g_new

                # (4) spike -> reset, refractory countdown; propagation deferred to pass 2
                if v_new > v_th:
                    v[b, i] = V_RST
                    g[b, i] = 0.0
                    rf[b, i] = R
                    spike_count[b, i] += 1
                    spiked[b, i] = True
                    if record_mask[i]:
                        rec_step.append(step_global)
                        rec_b.append(b)
                        rec_n.append(i)
        # pass 2: propagate this step's spikes into the delay buffer (arrives D steps later)
        for b in range(B):
            for i in range(N):
                if spiked[b, i]:
                    spiked[b, i] = False
                    for p in range(ip[i], ip[i + 1]):
                        buf[slot, b, ix[p]] += dat[p]
    return rec_step, rec_b, rec_n


class Simulator:
    """Batched LIF simulator over a frozen signed sub-circuit. State persists across
    run()/run_record() calls (Step 5 calls run() once per 25 ms game frame)."""

    def __init__(self, W, n_candidates: int, dt: float = DT_DEFAULT, seeds=None, i_ext=None):
        W_csr = W.tocsr() if not sparse.isspmatrix_csr(W) else W
        self.N = W_csr.shape[0]
        self.B = n_candidates
        self.dt = float(dt)

        self.ip = W_csr.indptr.astype(np.int64)
        self.ix = W_csr.indices.astype(np.int64)
        self.dat = (W_csr.data.astype(np.float32) * np.float32(W_SYN)).astype(np.float32)

        self.D = _steps(T_DLY, self.dt)
        self.R = _steps(T_RFC, self.dt)
        self.e_g = float(np.exp(-self.dt / TAU))
        self.e_m = float(np.exp(-self.dt / T_MBR))
        self.g_coef = TAU / (TAU - T_MBR)

        self.i_ext = (
            np.zeros(self.N, dtype=np.float32)
            if i_ext is None
            else np.asarray(i_ext, dtype=np.float32)
        )
        assert self.i_ext.shape == (self.N,)

        if seeds is None:
            seeds = np.arange(self.B, dtype=np.uint64)
        self.reset(seeds)

    def reset(self, seeds) -> None:
        """v=v_0, g=0, rf=0, buf=0, step_global=0; sets the per-candidate RNG seeds."""
        seeds = np.asarray(seeds, dtype=np.uint64)
        assert seeds.shape == (self.B,), f"expected {self.B} seeds, got {seeds.shape}"
        self.seeds = seeds
        self.v = np.full((self.B, self.N), V_0, dtype=np.float32)
        self.g = np.zeros((self.B, self.N), dtype=np.float32)
        self.rf = np.zeros((self.B, self.N), dtype=np.int32)
        self.buf = np.zeros((self.D, self.B, self.N), dtype=np.float32)
        self._spiked = np.zeros((self.B, self.N), dtype=np.bool_)  # reusable scratch, pass 2
        self.step_global = 0

    def _run(self, n_steps: int, rates, record_mask):
        rates = np.ascontiguousarray(rates, dtype=np.float32)
        assert rates.shape == (self.B, self.N)
        spike_count = np.zeros((self.B, self.N), dtype=np.int32)
        rec_step, rec_b, rec_n = _run_kernel(
            self.v, self.g, self.rf, self.buf, spike_count,
            self.ip, self.ix, self.dat, rates, self.seeds,
            self.step_global, n_steps, self.dt,
            V_0, V_TH, self.e_g, self.e_m, self.g_coef, POISSON_STEP,
            self.i_ext, self.D, self.R, record_mask, self._spiked,
        )
        self.step_global += n_steps
        return spike_count, rec_step, rec_b, rec_n

    def run(self, n_steps: int, rates) -> np.ndarray:
        """Advance n_steps with the given (B, N) float32 Hz rate array. Returns int32
        spike counts (B, N) for this window."""
        no_record = np.zeros(self.N, dtype=np.bool_)
        spike_count, _, _, _ = self._run(n_steps, rates, no_record)
        return spike_count

    def run_record(self, n_steps: int, rates, record_idx) -> list[tuple[int, int, int]]:
        """Like run(), but also returns [(step, candidate, neuron), ...] spike events for
        every neuron in record_idx (step counts from episode start, not from this call)."""
        mask = np.zeros(self.N, dtype=np.bool_)
        mask[np.asarray(record_idx, dtype=np.int64)] = True
        _, rec_step, rec_b, rec_n = self._run(n_steps, rates, mask)
        return list(zip(rec_step, rec_b, rec_n))


# --- Benchmark (PLAN Step 3 performance gate: >= 400 sim-s/wall-min/process) ---


def _bench() -> None:
    W, neurons = load_network()
    N = W.shape[0]
    B = 16
    sim_seconds = 10.0

    input_mask = (neurons["role"] == "input_seed").to_numpy()
    rates = np.zeros((B, N), dtype=np.float32)
    rates[:, input_mask] = 50.0

    sim = Simulator(W, B, dt=DT_DEFAULT)
    # warm-up / compile call, excluded from timing
    sim.run(5, rates)
    sim.reset(np.arange(B, dtype=np.uint64))

    n_steps = _steps(sim_seconds * 1000.0, sim.dt)
    t0 = time.perf_counter()
    spikes = sim.run(n_steps, rates)
    elapsed = time.perf_counter() - t0

    sim_s_per_wallmin = B * sim_seconds * 60.0 / elapsed
    mean_rate = spikes.sum() / (B * N * sim_seconds)

    print(f"benchmark: dt={sim.dt} ms, B={B}, N={N}, steps={n_steps}, wall={elapsed:.3f} s")
    print(f"sim-s per wall-min per process = {sim_s_per_wallmin:.1f}")
    print(f"mean spikes/neuron/s = {mean_rate:.4f}")
    print(f"gate (>=400): {'PASS' if sim_s_per_wallmin >= 400 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="flybrain LIF simulator")
    parser.add_argument("--bench", action="store_true", help="run the Step 3 performance benchmark")
    args = parser.parse_args()
    if args.bench:
        _bench()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
