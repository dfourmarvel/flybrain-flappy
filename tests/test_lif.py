"""Tests for Step 3 LIF simulator (src/flybrain/lif.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from flybrain.lif import (
    DT_DEFAULT,
    T_MBR,
    T_RFC,
    TAU,
    V_0,
    V_TH,
    Simulator,
    load_network,
)


def _analytic_rate_hz(i_ext: float) -> float:
    """Constant-current isolated-neuron firing rate (no synaptic input): the neuron
    integrates from v_rst to v_th under dv/dt = (v_0 - v + I)/t_mbr, then refracts."""
    gap = V_TH - V_0
    period_ms = T_RFC + T_MBR * math.log(i_ext / (i_ext - gap))
    return 1000.0 / period_ms


# --- Test 1: isolated neuron, constant current -----------------------------------------


@pytest.mark.parametrize("dt", [0.1, 0.2])
def test_isolated_neuron_constant_current_matches_analytic_rate(dt):
    W = sparse.csr_matrix((1, 1), dtype=np.float32)
    sim = Simulator(W, n_candidates=1, dt=dt, i_ext=np.array([10.0], dtype=np.float32))
    sim_seconds = 5.0
    n_steps = round(sim_seconds * 1000.0 / dt)
    rates = np.zeros((1, 1), dtype=np.float32)
    spikes = sim.run(n_steps, rates)
    measured = spikes[0, 0] / sim_seconds
    expected = _analytic_rate_hz(10.0)
    print(f"dt={dt}: measured {measured:.3f} Hz, analytic {expected:.3f} Hz")
    assert measured == pytest.approx(expected, rel=0.05)


def test_isolated_neuron_below_threshold_never_spikes():
    W = sparse.csr_matrix((1, 1), dtype=np.float32)
    sim = Simulator(W, n_candidates=1, dt=DT_DEFAULT, i_ext=np.array([5.0], dtype=np.float32))
    n_steps = round(5000.0 / DT_DEFAULT)
    rates = np.zeros((1, 1), dtype=np.float32)
    spikes = sim.run(n_steps, rates)
    assert spikes[0, 0] == 0


# --- Test 2: exact integrator vs a fine-grained Euler reference ------------------------


def test_exact_integrator_matches_fine_euler_reference():
    dt = 0.2
    i_ext_val = 10.0
    v_start, g_start = -50.0, 5.0

    W = sparse.csr_matrix((1, 1), dtype=np.float32)
    sim = Simulator(W, n_candidates=1, dt=dt, i_ext=np.array([i_ext_val], dtype=np.float32))
    sim.v[0, 0] = np.float32(v_start)
    sim.g[0, 0] = np.float32(g_start)
    rates = np.zeros((1, 1), dtype=np.float32)
    sim.run(1, rates)
    v_kernel, g_kernel = float(sim.v[0, 0]), float(sim.g[0, 0])
    assert v_kernel <= V_TH, "reference values must not cross threshold mid-test"

    sub_dt = 1e-4
    n_sub = round(dt / sub_dt)
    v, g = v_start, g_start
    for _ in range(n_sub):
        dv = (V_0 - v + g + i_ext_val) / T_MBR * sub_dt
        dg = -g / TAU * sub_dt
        v, g = v + dv, g + dg

    rel_err_v = abs(v_kernel - v) / abs(v)
    rel_err_g = abs(g_kernel - g) / abs(g)
    print(f"v: kernel={v_kernel:.6f} euler={v:.6f} rel_err={rel_err_v:.2e}")
    print(f"g: kernel={g_kernel:.6f} euler={g:.6f} rel_err={rel_err_g:.2e}")
    assert rel_err_v < 1e-4
    assert rel_err_g < 1e-4


# --- Test 3: zero input on the real network is silent -----------------------------------


def test_real_network_zero_input_is_silent():
    W, neurons = load_network()
    N = W.shape[0]
    sim = Simulator(W, n_candidates=1, dt=DT_DEFAULT)
    n_steps = round(5000.0 / DT_DEFAULT)
    rates = np.zeros((1, N), dtype=np.float32)
    spikes = sim.run(n_steps, rates)
    assert spikes.sum() == 0


# --- Test 4: conduction gate (PLAN Step 3) -----------------------------------------------


@pytest.mark.slow
def test_conduction_gate_lc4_drives_dnp01():
    W, neurons = load_network()
    N = W.shape[0]
    lc4_idx = neurons.index[(neurons["role"] == "input_seed") & (neurons["type"] == "LC4")].to_numpy()
    dnp01_idx = neurons.index[neurons["type"] == "DNp01"].to_numpy()
    assert len(dnp01_idx) == 2  # both hemispheres

    n_seeds = 10
    dt = DT_DEFAULT
    n_steps = round(100.0 / dt)
    sim = Simulator(W, n_candidates=n_seeds, dt=dt, seeds=np.arange(n_seeds, dtype=np.uint64))
    rates = np.zeros((n_seeds, N), dtype=np.float32)
    rates[:, lc4_idx] = 150.0

    events = sim.run_record(n_steps, rates, dnp01_idx)
    first_latency_ms = {}
    for step, b, _n in events:
        t_ms = step * dt
        if b not in first_latency_ms or t_ms < first_latency_ms[b]:
            first_latency_ms[b] = t_ms

    n_conducted = len(first_latency_ms)
    print("conduction gate: LC4 @150 Hz, first DNp01 spike latency per seed (ms):")
    for b in range(n_seeds):
        lat = first_latency_ms.get(b, None)
        print(f"  seed {b}: {'no spike' if lat is None else f'{lat:.2f} ms'}")
    print(f"conducted in {n_conducted}/{n_seeds} seeds within 100 ms")

    assert n_conducted >= 9, (
        f"conduction gate FAILED: only {n_conducted}/10 seeds spiked DNp01 within 100 ms "
        f"-- do not change network/parameters/protocol, report BLOCKED with these numbers"
    )


# --- Test 5: dt sensitivity check (PLAN amendment 2026-09-22) ---------------------------


@pytest.mark.slow
def test_dt_sensitivity_output_rates_within_tolerance():
    W, neurons = load_network()
    N = W.shape[0]
    lc4_mask = (neurons["type"] == "LC4").to_numpy()  # PLAN Step 3 amendment: LC4-stimulation protocol
    output_idx = neurons.index[neurons["role"] == "output_seed"].to_numpy()
    output_types = neurons.loc[output_idx, "type"].to_numpy()
    output_sides = neurons.loc[output_idx, "somaSide"].to_numpy()

    # 40 trials: at 10 the least active neuron's +10% bias and trial noise were indistinguishable.
    n_seeds = 40
    sim_seconds = 1.0
    rates_by_dt = {}
    for dt in (0.1, 0.2):
        n_steps = round(sim_seconds * 1000.0 / dt)
        sim = Simulator(W, n_candidates=n_seeds, dt=dt, seeds=np.arange(10_000, 10_000 + n_seeds, dtype=np.uint64))
        rates = np.zeros((n_seeds, N), dtype=np.float32)
        rates[:, lc4_mask] = 150.0
        spikes = sim.run(n_steps, rates)
        mean_rate_hz = spikes[:, output_idx].mean(axis=0) / sim_seconds
        rates_by_dt[dt] = mean_rate_hz

    r01, r02 = rates_by_dt[0.1], rates_by_dt[0.2]
    print(f"dt-check: LC4 @150 Hz, 1 s, mean rate per output seed ({n_seeds} seeds)")
    print(f"{'type':8s} {'side':5s} {'rate@0.1':>10s} {'rate@0.2':>10s} {'diff':>8s} {'ok':>4s}")
    all_ok = True
    for i in range(len(output_idx)):
        diff = r02[i] - r01[i]
        # Sanity bound only, NOT the fidelity claim. The measured bias (120 trials, 95% CI) is in
        # docs/DATA.md: <= 1.8% for 8 of 10 output neurons, +11.9% (CI +9.9..+13.9%) for DNp06 L.
        # A +-12% gate set from one seed block passed or failed by seed luck, so it was dropped.
        tol = max(0.20 * r01[i], 2.0)
        ok = abs(diff) <= tol
        all_ok &= ok
        print(f"{output_types[i]:8s} {output_sides[i]:5s} {r01[i]:10.3f} {r02[i]:10.3f} {diff:8.3f} {str(ok):>4s}")

    assert all_ok, "dt sensitivity outside tolerance (see table above)"


# --- Test 6: batched vs single-candidate bit-identical ------------------------------------


def test_batched_matches_single_candidate():
    W, neurons = load_network()
    N = W.shape[0]
    input_mask = (neurons["role"] == "input_seed").to_numpy()
    rng = np.random.default_rng(0)

    B = 4
    seeds = np.array([11, 22, 33, 44], dtype=np.uint64)
    rates_all = np.zeros((B, N), dtype=np.float32)
    for b in range(B):
        rates_all[b, input_mask] = rng.choice([0.0, 80.0, 150.0], size=input_mask.sum())

    n_steps = 40  # 8 ms at dt=0.2

    sim_batch = Simulator(W, n_candidates=B, dt=DT_DEFAULT, seeds=seeds)
    spikes_batch = sim_batch.run(n_steps, rates_all)
    v_batch = sim_batch.v.copy()

    for b in range(B):
        sim_single = Simulator(W, n_candidates=1, dt=DT_DEFAULT, seeds=seeds[b:b + 1])
        spikes_single = sim_single.run(n_steps, rates_all[b:b + 1])
        assert np.array_equal(spikes_batch[b], spikes_single[0])
        assert np.array_equal(v_batch[b], sim_single.v[0])


# --- Test 7: synaptic delay ---------------------------------------------------------------


def test_synaptic_delay_is_exactly_D_steps():
    dt = DT_DEFAULT
    W = sparse.csr_matrix(
        (np.array([5000.0], dtype=np.float32), (np.array([0]), np.array([1]))),
        shape=(2, 2),
        dtype=np.float32,
    )
    sim = Simulator(W, n_candidates=1, dt=dt, i_ext=np.array([10.0, 0.0], dtype=np.float32))
    # neuron 0 (v_rst=-52, v_th=-45, I_ext=10) first crosses threshold at ~24.1 ms
    # (v(t) = v0+I - 10*exp(-t/t_mbr)); 40 ms is plenty of margin for it and the reaction.
    n_steps = round(40.0 / dt)
    rates = np.zeros((1, 2), dtype=np.float32)
    events = sim.run_record(n_steps, rates, record_idx=[0, 1])

    step0 = min(s for s, b, n in events if n == 0)
    step1 = min(s for s, b, n in events if n == 1)
    print(f"neuron 0 first spike at step {step0}, neuron 1 first spike at step {step1}, D={sim.D}")
    # the 1375 mV kick makes neuron 1 fire in the very step it arrives, so the gap is exactly D
    assert step1 == step0 + sim.D


# --- Test 8: continuity across calls (state + RNG stream) ---------------------------------


def test_run_continuity_matches_single_call():
    W, neurons = load_network()
    N = W.shape[0]
    input_mask = (neurons["role"] == "input_seed").to_numpy()
    rates = np.zeros((1, N), dtype=np.float32)
    rates[:, input_mask] = 100.0

    sim_split = Simulator(W, n_candidates=1, dt=DT_DEFAULT, seeds=np.array([7], dtype=np.uint64))
    s1 = sim_split.run(125, rates)
    s2 = sim_split.run(125, rates)
    total_split = s1 + s2

    sim_once = Simulator(W, n_candidates=1, dt=DT_DEFAULT, seeds=np.array([7], dtype=np.uint64))
    total_once = sim_once.run(250, rates)

    assert np.array_equal(total_split, total_once)
    assert np.array_equal(sim_split.v, sim_once.v)
    assert np.array_equal(sim_split.g, sim_once.g)
