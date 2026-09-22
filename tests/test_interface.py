"""Tests for Step 5 interface (src/flybrain/interface.py).

Tests 4-8 exercise the real sub-circuit and are skipped if data/derived/*.npz is missing (same
convention as tests/test_control.py). max_frames is kept small (<=120) so the whole file runs
well under 90 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybrain.game import GameConfig, Obs
from flybrain.interface import (
    N_PARAMS,
    BatchResult,
    InterfaceParams,
    compute_input_rates,
    fitness,
    flap_decision,
    from_vector,
    play_batch,
    to_vector,
    update_trace,
)

DERIVED_DIR = Path(__file__).parent.parent / "data" / "derived"


@pytest.fixture(scope="module")
def network():
    """Load the real frozen sub-circuit; skip if not built yet."""
    if not (DERIVED_DIR / "subcircuit.npz").exists():
        pytest.skip("data/derived/subcircuit.npz not found")
    from flybrain.lif import load_network

    return load_network()


# --- 1. Parameter vector <-> named parameters ---------------------------------------------


class TestVectorRoundTrip:
    def test_to_from_round_trip_random(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            x = rng.normal(0.0, 2.5, size=N_PARAMS)
            x2 = to_vector(from_vector(x))
            np.testing.assert_allclose(x2, x, atol=1e-9)

    def test_from_to_round_trip_hand_built(self):
        p = InterfaceParams(
            G=120.0, lam=90.0, kappa=1.5, r0=15.0,
            w=np.linspace(-0.3, 0.3, 10), tau_ms=40.0, bias=0.25,
        )
        x = to_vector(p)
        p2 = from_vector(x)
        assert p2.G == pytest.approx(p.G, abs=1e-9)
        assert p2.lam == pytest.approx(p.lam, abs=1e-9)
        assert p2.kappa == pytest.approx(p.kappa, abs=1e-9)
        assert p2.r0 == pytest.approx(p.r0, abs=1e-9)
        np.testing.assert_allclose(p2.w, p.w, atol=1e-9)
        assert p2.tau_ms == pytest.approx(p.tau_ms, abs=1e-9)
        assert p2.bias == pytest.approx(p.bias, abs=1e-9)

    @pytest.mark.parametrize("xv", [-50.0, 50.0])
    def test_derived_quantities_in_bounds_at_extremes(self, xv):
        x = np.full(N_PARAMS, xv)
        p = from_vector(x)
        assert 0.0 <= p.G <= 300.0
        assert 20.0 <= p.lam <= 300.0
        assert p.kappa > 0.0
        assert 0.0 <= p.r0 <= 50.0
        assert 5.0 <= p.tau_ms <= 200.0
        # w and bias are unbounded by design (x[4:14]/100, x15 directly)
        assert np.isfinite(p.w).all()
        assert np.isfinite(p.bias)

    def test_wrong_length_rejected(self):
        with pytest.raises(AssertionError):
            from_vector(np.zeros(15))


# --- 2. Input mapping -----------------------------------------------------------------------


class TestInputMapping:
    @pytest.fixture
    def masks(self):
        # 4 neurons: [input-L, input-R, interneuron, output-seed]
        side_L = np.array([True, False, False, False])
        side_R = np.array([False, True, False, False])
        return side_L, side_R

    def test_proximity_monotone_in_dx(self, masks):
        side_L, side_R = masks
        G = np.array([100.0])
        lam = np.array([100.0])
        kappa = np.array([1.0])
        r0 = np.array([0.0])
        offset = np.array([0.0])
        near = compute_input_rates(G, lam, kappa, r0, np.array([0.0]), offset, side_L, side_R)
        far = compute_input_rates(G, lam, kappa, r0, np.array([500.0]), offset, side_L, side_R)
        # offset=0 -> rate_L == rate_R; nearer pipe -> larger drive
        assert near[0, 0] > far[0, 0] > 0.0

    def test_rate_l_gt_r_when_below_gap_and_reverse_above(self, masks):
        side_L, side_R = masks
        G = np.array([100.0, 100.0, 100.0])
        lam = np.array([100.0, 100.0, 100.0])
        kappa = np.array([1.0, 1.0, 1.0])
        r0 = np.array([10.0, 10.0, 10.0])
        dx = np.array([50.0, 50.0, 50.0])
        offset = np.array([2.0, -2.0, 0.0])  # below, above, level
        rates = compute_input_rates(G, lam, kappa, r0, dx, offset, side_L, side_R)
        rate_L = rates[:, 0]
        rate_R = rates[:, 1]
        assert rate_L[0] > rate_R[0]  # below gap (positive offset) -> L wins
        assert rate_L[1] < rate_R[1]  # above gap (negative offset) -> R wins
        assert rate_L[2] == pytest.approx(rate_R[2])  # offset 0 -> equal

    def test_non_input_neurons_exactly_zero(self, masks):
        side_L, side_R = masks
        G = np.array([300.0])
        lam = np.array([20.0])
        kappa = np.array([10.0])
        r0 = np.array([50.0])
        rates = compute_input_rates(
            G, lam, kappa, r0, np.array([0.0]), np.array([5.0]), side_L, side_R
        )
        assert rates[0, 2] == 0.0  # interneuron
        assert rates[0, 3] == 0.0  # output-seed

    def test_clipping_holds(self, masks):
        side_L, side_R = masks
        # r0=50 (max) + G=300 (max) can exceed 300 Hz before clipping
        G = np.array([300.0])
        lam = np.array([20.0])
        kappa = np.array([50.0])  # saturate sigmoid
        r0 = np.array([50.0])
        rates = compute_input_rates(
            G, lam, kappa, r0, np.array([0.0]), np.array([10.0]), side_L, side_R
        )
        assert rates.max() <= 300.0
        assert rates.min() >= 0.0

    def test_input_seed_bad_soma_side_raises(self):
        import pandas as pd
        from flybrain.interface import _input_seed_side_masks

        neurons = pd.DataFrame({
            "role": ["input_seed", "input_seed", "interneuron"],
            "somaSide": ["L", "M", "L"],
        })
        with pytest.raises(ValueError):
            _input_seed_side_masks(neurons)


# --- 3. Readout -------------------------------------------------------------------------------


class TestReadout:
    def test_trace_decay_exact(self):
        tau_ms = np.array([50.0, 20.0])
        trace0 = np.array([[10.0] * 10, [4.0] * 10])
        counts = np.zeros((2, 10))
        new_trace = update_trace(trace0, counts, tau_ms, frame_ms=25.0)
        expected = trace0 * np.exp(-25.0 / tau_ms)[:, None]
        np.testing.assert_allclose(new_trace, expected)

    def test_flap_iff_weighted_sum_positive(self):
        trace = np.array([[1.0] * 10, [1.0] * 10, [1.0] * 10])
        w = np.array([[0.1] * 10, [-0.1] * 10, [0.0] * 10])
        bias = np.array([0.0, 0.0, 0.001])
        flap = flap_decision(trace, w, bias)
        assert flap[0] == True   # 10*0.1 = 1.0 > 0
        assert flap[1] == False  # 10*-0.1 = -1.0 < 0
        assert flap[2] == True   # 0 + 0.001 > 0


# --- 4-8. Episode runner (real network) --------------------------------------------------------


def _make_x(rng, n=1, **overrides):
    xs = rng.normal(0.0, 1.0, size=(n, N_PARAMS))
    return xs


def _bias_x(bias):
    p = InterfaceParams(G=50.0, lam=100.0, kappa=1.0, r0=10.0, w=np.zeros(10), tau_ms=25.0, bias=bias)
    return to_vector(p)


class TestPlayBatch:
    def test_determinism(self, network):
        W, neurons = network
        rng = np.random.default_rng(3)
        X = _make_x(rng, n=3)
        res1 = play_batch(W, neurons, X, game_seed=1, sim_seed=2, max_frames=60)
        res2 = play_batch(W, neurons, X, game_seed=1, sim_seed=2, max_frames=60)
        np.testing.assert_array_equal(res1.scores, res2.scores)
        np.testing.assert_array_equal(res1.frames_survived, res2.frames_survived)

    def test_batch_independence(self, network):
        W, neurons = network
        rng = np.random.default_rng(4)
        X = _make_x(rng, n=4)
        res_batch = play_batch(W, neurons, X, game_seed=5, sim_seed=7, max_frames=60, record=True)
        res_solo = play_batch(W, neurons, X[2:3], game_seed=5, sim_seed=7, max_frames=60, record=True)
        assert res_batch.scores[2] == res_solo.scores[0]
        assert res_batch.frames_survived[2] == res_solo.frames_survived[0]
        # recorded arrays can differ in length (T = whole-batch runtime, not this candidate's own
        # lifetime), so compare only over the frames this candidate was actually alive
        fs = res_solo.frames_survived[0]
        np.testing.assert_array_equal(res_batch.flaps[2, :fs], res_solo.flaps[0, :fs])
        np.testing.assert_array_equal(res_batch.output_counts[2, :fs], res_solo.output_counts[0, :fs])

    def test_dead_candidate_frames_survived_others_continue(self, network):
        W, neurons = network
        rng = np.random.default_rng(9)
        other_x = _make_x(rng, n=1)[0]
        X = np.stack([_bias_x(-1e6), other_x])
        res = play_batch(W, neurons, X, game_seed=0, sim_seed=0, max_frames=120)
        # never-flap candidate dies around frame 34
        assert abs(res.frames_survived[0] - 34) <= 3
        # the other candidate's own trajectory is unaffected by candidate 0's death: it matches
        # what it gets when run completely alone
        res_solo = play_batch(
            W, neurons, other_x.reshape(1, N_PARAMS), game_seed=0, sim_seed=0, max_frames=120
        )
        assert res.frames_survived[1] == res_solo.frames_survived[0]
        assert res.scores[1] == res_solo.scores[0]

    def test_always_flap_dies_at_ceiling(self, network):
        W, neurons = network
        X = _bias_x(1e6).reshape(1, N_PARAMS)
        res = play_batch(W, neurons, X, game_seed=0, sim_seed=0, max_frames=120)
        assert abs(res.frames_survived[0] - 33) <= 3

    def test_record_true_shapes_and_dnp01(self, network):
        W, neurons = network
        rng = np.random.default_rng(11)
        X = _make_x(rng, n=2)
        res = play_batch(W, neurons, X, game_seed=0, sim_seed=0, max_frames=30, record=True)
        B = 2
        T = res.obs.shape[1]
        assert res.obs.shape == (B, T)
        assert res.flaps.shape == (B, T)
        assert res.output_counts.shape == (B, T, 10)
        assert res.dnp01_counts.shape == (B, T, 2)
        assert isinstance(res.obs[0, 0], Obs)
        assert res.dnp01_counts.shape[-1] == 2  # both DNp01 neurons present

    def test_fitness_tie_break_smaller_than_one_point(self):
        scores = np.array([2, 2])
        frames = np.array([600, 1])
        f = fitness(scores, frames, max_frames=600)
        assert f[0] > f[1]
        assert (f[0] - scores[0]) < 1.0
        assert (f[1] - scores[1]) < 1.0
