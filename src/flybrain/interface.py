"""Step 5 -- interface between the game (game.py) and the frozen brain (lif.py).

The only learned part of this project (docs/PLAN.md L2). A 16-element real vector, searched
unbounded by CMA-ES, maps to bounded interface parameters via the fixed transforms in
`from_vector` / `to_vector`. Everything here is a *convention* for wiring an arbitrary game to
an arbitrary sub-circuit -- none of it is a claim about fly biology.

INTERFACE CONVENTION (docs/PLAN.md Step 5): the left/right somaSide split used below to route
"bird below gap" vs "bird above gap" drive onto the input-seed neurons is an arbitrary interface
choice with no biological meaning. In the real fly, left/right corresponds to visual-field side,
not vertical position in a game. docs/RESULTS.md must repeat this caveat.

Sensorimotor latency: `play_batch` computes each frame's input rates from the Obs at the *start*
of that frame, then simulates the brain for the frame's duration, then applies the resulting flap
decision when stepping the game. The brain therefore always acts on a 25 ms (one frame) stale
observation -- a fixed 25 ms sensorimotor latency, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flybrain.game import Game, GameConfig, Obs
from flybrain.lif import DT_DEFAULT, Simulator

N_PARAMS = 16
RATE_MAX_HZ = 300.0

OUTPUT_SEED_TYPES = ("DNp01", "DNp02", "DNp04", "DNp06", "DNp11")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(y: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return np.log(y / (1.0 - y))


# --- Parameter vector <-> named parameters -----------------------------------------------


@dataclass
class InterfaceParams:
    """Bounded interface parameters, derived from an unbounded 16-vector (see `from_vector`)."""

    G: float          # Hz, max looming drive (gain)
    lam: float        # px, distance-sensitivity length scale
    kappa: float       # above/below-gap offset sensitivity
    r0: float          # Hz, baseline rate
    w: np.ndarray       # (10,) readout weights, one per output-seed neuron (row order below)
    tau_ms: float       # ms, spike-trace decay time constant
    bias: float          # readout bias


def from_vector(x: np.ndarray) -> InterfaceParams:
    """Map an unbounded 16-vector to bounded InterfaceParams (the lead's fixed transform,
    docs/PLAN.md Step 5)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (N_PARAMS,), f"expected a length-{N_PARAMS} vector, got {x.shape}"

    G = 300.0 * _sigmoid(x[0])
    lam = 20.0 + 280.0 * _sigmoid(x[1])
    kappa = np.exp(x[2])
    r0 = 50.0 * _sigmoid(x[3])
    w = (x[4:14] / 100.0).copy()
    tau_ms = 5.0 + 195.0 * _sigmoid(x[14])
    bias = float(x[15])

    return InterfaceParams(
        G=float(G), lam=float(lam), kappa=float(kappa), r0=float(r0),
        w=w, tau_ms=float(tau_ms), bias=bias,
    )


def to_vector(p: InterfaceParams) -> np.ndarray:
    """Exact inverse of `from_vector`."""
    x = np.empty(N_PARAMS, dtype=np.float64)
    x[0] = _logit(np.array(p.G / 300.0))
    x[1] = _logit(np.array((p.lam - 20.0) / 280.0))
    x[2] = np.log(p.kappa)
    x[3] = _logit(np.array(p.r0 / 50.0))
    x[4:14] = np.asarray(p.w, dtype=np.float64) * 100.0
    x[14] = _logit(np.array((p.tau_ms - 5.0) / 195.0))
    x[15] = p.bias
    # CMA-ES can propose |x| large enough to saturate the sigmoids; clip so the inverse stays
    # finite (from_vector(to_vector(p)) is then exact to float tolerance, not bit-identical).
    x[:] = np.clip(x, -700.0, 700.0)
    return x


def describe(p: InterfaceParams) -> str:
    """Human-readable one-liner for logs."""
    w_str = ", ".join(f"{v:+.3f}" for v in p.w)
    return (
        f"G={p.G:.1f}Hz lam={p.lam:.1f}px kappa={p.kappa:.4f} r0={p.r0:.1f}Hz "
        f"tau={p.tau_ms:.1f}ms bias={p.bias:+.3f} w=[{w_str}]"
    )


# --- Input mapping: game observation -> per-neuron Poisson rate --------------------------


def _input_seed_side_masks(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(side_L, side_R): boolean masks, length N, True only for input-seed neurons on that
    soma side. Raises if an input-seed neuron has a somaSide other than 'L'/'R'."""
    role = neurons["role"].to_numpy()
    soma = neurons["somaSide"].to_numpy()
    is_input = role == "input_seed"

    valid = np.isin(soma, ["L", "R"])
    bad = is_input & ~valid
    if bad.any():
        bad_soma = pd.Series(soma[bad]).value_counts(dropna=False).to_dict()
        raise ValueError(
            f"input_seed neurons with unexpected somaSide (expected only 'L'/'R'): {bad_soma}"
        )

    side_L = is_input & (soma == "L")
    side_R = is_input & (soma == "R")
    return side_L, side_R


def compute_input_rates(
    G: np.ndarray, lam: np.ndarray, kappa: np.ndarray, r0: np.ndarray,
    dx: np.ndarray, offset: np.ndarray,
    side_L: np.ndarray, side_R: np.ndarray,
) -> np.ndarray:
    """Vectorised input mapping (docs/PLAN.md Step 5, lead's formula). All of G/lam/kappa/
    r0/dx/offset are (B,); side_L/side_R are (N,) boolean masks (see `_input_seed_side_masks`).
    Returns float32 (B, N), clipped to [0, RATE_MAX_HZ]; non-input-seed columns are exactly 0.
    """
    G = np.asarray(G, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)

    proximity = np.exp(-dx / lam)
    rate_L = r0 + G * proximity * _sigmoid(kappa * offset)
    rate_R = r0 + G * proximity * _sigmoid(-kappa * offset)

    B = G.shape[0]
    N = side_L.shape[0]
    rates = np.zeros((B, N), dtype=np.float64)
    rates[:, side_L] = rate_L[:, None]
    rates[:, side_R] = rate_R[:, None]
    np.clip(rates, 0.0, RATE_MAX_HZ, out=rates)
    return rates.astype(np.float32)


def obs_batch_rates(
    G: np.ndarray, lam: np.ndarray, kappa: np.ndarray, r0: np.ndarray,
    obs_list: list[Obs], config: GameConfig,
    side_L: np.ndarray, side_R: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: extract dx/offset from a list of per-candidate Obs and call
    `compute_input_rates`."""
    dx = np.array([max(o.next_pipe_dx, 0.0) for o in obs_list], dtype=np.float64)
    offset = np.array(
        [(o.bird_y - o.next_gap_y) / (config.gap_height / 2.0) for o in obs_list],
        dtype=np.float64,
    )
    return compute_input_rates(G, lam, kappa, r0, dx, offset, side_L, side_R)


# --- Readout: output-seed spike counts -> exponentially-decaying trace -> flap -----------


def update_trace(trace: np.ndarray, counts: np.ndarray, tau_ms: np.ndarray, frame_ms: float) -> np.ndarray:
    """trace <- trace * exp(-frame_ms/tau_ms) + counts. trace/counts: (B,10); tau_ms: (B,)."""
    tau_ms = np.asarray(tau_ms, dtype=np.float64)
    decay = np.exp(-frame_ms / tau_ms)
    return trace * decay[:, None] + counts


def flap_decision(trace: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """flap = (trace . w + bias) > 0, per candidate. trace/w: (B,10); bias: (B,)."""
    return (np.sum(trace * w, axis=1) + np.asarray(bias, dtype=np.float64)) > 0.0


def fitness(scores: np.ndarray, frames_survived: np.ndarray, max_frames: int) -> np.ndarray:
    """Step 6 fitness: score plus a small survival-time tie-break bonus (< 1 point)."""
    scores = np.asarray(scores, dtype=np.float64)
    frames_survived = np.asarray(frames_survived, dtype=np.float64)
    return scores + 0.1 * frames_survived / max_frames


# --- Episode runner ------------------------------------------------------------------------


@dataclass
class BatchResult:
    """Result of `play_batch`. scores/frames_survived: (B,) int. When record=True, the
    remaining fields are populated with per-candidate, per-frame arrays of length
    T = number of frames the batch actually ran (all share the same T; frames after a
    candidate's death simply repeat its last Obs / a False flap / that frame's real, still-
    running brain counts, since its brain is never stopped)."""

    scores: np.ndarray
    frames_survived: np.ndarray
    obs: np.ndarray | None = None              # (B, T) object array of Obs
    flaps: np.ndarray | None = None             # (B, T) bool
    output_counts: np.ndarray | None = None      # (B, T, 10) int32, the 10 output seeds
    dnp01_counts: np.ndarray | None = None        # (B, T, 2) int32, DNp01 L/R (secondary channel)


def play_batch(
    W, neurons: pd.DataFrame, X: np.ndarray,
    game_seed: int, sim_seed: int, max_frames: int,
    config: GameConfig = GameConfig(), record: bool = False,
) -> BatchResult:
    """Run B candidates (one interface-parameter vector each, X: (B, 16)) in lockstep, one
    Simulator with common random numbers (every candidate gets the same Poisson seed, so they
    differ only through their interface) and one Game per candidate (all the same game_seed).
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == N_PARAMS, f"X must be (B, {N_PARAMS}), got {X.shape}"
    B = X.shape[0]

    params = [from_vector(X[b]) for b in range(B)]
    G = np.array([p.G for p in params], dtype=np.float64)
    lam = np.array([p.lam for p in params], dtype=np.float64)
    kappa = np.array([p.kappa for p in params], dtype=np.float64)
    r0 = np.array([p.r0 for p in params], dtype=np.float64)
    w_readout = np.stack([p.w for p in params], axis=0)  # (B, 10)
    tau_ms = np.array([p.tau_ms for p in params], dtype=np.float64)
    bias = np.array([p.bias for p in params], dtype=np.float64)

    side_L, side_R = _input_seed_side_masks(neurons)

    role = neurons["role"].to_numpy()
    output_idx = np.flatnonzero(role == "output_seed")
    assert output_idx.shape[0] == 10, f"expected 10 output-seed neurons, got {output_idx.shape[0]}"

    ntype = neurons["type"].to_numpy()
    dnp01_idx = np.flatnonzero(ntype == "DNp01")
    assert dnp01_idx.shape[0] == 2, f"expected 2 DNp01 neurons, got {dnp01_idx.shape[0]}"

    seeds = np.full(B, sim_seed, dtype=np.uint64)
    sim = Simulator(W, n_candidates=B, dt=DT_DEFAULT, seeds=seeds)

    steps_per_frame_exact = config.frame_ms / sim.dt
    steps_per_frame = round(steps_per_frame_exact)
    assert abs(steps_per_frame_exact - steps_per_frame) < 1e-9, (
        f"frame_ms/dt must be a whole number of steps, got {steps_per_frame_exact}"
    )
    assert abs(config.frame_ms / sim.dt - steps_per_frame) < 1e-9, (
        f"frame_ms {config.frame_ms} is not a whole number of {sim.dt} ms steps")

    games = [Game(config, seed=game_seed) for _ in range(B)]
    obs_list = [g.reset() for g in games]

    alive = np.ones(B, dtype=bool)
    frames_survived = np.zeros(B, dtype=np.int64)
    scores = np.zeros(B, dtype=np.int64)
    trace = np.zeros((B, 10), dtype=np.float64)

    rec_obs: list[list[Obs]] = [] if record else None
    rec_flap: list[np.ndarray] = [] if record else None
    rec_counts: list[np.ndarray] = [] if record else None
    rec_dnp01: list[np.ndarray] = [] if record else None

    frame = 0
    while alive.any() and frame < max_frames:
        # obs from the START of this frame -> 25 ms sensorimotor latency (see module docstring)
        rates = obs_batch_rates(G, lam, kappa, r0, obs_list, config, side_L, side_R)
        rates[~alive] = 0.0

        # Freeze finished candidates: their results are already recorded, and at population 16
        # a single long survivor would otherwise pay for 15 dead brains every frame.
        counts = sim.run(steps_per_frame, rates, active=alive)
        out_counts = counts[:, output_idx].astype(np.float64)
        trace = update_trace(trace, out_counts, tau_ms, config.frame_ms)
        flap = flap_decision(trace, w_readout, bias)

        if record:
            rec_obs.append(list(obs_list))
            rec_flap.append(flap.copy())
            rec_counts.append(counts[:, output_idx].copy())
            rec_dnp01.append(counts[:, dnp01_idx].copy())

        for b in range(B):
            if alive[b]:
                obs_b, _points, done = games[b].step(bool(flap[b]))
                obs_list[b] = obs_b
                frames_survived[b] += 1
                scores[b] = games[b].score
                if done:
                    alive[b] = False

        frame += 1

    result = BatchResult(scores=scores, frames_survived=frames_survived)
    if record:
        T = len(rec_obs)
        obs_arr = np.empty((B, T), dtype=object)
        for t in range(T):
            for b in range(B):
                obs_arr[b, t] = rec_obs[t][b]
        result.obs = obs_arr
        result.flaps = np.array(rec_flap, dtype=bool).T  # (T,B) -> (B,T)
        result.output_counts = np.stack(rec_counts, axis=1).astype(np.int32)  # (B,T,10)
        result.dnp01_counts = np.stack(rec_dnp01, axis=1).astype(np.int32)    # (B,T,2)

    return result
