"""Step 9 -- replay recorder (docs/PLAN.md Step 9, with the 2026-09-22 amendment).

Re-runs the best real-connectome parameter vector on a held-out seed and writes a single
self-contained JSON (`docs-site/replay/best.json`) that the Step 10 web page reads. The page must
hard-code NOTHING about the game: this file carries every `GameConfig` constant, the exact ordered
pipe list (the LEVEL), the fly's full per-frame trajectory, and a downsampled neuron-activity trace.

Two things are verified before anything is written:
  1. `interface.play_batch(..., record=True)` is used to reproduce the episode with the run's
     `best_x.npy` and `train.sim_seed_for`'s sim-seed rule; its score is asserted equal to the
     score `heldout.json` reports for that seed -- fail loudly if it differs.
  2. A second, hand-rolled pass over the same `Simulator`/interface primitives captures full
     per-neuron spike counts every frame (`play_batch` only returns the 10 output-seed + 2 DNp01
     channels), and its flap sequence and final score are asserted identical to pass 1's -- both
     runs are fully deterministic given the same seeds, so any mismatch means the two passes have
     drifted apart and the recording is not trustworthy.

LEVEL convention (2026-09-22 amendment): pipe positions are stored as (x0, gap_centre), where x0 is
the pipe's left edge extrapolated back to "frame 0" -- the state at `Game.reset()`, before any
`step()` has run. Recorded per-frame arrays (`fly.bird_y` etc.) are POST-step state: array index i
(0-based) is the state after (i+1) physics steps. So a pipe's on-screen position at array index i is
`x0 - (i + 1) * config.pipe_speed`. This lets the browser place every pipe with no RNG -- see
`meta.schema` in the written JSON, and `tests/test_record.py`'s independent re-simulation, which is
the contract this convention exists to satisfy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flybrain import interface, lif, train
from flybrain.game import Game, GameConfig
from flybrain.lif import DT_DEFAULT, Simulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = 1
CONNECTOME_NAME = "MaleCNS v1.0"
MAX_ACTIVITY_NEURONS = 200
REAL_N_NEURONS = 2493        # the real sub-circuit's size; guards provenance claims in meta
N_INPUT_SEEDS_KEPT = 40      # a sample of looming (LC4/LPLC2) neurons, so the page can show the input
N_EXTRA_INTERNEURONS = 140
MAX_FILE_BYTES = 2 * 1024 * 1024
ROUND_DECIMALS = 3

DESCRIPTION = (
    "One recorded episode of a Drosophila male-CNS visual sub-circuit (frozen connectome, "
    "fitted interface only) playing Flappy Bird, alongside enough level and physics data for "
    "a browser to replay the identical level and let a human play it too. `fly` gives the "
    "bird's exact per-frame trajectory and flap decisions; `activity` gives per-frame spike "
    "counts for a subset of neurons (all 10 output-seed descending neurons plus the "
    "highest-traffic interneurons this episode); `gf` isolates the two DNp01 giant-fiber "
    "neurons for the habituation result. See docs/RESULTS.md for what is frozen vs fitted."
)

SCHEMA = {
    "meta.schema_version": "int, this file's schema version.",
    "meta.connectome": (
        "str, connectome name/version, or 'SYNTHETIC PLACEHOLDER (...)' when this replay was "
        "NOT recorded from the real sub-circuit -- check meta.is_real_connectome before making "
        "any provenance claim in a UI."
    ),
    "meta.is_real_connectome": (
        "bool, true only when this episode ran on the real MaleCNS sub-circuit (network 'real' "
        "with the expected neuron count). A page must not claim a real fly brain when false."
    ),
    "meta.n_neurons": "int, neuron count in the recorded network (read from the network file).",
    "meta.n_edges": "int, nonzero-edge count in the recorded network (read from the network file).",
    "meta.network": "str, the run's network id ('real' or 'control_NN').",
    "meta.run_seed": "int, the CMA-ES run seed that produced best_x.",
    "meta.git_commit": "str, git commit of the training run (from its config.json).",
    "meta.heldout_mean": "float, the run's held-out mean score over its 20 held-out seeds.",
    "meta.game_seed": "int, the held-out seed used for this specific episode.",
    "meta.score": "int, the fly's final score this episode.",
    "meta.frames": "int, number of frames recorded (episode length; capped at config.max_frames).",
    "meta.description": "str, one-paragraph plain-language summary of this file.",
    "meta.schema": "object, this field-by-field description.",
    "config": (
        "object, every GameConfig field (docs/PLAN.md Step 4), all numbers. config.frame_ms is "
        "simulated BRAIN time per frame (25 ms), not a display frame rate."
    ),
    "level": (
        "array of {x0, gap_centre}, ordered by x0 ascending. x0 is the pipe's left edge at "
        "frame 0 (Game.reset(), before any step()). Pipe position at recorded array index i "
        "(0-based) is x0 - (i + 1) * config.pipe_speed. Enough pipes are included to cover "
        "config.max_frames."
    ),
    "fly.bird_y": "float[frames], rounded to 3 decimals. Post-step bird y each recorded frame.",
    "fly.bird_vy": "float[frames], rounded to 3 decimals. Post-step bird vertical velocity.",
    "fly.flap": (
        "int[frames], 0/1: whether the brain flapped on that frame. Physics for a flap: it SETS "
        "bird_vy to config.flap_velocity (it does not add to it). Per frame, in order: apply "
        "flap; vy = min(vy + config.gravity, config.max_fall_speed); y += vy; pipes move left by "
        "config.pipe_speed; a pipe scores once, the first frame its right edge (x + pipe_width) "
        "is left of config.bird_x - config.bird_radius; death = ceiling (y - r <= 0), floor "
        "(y + r >= config.height), or circle-vs-rectangle overlap with either pipe rectangle."
    ),
    "fly.score": "int[frames]. Cumulative score after that frame.",
    "activity.ids": (
        "int[n_neurons_kept], bodyId of each kept neuron, ordered: the 10 output-seed neurons "
        "(network row order), then a sample of input-seed (looming) neurons, then interneurons "
        "ranked by total spikes this episode. Group with activity.roles, never by position."
    ),
    "activity.roles": (
        "str[n_neurons_kept], one of 'output_seed' | 'input_seed' | 'interneuron'."
    ),
    "activity.types": (
        "str[n_neurons_kept]. For the 10 output seeds: '<DN type>_<L|R>' (e.g. 'DNp01_R'). "
        "For all others: the neuron's `type` column value (e.g. 'LC4', 'LPLC2')."
    ),
    "activity.counts": "int[frames][n_neurons_kept]. Spike count of each kept neuron, per frame.",
    "gf.ids": "int[2], bodyId of the two DNp01 neurons (giant-fiber channel), R then L.",
    "gf.sides": "str[2], ['R', 'L'] -- soma side matching gf.ids.",
    "gf.counts": "int[frames][2]. DNp01 spike counts per frame, R then L.",
}


# --- picking a run ---------------------------------------------------------------------------


def find_best_auto_run(runs_root: Path) -> Path:
    """The finished ('real', *) run (has heldout.json) with the highest held-out mean.
    Runs still in progress (no heldout.json yet) are ignored, not treated as an error."""
    heldout_paths = sorted((runs_root / "real").glob("seed_*/heldout.json"))
    if not heldout_paths:
        raise SystemExit(
            f"--run auto: no finished runs under {runs_root / 'real'} "
            f"(no seed_*/heldout.json found -- the sweep may still be running)"
        )
    best_dir, best_mean = None, -np.inf
    for p in heldout_paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["mean"] > best_mean:
            best_mean, best_dir = d["mean"], p.parent
    return best_dir


def best_seed_of_run(heldout: dict) -> int:
    """The held-out seed with the highest recorded score in this run's heldout.json.
    Ties break to the lower seed number, for a deterministic choice."""
    best = max(heldout["seeds"], key=lambda d: (d["score"], -d["seed"]))
    return int(best["seed"])


def load_network(network_path: Path) -> tuple[sparse.spmatrix, pd.DataFrame]:
    """(W, neurons) for an arbitrary network file. neurons.parquet is shared across the real
    network and every control (train.py's `_load_for_network` docstring), so it is always read
    from `lif.DERIVED_DIR` -- read dynamically (not imported by name) so tests can monkeypatch
    `flybrain.lif.DERIVED_DIR` to point at a fake network + neurons table."""
    W = sparse.load_npz(network_path)
    neurons = pd.read_parquet(lif.DERIVED_DIR / "neurons.parquet")
    return W, neurons


# --- level (pipe list independent of the fly's actions) --------------------------------------


def build_level(config: GameConfig, game_seed: int, max_frames: int) -> list[dict]:
    """The ordered pipe list, extrapolated back to frame 0 (see module docstring). Pipe motion
    and spawning never depend on the bird's actions, so this is simulated with a throwaway,
    never-flapping Game and does not need to agree with the fly's actual (possibly shorter)
    episode -- it must cover the full `max_frames`, per docs/PLAN.md Step 9's amendment."""
    game = Game(config, seed=game_seed)
    game.reset()

    level = [
        {"x0": float(x), "gap_centre": float((gap_top + gap_bottom) / 2.0)}
        for x, gap_top, gap_bottom in game.pipes
    ]
    prev_total = len(game.pipes) + game._dropped

    for frame_i in range(1, max_frames + 1):
        game.step(False)
        total = len(game.pipes) + game._dropped
        n_new = total - prev_total
        if n_new > 0:
            for x, gap_top, gap_bottom in game.pipes[-n_new:]:
                x0 = float(x + frame_i * config.pipe_speed)
                gap_centre = float((gap_top + gap_bottom) / 2.0)
                level.append({"x0": x0, "gap_centre": gap_centre})
        prev_total = total

    return level


# --- episode: full per-neuron activity pass ---------------------------------------------------


def _output_and_dnp01_idx(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    role = neurons["role"].to_numpy()
    output_idx = np.flatnonzero(role == "output_seed")
    assert output_idx.shape[0] == 10, f"expected 10 output-seed neurons, got {output_idx.shape[0]}"

    ntype = neurons["type"].to_numpy()
    dnp01_idx = np.flatnonzero(ntype == "DNp01")
    assert dnp01_idx.shape[0] == 2, f"expected 2 DNp01 neurons, got {dnp01_idx.shape[0]}"
    return output_idx, dnp01_idx


def run_full_episode(
    W, neurons: pd.DataFrame, best_x: np.ndarray,
    game_seed: int, sim_seed: int, max_frames: int, config: GameConfig,
) -> dict:
    """Hand-rolled single-candidate replay of interface.play_batch's loop, built from the same
    primitives (obs_batch_rates / update_trace / flap_decision / Simulator), but also keeping the
    FULL per-neuron spike-count array every frame -- play_batch's BatchResult only keeps the 10
    output-seed + 2 DNp01 channels, and interface.py is out of scope for this ticket to change.
    """
    x = np.asarray(best_x, dtype=np.float64)
    params = interface.from_vector(x)
    G = np.array([params.G])
    lam = np.array([params.lam])
    kappa = np.array([params.kappa])
    r0 = np.array([params.r0])
    w_readout = params.w[None, :]
    tau_ms = np.array([params.tau_ms])
    bias = np.array([params.bias])

    side_L, side_R = interface._input_seed_side_masks(neurons)
    output_idx, dnp01_idx = _output_and_dnp01_idx(neurons)

    seeds = np.full(1, sim_seed, dtype=np.uint64)
    sim = Simulator(W, n_candidates=1, dt=DT_DEFAULT, seeds=seeds)
    steps_per_frame = round(config.frame_ms / sim.dt)
    assert abs(config.frame_ms / sim.dt - steps_per_frame) < 1e-9

    game = Game(config, seed=game_seed)
    obs = game.reset()

    alive = np.ones(1, dtype=bool)
    trace = np.zeros((1, 10), dtype=np.float64)

    bird_y, bird_vy, flap_seq, score_seq = [], [], [], []
    full_counts, output_counts, dnp01_counts = [], [], []

    frame = 0
    while alive.any() and frame < max_frames:
        rates = interface.obs_batch_rates(G, lam, kappa, r0, [obs], config, side_L, side_R)
        rates[~alive] = 0.0

        counts = sim.run(steps_per_frame, rates, active=alive)  # (1, N)
        out_counts = counts[:, output_idx].astype(np.float64)
        trace = interface.update_trace(trace, out_counts, tau_ms, config.frame_ms)
        flap = interface.flap_decision(trace, w_readout, bias)

        post_obs, _points, done = game.step(bool(flap[0]))

        bird_y.append(post_obs.bird_y)
        bird_vy.append(post_obs.bird_vy)
        flap_seq.append(bool(flap[0]))
        score_seq.append(int(post_obs.score))
        full_counts.append(counts[0].astype(np.int64).copy())
        output_counts.append(counts[0, output_idx].astype(np.int64).copy())
        dnp01_counts.append(counts[0, dnp01_idx].astype(np.int64).copy())

        obs = post_obs
        if done:
            alive[0] = False
        frame += 1

    return {
        "bird_y": np.array(bird_y, dtype=np.float64),
        "bird_vy": np.array(bird_vy, dtype=np.float64),
        "flap": np.array(flap_seq, dtype=bool),
        "score": np.array(score_seq, dtype=np.int64),
        "full_counts": np.array(full_counts, dtype=np.int64) if full_counts else np.zeros((0, W.shape[0]), dtype=np.int64),
        "output_idx": output_idx,
        "dnp01_idx": dnp01_idx,
        "final_score": int(game.score),
        "frames": frame,
    }


# --- assembling the JSON -----------------------------------------------------------------------


def _select_activity_neurons(neurons: pd.DataFrame, full_counts: np.ndarray,
                              output_idx: np.ndarray, max_neurons: int) -> np.ndarray:
    """Neuron indices to keep for `activity`, in this order: every output seed, then a sample of
    input seeds (the looming neurons -- PLAN Step 9 asks for input seeds and Step 10's panel shows
    them), then interneurons ranked by total spikes this episode. Capped at max_neurons."""
    role = neurons["role"].to_numpy()
    totals = full_counts.sum(axis=0) if full_counts.size else np.zeros(neurons.shape[0])

    input_idx = np.flatnonzero(role == "input_seed")
    # evenly spaced through the input-seed block so both LC4 and LPLC2, and both sides, appear
    if input_idx.size > N_INPUT_SEEDS_KEPT:
        take = np.linspace(0, input_idx.size - 1, N_INPUT_SEEDS_KEPT).round().astype(int)
        input_idx = input_idx[np.unique(take)]

    interneuron_idx = np.flatnonzero(role == "interneuron")
    order = interneuron_idx[np.argsort(-totals[interneuron_idx], kind="stable")]

    room = max(0, max_neurons - output_idx.shape[0] - input_idx.size)
    kept_extra = order[:min(N_EXTRA_INTERNEURONS, room)]
    return np.concatenate([output_idx, input_idx, kept_extra])


def _neuron_type_label(neurons: pd.DataFrame, idx: int, is_output: bool) -> str:
    ntype = str(neurons["type"].iat[idx])
    if is_output:
        side = str(neurons["somaSide"].iat[idx])
        return f"{ntype}_{side}"
    return ntype


def build_replay(run_dir: Path, seed: int | None = None, max_frames: int | None = None) -> dict:
    """Build the full replay dict for `run_dir` (a training run directory containing
    config.json / best_x.npy / heldout.json). `max_frames` overrides `GameConfig().max_frames`
    for fast tests only -- the CLI never passes it, so production recordings always use the
    full 1,500-frame game (docs/PLAN.md Step 9)."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    best_x_path = run_dir / "best_x.npy"
    heldout_path = run_dir / "heldout.json"
    for p in (config_path, best_x_path, heldout_path):
        if not p.exists():
            raise SystemExit(f"missing {p} -- is {run_dir} a finished training run?")

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    best_x = np.load(best_x_path)

    game_seed = seed if seed is not None else best_seed_of_run(heldout)
    detail = next((d for d in heldout["seeds"] if d["seed"] == game_seed), None)
    if detail is None:
        raise SystemExit(
            f"seed {game_seed} is not one of this run's held-out seeds "
            f"({[d['seed'] for d in heldout['seeds']]})"
        )
    expected_score = int(detail["score"])

    W, neurons = load_network(Path(run_config["network_path"]))
    game_config = GameConfig()
    frames_cap = max_frames if max_frames is not None else game_config.max_frames

    sim_seed = train.sim_seed_for(game_seed)

    # Pass 1: interface.play_batch itself, the MUST-DO reproduction + score assertion.
    pb_result = interface.play_batch(
        W, neurons, best_x[None, :],
        game_seed=game_seed, sim_seed=sim_seed, max_frames=frames_cap,
        config=game_config, record=True,
    )
    pb_score = int(pb_result.scores[0])
    assert pb_score == expected_score, (
        f"play_batch score ({pb_score}) != heldout.json score for seed {game_seed} "
        f"({expected_score}) -- recording is not a faithful reproduction of the training run"
    )

    # Pass 2: full per-neuron activity, built from the same primitives (see run_full_episode).
    episode = run_full_episode(
        W, neurons, best_x, game_seed=game_seed, sim_seed=sim_seed,
        max_frames=frames_cap, config=game_config,
    )
    assert episode["final_score"] == pb_score, (
        f"full-activity pass score ({episode['final_score']}) != play_batch score ({pb_score}) "
        f"-- the two passes have diverged, recording aborted"
    )
    pb_flaps = pb_result.flaps[0, : episode["frames"]]
    assert np.array_equal(pb_flaps, episode["flap"]), (
        "full-activity pass flap sequence != play_batch's flap sequence -- diverged, aborted"
    )

    level = build_level(game_config, game_seed, frames_cap)

    output_idx, dnp01_idx = episode["output_idx"], episode["dnp01_idx"]
    kept_idx = _select_activity_neurons(neurons, episode["full_counts"], output_idx, MAX_ACTIVITY_NEURONS)
    is_output = np.isin(kept_idx, output_idx)

    body_ids = neurons["bodyId"].to_numpy()
    roles = [str(neurons["role"].iat[int(i)]) for i in kept_idx]
    activity = {
        "ids": [int(body_ids[i]) for i in kept_idx],
        "types": [_neuron_type_label(neurons, int(i), bool(o)) for i, o in zip(kept_idx, is_output)],
        # "output_seed" | "input_seed" | "interneuron" -- the page groups the panel by this
        # instead of assuming "the first ten are the outputs".
        "roles": roles,
        "counts": episode["full_counts"][:, kept_idx].astype(np.int64).tolist(),
    }

    dnp01_sides = [str(neurons["somaSide"].iat[int(i)]) for i in dnp01_idx]
    gf = {
        "ids": [int(body_ids[int(i)]) for i in dnp01_idx],
        "sides": dnp01_sides,
        "counts": episode["full_counts"][:, dnp01_idx].astype(np.int64).tolist(),
    }

    fly = {
        "bird_y": np.round(episode["bird_y"], ROUND_DECIMALS).tolist(),
        "bird_vy": np.round(episode["bird_vy"], ROUND_DECIMALS).tolist(),
        "flap": episode["flap"].astype(np.int64).tolist(),
        "score": episode["score"].tolist(),
    }

    replay = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            # A replay built from anything but the real sub-circuit must not advertise the
            # connectome's name: a placeholder that claims "MaleCNS v1.0" is a false provenance
            # claim the moment the page ships it.
            "connectome": (CONNECTOME_NAME if str(run_config["network"]) == "real"
                           and int(W.shape[0]) == REAL_N_NEURONS
                           else f"SYNTHETIC PLACEHOLDER (not {CONNECTOME_NAME})"),
            "is_real_connectome": bool(str(run_config["network"]) == "real"
                                       and int(W.shape[0]) == REAL_N_NEURONS),
            "n_neurons": int(W.shape[0]),
            "n_edges": int(W.nnz),
            "network": str(run_config["network"]),
            "run_seed": int(run_config["run_seed"]),
            "git_commit": str(run_config.get("git_commit", "unknown")),
            "heldout_mean": float(heldout["mean"]),
            "game_seed": int(game_seed),
            "score": int(episode["final_score"]),
            "frames": int(episode["frames"]),
            "description": DESCRIPTION,
            "schema": SCHEMA,
        },
        "config": {k: v for k, v in asdict(game_config).items()},
        "level": level,
        "fly": fly,
        "activity": activity,
        "gf": gf,
    }
    return replay


def write_replay(replay: dict, out_path: Path) -> int:
    """Write `replay` as compact JSON to out_path (parents created). Returns the file size in
    bytes. Uses separators without spaces to keep the file small (no functional difference)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(replay, separators=(",", ":"))
    out_path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


# --- CLI -----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 9 -- record an episode as a self-contained replay JSON for docs-site/."
    )
    parser.add_argument(
        "--run", required=True,
        help="training run directory (e.g. runs/real/seed_3), or 'auto' to pick the finished "
             "real run with the highest held-out mean",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="held-out game seed to record (default: the run's best-scoring held-out seed)",
    )
    parser.add_argument(
        "--out", type=str, default=str(PROJECT_ROOT / "docs-site" / "replay" / "best.json"),
        help="output JSON path (default: docs-site/replay/best.json)",
    )
    parser.add_argument(
        "--runs-root", type=str, default=str(PROJECT_ROOT / "runs"),
        help="root of the runs/ tree, used only by --run auto (default: runs/)",
    )
    args = parser.parse_args(argv)

    run_dir = find_best_auto_run(Path(args.runs_root)) if args.run == "auto" else Path(args.run)

    replay = build_replay(run_dir, seed=args.seed)
    # check the cap before writing, so an over-cap run never leaves a stale artefact behind
    probe = len(json.dumps(replay, separators=(",", ":")).encode("utf-8"))
    if probe > MAX_FILE_BYTES:
        raise SystemExit(
            f"replay JSON would be {probe} bytes, over the {MAX_FILE_BYTES}-byte cap -- "
            f"reduce N_EXTRA_INTERNEURONS in record.py and re-run"
        )
    size_bytes = write_replay(replay, Path(args.out))

    n_neurons_kept = len(replay["activity"]["ids"])
    print(f"wrote {args.out}")
    print(f"run: {run_dir}  network={replay['meta']['network']}  run_seed={replay['meta']['run_seed']}")
    print(f"game_seed={replay['meta']['game_seed']}  score={replay['meta']['score']}  frames={replay['meta']['frames']}")
    print(f"file size: {size_bytes} bytes ({size_bytes / 1024 / 1024:.3f} MB, cap {MAX_FILE_BYTES / 1024 / 1024:.0f} MB)")
    print(f"activity neurons kept: {n_neurons_kept} (cap {MAX_ACTIVITY_NEURONS})")

    if size_bytes > MAX_FILE_BYTES:
        raise SystemExit(
            f"replay JSON is {size_bytes} bytes, over the {MAX_FILE_BYTES}-byte cap -- "
            f"reduce N_EXTRA_INTERNEURONS in record.py and re-run"
        )


if __name__ == "__main__":
    main()
