"""Tests for Step 9 replay recorder (src/flybrain/record.py).

Uses only small, self-built fake networks/runs (never the real 20 h sweep in progress under
runs/), so this file runs in well under 60 s and does not depend on the sweep finishing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flybrain import interface, lif, record, train
from flybrain.game import GameConfig

MAX_FRAMES_TEST = 60  # short episode -- "short max_frames" fake run (ticket instruction)
OUTPUT_TYPES = ("DNp01", "DNp02", "DNp04", "DNp06", "DNp11")


# --- fake network + run builders ---------------------------------------------------------------


def _make_fake_neurons(n_input=4, n_interneuron=18) -> pd.DataFrame:
    """4 input_seed (2 L / 2 R) + 10 output_seed (5 types x L/R, includes the 2 DNp01) +
    n_interneuron interneurons. Row order: input, output, interneuron (idx == row position)."""
    rows = []
    idx = 0
    for i in range(n_input):
        rows.append(dict(idx=idx, bodyId=1000 + idx, type="input_dummy",
                          somaSide="L" if i % 2 == 0 else "R",
                          consensus_nt="ACh", sign=1, role="input_seed"))
        idx += 1
    for t in OUTPUT_TYPES:
        for side in ("R", "L"):
            rows.append(dict(idx=idx, bodyId=2000 + idx, type=t, somaSide=side,
                              consensus_nt="ACh", sign=-1, role="output_seed"))
            idx += 1
    for i in range(n_interneuron):
        rows.append(dict(idx=idx, bodyId=3000 + idx, type=f"inter_{i}",
                          somaSide="L" if i % 2 == 0 else "R",
                          consensus_nt="GABA", sign=-1, role="interneuron"))
        idx += 1
    return pd.DataFrame(rows)


def _make_fake_network(rng: np.random.Generator, n: int) -> sparse.csr_matrix:
    density = 0.15
    W = sparse.random(n, n, density=density, format="csr", random_state=rng,
                       data_rvs=lambda size: rng.integers(1, 6, size=size).astype(np.float32))
    signs = rng.choice([-1.0, 1.0], size=W.data.shape)
    W.data = (W.data * signs).astype(np.float32)
    return W


def make_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_seed: int = 1,
                   network: str = "real", n_interneuron: int = 18,
                   heldout_seeds=(301, 302, 303)) -> Path:
    """Builds a self-contained fake network + a training-run directory that record.py can read,
    with heldout.json scores computed by actually running interface.play_batch (so
    build_replay's play_batch-vs-heldout assertion holds). Monkeypatches flybrain.lif.DERIVED_DIR
    to a tmp 'derived' dir holding neurons.parquet, per record.load_network's convention."""
    derived_dir = tmp_path / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lif, "DERIVED_DIR", derived_dir)

    neurons = _make_fake_neurons(n_interneuron=n_interneuron)
    neurons.to_parquet(derived_dir / "neurons.parquet")

    rng = np.random.default_rng(run_seed)
    W = _make_fake_network(rng, n=len(neurons))
    net_path = derived_dir / f"{network}.npz"
    sparse.save_npz(net_path, W)

    best_x = rng.normal(size=interface.N_PARAMS)

    seed_details = []
    for s in heldout_seeds:
        result = interface.play_batch(
            W, neurons, best_x[None, :],
            game_seed=s, sim_seed=train.sim_seed_for(s), max_frames=MAX_FRAMES_TEST,
        )
        seed_details.append({
            "seed": int(s),
            "score": int(result.scores[0]),
            "frames_survived": int(result.frames_survived[0]),
        })

    run_dir = tmp_path / "runs" / network / f"seed_{run_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "network": network,
        "run_seed": run_seed,
        "git_commit": "deadbeef",
        "network_path": str(net_path),
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    np.save(run_dir / "best_x.npy", best_x)

    heldout = {
        "network": network,
        "run_seed": run_seed,
        "seeds": seed_details,
        "mean": float(np.mean([d["score"] for d in seed_details])),
        "median": float(np.median([d["score"] for d in seed_details])),
    }
    (run_dir / "heldout.json").write_text(json.dumps(heldout), encoding="utf-8")

    return run_dir


@pytest.fixture()
def fake_run(tmp_path, monkeypatch) -> Path:
    return make_fake_run(tmp_path, monkeypatch, run_seed=7)


@pytest.fixture()
def replay(fake_run) -> dict:
    return record.build_replay(fake_run, max_frames=MAX_FRAMES_TEST)


# --- test 1: schema ------------------------------------------------------------------------------


def test_schema_keys_types_and_lengths(replay):
    meta = replay["meta"]
    for key in ("schema_version", "connectome", "n_neurons", "n_edges", "network", "run_seed",
                "git_commit", "heldout_mean", "game_seed", "score", "frames", "description",
                "schema"):
        assert key in meta, f"meta.{key} missing"

    assert isinstance(meta["schema_version"], int)
    assert isinstance(meta["connectome"], str)
    assert isinstance(meta["n_neurons"], int) and meta["n_neurons"] > 0
    assert isinstance(meta["n_edges"], int) and meta["n_edges"] > 0
    assert isinstance(meta["network"], str)
    assert isinstance(meta["run_seed"], int)
    assert isinstance(meta["game_seed"], int)
    assert isinstance(meta["score"], int)
    assert isinstance(meta["frames"], int) and meta["frames"] > 0
    assert isinstance(meta["description"], str) and len(meta["description"]) > 0
    assert isinstance(meta["schema"], dict) and len(meta["schema"]) > 0

    frames = meta["frames"]
    assert frames == len(replay["fly"]["bird_y"])

    assert set(asdict(GameConfig()).keys()) <= set(replay["config"].keys())
    for v in replay["config"].values():
        assert isinstance(v, (int, float))

    for key in ("x0", "gap_centre"):
        assert all(key in p for p in replay["level"])
    assert len(replay["level"]) > 0

    fly = replay["fly"]
    for key in ("bird_y", "bird_vy", "flap", "score"):
        assert key in fly
        assert len(fly[key]) == frames
    assert all(v in (0, 1) for v in fly["flap"])

    activity = replay["activity"]
    for key in ("ids", "types", "counts"):
        assert key in activity
    n_kept = len(activity["ids"])
    assert len(activity["types"]) == n_kept
    assert len(activity["counts"]) == frames
    assert all(len(row) == n_kept for row in activity["counts"])
    assert n_kept <= record.MAX_ACTIVITY_NEURONS

    gf = replay["gf"]
    assert len(gf["ids"]) == 2
    assert len(gf["counts"]) == frames
    assert all(len(row) == 2 for row in gf["counts"])


# --- test 2: level replay (the important one) -----------------------------------------------------


def _circle_rect_overlap(bird_x, bird_y, bird_r, rect_x, rect_y, rect_w, rect_h) -> bool:
    closest_x = max(rect_x, min(bird_x, rect_x + rect_w))
    closest_y = max(rect_y, min(bird_y, rect_y + rect_h))
    dx = bird_x - closest_x
    dy = bird_y - closest_y
    return (dx * dx + dy * dy) ** 0.5 < bird_r


def _pure_python_resim(config: dict, level: list[dict], flap_seq: list[int]):
    """Independent reimplementation of game.py's physics/scoring/collision -- deliberately not
    importing flybrain.game, per the ticket's requirement that this is the contract the JS page
    relies on. Returns (bird_y[], bird_vy[], score[], collided_at_frame_or_None)."""
    bird_x = config["bird_x"]
    bird_r = config["bird_radius"]
    height = config["height"]
    gravity = config["gravity"]
    flap_v = config["flap_velocity"]
    max_fall = config["max_fall_speed"]
    pipe_w = config["pipe_width"]
    gap_h = config["gap_height"]
    speed = config["pipe_speed"]
    threshold = bird_x - bird_r

    y = config["start_y"]
    vy = 0.0
    bird_ys, bird_vys, scores = [], [], []
    collided_at = None

    for i, flap in enumerate(flap_seq):
        if flap:
            vy = flap_v
        vy = min(vy + gravity, max_fall)
        y += vy

        frame_n = i + 1
        score = 0
        cur_pipes = []
        for p in level:
            x = p["x0"] - frame_n * speed
            if x + pipe_w < threshold:
                score += 1
            cur_pipes.append((x, p["gap_centre"]))

        collided = y - bird_r <= 0 or y + bird_r >= height
        if not collided:
            for x, gc in cur_pipes:
                gap_top = gc - gap_h / 2.0
                gap_bot = gc + gap_h / 2.0
                if _circle_rect_overlap(bird_x, y, bird_r, x, 0, pipe_w, gap_top):
                    collided = True
                    break
                if _circle_rect_overlap(bird_x, y, bird_r, x, gap_bot, pipe_w, height - gap_bot):
                    collided = True
                    break

        bird_ys.append(y)
        bird_vys.append(vy)
        scores.append(score)
        if collided:
            collided_at = i
            break

    return bird_ys, bird_vys, scores, collided_at


def test_level_replay_matches_recorded_trajectory(replay):
    config = replay["config"]
    level = replay["level"]
    flap_seq = replay["fly"]["flap"]

    bird_ys, bird_vys, scores, collided_at = _pure_python_resim(config, level, flap_seq)

    n = len(bird_ys)
    assert n == len(replay["fly"]["bird_y"]), "reimplementation ran a different number of frames"

    recorded_y = replay["fly"]["bird_y"]
    recorded_vy = replay["fly"]["bird_vy"]
    recorded_score = replay["fly"]["score"]

    got_y = [round(v, record.ROUND_DECIMALS) for v in bird_ys]
    got_vy = [round(v, record.ROUND_DECIMALS) for v in bird_vys]

    assert got_y == recorded_y, "bird_y trajectory diverged from the recorded episode"
    assert got_vy == recorded_vy, "bird_vy trajectory diverged from the recorded episode"
    assert scores == recorded_score, "score sequence diverged from the recorded episode"

    # collision frame (if the episode ended in a collision, not the frame cap) should line up
    # with the last recorded frame.
    if replay["meta"]["frames"] < config["max_frames"]:
        assert collided_at == n - 1


# --- test 3: file size + neuron count ---------------------------------------------------------


def test_file_size_and_neuron_count(fake_run, tmp_path):
    replay = record.build_replay(fake_run, max_frames=MAX_FRAMES_TEST)
    out_path = tmp_path / "best.json"
    size = record.write_replay(replay, out_path)

    assert out_path.exists()
    assert size == out_path.stat().st_size
    assert size < record.MAX_FILE_BYTES
    assert len(replay["activity"]["ids"]) <= 200


# --- test 4: --run auto picks the highest held-out mean -----------------------------------------


def test_auto_picks_highest_heldout_mean(tmp_path):
    runs_root = tmp_path / "runs"
    means = {1: 3.0, 2: 9.5, 3: 7.0}
    for seed, mean in means.items():
        run_dir = runs_root / "real" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "heldout.json").write_text(
            json.dumps({"mean": mean, "seeds": [{"seed": 301, "score": int(mean), "frames_survived": 10}]}),
            encoding="utf-8",
        )
    # an in-progress run (no heldout.json yet) must be ignored, not picked or errored on
    (runs_root / "real" / "seed_4").mkdir(parents=True)

    best_dir = record.find_best_auto_run(runs_root)
    assert best_dir == runs_root / "real" / "seed_2"


def test_auto_raises_when_nothing_finished(tmp_path):
    runs_root = tmp_path / "runs"
    (runs_root / "real" / "seed_1").mkdir(parents=True)
    with pytest.raises(SystemExit):
        record.find_best_auto_run(runs_root)


# --- test 5: rounding does not change score or flap sequence -------------------------------------


def test_rounding_preserves_score_and_flaps(replay):
    raw_bird_y = replay["fly"]["bird_y"]
    rounded_again = [round(v, record.ROUND_DECIMALS) for v in raw_bird_y]
    assert rounded_again == raw_bird_y, "bird_y is not stably rounded to ROUND_DECIMALS"

    # flap/score are integers already -- rounding the float channels must not have touched them.
    assert all(isinstance(v, int) for v in replay["fly"]["flap"])
    assert all(isinstance(v, int) for v in replay["fly"]["score"])
    assert replay["meta"]["score"] == replay["fly"]["score"][-1]


# --- The level contract, exercised over a FULL surviving episode -------------------------------


def test_level_survives_a_long_episode_with_spawns_and_scoring():
    """`build_level` must hold up over a full-length game, not just the 30-odd frames an
    untrained fly survives: pipes that spawn mid-episode, pipes scrolling off the left, and
    every scoring event. This is the contract the browser relies on to let a human play the
    fly's exact level, so it is checked against a scripted policy that actually survives.
    """
    from flybrain.game import Game, GameConfig as RealConfig  # generation only

    cfg = RealConfig()
    game = Game(cfg, seed=7)
    obs = game.reset()
    flaps: list[int] = []
    while obs.alive and obs.frame < cfg.max_frames:
        flap = obs.bird_y > obs.next_gap_y + 10  # naive but survives many pipes
        flaps.append(int(flap))
        obs, _points, done = game.step(bool(flap))
        if done:
            break

    assert game.score >= 5, f"scripted policy only scored {game.score}; test would be vacuous"
    assert len(flaps) > 300, f"episode too short ({len(flaps)} frames) to cover pipe spawns"

    level = record.build_level(cfg, game_seed=7, max_frames=cfg.max_frames)
    config = {k: v for k, v in asdict(cfg).items()}

    bird_ys, _vys, scores, _collided = _pure_python_resim(config, level, flaps)

    assert scores[-1] == game.score
    assert len(bird_ys) == len(flaps)
    assert bird_ys[-1] == pytest.approx(game.frame_log[-1].bird_y, abs=1e-9)
    for i, obs_i in enumerate(game.frame_log):
        assert bird_ys[i] == pytest.approx(obs_i.bird_y, abs=1e-9), f"frame {i}"
        assert scores[i] == obs_i.score, f"score diverged at frame {i}"
