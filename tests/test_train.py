"""Tests for Step 6 training (src/flybrain/train.py).

Runs entirely against a tiny synthetic sub-circuit (12 neurons: 2 input-seed, 10 output-seed --
the minimum shape interface.play_batch requires), never the real 2,493-neuron connectome, and
uses tiny budgets (budget_gens=2, small popsize, few frames) throughout. No full training is run.
Whole file runs in well under 3 minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flybrain import train

# --- Tiny synthetic network shared by every test in this file --------------------------------

OUTPUT_TYPES = ("DNp01", "DNp02", "DNp04", "DNp06", "DNp11")


def _build_tiny_network(derived_dir: Path) -> None:
    """Writes a 14-neuron subcircuit.npz + neurons.parquet under derived_dir (matching the
    Step 2 schema), plus a copy at derived_dir/controls/control_00.npz."""
    derived_dir.mkdir(parents=True, exist_ok=True)
    (derived_dir / "controls").mkdir(parents=True, exist_ok=True)

    rows = []
    body_id = 1
    idx = 0
    for side in ("L", "R"):
        rows.append({
            "idx": idx, "bodyId": body_id, "type": "LC4", "somaSide": side,
            "consensus_nt": "acetylcholine", "sign": 1, "role": "input_seed",
        })
        idx += 1
        body_id += 1

    for t in OUTPUT_TYPES:
        for side in ("L", "R"):
            rows.append({
                "idx": idx, "bodyId": body_id, "type": t, "somaSide": side,
                "consensus_nt": "acetylcholine", "sign": 1, "role": "output_seed",
            })
            idx += 1
            body_id += 1

    neurons = pd.DataFrame(rows)
    assert len(neurons) == 12  # 2 input_seed (L/R) + 10 output_seed (5 types x L/R)
    neurons.to_parquet(derived_dir / "neurons.parquet")

    n = len(neurons)
    input_idx = neurons.index[neurons["role"] == "input_seed"].to_numpy()
    output_idx = neurons.index[neurons["role"] == "output_seed"].to_numpy()

    rng = np.random.default_rng(0)
    row_i, col_i, dat = [], [], []
    for i in input_idx:
        for o in output_idx:
            row_i.append(i)
            col_i.append(o)
            dat.append(float(rng.integers(5, 15)))
    W = sparse.csr_matrix((dat, (row_i, col_i)), shape=(n, n), dtype=np.float32)

    sparse.save_npz(derived_dir / "subcircuit.npz", W)
    sparse.save_npz(derived_dir / "controls" / "control_00.npz", W)


@pytest.fixture(scope="module")
def net_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("derived")
    _build_tiny_network(root)
    return root


@pytest.fixture()
def patched(monkeypatch, net_root):
    """Point train.py's network locations at the tiny synthetic network."""
    monkeypatch.setattr(train, "DERIVED_DIR", net_root)
    monkeypatch.setattr(train, "REAL_NETWORK_PATH", net_root / "subcircuit.npz")
    monkeypatch.setattr(train, "CONTROLS_DIR", net_root / "controls")
    return net_root


TINY_KWARGS = dict(
    popsize=3,
    max_frames_train=50,   # long enough that a do-nothing candidate actually dies (game.py)
    max_frames_full=50,
    probe_every=5,
)


# --- 1. Full artefact set from a 2-generation run ---------------------------------------------


def test_run_writes_all_artefacts(patched, tmp_path):
    run_dir = train.run_training(
        "real", run_seed=1, out_dir=tmp_path, budget_gens=2, budget_minutes=60, **TINY_KWARGS,
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "best_x.npy").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "heldout.json").exists()
    assert (run_dir / "stop_reason").exists()

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["network"] == "real"
    assert config["run_seed"] == 1
    assert config["git_commit"]
    assert len(config["git_commit"]) == 40  # a real sha, not the "unknown (...)" fallback
    assert "versions" in config and "numpy" in config["versions"]
    assert config["budget_gens"] == 2
    assert config["train_seeds"] == list(train.TRAIN_SEEDS)
    assert config["probe_seeds"] == list(train.PROBE_SEEDS)
    assert config["heldout_seeds"] == list(train.HELDOUT_SEEDS)
    expected_sha = train.sha256_file(train.REAL_NETWORK_PATH)
    assert config["network_sha256"] == expected_sha

    best_x = np.load(run_dir / "best_x.npy")
    assert best_x.shape == (train.N_DIM,)

    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8")
    assert list(history["generation"]) == [1, 2]
    for col in ("generation", "best_fitness", "mean_fitness", "elapsed_s", "probe_mean"):
        assert col in history.columns
    assert not pd.isna(history.loc[history["generation"] == 1, "probe_mean"]).any()

    heldout = json.loads((run_dir / "heldout.json").read_text(encoding="utf-8"))
    assert len(heldout["seeds"]) == 20
    assert {d["seed"] for d in heldout["seeds"]} == set(train.HELDOUT_SEEDS)
    assert "mean" in heldout and "median" in heldout and "params" in heldout

    stop_reason = (run_dir / "stop_reason").read_text(encoding="utf-8").strip()
    assert stop_reason in ("perfect", "budget_gens", "budget_minutes")
    assert stop_reason == config["stop_reason"]


# --- 2. Resumable skip ---------------------------------------------------------------------


def test_rerun_skips_and_does_not_modify(patched, tmp_path, capsys):
    run_dir = train.run_training(
        "real", run_seed=2, out_dir=tmp_path, budget_gens=2, budget_minutes=60, **TINY_KWARGS,
    )
    capsys.readouterr()

    files = ["config.json", "best_x.npy", "history.csv", "heldout.json", "stop_reason"]
    before_mtimes = {f: (run_dir / f).stat().st_mtime_ns for f in files}
    before_hashes = {f: train.sha256_file(run_dir / f) for f in files}

    result_dir = train.run_training(
        "real", run_seed=2, out_dir=tmp_path, budget_gens=2, budget_minutes=60, **TINY_KWARGS,
    )
    out = capsys.readouterr().out
    assert "skip" in out
    assert result_dir == run_dir

    for f in files:
        assert (run_dir / f).stat().st_mtime_ns == before_mtimes[f]
        assert train.sha256_file(run_dir / f) == before_hashes[f]


# --- 3. Determinism by run_seed ---------------------------------------------------------------


def test_same_seed_reproduces_history(patched, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    train.run_training("real", run_seed=5, out_dir=dir_a, budget_gens=2, budget_minutes=60, **TINY_KWARGS)
    train.run_training("real", run_seed=5, out_dir=dir_b, budget_gens=2, budget_minutes=60, **TINY_KWARGS)

    hist_a = pd.read_csv(dir_a / "real" / "seed_5" / "history.csv", encoding="utf-8")
    hist_b = pd.read_csv(dir_b / "real" / "seed_5" / "history.csv", encoding="utf-8")
    pd.testing.assert_frame_equal(hist_a[["generation", "best_fitness", "mean_fitness"]],
                                   hist_b[["generation", "best_fitness", "mean_fitness"]])


def test_different_seed_differs(patched, tmp_path):
    dir_a = tmp_path / "a"
    dir_c = tmp_path / "c"
    train.run_training("real", run_seed=6, out_dir=dir_a, budget_gens=2, budget_minutes=60, **TINY_KWARGS)
    train.run_training("real", run_seed=7, out_dir=dir_c, budget_gens=2, budget_minutes=60, **TINY_KWARGS)

    hist_a = pd.read_csv(dir_a / "real" / "seed_6" / "history.csv", encoding="utf-8")
    hist_c = pd.read_csv(dir_c / "real" / "seed_7" / "history.csv", encoding="utf-8")
    # mean_fitness averages over the whole (larger) population, so a coincidental tie between
    # two differently-seeded runs is far less likely than for best_fitness alone.
    assert not hist_a["mean_fitness"].equals(hist_c["mean_fitness"])


# --- 4. Seed sets disjoint -----------------------------------------------------------------


def test_seed_sets_disjoint():
    train_set = set(train.TRAIN_SEEDS)
    probe_set = set(train.PROBE_SEEDS)
    heldout_set = set(train.HELDOUT_SEEDS)
    assert train_set & probe_set == set()
    assert train_set & heldout_set == set()
    assert probe_set & heldout_set == set()
    assert len(train.HELDOUT_SEEDS) == 20
    assert len(train.PROBE_SEEDS) == 5
    assert len(train.TRAIN_SEEDS) == 3


# --- 5. generations_to_competence -----------------------------------------------------------


def test_generations_to_competence_reached():
    history = [
        {"generation": 1, "probe_mean": 2.0},
        {"generation": 2, "probe_mean": None},
        {"generation": 3, "probe_mean": None},
        {"generation": 4, "probe_mean": None},
        {"generation": 5, "probe_mean": 9.0},
        {"generation": 6, "probe_mean": None},
        {"generation": 10, "probe_mean": 15.0},
    ]
    assert train.generations_to_competence(history) == 10


def test_generations_to_competence_reached_then_lost_still_counts_first_hit():
    history = [
        {"generation": 1, "probe_mean": 3.0},
        {"generation": 5, "probe_mean": 12.0},   # reached
        {"generation": 10, "probe_mean": 4.0},   # lost again
    ]
    assert train.generations_to_competence(history) == 5


def test_generations_to_competence_never_reached():
    history = [
        {"generation": 1, "probe_mean": 1.0},
        {"generation": 5, "probe_mean": 5.0},
        {"generation": 10, "probe_mean": 9.9},
    ]
    assert train.generations_to_competence(history) is None


def test_generations_to_competence_accepts_dataframe():
    df = pd.DataFrame([
        {"generation": 1, "probe_mean": 2.0},
        {"generation": 5, "probe_mean": 22.0},
    ])
    assert train.generations_to_competence(df) == 5


# --- 6. --all pairing -----------------------------------------------------------------------


def test_all_pairs_60_unique_each_control_once():
    pairs = train.all_pairs()
    assert len(pairs) == 60
    assert len(set(pairs)) == 60

    real_pairs = [p for p in pairs if p[0] == "real"]
    control_pairs = [p for p in pairs if p[0] != "real"]
    assert len(real_pairs) == 30
    assert len(control_pairs) == 30

    controls_used = [net for net, _ in control_pairs]
    expected_controls = {f"control_{nn:02d}" for nn in range(30)}
    assert set(controls_used) == expected_controls
    assert len(controls_used) == len(set(controls_used))  # each control exactly once

    real_seeds = sorted(seed for _, seed in real_pairs)
    assert real_seeds == list(range(1, 31))


# --- 7. Network file never modified ----------------------------------------------------------


def test_network_file_not_modified(patched, tmp_path):
    net_path = train.REAL_NETWORK_PATH
    sha_before = train.sha256_file(net_path)

    train.run_training(
        "real", run_seed=9, out_dir=tmp_path, budget_gens=2, budget_minutes=60, **TINY_KWARGS,
    )

    sha_after = train.sha256_file(net_path)
    assert sha_before == sha_after


# --- 8. Both arms are treated identically (the C2 validity condition) --------------------------


def test_real_and_control_arms_are_symmetric(patched, tmp_path):
    """A control run differs from a real run only in which network file is loaded.

    Without this, an asymmetry between the arms could bias the real-vs-control comparison
    that the whole experiment rests on (PLAN claim C2).
    """
    import json

    real_dir = train.run_training(
        "real", run_seed=4, out_dir=tmp_path, budget_gens=2, budget_minutes=60, **TINY_KWARGS,
    )
    ctrl_dir = train.run_training(
        "control_00", run_seed=4, out_dir=tmp_path, budget_gens=2, budget_minutes=60,
        **TINY_KWARGS,
    )

    real_cfg = json.loads((real_dir / "config.json").read_text(encoding="utf-8"))
    ctrl_cfg = json.loads((ctrl_dir / "config.json").read_text(encoding="utf-8"))

    # Everything except the network identity and timings must match exactly.
    varying = {"network", "network_path", "network_sha256", "elapsed_s", "started_at",
               "finished_at", "n_generations_run", "stop_reason"}
    for key in set(real_cfg) | set(ctrl_cfg):
        if key in varying:
            continue
        assert real_cfg[key] == ctrl_cfg[key], f"arms differ in config key {key!r}"

    # In this fixture the "control" file is a byte-identical copy of the real one, so the two
    # arms must also produce identical RESULTS -- the strongest possible symmetry check: any
    # branch on the network name would show up here.
    assert real_cfg["network_sha256"] == ctrl_cfg["network_sha256"]
    def _history_without_timings(path):
        rows = path.read_text(encoding="utf-8").strip().splitlines()
        header = rows[0].split(",")
        keep = [i for i, name in enumerate(header) if name != "elapsed_s"]
        return [[r.split(",")[i] for i in keep] for r in rows]

    assert _history_without_timings(real_dir / "history.csv") ==         _history_without_timings(ctrl_dir / "history.csv")
    import numpy as np
    assert np.array_equal(np.load(real_dir / "best_x.npy"), np.load(ctrl_dir / "best_x.npy"))

    for d in (real_dir, ctrl_dir):
        assert (d / "heldout.json").exists()
        heldout = json.loads((d / "heldout.json").read_text(encoding="utf-8"))
        assert [s["seed"] for s in heldout["seeds"]] == list(train.HELDOUT_SEEDS)
