"""Step 6 -- training (docs/PLAN.md Step 6, with the "Amended 2026-09-22" budgets/pop below).

CMA-ES over the 16-parameter interface vector (interface.py). Fitness is the mean of
interface.fitness(...) over 3 fixed training seeds, training episodes capped at 600 frames.
Every 5 generations (and at generation 1) the current best candidate is re-evaluated on 5 fixed
probe seeds at the full 1,500-frame game -- this feeds the pre-registered secondary measure
("generations to competence", PLAN section 1). Final evaluation is on 20 held-out seeds, never
used during training or probing, at 1,500 frames -- the only number PLAN section 6 says is
reported anywhere.

One run = one process, one `out_dir/<network>/seed_<run_seed>/`. `--all` fans 60 runs (30 real +
30 control, each control used exactly once, paired with its matching run seed) across worker
processes with ProcessPoolExecutor, one run per process, resumable via the heldout.json marker.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cma
import numpy as np

from flybrain import interface
from flybrain.lif import DERIVED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Fixed seed sets (PLAN Step 6 + section 1). Disjoint by construction -- see test 4. ---
TRAIN_SEEDS = (11, 22, 33)
PROBE_SEEDS = (201, 202, 203, 204, 205)
HELDOUT_SEEDS = tuple(range(301, 321))  # 301..320, 20 seeds

TRAIN_MAX_FRAMES = 600
FULL_MAX_FRAMES = 1500
MAX_SCORE = 22.0  # 1500 frames / fixed pipe spacing -- the lead pilot's observed ceiling
COMPETENCE_THRESHOLD = 10.0

POPSIZE = 16
SIGMA0 = 1.0
N_DIM = interface.N_PARAMS  # 16

PROBE_EVERY = 5
DEFAULT_BUDGET_GENS = 150
DEFAULT_BUDGET_MINUTES = 90  # PLAN Step 6 (amended 2026-09-22) is authoritative; was 60 here

REAL_NETWORK_PATH = DERIVED_DIR / "subcircuit.npz"
CONTROLS_DIR = DERIVED_DIR / "controls"


def network_path(network: str) -> Path:
    if network == "real":
        return REAL_NETWORK_PATH
    if network.startswith("control_"):
        return CONTROLS_DIR / f"{network}.npz"
    raise ValueError(f"unknown network {network!r}; expected 'real' or 'control_NN'")


def _load_for_network(network: str):
    """(W, neurons) for `network`. All controls (Step 7) and the real sub-circuit share the
    same neurons.parquet -- roles and row/col order are identical, only the wiring in W
    differs (PLAN Step 7) -- so this is just sparse.load_npz + the shared neurons table."""
    from scipy import sparse
    import pandas as pd
    W = sparse.load_npz(network_path(network))
    neurons = pd.read_parquet(DERIVED_DIR / "neurons.parquet")
    return W, neurons


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sim_seed_for(game_seed: int) -> int:
    """Lead pilot's fixed transform: sim_seed = 1_000_003*game_seed + 17 (docs)."""
    return 1_000_003 * game_seed + 17


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        return f"unknown ({exc})"


def _versions() -> dict:
    import scipy
    import pandas
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "cma": cma.__version__,
    }


# --- Evaluation helpers ---------------------------------------------------------------------


def evaluate_mean_fitness(W, neurons, x: np.ndarray, seeds, max_frames: int) -> float:
    """Mean interface.fitness(...) over `seeds` for a single candidate vector x (16,)."""
    scores = []
    for s in seeds:
        result = interface.play_batch(
            W, neurons, x[None, :],
            game_seed=s, sim_seed=sim_seed_for(s), max_frames=max_frames,
        )
        fit = interface.fitness(result.scores, result.frames_survived, max_frames)
        scores.append(float(fit[0]))
    return float(np.mean(scores))


def evaluate_score_mean(W, neurons, x: np.ndarray, seeds, max_frames: int) -> tuple[float, list[dict]]:
    """Mean raw score (not fitness) over `seeds`, plus a per-seed detail list."""
    details = []
    for s in seeds:
        result = interface.play_batch(
            W, neurons, x[None, :],
            game_seed=s, sim_seed=sim_seed_for(s), max_frames=max_frames,
        )
        details.append({
            "seed": int(s),
            "score": int(result.scores[0]),
            "frames_survived": int(result.frames_survived[0]),
        })
    mean = float(np.mean([d["score"] for d in details]))
    return mean, details


def generations_to_competence(history) -> int | None:
    """First generation (from `history`, a list of row-dicts or a pandas DataFrame with
    columns 'generation' and 'probe_mean') whose probe_mean >= COMPETENCE_THRESHOLD, or None
    if never reached. Rows without a probe_mean (not a probe generation) are skipped."""
    rows = history.to_dict("records") if hasattr(history, "to_dict") else history
    for row in rows:
        pm = row.get("probe_mean")
        if pm is None:
            continue
        try:
            if float(pm) >= COMPETENCE_THRESHOLD:
                return int(row["generation"])
        except (TypeError, ValueError):
            continue
    return None


# --- Single run -------------------------------------------------------------------------------


@dataclass
class RunConfig:
    network: str
    run_seed: int
    out_dir: Path
    budget_gens: int = DEFAULT_BUDGET_GENS
    budget_minutes: float = DEFAULT_BUDGET_MINUTES
    popsize: int = POPSIZE
    sigma0: float = SIGMA0
    probe_every: int = PROBE_EVERY
    max_frames_train: int = TRAIN_MAX_FRAMES
    max_frames_full: int = FULL_MAX_FRAMES
    train_seeds: tuple = TRAIN_SEEDS
    probe_seeds: tuple = PROBE_SEEDS
    heldout_seeds: tuple = HELDOUT_SEEDS


def run_training(
    network: str,
    run_seed: int,
    out_dir: str | Path,
    budget_gens: int = DEFAULT_BUDGET_GENS,
    budget_minutes: float = DEFAULT_BUDGET_MINUTES,
    popsize: int = POPSIZE,
    sigma0: float = SIGMA0,
    probe_every: int = PROBE_EVERY,
    max_frames_train: int = TRAIN_MAX_FRAMES,
    max_frames_full: int = FULL_MAX_FRAMES,
) -> Path:
    """Run one CMA-ES training run and write its artefacts. Returns the run directory.

    Resumable-safe: if `heldout.json` already exists in the run directory, skips (prints
    "skip") and returns immediately without touching any existing file.
    """
    run_dir = Path(out_dir) / network / f"seed_{run_seed}"
    heldout_path = run_dir / "heldout.json"
    if heldout_path.exists():
        print(f"skip {network} seed_{run_seed} (heldout.json exists)")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    net_path = network_path(network)
    sha_before = sha256_file(net_path)
    W, neurons = _load_for_network(network)

    es = cma.CMAEvolutionStrategy(
        list(np.zeros(N_DIM)), sigma0,
        {"popsize": popsize, "seed": run_seed, "verbose": -9},
    )

    history_rows: list[dict] = []
    best_x = np.zeros(N_DIM)
    best_fitness_ever = -np.inf
    stop_reason = "budget_gens"
    t0 = time.monotonic()
    generation = 0
    perfect_streak = 0

    while True:
        # Stop conditions are exactly the two PLAN Step 6 budgets, plus "perfect" below --
        # CMA-ES's own internal es.stop() heuristics (tolfun/flat-fitness/...) are NOT
        # consulted, since a fitness plateau is an expected, not exceptional, outcome here.
        generation += 1
        if generation > budget_gens:
            generation -= 1
            stop_reason = "budget_gens"
            break

        elapsed_s = time.monotonic() - t0
        if elapsed_s > budget_minutes * 60.0:
            generation -= 1
            stop_reason = "budget_minutes"
            break

        X = es.ask()
        fits = np.array([
            evaluate_mean_fitness(W, neurons, np.asarray(x), TRAIN_SEEDS, max_frames_train)
            for x in X
        ])
        es.tell(list(X), list(-fits))

        gen_best_idx = int(np.argmax(fits))
        gen_best_fitness = float(fits[gen_best_idx])
        if gen_best_fitness > best_fitness_ever:
            best_fitness_ever = gen_best_fitness
            best_x = np.asarray(X[gen_best_idx], dtype=np.float64).copy()

        row = {
            "generation": generation,
            "best_fitness": gen_best_fitness,
            "mean_fitness": float(np.mean(fits)),
            "elapsed_s": time.monotonic() - t0,
            "probe_mean": None,
        }

        do_probe = (generation == 1) or (generation % probe_every == 0)
        if do_probe:
            probe_mean, _ = evaluate_score_mean(
                W, neurons, best_x, PROBE_SEEDS, max_frames_full,
            )
            row["probe_mean"] = probe_mean
            if probe_mean >= MAX_SCORE:
                perfect_streak += 1
            else:
                perfect_streak = 0
            if perfect_streak >= 2:
                history_rows.append(row)
                stop_reason = "perfect"
                break

        history_rows.append(row)

    total_elapsed_s = time.monotonic() - t0

    sha_after = sha256_file(net_path)
    assert sha_before == sha_after, (
        f"network file {net_path} changed during training "
        f"({sha_before} -> {sha_after}) -- the connectome must be frozen (PLAN L1)"
    )

    heldout_mean, heldout_details = evaluate_score_mean(
        W, neurons, best_x, HELDOUT_SEEDS, max_frames_full,
    )
    heldout_scores = [d["score"] for d in heldout_details]

    params = interface.from_vector(best_x)

    config = {
        "network": network,
        "run_seed": run_seed,
        "git_commit": _git_commit(),
        "versions": _versions(),
        "budget_gens": budget_gens,
        "budget_minutes": budget_minutes,
        "popsize": popsize,
        "sigma0": sigma0,
        "probe_every": probe_every,
        "max_frames_train": max_frames_train,
        "max_frames_full": max_frames_full,
        "train_seeds": list(TRAIN_SEEDS),
        "probe_seeds": list(PROBE_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "network_path": str(net_path),
        "network_sha256": sha_after,
        "n_generations_run": generation,
        "elapsed_s": total_elapsed_s,
        "stop_reason": stop_reason,
    }

    # --- write atomically ---
    _atomic_write_text(run_dir / "config.json", json.dumps(config, indent=2))

    # np.save appends .npy unless the name already ends with it, so use a .npy-suffixed tmp name.
    best_x_tmp = run_dir / "best_x.tmp.npy"
    np.save(best_x_tmp, best_x)
    best_x_tmp.replace(run_dir / "best_x.npy")

    import pandas as pd
    history_df = pd.DataFrame(history_rows)
    csv_tmp = run_dir / "history.csv.tmp"
    history_df.to_csv(csv_tmp, index=False, encoding="utf-8")
    csv_tmp.replace(run_dir / "history.csv")

    heldout = {
        "network": network,
        "run_seed": run_seed,
        "seeds": heldout_details,
        "mean": heldout_mean,
        "median": float(np.median(heldout_scores)),
        "params": interface.describe(params),
    }
    # stop_reason first: heldout.json is the skip marker, so it must be written last or a kill
    # between the two leaves a run that is skipped for ever but missing its stop_reason.
    _atomic_write_text(run_dir / "stop_reason", stop_reason)
    _atomic_write_text(run_dir / "heldout.json", json.dumps(heldout, indent=2))

    return run_dir


# --- --all pairing + orchestration -------------------------------------------------------------


def all_pairs() -> list[tuple[str, int]]:
    """60 (network, run_seed) pairs: real x seeds 1..30, plus control_00..control_29 each
    paired with its matching seed NN+1 (control_00 -> seed 1, ..., control_29 -> seed 30), so
    every control is used exactly once."""
    pairs = [("real", s) for s in range(1, 31)]
    pairs += [(f"control_{nn:02d}", nn + 1) for nn in range(30)]
    return pairs


def _run_one(args: tuple) -> str:
    network, run_seed, out_dir, budget_gens, budget_minutes = args
    run_training(network, run_seed, out_dir, budget_gens=budget_gens, budget_minutes=budget_minutes)
    return f"{network} seed_{run_seed}"


def run_all(out_dir: str | Path, workers: int = 6,
            budget_gens: int = DEFAULT_BUDGET_GENS, budget_minutes: float = DEFAULT_BUDGET_MINUTES) -> None:
    pairs = all_pairs()
    tasks = [(net, seed, out_dir, budget_gens, budget_minutes) for net, seed in pairs]
    done = 0
    total = len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, t): t for t in tasks}
        failures: list[tuple[tuple, str]] = []
        for fut in concurrent.futures.as_completed(futures):
            task = futures[fut]
            done += 1
            try:
                label = fut.result()
            except Exception as exc:  # one bad run must not silence the other 59
                failures.append((task, repr(exc)))
                print(f"[{done}/{total}] FAILED: {task[0]} seed {task[1]}: {exc!r}", flush=True)
            else:
                print(f"[{done}/{total}] done: {label}", flush=True)

    missing = [(net, seed) for net, seed, *_ in tasks
               if not (Path(out_dir) / net / f"seed_{seed}" / "heldout.json").exists()]
    print(f"\nsweep finished: {total - len(missing)}/{total} runs have heldout.json")
    for task, exc in failures:
        print(f"  failed: {task[0]} seed {task[1]}: {exc}")
    for net, seed in missing:
        print(f"  missing: {net} seed {seed}")
    if missing:
        print("re-run the same command to resume; finished runs are skipped.")


# --- --project gate -----------------------------------------------------------------------------


def _short_run(args: tuple) -> tuple[str, float]:
    """Runs 2 generations, returns (network, seconds/generation). Used under concurrent load
    by --project -- each job is a full-size run (default popsize/seeds/max_frames), only the
    generation budget is cut to 2, so the measured rate reflects real per-generation cost."""
    network, seed, tmp_root = args
    run_dir = Path(tmp_root) / network / f"seed_{seed}"
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
    t0 = time.monotonic()
    run_training(network, seed, tmp_root, budget_gens=2, budget_minutes=1e9)
    elapsed = time.monotonic() - t0
    return network, elapsed / 2.0


def project(out_dir: str | Path = "runs", workers: int = 6,
            budget_gens: int = DEFAULT_BUDGET_GENS,
            budget_minutes: float = DEFAULT_BUDGET_MINUTES) -> int:
    """PLAN gate: run 2 generations on the real network and 2 on control_00, under a
    `workers`-process load (fills the remaining slots with more real/control_00 jobs so the
    measurement reflects realistic contention), measure seconds/generation, project
    worst-case hours for 60 runs, and exit non-zero if that exceeds 20 h."""
    import tempfile
    n = max(workers, 2)
    half = n // 2
    jobs = [("real", 9001 + i, "") for i in range(half)]
    jobs += [("control_00", 9101 + i, "") for i in range(n - half)]

    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(net, seed, tmp) for net, seed, _ in jobs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_short_run, jobs))

    spg_by_net: dict[str, list[float]] = {}
    for net, spg in results:
        spg_by_net.setdefault(net, []).append(spg)
    spg = float(np.mean([v for _, v in results]))

    hours_budget_gens = spg * budget_gens / 3600.0
    hours_budget_minutes = budget_minutes / 60.0
    # a run stops at budget_gens OR budget_minutes, whichever comes first (PLAN Step 6)
    hours_per_run = min(hours_budget_gens, hours_budget_minutes)
    worst_case_hours = hours_per_run * 60 / workers
    # Honesty note (blind-verification finding): s/generation is measured on the first two
    # generations, when most candidates die within ~1 s of game time. Later generations cost
    # several times more, so hours_budget_gens is a floor, not a forecast; the wall-clock
    # budget is the real cap. The held-out evaluation runs outside the budget on top.

    print(f"measured s/generation, real: {np.mean(spg_by_net.get('real', [float('nan')])):.3f}")
    print(f"measured s/generation, control_00: {np.mean(spg_by_net.get('control_00', [float('nan')])):.3f}")
    print(f"measured s/generation, overall mean ({workers}-process load): {spg:.3f}")
    print(f"projected hours/run at budget_gens={budget_gens}: {hours_budget_gens:.2f} h")
    print(f"projected hours/run at budget_minutes={budget_minutes}: {hours_budget_minutes:.2f} h")
    print(f"projected worst-case total for 60 runs, {workers} workers: {worst_case_hours:.2f} h")

    if worst_case_hours > 20.0:
        print("GATE: projection exceeds 20 h -- report to the lead before launching")
        return 1
    print(f"GATE: projection within 20 h ({worst_case_hours:.2f} h) -- OK to launch")
    return 0


# --- CLI -----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="flybrain Step 6 training")
    parser.add_argument("--network", type=str, help="'real' or 'control_NN'")
    parser.add_argument("--seed", type=int, help="run seed")
    parser.add_argument("--out", type=str, default="runs", help="output directory")
    parser.add_argument("--all", action="store_true", help="run all 60 real/control runs")
    parser.add_argument("--workers", type=int, default=6, help="parallel processes for --all/--project")
    parser.add_argument("--project", action="store_true", help="PLAN compute-budget gate")
    parser.add_argument("--budget-gens", type=int, default=DEFAULT_BUDGET_GENS)
    parser.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES)
    args = parser.parse_args()

    if args.project:
        sys.exit(project(args.out, workers=args.workers,
                         budget_gens=args.budget_gens, budget_minutes=args.budget_minutes))
    elif args.all:
        run_all(args.out, workers=args.workers, budget_gens=args.budget_gens, budget_minutes=args.budget_minutes)
    else:
        if not args.network or args.seed is None:
            parser.error("--network and --seed are required unless --all or --project is given")
        run_training(
            args.network, args.seed, args.out,
            budget_gens=args.budget_gens, budget_minutes=args.budget_minutes,
        )


if __name__ == "__main__":
    main()
