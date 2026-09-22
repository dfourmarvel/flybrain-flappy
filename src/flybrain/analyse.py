"""Step 8 -- analysis (docs/PLAN.md Step 8) and docs/RESULTS.md.

Reads every finished run under `runs/<network>/seed_<N>/heldout.json` (+ `history.csv`),
compares the pre-registered outcome measures (PLAN section 1) between the real connectome (30
runs) and the shuffled controls (30 runs), writes four figures to `docs-site/figures/`, and
writes `docs/RESULTS.md`. No number in RESULTS.md is typed by hand -- every one comes from this
script's own computation, printed to stdout in the same order the doc uses them.

Usage: `python -m flybrain.analyse --runs runs --out docs`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

N_REAL_EXPECTED = 30
N_CONTROL_EXPECTED = 30
COMPETENCE_THRESHOLD = 10.0
CEILING_SCORE = 22.0
FIGURE_DPI = 2 * 100  # "2x resolution" (PLAN Step 8 checklist)

# --- Limitations (PLAN Step 8 checklist: "names all six items above" -- section 1 + LIF-dt
# amendment add two more, so eight bullets total; every one is required by the ticket). ---
LIMITATIONS = [
    "This is a sub-circuit of 2,493 neurons on the path from the looming-detector inputs to the "
    "takeoff descending-neuron outputs -- not the whole *Drosophila* brain.",
    "Edge signs come from predicted, not measured, neurotransmitters (Eckstein et al. 2024 "
    "confidence-scored predictions).",
    "No neuromodulation: dopamine, serotonin and octopamine edges are excluded from the fast "
    "simulation entirely (see docs/DATA.md for the excluded-edge count), so no reward or "
    "arousal signalling is modelled.",
    "No learning happens inside the connectome -- weights, signs and wiring are frozen (PLAN "
    "L1). Only the 16-parameter game<->neuron interface (Step 5) was fitted.",
    "Synapse counts are used as a proxy for connection strength, not a measured physiological "
    "weight.",
    "The dt=0.2 ms simulation step (vs. the paper's 0.1 ms) carries a measured timing bias: "
    "<= 1.8% for 8 of 10 output neurons, and +11.9% for DNp06 L (docs/DATA.md).",
    "The left/right split of input seeds for above-gap/below-gap drive is an interface "
    "convention with no biological meaning -- left/right is really visual-field side.",
    "The interface (Step 5) was fitted specifically for this task; it is not a general-purpose "
    "readout of the connectome.",
]


# --- Loading ---------------------------------------------------------------------------------


@dataclass
class LoadedRuns:
    real: pd.DataFrame  # one row per finished real run
    control: pd.DataFrame  # one row per finished control run
    n_real_found: int
    n_control_found: int
    n_real_expected: int = N_REAL_EXPECTED
    n_control_expected: int = N_CONTROL_EXPECTED
    skipped: list = field(default_factory=list)  # (network, seed_dir, reason)

    def status_line(self) -> str:
        return (
            f"found {self.n_real_found}/{self.n_real_expected} real runs, "
            f"{self.n_control_found}/{self.n_control_expected} control runs finished"
        )


def _read_run(run_dir: Path, network_label: str) -> dict | None:
    """One run's row, or None (+ appended to `skipped`) if it is missing/unfinished/unreadable."""
    heldout_path = run_dir / "heldout.json"
    if not heldout_path.exists():
        return None
    try:
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # A malformed heldout.json (no "mean" key) or a genuinely NaN mean are both treated as
    # "no usable held-out mean" -- excluded and named in RESULTS.md, never a silent 0/NaN.
    try:
        heldout_mean = float(heldout["mean"])
    except (KeyError, TypeError, ValueError):
        heldout_mean = float("nan")

    row = {
        "network": network_label,
        "run_seed": heldout.get("run_seed"),
        "run_dir": str(run_dir),
        "heldout_mean": heldout_mean,
    }

    hist_path = run_dir / "history.csv"
    gen_to_comp = None
    last_generation = None
    usable_history = False
    if hist_path.exists():
        try:
            hist = pd.read_csv(hist_path, encoding="utf-8")
        except (OSError, pd.errors.EmptyDataError):
            hist = pd.DataFrame()
        if not hist.empty and "generation" in hist.columns:
            usable_history = True
            last_generation = int(hist["generation"].max())
            probed = hist.dropna(subset=["probe_mean"]) if "probe_mean" in hist.columns else hist.iloc[0:0]
            reached = probed[probed["probe_mean"] >= COMPETENCE_THRESHOLD]
            if not reached.empty:
                gen_to_comp = int(reached["generation"].min())
        row["history"] = hist
    else:
        row["history"] = pd.DataFrame()

    # A missing/empty/unreadable history.csv is a DATA problem, not a censored observation:
    # it must never be entered into the secondary (survival) analysis as "censored at 0".
    row["generations_to_competence"] = gen_to_comp
    row["last_generation"] = last_generation
    row["usable_history"] = usable_history
    row["censored"] = usable_history and gen_to_comp is None
    return row


def load_runs(runs_dir: Path) -> LoadedRuns:
    """Load every finished run under `runs_dir`. Tolerates a partially-finished sweep: missing
    or in-progress run directories (no heldout.json yet) are simply skipped and counted."""
    runs_dir = Path(runs_dir)
    skipped: list[tuple[str, str, str]] = []
    real_rows: list[dict] = []
    control_rows: list[dict] = []

    real_root = runs_dir / "real"
    if real_root.is_dir():
        for seed_dir in sorted(real_root.glob("seed_*")):
            row = _read_run(seed_dir, "real")
            if row is None:
                skipped.append(("real", seed_dir.name, "no heldout.json (missing or in-progress)"))
            else:
                real_rows.append(row)

    for control_dir in sorted(runs_dir.glob("control_*")):
        if not control_dir.is_dir():
            continue
        for seed_dir in sorted(control_dir.glob("seed_*")):
            row = _read_run(seed_dir, "control")
            if row is None:
                skipped.append((control_dir.name, seed_dir.name, "no heldout.json (missing or in-progress)"))
            else:
                control_rows.append(row)

    real_df = pd.DataFrame(real_rows)
    control_df = pd.DataFrame(control_rows)
    return LoadedRuns(
        real=real_df,
        control=control_df,
        n_real_found=len(real_df),
        n_control_found=len(control_df),
        skipped=skipped,
    )


# --- Primary: Mann-Whitney U + rank-biserial ------------------------------------------------


@dataclass
class PrimaryResult:
    n_real: int
    n_control: int
    n_excluded_real: int  # NaN/missing held-out mean, dropped from stats below
    n_excluded_control: int
    median_real: float
    median_control: float
    iqr_real: tuple
    iqr_control: tuple
    median_diff: float
    mean_real: float
    mean_control: float
    ceiling_count_real: int
    ceiling_count_control: int
    u_statistic: float
    p_value: float
    method: str  # "exact", "asymptotic (ties present)", or "n/a (insufficient data)"
    rank_biserial: float
    direction: str
    uninformative: bool


def analyse_primary(real_scores: np.ndarray, control_scores: np.ndarray) -> PrimaryResult:
    """Mann-Whitney U + rank-biserial effect size on the real vs control held-out means (PLAN
    section 1 primary measure). A NaN/missing held-out mean is excluded from the statistics
    (counted, never silently averaged in or turned into "n/a" for the whole result). Uses the
    exact null distribution when the pooled sample has no ties, the asymptotic (normal)
    approximation when it does -- scipy's exact method assumes no ties."""
    real_scores = np.asarray(real_scores, dtype=float)
    control_scores = np.asarray(control_scores, dtype=float)

    real_nan = np.isnan(real_scores)
    control_nan = np.isnan(control_scores)
    n_excluded_real = int(real_nan.sum())
    n_excluded_control = int(control_nan.sum())
    real_f = real_scores[~real_nan]
    control_f = control_scores[~control_nan]
    n_real, n_control = len(real_f), len(control_f)

    if n_real and n_control:
        pooled = np.concatenate([real_f, control_f])
        has_ties = len(np.unique(pooled)) < len(pooled)
        result = stats.mannwhitneyu(
            real_f, control_f, alternative="two-sided",
            method="asymptotic" if has_ties else "exact",
        )
        u = float(result.statistic)
        p = float(result.pvalue)
        method = "asymptotic (ties present)" if has_ties else "exact"
        # rank-biserial = 2U/(n1*n2) - 1 (Wendt 1972); positive => real tends to rank higher.
        rank_biserial = float(2 * u / (n_real * n_control) - 1)
    else:
        u = float("nan")
        p = float("nan")
        method = "n/a (insufficient data)"
        rank_biserial = float("nan")

    if rank_biserial > 0:
        direction = "real ranks higher than control"
    elif rank_biserial < 0:
        direction = "control ranks higher than real"
    else:
        direction = "no rank difference"

    def iqr(a):
        q1, q3 = np.percentile(a, [25, 75]) if len(a) else (float("nan"), float("nan"))
        return (float(q1), float(q3))

    ceiling_real = int(np.sum(real_f >= CEILING_SCORE))
    ceiling_control = int(np.sum(control_f >= CEILING_SCORE))
    uninformative = bool(
        n_real and n_control and ceiling_real == n_real and ceiling_control == n_control
    )

    return PrimaryResult(
        n_real=n_real, n_control=n_control,
        n_excluded_real=n_excluded_real, n_excluded_control=n_excluded_control,
        median_real=float(np.median(real_f)) if n_real else float("nan"),
        median_control=float(np.median(control_f)) if n_control else float("nan"),
        iqr_real=iqr(real_f), iqr_control=iqr(control_f),
        median_diff=(float(np.median(real_f) - np.median(control_f))
                     if n_real and n_control else float("nan")),
        mean_real=float(np.mean(real_f)) if n_real else float("nan"),
        mean_control=float(np.mean(control_f)) if n_control else float("nan"),
        ceiling_count_real=ceiling_real,
        ceiling_count_control=ceiling_control,
        u_statistic=u, p_value=p, method=method, rank_biserial=rank_biserial, direction=direction,
        uninformative=uninformative,
    )


# --- Secondary: censored generations-to-competence, log-rank test ---------------------------


@dataclass
class LogRankResult:
    chi2: float
    p_value: float
    n_real: int
    n_control: int
    n_censored_real: int
    n_censored_control: int
    median_real: float | None  # Kaplan-Meier median; None means "not reached"
    median_control: float | None


def kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> list[tuple[float, float]]:
    """Kaplan-Meier survival curve S(t) from (time, event) pairs (event=1 observed, 0
    right-censored). Returns [(t, S(t)), ...] at each distinct OBSERVED-event time, in
    ascending order. Standard product-limit estimator:
        S(t) = prod over event times t_i <= t of (1 - d_i / n_i)
    where n_i is the number still at risk at t_i (time >= t_i, censored or not) and d_i is the
    number of events exactly at t_i."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if len(times) == 0:
        return []
    event_times = np.unique(times[events == 1])
    s = 1.0
    curve: list[tuple[float, float]] = []
    for t in event_times:
        n_i = int(np.sum(times >= t))
        d_i = int(np.sum((times == t) & (events == 1)))
        if n_i > 0:
            s *= 1.0 - d_i / n_i
        curve.append((float(t), float(s)))
    return curve


def km_median(times: np.ndarray, events: np.ndarray) -> float | None:
    """Kaplan-Meier median survival time: the smallest observed time t where S(t) <= 0.5, or
    None ("not reached") if the KM curve never drops to or below 0.5."""
    for t, s in kaplan_meier_curve(times, events):
        if s <= 0.5:
            return t
    return None


def log_rank_test(
    time_real: np.ndarray, event_real: np.ndarray,
    time_control: np.ndarray, event_control: np.ndarray,
) -> LogRankResult:
    """Two-sample log-rank test, implemented directly (no lifelines dependency -- PLAN forbids
    a new dependency). `event` is 1 if the event (competence reached) was observed, 0 if the
    run is right-censored at `time` (never reached, censored at its last recorded generation).

    Standard construction (Mantel-Haenszel / Peto form): at each distinct event time t_i, with
    n_1i, n_2i at risk in each arm (n_i = n_1i + n_2i total at risk) and d_i total events,
    the expected events in arm 1 is e_1i = d_i * n_1i / n_i, and the hypergeometric variance is
        v_i = d_i * (n_1i/n_i) * (n_2i/n_i) * (n_i - d_i) / (n_i - 1)   [0 when n_i == 1]
    O_1 = sum(observed events in arm 1), E_1 = sum(e_1i), V = sum(v_i).
    chi2 = (O_1 - E_1)^2 / V, compared against chi2(df=1) via scipy.stats.chi2.
    """
    time_real = np.asarray(time_real, dtype=float)
    event_real = np.asarray(event_real, dtype=int)
    time_control = np.asarray(time_control, dtype=float)
    event_control = np.asarray(event_control, dtype=int)

    times = np.concatenate([time_real, time_control])
    events = np.concatenate([event_real, event_control])
    arm1 = np.concatenate([np.ones(len(time_real), dtype=int), np.zeros(len(time_control), dtype=int)])

    n1_total, n2_total = len(time_real), len(time_control)

    O1 = float(event_real.sum())
    event_times = np.unique(times[events == 1])

    E1 = 0.0
    V = 0.0
    for t in event_times:
        at_risk = times >= t
        n_i = int(at_risk.sum())
        n_1i = int((at_risk & (arm1 == 1)).sum())
        n_2i = n_i - n_1i
        d_i = int(((times == t) & (events == 1)).sum())
        if n_i == 0:
            continue
        e_1i = d_i * n_1i / n_i
        E1 += e_1i
        if n_i > 1:
            v_i = d_i * (n_1i / n_i) * (n_2i / n_i) * (n_i - d_i) / (n_i - 1)
            V += v_i

    if V <= 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (O1 - E1) ** 2 / V
        p_value = float(stats.chi2.sf(chi2_stat, df=1))

    return LogRankResult(
        chi2=float(chi2_stat), p_value=p_value,
        n_real=n1_total, n_control=n2_total,
        n_censored_real=int((event_real == 0).sum()),
        n_censored_control=int((event_control == 0).sum()),
        median_real=km_median(time_real, event_real),
        median_control=km_median(time_control, event_control),
    )


def _run_label(row) -> str:
    """Short id for a run, e.g. "real/seed_3" or "control_00/seed_5", for naming runs in
    RESULTS.md warnings."""
    p = Path(row["run_dir"])
    return f"{p.parent.name}/{p.name}"


def build_survival_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """(time, event, excluded_labels) for the secondary measure from a loaded-runs dataframe.
    time is generations_to_competence if reached, else last_generation (censoring time); event
    is 1 if reached, 0 if censored (PLAN section 1: "Censored at the run's budget ... use a
    survival/rank test, not a mean"). A run with no usable history.csv (missing, empty or
    unreadable) is a DATA problem, not a censored observation -- it is excluded here and its
    label returned separately, never entered as "censored at generation 0"."""
    if df.empty:
        return np.array([]), np.array([]), []
    usable = df[df["usable_history"]]
    excluded = df[~df["usable_history"]]
    excluded_labels = [_run_label(row) for _, row in excluded.iterrows()]
    if usable.empty:
        return np.array([]), np.array([]), excluded_labels
    reached = usable["generations_to_competence"]
    last_gen = usable["last_generation"]
    time = np.where(reached.notna(), reached, last_gen).astype(float)
    event = reached.notna().astype(int).to_numpy()
    return time, event, excluded_labels


# --- Figures -----------------------------------------------------------------------------------


def _iqr_band(hist_list: list[pd.DataFrame], value_col: str, gen_col: str = "generation"):
    """Median + IQR of `value_col` across runs, aligned on `gen_col`. Returns (gens, median,
    q1, q3); runs are aligned on their common generation values (inner union with NaN padding)."""
    frames = []
    for h in hist_list:
        if h is None or h.empty or value_col not in h.columns:
            continue
        sub = h[[gen_col, value_col]].dropna(subset=[value_col])
        if sub.empty:
            continue
        frames.append(sub.set_index(gen_col)[value_col])
    if not frames:
        return np.array([]), np.array([]), np.array([]), np.array([])
    wide = pd.concat(frames, axis=1)
    wide = wide.sort_index()
    gens = wide.index.to_numpy(dtype=float)
    median = wide.median(axis=1, skipna=True).to_numpy()
    q1 = wide.quantile(0.25, axis=1).to_numpy()
    q3 = wide.quantile(0.75, axis=1).to_numpy()
    return gens, median, q1, q3


def fig_strip_plot(real_scores: np.ndarray, control_scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rng = np.random.default_rng(0)
    for x0, scores, color, label in ((0, real_scores, "#2b6cb0", "real"),
                                      (1, control_scores, "#c05621", "control")):
        if len(scores) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(np.full(len(scores), x0) + jitter, scores, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.5, s=45, zorder=3, label=label)
        ax.hlines(np.median(scores), x0 - 0.22, x0 + 0.22, color=color, linewidth=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"real (n={len(real_scores)})", f"control (n={len(control_scores)})"])
    ax.set_ylabel("held-out mean score (20 seeds)")
    ax.set_title("Held-out score by network")
    ax.set_ylim(bottom=-0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def fig_fitness_curves(real_hist: list[pd.DataFrame], control_hist: list[pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hists, color, label in ((real_hist, "#2b6cb0", "real"), (control_hist, "#c05621", "control")):
        gens, med, q1, q3 = _iqr_band(hists, "mean_fitness")
        if len(gens) == 0:
            continue
        ax.plot(gens, med, color=color, label=f"{label} (median)", linewidth=2)
        ax.fill_between(gens, q1, q3, color=color, alpha=0.2, label=f"{label} IQR")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean training fitness")
    ax.set_title("Mean fitness across generations")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def fig_probe_curves(real_hist: list[pd.DataFrame], control_hist: list[pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hists, color, label in ((real_hist, "#2b6cb0", "real"), (control_hist, "#c05621", "control")):
        gens, med, q1, q3 = _iqr_band(hists, "probe_mean")
        if len(gens) == 0:
            continue
        ax.plot(gens, med, color=color, label=f"{label} (median)", linewidth=2, marker="o", markersize=3)
        ax.fill_between(gens, q1, q3, color=color, alpha=0.2, label=f"{label} IQR")
    ax.axhline(COMPETENCE_THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"competence threshold ({COMPETENCE_THRESHOLD:g})")
    ax.set_xlabel("generation")
    ax.set_ylabel("probe mean score (5 probe seeds, full game)")
    ax.set_title("Probe-score curves")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


GF_SKIP_MESSAGES = {
    "no_replay_file": "`docs-site/replay/best.json` does not exist (or could not be read) yet.",
    "not_real_connectome": (
        "the replay at `docs-site/replay/best.json` is not recorded from the real connectome "
        "(`meta.is_real_connectome` is false)."
    ),
    "no_gf_block": (
        "the replay at `docs-site/replay/best.json` has no `gf` spike-count block to plot."
    ),
}


def fig_gf_habituation(replay_path: Path, out_path: Path) -> tuple[bool, str]:
    """Writes the giant-fiber habituation figure from a replay JSON if it exists AND is a
    real-connectome recording. Returns (written, reason): reason is "" if written, else one of
    GF_SKIP_MESSAGES's keys naming why it was skipped ("no_replay_file", "not_real_connectome",
    "no_gf_block") -- the three reasons must never be collapsed into one message."""
    if not replay_path.exists():
        return False, "no_replay_file"
    try:
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "no_replay_file"
    meta = replay.get("meta", {})
    if not meta.get("is_real_connectome"):
        return False, "not_real_connectome"
    gf = replay.get("gf")
    if not gf or not gf.get("counts"):
        return False, "no_gf_block"

    counts = np.asarray(gf["counts"], dtype=float)  # (frames, 2)
    sides = gf.get("sides", ["R", "L"])
    frames = np.arange(counts.shape[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"R": "#2b6cb0", "L": "#c05621"}
    for i, side in enumerate(sides):
        ax.plot(frames, counts[:, i], color=colors.get(side, "black"), label=f"DNp01 {side}",
                linewidth=1, alpha=0.85)
    ax.set_xlabel("frame")
    ax.set_ylabel("DNp01 spike count")
    ax.set_title("Giant-fiber (DNp01) spiking across the episode")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    return True, ""


# --- RESULTS.md --------------------------------------------------------------------------------


MIN_FINISHED_FOR_VERDICT = 3  # below this per arm, RESULTS.md prints no C2 verdict sentence


def _fmt(x, nd=3) -> str:
    if x is None:
        return "not reached"
    if isinstance(x, float) and (np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def _fmt_p(p: float) -> str:
    """"p < 0.0001" below that floor (never "p = 0.0000"), else "p = 0.xxxx"."""
    if np.isnan(p):
        return "p = n/a"
    if p < 0.0001:
        return "p < 0.0001"
    return f"p = {p:.4f}"


def render_results_md(
    loaded: LoadedRuns,
    primary: PrimaryResult,
    secondary: LogRankResult,
    gf_figure_written: bool,
    gf_skip_reason: str,
    figures_rel: dict,
    excluded_mean_real: list,
    excluded_mean_control: list,
    excluded_history_real: list,
    excluded_history_control: list,
) -> str:
    lines: list[str] = []
    lines.append("# Results")
    lines.append("")
    lines.append("*Generated entirely by `python -m flybrain.analyse` -- every number below is")
    lines.append("computed by the script, none is typed by hand.*")
    lines.append("")

    lines.append("## What was tested")
    lines.append("")
    lines.append(
        "- **C1 (demo):** does the fly's own looming-to-escape circuit, unmodified, play "
        "Flappy Bird above chance?"
    )
    lines.append(
        "- **C2 (experiment):** does the *real* wiring of that circuit play better than a "
        "degree-preserving shuffled control network of identical size, degree sequence and "
        "neurotransmitter composition?"
    )
    lines.append("")

    lines.append("## How")
    lines.append("")
    lines.append(
        f"{loaded.n_real_found}/{loaded.n_real_expected} real-connectome runs and "
        f"{loaded.n_control_found}/{loaded.n_control_expected} shuffled-control runs "
        "(PLAN L5: 30 independent training seeds per arm) were trained by CMA-ES over the "
        "16-parameter game-to-neuron interface (PLAN Step 5), then each run's best parameter "
        "vector was scored on 20 held-out seeds never used during training. This report "
        "compares those held-out scores between arms."
    )
    lines.append("")

    lines.append("## Pre-registered measures")
    lines.append("")
    lines.append(
        "Fixed 2026-09-22, before any control network was trained (PLAN section 1); neither "
        "measure nor its threshold changed after seeing results."
    )
    lines.append("")
    lines.append(
        "- **Primary:** held-out mean score per run (mean over 20 held-out seeds), 30 real vs "
        "30 control."
    )
    lines.append(
        f"- **Secondary:** generations to competence -- first generation whose probe_mean >= "
        f"{COMPETENCE_THRESHOLD:g}, censored (as \"not reached\") when a run never reaches it."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(f"Runs found: {loaded.status_line()}.")
    lines.append("")

    lines.append("### Primary -- held-out mean score")
    lines.append("")
    if excluded_mean_real or excluded_mean_control:
        excluded_all = excluded_mean_real + excluded_mean_control
        lines.append(
            f"**Warning:** {len(excluded_all)} run(s) have a NaN or missing held-out mean and "
            f"are excluded from the statistics below: {', '.join(excluded_all)}."
        )
        lines.append("")
    lines.append("| | real | control |")
    lines.append("|---|---|---|")
    lines.append(f"| n | {primary.n_real} | {primary.n_control} |")
    lines.append(f"| median | {_fmt(primary.median_real)} | {_fmt(primary.median_control)} |")
    lines.append(
        f"| IQR | [{_fmt(primary.iqr_real[0])}, {_fmt(primary.iqr_real[1])}] "
        f"| [{_fmt(primary.iqr_control[0])}, {_fmt(primary.iqr_control[1])}] |"
    )
    lines.append(f"| mean | {_fmt(primary.mean_real)} | {_fmt(primary.mean_control)} |")
    lines.append(
        f"| at ceiling ({CEILING_SCORE:g}) | {primary.ceiling_count_real}/{primary.n_real} "
        f"| {primary.ceiling_count_control}/{primary.n_control} |"
    )
    lines.append("")
    lines.append(f"Difference in medians (real - control): {_fmt(primary.median_diff)}.")
    lines.append("")
    lines.append(
        f"Mann-Whitney U = {_fmt(primary.u_statistic)}, two-sided {_fmt_p(primary.p_value)} "
        f"(method: {primary.method}). Rank-biserial correlation = {_fmt(primary.rank_biserial)} "
        f"({primary.direction})."
    )
    lines.append("")
    if primary.n_real < MIN_FINISHED_FOR_VERDICT or primary.n_control < MIN_FINISHED_FOR_VERDICT:
        lines.append(
            f"Not enough finished runs for a verdict (real: {loaded.n_real_found}/"
            f"{loaded.n_real_expected}, control: {loaded.n_control_found}/{loaded.n_control_expected})."
        )
    elif primary.uninformative:
        lines.append(
            "**Both arms are at the 22-point ceiling on every run.** The primary comparison is "
            "uninformative -- there is no headroom left for the real connectome to distinguish "
            "itself from control on held-out score, so the result rests on the secondary "
            "measure below."
        )
    elif primary.p_value < 0.05 and primary.rank_biserial > 0:
        lines.append("Real outperforms control on held-out score (C2 supported).")
    elif primary.p_value < 0.05 and primary.rank_biserial < 0:
        lines.append(
            "Control outperforms real on held-out score. **C2 fails**: the real wiring does "
            "not beat the shuffled control on the primary measure."
        )
    else:
        lines.append(
            "No statistically significant difference between real and control on held-out "
            "score. **C2 is not supported** by the primary measure."
        )
    lines.append("")

    lines.append("### Secondary -- generations to competence (censored)")
    lines.append("")
    if excluded_history_real or excluded_history_control:
        excluded_all_h = excluded_history_real + excluded_history_control
        lines.append(
            f"{len(excluded_all_h)} run(s) excluded from the secondary measure (no usable "
            f"history): {', '.join(excluded_all_h)}."
        )
        lines.append("")
    lines.append("| | real | control |")
    lines.append("|---|---|---|")
    lines.append(f"| n | {secondary.n_real} | {secondary.n_control} |")
    lines.append(
        f"| median generations (Kaplan-Meier) | {_fmt(secondary.median_real)} "
        f"| {_fmt(secondary.median_control)} |"
    )
    lines.append(
        f"| censored (never reached) | {secondary.n_censored_real}/{secondary.n_real} "
        f"| {secondary.n_censored_control}/{secondary.n_control} |"
    )
    lines.append("")
    lines.append(
        f"Log-rank chi-square = {_fmt(secondary.chi2)}, {_fmt_p(secondary.p_value)}."
    )
    lines.append("")
    if secondary.n_real < MIN_FINISHED_FOR_VERDICT or secondary.n_control < MIN_FINISHED_FOR_VERDICT:
        lines.append(
            f"Not enough finished runs for a verdict (real: {loaded.n_real_found}/"
            f"{loaded.n_real_expected}, control: {loaded.n_control_found}/{loaded.n_control_expected})."
        )
        lines.append("")

    lines.append("### Figures")
    lines.append("")
    lines.append(f"1. ![Held-out score by network]({figures_rel['strip']})")
    lines.append(f"2. ![Mean fitness across generations]({figures_rel['fitness']})")
    lines.append(f"3. ![Probe-score curves]({figures_rel['probe']})")
    if gf_figure_written:
        lines.append(f"4. ![Giant-fiber habituation]({figures_rel['gf']})")
    else:
        reason_text = GF_SKIP_MESSAGES.get(gf_skip_reason, "the replay data is unavailable.")
        lines.append(f"4. Giant-fiber habituation figure skipped: {reason_text}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines) + "\n"


# --- Orchestration ----------------------------------------------------------------------------


def run_analysis(runs_dir: str | Path, out_dir: str | Path,
                  project_root: Path | None = None) -> Path:
    """Loads every finished run, runs both pre-registered comparisons, writes figures + RESULTS.md.
    Returns the path to RESULTS.md. `project_root` (default: two parents up from this file, i.e.
    the repo root) anchors `docs-site/figures/` and `docs-site/replay/best.json`."""
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    loaded = load_runs(runs_dir)
    print(loaded.status_line())
    for net, seed_name, reason in loaded.skipped:
        print(f"  skipped: {net}/{seed_name}: {reason}")

    excluded_mean_real = (
        [_run_label(row) for _, row in loaded.real[loaded.real["heldout_mean"].isna()].iterrows()]
        if not loaded.real.empty else []
    )
    excluded_mean_control = (
        [_run_label(row) for _, row in loaded.control[loaded.control["heldout_mean"].isna()].iterrows()]
        if not loaded.control.empty else []
    )
    for label in excluded_mean_real + excluded_mean_control:
        print(f"  excluded (NaN/missing held-out mean): {label}")

    real_scores = loaded.real["heldout_mean"].to_numpy() if not loaded.real.empty else np.array([])
    control_scores = loaded.control["heldout_mean"].to_numpy() if not loaded.control.empty else np.array([])
    primary = analyse_primary(real_scores, control_scores)
    print(f"primary: n real={primary.n_real} control={primary.n_control}")
    print(f"primary: median real={primary.median_real} control={primary.median_control}")
    print(f"primary: IQR real={primary.iqr_real} control={primary.iqr_control}")
    print(f"primary: mean real={primary.mean_real} control={primary.mean_control}")
    print(
        f"primary: at ceiling real={primary.ceiling_count_real}/{primary.n_real} "
        f"control={primary.ceiling_count_control}/{primary.n_control}"
    )
    print(f"primary: median_diff={primary.median_diff}")
    print(
        f"primary: U={primary.u_statistic} p={primary.p_value} method={primary.method} "
        f"rank_biserial={primary.rank_biserial} direction={primary.direction}"
    )

    time_real, event_real, excluded_hist_real = build_survival_arrays(loaded.real)
    time_control, event_control, excluded_hist_control = build_survival_arrays(loaded.control)
    for label in excluded_hist_real + excluded_hist_control:
        print(f"  excluded (no usable history): {label}")
    secondary = log_rank_test(time_real, event_real, time_control, event_control)
    print(f"secondary: n real={secondary.n_real} control={secondary.n_control}")
    print(
        f"secondary: median_gens (Kaplan-Meier) real={secondary.median_real} "
        f"control={secondary.median_control}"
    )
    print(
        f"secondary: censored real={secondary.n_censored_real}/{secondary.n_real} "
        f"control={secondary.n_censored_control}/{secondary.n_control}"
    )
    print(f"secondary: chi2={secondary.chi2} p={secondary.p_value}")

    figures_dir = project_root / "docs-site" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_strip_plot(real_scores, control_scores, figures_dir / "primary_strip.png")
    fig_fitness_curves(
        loaded.real["history"].tolist() if "history" in loaded.real else [],
        loaded.control["history"].tolist() if "history" in loaded.control else [],
        figures_dir / "fitness_curves.png",
    )
    fig_probe_curves(
        loaded.real["history"].tolist() if "history" in loaded.real else [],
        loaded.control["history"].tolist() if "history" in loaded.control else [],
        figures_dir / "probe_curves.png",
    )
    replay_path = project_root / "docs-site" / "replay" / "best.json"
    gf_path = figures_dir / "gf_habituation.png"
    gf_written, gf_reason = fig_gf_habituation(replay_path, gf_path)

    figures_rel = {
        "strip": "../docs-site/figures/primary_strip.png",
        "fitness": "../docs-site/figures/fitness_curves.png",
        "probe": "../docs-site/figures/probe_curves.png",
        "gf": "../docs-site/figures/gf_habituation.png",
    }

    md = render_results_md(
        loaded, primary, secondary, gf_written, gf_reason, figures_rel,
        excluded_mean_real, excluded_mean_control, excluded_hist_real, excluded_hist_control,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "RESULTS.md"
    results_path.write_text(md, encoding="utf-8")
    print(f"wrote {results_path}")
    return results_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Step 8 analysis: compare real vs control runs.")
    parser.add_argument("--runs", default="runs", help="root directory of run outputs (default: runs)")
    parser.add_argument("--out", default="docs", help="directory to write RESULTS.md into (default: docs)")
    args = parser.parse_args(argv)
    run_analysis(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
