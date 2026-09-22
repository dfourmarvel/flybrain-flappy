"""Tests for Step 8 analysis (src/flybrain/analyse.py).

Entirely synthetic: builds small runs/ trees under tmp_path with known numbers, never touches
the real sweep under the repo's runs/ directory and never imports anything that reads it. Whole
file runs in well under 60 s.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flybrain import analyse

# --------------------------------------------------------------------------------------------
# Helpers to build a synthetic runs/ tree
# --------------------------------------------------------------------------------------------


def _write_run(runs_root, network, seed, *, heldout_mean, history_rows=None, finished=True):
    """Writes runs/<network>/seed_<seed>/{heldout.json, history.csv} (or leaves it unfinished
    if finished=False, matching an in-progress or missing run: no heldout.json)."""
    run_dir = runs_root / network / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
    if finished:
        heldout = {
            "network": network,
            "run_seed": seed,
            "seeds": [{"seed": s, "score": 0, "frames_survived": 0} for s in range(301, 321)],
            "mean": heldout_mean,
            "median": heldout_mean,
            "params": {},
        }
        (run_dir / "heldout.json").write_text(json.dumps(heldout), encoding="utf-8")
    return run_dir


def _probe_rows(gens_and_probes):
    """[(generation, probe_mean_or_None), ...] -> history rows list of dicts."""
    rows = []
    last_gen = 0
    for gen, probe in gens_and_probes:
        rows.append({
            "generation": gen, "best_fitness": 1.0, "mean_fitness": 1.0,
            "elapsed_s": 1.0, "probe_mean": probe,
        })
        last_gen = gen
    return rows


# --------------------------------------------------------------------------------------------
# 1. Loading tolerates missing/in-progress runs and reports counts correctly
# --------------------------------------------------------------------------------------------


def test_load_runs_counts_and_tolerates_missing(tmp_path):
    runs_root = tmp_path / "runs"
    # 2 finished real runs, 1 in-progress real run (no heldout.json), 1 missing entirely.
    _write_run(runs_root, "real", 1, heldout_mean=12.0, history_rows=_probe_rows([(1, 2.0)]))
    _write_run(runs_root, "real", 2, heldout_mean=15.0, history_rows=_probe_rows([(1, 3.0)]))
    _write_run(runs_root, "real", 3, heldout_mean=None, finished=False)  # in-progress
    # seed 4 never created at all -> simply absent

    # 1 finished control run.
    _write_run(runs_root, "control_00", 1, heldout_mean=8.0, history_rows=_probe_rows([(1, 1.0)]))

    loaded = analyse.load_runs(runs_root)

    assert loaded.n_real_found == 2
    assert loaded.n_control_found == 1
    assert loaded.n_real_expected == 30
    assert loaded.n_control_expected == 30
    assert "2/30 real" in loaded.status_line()
    assert "1/30 control" in loaded.status_line()
    # the in-progress run must show up as skipped, not crash the load
    assert any(name == "seed_3" for _, name, _ in loaded.skipped)
    assert sorted(loaded.real["heldout_mean"]) == [12.0, 15.0]


def test_load_runs_empty_tree(tmp_path):
    loaded = analyse.load_runs(tmp_path / "runs")  # directory doesn't even exist yet
    assert loaded.n_real_found == 0
    assert loaded.n_control_found == 0
    assert loaded.real.empty
    assert loaded.control.empty


# --------------------------------------------------------------------------------------------
# 2. Mann-Whitney U and rank-biserial match hand-computed values on a tiny fixture
# --------------------------------------------------------------------------------------------


def test_mannwhitney_and_rank_biserial_hand_computed():
    # real = [1, 3, 5], control = [2, 4, 6]. Combined ranks 1..6:
    #   real  -> ranks 1, 3, 5  (sum = 9)
    #   control -> ranks 2, 4, 6 (sum = 12)
    # U_real = sum(rank_real) - n1(n1+1)/2 = 9 - 3*4/2 = 3
    # Cross-check by direct pairwise count (U = #(x_i > y_j) + 0.5*#(x_i == y_j)):
    #   x=1: beats none of [2,4,6] -> 0
    #   x=3: beats 2 only          -> 1
    #   x=5: beats 2,4             -> 2
    #   U_real = 0 + 1 + 2 = 3  (matches)
    # rank_biserial = 2*U/(n1*n2) - 1 = 2*3/9 - 1 = -1/3
    real = np.array([1.0, 3.0, 5.0])
    control = np.array([2.0, 4.0, 6.0])

    result = analyse.analyse_primary(real, control)

    assert result.u_statistic == pytest.approx(3.0)
    assert result.rank_biserial == pytest.approx(-1.0 / 3.0)
    assert result.direction == "control ranks higher than real"
    assert result.method == "exact"  # no ties in the pooled [1,2,3,4,5,6]

    assert result.median_real == pytest.approx(3.0)
    assert result.median_control == pytest.approx(4.0)
    assert result.median_diff == pytest.approx(-1.0)
    assert result.ceiling_count_real == 0
    assert result.ceiling_count_control == 0


# --------------------------------------------------------------------------------------------
# 3. Log-rank test: hand-worked example, no censoring, all-censored arm, ties
# --------------------------------------------------------------------------------------------


def test_log_rank_hand_worked_example_no_censoring():
    # Arm A (real): event times 2, 4 (both observed). Arm B (control): event times 1, 3 (both
    # observed). No censoring in either arm.
    #
    # Distinct event times: 1, 2, 3, 4. At each, with n = total at risk, n1/n2 at risk per arm,
    # d = total events at that time (Mantel-Haenszel / Peto log-rank construction):
    #
    #   t=1: at risk A:{2,4}=2, B:{1,3}=2, n=4;  event at t=1 is in B -> d=1
    #        e1 = d*n1/n = 1*2/4 = 0.5
    #        v1 = d*(n1/n)*(n2/n)*(n-d)/(n-1) = 1*0.5*0.5*3/3 = 0.25
    #   t=2: at risk A:{2,4}=2, B:{3}=1, n=3;    event at t=2 is in A -> d=1
    #        e2 = 1*2/3 = 0.6667
    #        v2 = 1*(2/3)*(1/3)*(3-1)/(3-1) = 1*(2/9) = 0.2222
    #   t=3: at risk A:{4}=1, B:{3}=1, n=2;      event at t=3 is in B -> d=1
    #        e3 = 1*1/2 = 0.5
    #        v3 = 1*0.5*0.5*(2-1)/(2-1) = 0.25
    #   t=4: at risk A:{4}=1, B:{}=0, n=1;       event at t=4 is in A -> d=1
    #        e4 = 1*1/1 = 1.0
    #        v4 = 0 (n=1, so the (n-1) denominator case is skipped -> contributes 0 variance)
    #
    # O1 (observed events in arm A) = 2 (both A's events)
    # E1 = 0.5 + 0.6667 + 0.5 + 1.0 = 2.6667
    # V  = 0.25 + 0.2222 + 0.25 + 0 = 0.7222
    # chi2 = (O1 - E1)^2 / V = (2 - 2.6667)^2 / 0.7222 = 0.44444 / 0.72222 = 0.61538
    time_a = np.array([2.0, 4.0])
    event_a = np.array([1, 1])
    time_b = np.array([1.0, 3.0])
    event_b = np.array([1, 1])

    result = analyse.log_rank_test(time_a, event_a, time_b, event_b)

    assert result.chi2 == pytest.approx(0.61538, abs=1e-4)
    assert result.p_value == pytest.approx(float(stats.chi2.sf(0.61538, df=1)), abs=1e-4)
    assert result.n_censored_real == 0
    assert result.n_censored_control == 0
    # Kaplan-Meier median (fix #1): smallest t where S(t) <= 0.5, not a raw order statistic.
    # Arm A [2,4]: at t=2, n=2, d=1 -> S=0.5 <= 0.5 -> median=2.0.
    assert result.median_real == pytest.approx(2.0)
    # Arm B [1,3]: at t=1, n=2, d=1 -> S=0.5 <= 0.5 -> median=1.0.
    assert result.median_control == pytest.approx(1.0)


def test_log_rank_all_censored_in_one_arm():
    # Control arm never reaches the event at all -> median must read "not reached" (None), and
    # the test must not divide by zero or crash.
    time_real = np.array([1.0, 2.0])
    event_real = np.array([1, 1])
    time_control = np.array([5.0, 5.0])
    event_control = np.array([0, 0])  # both censored (never reached)

    result = analyse.log_rank_test(time_real, event_real, time_control, event_control)

    assert result.n_censored_control == 2
    assert result.n_censored_real == 0
    assert result.median_control is None
    # Kaplan-Meier median: at t=1, n=2, d=1 -> S=0.5 <= 0.5 -> median=1.0.
    assert result.median_real == pytest.approx(1.0)
    assert 0.0 <= result.p_value <= 1.0
    assert np.isfinite(result.chi2)


def test_log_rank_ties_across_arms():
    # All four runs "reach competence" at exactly the same generation -- a tie across arms with
    # no censoring. n=4, n1=2, n2=2, d=4 at the single event time t=3:
    #   e1 = 4*2/4 = 2 = O1  ->  chi2 numerator is 0
    #   v1 = 4*(2/4)*(2/4)*(4-4)/(4-1) = 0 (n - d = 0) -> V = 0 -> the V<=0 branch reports
    #   chi2=0, p=1 (no information to discriminate the arms when everyone ties).
    time_real = np.array([3.0, 3.0])
    event_real = np.array([1, 1])
    time_control = np.array([3.0, 3.0])
    event_control = np.array([1, 1])

    result = analyse.log_rank_test(time_real, event_real, time_control, event_control)

    assert result.chi2 == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------
# 4. generations_to_competence censoring via build_survival_arrays
# --------------------------------------------------------------------------------------------


def test_build_survival_arrays_censoring(tmp_path):
    runs_root = tmp_path / "runs"
    # Run 1 reaches competence (probe_mean >= 10) at generation 10.
    _write_run(runs_root, "real", 1, heldout_mean=20.0,
               history_rows=_probe_rows([(1, 2.0), (5, 6.0), (10, 11.0), (15, 12.0)]))
    # Run 2 never reaches it -> censored at its last recorded generation (15).
    _write_run(runs_root, "real", 2, heldout_mean=5.0,
               history_rows=_probe_rows([(1, 1.0), (5, 3.0), (10, 4.0), (15, 5.0)]))

    loaded = analyse.load_runs(runs_root)
    row1 = loaded.real[loaded.real["run_seed"] == 1].iloc[0]
    row2 = loaded.real[loaded.real["run_seed"] == 2].iloc[0]
    assert row1["generations_to_competence"] == 10
    assert bool(row1["censored"]) is False
    assert pd.isna(row2["generations_to_competence"])
    assert bool(row2["censored"]) is True
    assert row2["last_generation"] == 15

    time, event, excluded = analyse.build_survival_arrays(loaded.real)
    time_by_seed = dict(zip(loaded.real["run_seed"], time))
    event_by_seed = dict(zip(loaded.real["run_seed"], event))
    assert time_by_seed[1] == 10
    assert event_by_seed[1] == 1
    assert time_by_seed[2] == 15  # censored at last generation, not dropped or imputed
    assert event_by_seed[2] == 0
    assert excluded == []


# --------------------------------------------------------------------------------------------
# 5. RESULTS.md contains every Limitations bullet, and no hand-typed statistic
# --------------------------------------------------------------------------------------------


def _run_full_analysis(tmp_path, real_means, control_means, *, seed_offset=0):
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "docs"
    for i, mean in enumerate(real_means):
        _write_run(runs_root, "real", seed_offset + i + 1, heldout_mean=mean,
                   history_rows=_probe_rows([(1, mean / 2), (5, mean)]))
    for i, mean in enumerate(control_means):
        _write_run(runs_root, f"control_{i:02d}", seed_offset + i + 1, heldout_mean=mean,
                   history_rows=_probe_rows([(1, mean / 2), (5, mean)]))
    results_path = analyse.run_analysis(runs_root, out_dir, project_root=tmp_path)
    return results_path.read_text(encoding="utf-8")


LIMITATIONS_KEYWORDS = [
    "sub-circuit",
    "neurotransmitter",
    "neuromodulation",
    "interface",
    "synapse count",
    "DNp06",
    "left/right",
    "fitted specifically for this task",
]


def test_results_md_has_eight_limitations_bullets_with_expected_topics(tmp_path):
    real_means = [10.0, 12.0, 14.0, 16.0, 9.0]
    control_means = [5.0, 6.0, 7.0, 4.0, 8.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)

    limitations_block = text.split("## Limitations")[1]
    bullets = [line for line in limitations_block.splitlines() if line.startswith("- ")]
    assert len(bullets) == 8

    lowered = limitations_block.lower()
    for keyword in LIMITATIONS_KEYWORDS:
        assert keyword.lower() in lowered, f"limitations section missing topic: {keyword!r}"


def test_results_md_no_hand_typed_numbers(tmp_path):
    real_means = [10.0, 12.0, 14.0, 16.0, 9.0]
    control_means = [5.0, 6.0, 7.0, 4.0, 8.0]
    text_a = _run_full_analysis(tmp_path, real_means, control_means)

    # Regenerate with DIFFERENT synthetic numbers into a fresh tmp dir and confirm the
    # numeric/statistical content of the doc actually changes (i.e. nothing is hand-typed).
    tmp_path_b = tmp_path / "b"
    tmp_path_b.mkdir()
    real_means_b = [1.0, 2.0, 1.0, 3.0, 2.0]
    control_means_b = [1.0, 1.0, 2.0, 1.0, 1.0]
    text_b = _run_full_analysis(tmp_path_b, real_means_b, control_means_b)

    def stats_block(text):
        # everything from "## Results" up to "## Limitations" -- the numeric content.
        return text.split("## Results")[1].split("## Limitations")[0]

    assert stats_block(text_a) != stats_block(text_b)
    # the Limitations section itself, by contrast, is fixed prose and must be identical.
    limitations_a = text_a.split("## Limitations")[1]
    limitations_b = text_b.split("## Limitations")[1]
    assert limitations_a == limitations_b


# --------------------------------------------------------------------------------------------
# 6. Ceiling case: both arms all-22 -> primary comparison flagged uninformative
# --------------------------------------------------------------------------------------------


def test_ceiling_case_flags_primary_uninformative(tmp_path):
    real_means = [22.0] * 5
    control_means = [22.0] * 5
    text = _run_full_analysis(tmp_path, real_means, control_means)

    assert "uninformative" in text.lower()
    # sanity: the ceiling-count table should show every run at ceiling
    assert "5/5" in text


def test_analyse_primary_ceiling_flag_direct():
    scores = np.array([22.0, 22.0, 22.0])
    result = analyse.analyse_primary(scores, scores.copy())
    assert result.uninformative is True
    assert result.ceiling_count_real == 3
    assert result.ceiling_count_control == 3


def test_analyse_primary_not_uninformative_when_below_ceiling():
    real = np.array([22.0, 22.0, 22.0])
    control = np.array([22.0, 10.0, 22.0])
    result = analyse.analyse_primary(real, control)
    assert result.uninformative is False


# --------------------------------------------------------------------------------------------
# 7. Kaplan-Meier median: the two pre-registered validation cases (fix #1)
# --------------------------------------------------------------------------------------------


def test_km_median_freireich_6mp_treated_arm():
    # Freireich et al. 1963 6-MP treated-arm remission-time data (a standard KM textbook
    # example). Times in order, asterisk = censored:
    # 6,6,6,6*,7,9*,10,10*,11*,13,16,17*,19*,20*,22,23,25*,32*,32*,34*,35*
    # Known KM median for this arm is 23. The old order-statistic code gave 16.0.
    times = np.array([6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35],
                      dtype=float)
    events = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    assert len(times) == 21
    assert events.sum() == 9  # 9 observed relapses, 12 censored

    median = analyse.km_median(times, events)

    assert median == pytest.approx(23.0)


def test_km_median_censored_block_then_sequential_events():
    # 15 runs censored at generation 30 (never reached), 15 runs that reached at generations
    # 35..49 (one each, no ties). Known KM median is 42.0. The old code, seeing >=half the
    # arm censored, printed "not reached" -- wrong, because the censoring happens BEFORE the
    # events, not spread through them.
    times = np.concatenate([np.full(15, 30.0), np.arange(35.0, 50.0)])
    events = np.concatenate([np.zeros(15, dtype=int), np.ones(15, dtype=int)])

    median = analyse.km_median(times, events)

    assert median == pytest.approx(42.0)


def test_km_median_none_when_survival_never_drops_to_half():
    # All censored -> survival curve never drops at all -> "not reached".
    times = np.array([5.0, 5.0, 5.0])
    events = np.array([0, 0, 0])
    assert analyse.km_median(times, events) is None


# --------------------------------------------------------------------------------------------
# 8. Missing/empty history.csv is a data problem, not a censored observation (fix #2)
# --------------------------------------------------------------------------------------------


def test_missing_and_empty_history_excluded_not_censored(tmp_path):
    runs_root = tmp_path / "runs"
    # seed 1: normal run, usable history.
    _write_run(runs_root, "real", 1, heldout_mean=15.0,
                history_rows=_probe_rows([(1, 2.0), (10, 11.0)]))
    # seed 2: finished (heldout.json present) but history.csv never written at all.
    _write_run(runs_root, "real", 2, heldout_mean=8.0, history_rows=None)
    # seed 3: finished, history.csv exists but is a zero-byte file.
    run_dir_3 = _write_run(runs_root, "real", 3, heldout_mean=9.0, history_rows=None)
    (run_dir_3 / "history.csv").write_text("", encoding="utf-8")

    loaded = analyse.load_runs(runs_root)
    assert loaded.n_real_found == 3  # all three have a readable heldout.json -> "found"

    time, event, excluded = analyse.build_survival_arrays(loaded.real)

    assert len(time) == 1  # only seed 1 contributes to the secondary analysis
    assert len(excluded) == 2
    assert any(label.endswith("seed_2") for label in excluded)
    assert any(label.endswith("seed_3") for label in excluded)
    # must never read as "censored at generation 0"
    assert 0.0 not in time


def test_results_md_names_excluded_history_runs(tmp_path):
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "docs"
    _write_run(runs_root, "real", 1, heldout_mean=15.0,
                history_rows=_probe_rows([(1, 2.0), (10, 11.0)]))
    _write_run(runs_root, "real", 2, heldout_mean=8.0, history_rows=None)
    for i, m in enumerate([4.0, 5.0, 6.0], start=1):
        _write_run(runs_root, "control_00", i, heldout_mean=m,
                    history_rows=_probe_rows([(1, m / 2), (5, m)]))

    results_path = analyse.run_analysis(runs_root, out_dir, project_root=tmp_path)
    text = results_path.read_text(encoding="utf-8")

    assert "excluded from the secondary measure (no usable history)" in text
    assert "real/seed_2" in text


# --------------------------------------------------------------------------------------------
# 9. p-value formatting and Mann-Whitney method naming (fix #3)
# --------------------------------------------------------------------------------------------


def test_fmt_p_below_floor_prints_less_than():
    assert analyse._fmt_p(0.00001) == "p < 0.0001"
    assert analyse._fmt_p(0.0) == "p < 0.0001"


def test_fmt_p_above_floor_prints_equals():
    assert analyse._fmt_p(0.0234) == "p = 0.0234"
    assert analyse._fmt_p(0.5) == "p = 0.5000"


def test_results_md_states_exact_method_when_no_ties(tmp_path):
    real_means = [10.0, 12.0, 14.0]
    control_means = [1.0, 2.0, 3.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "(method: exact)" in text


def test_results_md_states_asymptotic_method_when_ties_present(tmp_path):
    # A ceiling effect makes ties across the pooled sample likely; here duplicate 5.0s appear
    # in both arms, forcing method="asymptotic" per the ticket's ties rule.
    real_means = [5.0, 5.0, 5.0]
    control_means = [5.0, 3.0, 5.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "(method: asymptotic (ties present))" in text


# --------------------------------------------------------------------------------------------
# 10. No verdict without enough finished runs (fix #4)
# --------------------------------------------------------------------------------------------


def test_no_verdict_at_zero_finished_runs(tmp_path):
    text = _run_full_analysis(tmp_path, [], [])
    assert "Not enough finished runs for a verdict (real: 0/30, control: 0/30)." in text
    assert "C2 supported" not in text
    assert "C2 fails" not in text
    assert "C2 is not supported" not in text


def test_no_verdict_at_two_finished_runs(tmp_path):
    real_means = [12.0, 14.0]
    control_means = [5.0, 6.0, 7.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "Not enough finished runs for a verdict (real: 2/30, control: 3/30)." in text


def test_verdict_printed_at_three_finished_runs_each_side(tmp_path):
    real_means = [20.0, 21.0, 22.0]
    control_means = [1.0, 2.0, 3.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "Not enough finished runs for a verdict" not in text


# --------------------------------------------------------------------------------------------
# 11. NaN / missing held-out mean (fix #5)
# --------------------------------------------------------------------------------------------


def test_nan_and_missing_mean_excluded_named_and_stats_from_rest(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "real", 1, heldout_mean=15.0)
    # seed 2: heldout.json has an explicit NaN mean.
    run_dir_2 = runs_root / "real" / "seed_2"
    run_dir_2.mkdir(parents=True)
    (run_dir_2 / "heldout.json").write_text(
        json.dumps({"run_seed": 2, "mean": float("nan")}), encoding="utf-8"
    )
    # seed 3: heldout.json is malformed -- no "mean" key at all (must not raise KeyError).
    run_dir_3 = runs_root / "real" / "seed_3"
    run_dir_3.mkdir(parents=True)
    (run_dir_3 / "heldout.json").write_text(json.dumps({"run_seed": 3}), encoding="utf-8")
    for i, m in enumerate([4.0, 5.0, 6.0], start=1):
        _write_run(runs_root, "control_00", i, heldout_mean=m)

    loaded = analyse.load_runs(runs_root)
    assert loaded.n_real_found == 3  # readable heldout.json -> "found", regardless of mean

    real_scores = loaded.real["heldout_mean"].to_numpy()
    control_scores = loaded.control["heldout_mean"].to_numpy()
    primary = analyse.analyse_primary(real_scores, control_scores)
    assert primary.n_real == 1  # only seed 1 has a usable mean
    assert primary.n_excluded_real == 2
    assert primary.median_real == pytest.approx(15.0)

    out_dir = tmp_path / "docs"
    results_path = analyse.run_analysis(runs_root, out_dir, project_root=tmp_path)
    text = results_path.read_text(encoding="utf-8")
    assert "real/seed_2" in text
    assert "real/seed_3" in text
    assert "NaN or missing held-out mean" in text


# --------------------------------------------------------------------------------------------
# 12. Verdict wording: both directions (fix: existing tests never asserted the real-wins case)
# --------------------------------------------------------------------------------------------


def test_results_md_real_wins_wording(tmp_path):
    real_means = [20.0, 21.0, 22.0, 19.0, 21.0]
    control_means = [2.0, 3.0, 1.0, 2.0, 3.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "Real outperforms control on held-out score (C2 supported)." in text


def test_results_md_control_wins_wording(tmp_path):
    real_means = [2.0, 3.0, 1.0, 2.0, 3.0]
    control_means = [20.0, 21.0, 22.0, 19.0, 21.0]
    text = _run_full_analysis(tmp_path, real_means, control_means)
    assert "Control outperforms real on held-out score" in text
    assert "C2 fails" in text


# --------------------------------------------------------------------------------------------
# 13. Figure-4 skip reasons must be distinguishable (fix #6)
# --------------------------------------------------------------------------------------------


def test_gf_habituation_skip_reason_no_replay_file(tmp_path):
    written, reason = analyse.fig_gf_habituation(tmp_path / "missing.json", tmp_path / "out.png")
    assert written is False
    assert reason == "no_replay_file"


def test_gf_habituation_skip_reason_not_real_connectome(tmp_path):
    replay_path = tmp_path / "best.json"
    replay_path.write_text(json.dumps({"meta": {"is_real_connectome": False}, "gf": {"counts": [[1, 2]]}}),
                            encoding="utf-8")
    written, reason = analyse.fig_gf_habituation(replay_path, tmp_path / "out.png")
    assert written is False
    assert reason == "not_real_connectome"


def test_gf_habituation_skip_reason_no_gf_block(tmp_path):
    replay_path = tmp_path / "best.json"
    replay_path.write_text(json.dumps({"meta": {"is_real_connectome": True}}), encoding="utf-8")
    written, reason = analyse.fig_gf_habituation(replay_path, tmp_path / "out.png")
    assert written is False
    assert reason == "no_gf_block"
