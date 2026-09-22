# flybrain-flappy — Implementation Plan

A real fruit-fly brain (Janelia/Google **MaleCNS v1.0** connectome, Sept 2026) plays Flappy Bird.
The connectome is never modified. Only the interface between the game and the brain is learned.

**Spec-wins rule:** where this plan conflicts with the builder's judgment, this plan wins. Where this
plan is silent, take the simplest option consistent with what is written here and leave a
`# SPEC-GAP:` comment naming the gap. Do not invent biological parameters — every constant that
describes a neuron must be copied from a cited source, and its source named in a comment.

---

## 1. What this is

Flappy Bird is played by a leaky integrate-and-fire (LIF) simulation of a sub-circuit of the real
adult male *Drosophila* central nervous system. Pipes are converted into firing of the fly's real
looming-detector visual neurons. A flap happens when the fly's real takeoff descending neurons
(the escape pathway, including the giant fiber DNp01) cross a learned threshold.

Two claims are being tested:

- **C1 (demo):** the fly's own looming→escape circuit, unmodified, can play the game above chance.
- **C2 (experiment):** the *real* wiring plays better than a **degree-preserving shuffled** control
  network of identical size, degree sequence and neurotransmitter composition.

If C2 fails, that is a publishable-quality negative result and the write-up says so. **Do not tune
the experiment until C2 passes.** The control protocol is fixed before any result is seen.

---

## 2. Locked decisions

| # | Decision | Date |
|---|---|---|
| L1 | **Frozen brain.** Connectome edges, weights and signs are never altered by learning. | 2026-09-17 |
| L2 | **Learned interface only.** Learning touches the game→neuron input mapping and the neuron→flap readout, nothing else. | 2026-09-17 |
| L3 | **Input = looming neurons** (LC4, LPLC2 classes). Not raw pixels, not search-chosen neurons. | 2026-09-17 |
| L4 | **Output = continuous run, wider readout** over takeoff-related descending neurons. Brain state is never reset mid-episode. Giant-fiber-only behaviour is *recorded* as a secondary result, not used to drive the game. | 2026-09-17 |
| L5 | **Rigour: 30 vs 30.** 30 independent training seeds on the real connectome, 30 on independently shuffled controls. | 2026-09-17 |
| L6 | **Public repo + GitHub Pages** demo. Repo name `flybrain-flappy`. | 2026-09-17 |
| L7 | **Python trains offline; the browser replays.** The web demo replays a recorded real episode. It never claims to simulate live. | 2026-09-17 |
| L8 | Tier: **Light** — single planning doc, no database, no auth, no user data. | 2026-09-17 |

---

## 3. Explicitly NOT building

- No live brain simulation in the browser. (Replay only — see L7.)
- No learning inside the connectome: no Hebbian rules, no weight updates, no fine-tuning.
- No whole-brain simulation. A sub-circuit only (Step 2 defines it).
- No raw-pixel vision, no compound-eye model, no photoreceptor stage.
- No `flybody` / MuJoCo physical fly body. Wrong project, enormous scope.
- No leaderboard, no accounts, no backend, no analytics.
- No comparison against a conventional reinforcement-learning agent. Different claim, later phase if ever.
- No claim that the fly "learned" anything. The brain is frozen; the wording everywhere is
  "the interface was fitted", never "the fly was trained".

---

## 4. Stack

| Thing | Choice | Pin |
|---|---|---|
| Language | Python | 3.11 (3.11.15 present on this machine) |
| Arrays | `numpy` | `2.3.3` |
| Sparse matrices | `scipy` | `1.16.2` |
| Tables | `pandas` | `2.3.2` |
| Optimiser | `cma` (CMA-ES) | `4.0.0` |
| Stats | `scipy.stats` (from scipy above) | — |
| Plots | `matplotlib` | `3.10.6` |
| Tests | `pytest` | `8.4.2` |
| Data pull | `requests` | `2.32.5` |
| Web demo | vanilla HTML/CSS/JS + `<canvas>`, no framework, no build step | — |
| Hosting | GitHub Pages from `/docs-site` | — |

Env setup (literal):

```bash
cd flybrain-flappy
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/pip install numpy==2.3.3 scipy==1.16.2 pandas==2.3.2 cma==4.0.0 matplotlib==3.10.6 pytest==8.4.2 requests==2.32.5
.venv/Scripts/pip freeze > requirements.txt
```

`.gitignore` must contain at minimum: `.venv/`, `data/raw/`, `data/derived/`, `runs/`, `__pycache__/`,
`*.pyc`, `.env`, `.env.local`. **Raw connectome files are never committed** (size), only the derived
sub-circuit (Step 2) if it is under 20 MB.

---

## 5. Repo layout

```
flybrain-flappy/
  docs/PLAN.md              <- this file
  docs/RESULTS.md           <- written in Step 8
  docs-site/                <- GitHub Pages demo (index.html, app.js, style.css, replay/*.json)
  data/raw/                 <- gitignored: downloaded connectome CSVs
  data/derived/             <- gitignored: subcircuit.npz, neurons.parquet
  src/flybrain/
    __init__.py
    fetch.py                <- Step 1
    subcircuit.py           <- Step 2
    lif.py                  <- Step 3
    game.py                 <- Step 4
    interface.py            <- Step 5  (input mapping + readout)
    train.py                <- Step 6
    control.py              <- Step 7  (degree-preserving shuffle)
    analyse.py              <- Step 8
    record.py               <- Step 9  (episode -> replay JSON)
  tests/
  runs/                     <- gitignored: training outputs
  README.md
```

---

## 6. Build steps

Each step ends with the repo in a runnable state and every acceptance box tickable by someone who
did not write the step.

### Step 1 — Data acquisition (`fetch.py`)

Download MaleCNS v1.0 connectome data. Primary route is the **public bulk download**, which needs no
login: the flat connectome lives under `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`,
also reachable over plain HTTPS at `https://storage.googleapis.com/flyem-male-cns/v1.0/...`.

**Verified 2026-09-22 (lead spike).** Files are Apache Feather (read with `pandas.read_feather`,
needs `pyarrow`, pinned in `requirements.txt`). Download exactly these three, nothing else (the
syn-partners/syn-points/tbar files are 3–13 GB each and not needed):

| File | Size | Use |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14 MB | 211,577 rows; key cols `bodyId`, `type`, `instance`, `somaSide`, `superclass`, `class`, `status` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43 MB | key cols `body`, `cell_type`, `consensus_nt`, `predicted_nt_confidence` |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502 MB | edge table; record its real column names in `docs/DATA.md` |

Seed type labels confirmed in the `type` column: `LC4` (126), `LPLC2` (185), `DNp01`, `DNp02`,
`DNp04`, `DNp06`, `DNp11` (2 each, left and right). Use `consensus_nt` for edge signs in Step 2.

Fallback if the bulk files are unusable: `neuprint-python` against dataset `male-cns:v1.0`, which
**requires a personal token Daniel must create himself** at neuPrint (Google login). The builder must
not attempt to create an account or enter credentials — stop and report instead.

- [ ] `python -m flybrain.fetch` downloads to `data/raw/` and is idempotent (re-running skips existing files).
- [ ] `docs/DATA.md` records: source URLs, file sizes, sha256 of each file, column names, download date, licence/citation.
- [ ] Row counts are printed and sanity-checked: order ~1.4 x 10^5 neurons, ~10^7–10^8 synapse rows.
- [ ] Nothing in `data/` is tracked by git (`git status` clean after download).

### Step 2 — Sub-circuit extraction (`subcircuit.py`)

Build the frozen network the simulation runs on.

Seeds:
- **Input seeds:** all neurons whose cell type matches the looming-sensitive lobula columnar classes
  `LC4` and `LPLC2` (match on the type field, both hemispheres).
- **Output seeds:** takeoff-related descending neurons — `DNp01` (the giant fiber), plus `DNp02`,
  `DNp04`, `DNp06`, `DNp11`. Include every DN whose type matches these names.

Expansion: take the induced subgraph on all neurons lying on a path of at most **3 hops** from an
input seed to an output seed, keeping only edges with synapse weight >= **5** (the conventional
noise floor for this data; record the count at thresholds 3/5/10 in `docs/DATA.md`).

Signing: each edge's sign comes from the **presynaptic** neuron's predicted neurotransmitter
(Eckstein et al. 2024): acetylcholine → excitatory (+1); GABA and glutamate → inhibitory (−1);
dopamine, serotonin, octopamine → **excluded from the fast simulation** and logged as a count.
Unknown/low-confidence predictions → excluded and counted.

Output: `data/derived/subcircuit.npz` (scipy CSR sparse signed weight matrix) plus
`data/derived/neurons.parquet` (row index → body id, type, nt, role flag: `input_seed` /
`output_seed` / `interneuron`).

**Size gate:** if the subgraph exceeds **8,000 neurons**, reduce to 2 hops and record that. If it is
under 300 neurons, widen to 4 hops. The final number goes in `docs/DATA.md`.

- [ ] `python -m flybrain.subcircuit` writes both derived files and prints: neuron count, edge count, % excitatory, % inhibitory, count excluded by neurotransmitter, count of input seeds, count of output seeds.
- [ ] Every output seed is reachable from at least one input seed; the script fails loudly if not.
- [ ] Matrix is signed, sparse, and float32; row/col order matches `neurons.parquet` exactly (asserted in a test).
- [ ] `tests/test_subcircuit.py` verifies: no self-loops, no NaNs, sign of a hand-picked known-GABAergic neuron's outgoing edges is −1.

### Step 3 — LIF simulator (`lif.py`)

Reimplement the Shiu, Sterne, Spiller et al. 2024 (*Nature* 634:210–219) leaky integrate-and-fire
model in NumPy. Reference implementations to read first (do not vendor their code, do read their
parameter tables): `philshiu/Drosophila_brain_model` (the paper's own code), and the dependency-light
NumPy versions `vshapenko/flypoke` and `cfdgasman/flybrain-explorer`.

**Every neuron constant — resting potential, threshold, membrane time constant, refractory period,
synaptic time constant, per-synapse voltage step — is copied from the paper/repo and cited inline.
Inventing a value is a spec violation.** Record the full parameter table in `docs/DATA.md`.

Required design: the state update is a **batched** sparse mat-vec, shape `(n_candidates, n_neurons)`,
so an entire CMA-ES population steps in one call. This is what makes Step 6 affordable.

**Performance gate:** benchmark on this laptop and record it. The simulator must reach
**>= 2,000 simulated seconds of brain time per minute of wall clock** at population size 32.
If it does not: first raise `dt` toward 0.5 ms, then shrink the sub-circuit (Step 2 gate), then
stop and report. Do not proceed to Step 6 with a slower simulator — 60 training runs will not finish.

- [ ] `tests/test_lif.py`: a single isolated neuron driven by constant current fires at the analytically expected rate (±5%).
- [ ] With zero input, the network is silent (no spontaneous spiking) for 5 simulated seconds.
- [ ] Stimulating all LC4 inputs produces measurable spiking in DNp01 within 100 ms — i.e. the looming→escape pathway conducts. If it does not, Step 2's subgraph is wrong; fix that before continuing.
- [ ] Batched and single-candidate paths give bit-identical results for the same seed (asserted in a test).
- [ ] `docs/DATA.md` contains the benchmark number and the parameter table with citations.

### Step 4 — Flappy Bird (`game.py`)

Pure-Python, deterministic, seeded, headless. No pygame, no rendering in the training loop.

Fixed rules (do not "improve" these): bird x fixed; gravity and flap impulse constant; pipes at fixed
horizontal spacing with a fixed gap height and randomly placed gap centre drawn from the episode seed;
one point per pipe passed; episode ends on collision or at a **1,500-frame cap**. Frame duration is
**25 ms of simulated brain time** (40 fps). All constants live in one `GameConfig` dataclass.

- [ ] `tests/test_game.py`: same seed + same action sequence → identical score and identical frame log.
- [ ] A do-nothing agent dies within ~1 second; a flap-every-frame agent flies off the top. Both asserted.
- [ ] The frame log records, per frame: bird y, bird velocity, next pipe x distance, next pipe gap centre, score, alive flag.

### Step 5 — Interface (`interface.py`)

The only learned part. Parameters (all real-valued, CMA-ES optimises them):

**Input mapping (game → looming neurons).** Per frame, compute a looming drive from the next pipe's
distance and the vertical offset between bird and gap centre, then convert it to an injected current
per input-seed neuron. Parameters: base gain, distance sensitivity (how sharply drive rises as the
pipe nears), an offset-sensitivity term, and a baseline. The vertical offset splits the input seeds
into two groups by soma side (left/right hemisphere) so above-gap and below-gap situations drive
different neurons — this is the only structural choice in the mapping, and it is fixed, not learned.

**Readout (descending neurons → flap).** A weighted sum of the recent spike counts of the output-seed
descending neurons (exponentially-decaying spike trace, one time constant parameter shared across
neurons), plus a bias, compared against zero. Above zero → flap this frame. One weight per output-seed
neuron. **Weights are on the readout, not on the connectome.**

Total parameter count must be **< 100**. Record it. If the sub-circuit yields more than ~60 output
seeds, group them by cell type and learn one weight per type.

Brain state persists across frames for the whole episode (L4). The only reset is at episode start.

- [ ] Parameter vector ↔ named-parameter round-trip is tested (`to_vector` / `from_vector`).
- [ ] With a fixed random parameter vector and a fixed seed, an episode is exactly reproducible.
- [ ] A recorded secondary channel logs DNp01-only spike times every episode, unused by the flap decision (for the giant-fiber habituation result).

### Step 6 — Training (`train.py`)

CMA-ES over the Step 5 parameter vector. Fitness = mean score over **5 fixed evaluation seeds**
(the same 5 for every candidate and every run — no seed lottery), with a small survival-time bonus to
break ties among zero-score candidates. Population 32, budget **300 generations or 2 hours wall clock
per run, whichever comes first**. Each run writes `runs/<network>/<seed>/` containing: the config, the
best parameter vector, per-generation fitness history, and the final evaluation on **20 held-out seeds
never used during training**.

Held-out evaluation is the only number reported anywhere.

- [ ] One run completes end to end and writes all four outputs.
- [ ] Re-running the same seed reproduces the same best fitness (assert on the first 5 generations).
- [ ] Wall-clock per run is recorded, and the projected total for 60 runs is printed. If that exceeds ~20 hours, stop and report before launching the full set.
- [ ] Training never writes to `subcircuit.npz` — asserted by checking its sha256 before and after.

### Step 7 — Control networks (`control.py`)

Degree-preserving shuffle: rewire edges while preserving each neuron's in-degree and out-degree, the
weight multiset, and each neuron's neurotransmitter (so excitatory/inhibitory proportions are
identical). Input-seed and output-seed role labels stay on the same neuron indices, so the control has
the same number of input and output neurons in the same positions — only the wiring between them is
scrambled. Use a double-edge-swap approach with at least `10 × n_edges` successful swaps, seeded.

- [ ] `tests/test_control.py`: in-degree and out-degree sequences are exactly preserved; edge multiset size unchanged; at least 95% of edges actually changed.
- [ ] 30 independent controls generate with 30 different seeds and are stored under `data/derived/controls/`.
- [ ] A control network is never used to produce the demo replay.

### Step 8 — Analysis (`analyse.py`) and `docs/RESULTS.md`

Compare held-out scores: 30 real vs 30 control. Report median and interquartile range for each, a
Mann–Whitney U test with the exact p-value, and a rank-biserial effect size. Plot: paired strip plot
of the 60 held-out scores, plus mean fitness curves with an interquartile band.

Also report the secondary result: DNp01 firing across an episode, showing whether the giant fiber
habituates under repeated looming.

`docs/RESULTS.md` states the outcome plainly, including if the real connectome does **not** beat the
control, and lists the limitations honestly: sub-circuit not whole brain, predicted rather than
measured neurotransmitters, no neuromodulation, no learning, synapse counts as a proxy for strength,
and the fact that the interface was fitted specifically for this task.

- [ ] Figures render to `docs-site/figures/` at 2x resolution and are legible on a phone.
- [ ] Every number in `RESULTS.md` is produced by the script, not typed by hand; the script prints them in the same order the doc uses.
- [ ] Limitations section exists and names all six items above.

### Step 9 — Replay recorder (`record.py`)

Re-run the best real-connectome parameter vector on a held-out seed and record a replay JSON:
per frame — bird y, pipe positions, score, flap flag, plus a downsampled activity vector (spike counts
per frame for up to ~200 neurons, chosen as the input seeds, output seeds, and the highest-traffic
interneurons). Include the metadata block: connectome version, neuron count, edge count, seed,
held-out score. Cap the file at **2 MB**; downsample activity further if needed.

- [ ] `docs-site/replay/best.json` exists, is under 2 MB, and validates against a documented schema in the same file's `meta` block.
- [ ] The recorded score matches the score Step 6 reported for that seed.

### Step 10 — Web demo (`docs-site/`)

Single static page, no build step. Canvas replay of `best.json`: the game on the left, a live activity
panel on the right showing the input neurons, the descending neurons and a flap indicator, all driven
from the recorded data. Controls: play/pause, speed, restart. A short explainer above the fold: what
the connectome is, what is frozen, what was fitted, and a prominent line stating this is a **replay of
a real simulated run**, not a live simulation.

- [ ] Works on a 375px-wide phone with no horizontal scroll; canvas scales.
- [ ] Keyboard accessible (play/pause reachable and operable by keyboard), buttons have visible focus, and the activity panel is not the only way to understand the page.
- [ ] `prefers-reduced-motion` is respected: the replay does not autoplay when it is set.
- [ ] No external scripts, no fonts from a CDN, no analytics.
- [ ] Deployed to GitHub Pages and verified live in a real browser, not just locally.

---

## 7. Model routing (fable-foreman seats)

| Step | Seat | Why |
|---|---|---|
| 1 — Data acquisition | WORKHORSE (sonnet) | Unknown file formats; needs judgment when the bucket layout differs from the assumption. |
| 2 — Sub-circuit | WORKHORSE-DEEP (sonnet) | Biology-sensitive selection; wrong seeds silently ruin everything downstream. |
| 3 — LIF simulator | WORKHORSE-DEEP (sonnet) | Must reproduce a published model faithfully and hit a performance gate. |
| 4 — Game | WORKHORSE (haiku) | Fully specified, deterministic, test-vector driven — mechanical. |
| 5 — Interface | WORKHORSE-DEEP (sonnet) | Small but subtle; the parameterisation decides whether training can work at all. |
| 6 — Training | WORKHORSE (sonnet) | Standard CMA-ES wiring, clear contract. |
| 7 — Controls | WORKHORSE (haiku) | Textbook degree-preserving swap with exact tests. |
| 8 — Analysis | WORKHORSE (sonnet) | Stats plus prose that must not overclaim. |
| 9 — Recorder | WORKHORSE (haiku) | Serialisation against a stated schema. |
| 10 — Web demo | WORKHORSE-DEEP (sonnet) | Visual quality is the deliverable; a11y and mobile gates apply. |
| Any change to claims, controls or biology | FRONTIER (lead) | Judgment stays with the lead. |

Every step is blind-verified by `foreman-verifier` against its acceptance checklist before the next
step starts.

---

## 8. Risks and stop conditions

| Risk | Stop condition / mitigation |
|---|---|
| The looming→escape pathway does not conduct in the sub-circuit | Step 3 gate fails → revisit Step 2 seeds and hop count before any training. |
| Simulation too slow for 60 runs | Step 3 performance gate and Step 6 projection. Report before burning hours. |
| Giant fiber habituates and the fly stops flapping | Expected (L4). Wider readout is the mitigation; the habituation itself is a reported result. |
| Real connectome does not beat the shuffled control | Report it. Do not re-tune, do not re-pick controls, do not quietly widen the readout afterwards. |
| Data licensing for redistribution | Step 1 records the licence. Derived sub-circuit is only committed if the licence allows; otherwise the repo ships the extraction script instead. |
| Overclaiming in the write-up | Wording rules in section 3 and the limitations list in Step 8. |

## 9. What Daniel owes

- Nothing up front — the bulk data needs no account.
- Only if Step 1's fallback route is needed: a personal neuPrint token, created by him, never by an agent.
- A decision on whether `docs/RESULTS.md` becomes a LinkedIn post once the numbers exist.
