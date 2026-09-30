# Results

*Generated entirely by `python -m flybrain.analyse` -- every number below is
computed by the script, none is typed by hand.*

## What was tested

- **C1 (demo):** does the fly's own looming-to-escape circuit, unmodified, play Flappy Bird above chance?
- **C2 (experiment):** does the *real* wiring of that circuit play better than a degree-preserving shuffled control network of identical size, degree sequence and neurotransmitter composition?

## How

30/30 real-connectome runs and 30/30 shuffled-control runs (PLAN L5: 30 independent training seeds per arm) were trained by CMA-ES over the 16-parameter game-to-neuron interface (PLAN Step 5), then each run's best parameter vector was scored on 20 held-out seeds never used during training. This report compares those held-out scores between arms.

## Pre-registered measures

Fixed 2026-09-22, before any control network was trained (PLAN section 1); neither measure nor its threshold changed after seeing results.

- **Primary:** held-out mean score per run (mean over 20 held-out seeds), 30 real vs 30 control.
- **Secondary:** generations to competence -- first generation whose probe_mean >= 10, censored (as "not reached") when a run never reaches it.

## Results

Runs: found 30/30 real runs, 30/30 control runs finished.

### C1 -- does the real circuit play above chance?

29/30 real-connectome runs score above 0 on unseen levels (best run: held-out mean 21.00 of a possible 22). A brain-free interface scores 0: with the readout weights zeroed, or with the visual input switched off, the bird never passes a pipe. **C1 supported.**

### Primary -- held-out mean score

| | real | control |
|---|---|---|
| n | 30 | 30 |
| median | 18.125 | 0.000 |
| IQR | [8.412, 19.775] | [0.000, 0.000] |
| mean | 14.040 | 0.002 |
| at ceiling (22) | 0/30 | 0/30 |

Difference in medians (real - control): 18.125.

Mann-Whitney U = 884.000, two-sided p < 0.0001 (method: asymptotic (ties present)). Rank-biserial correlation = 0.964 (real ranks higher than control).

Real outperforms control on held-out score (C2 supported). This is conditional on the interface: it encodes above/below the gap as a left/right input split, so the task needs a network that keeps the two sides apart -- see [EXPLORATORY.md](EXPLORATORY.md) for the evidence that the real wiring does and the shuffled wiring does not.

### Secondary -- generations to competence (censored)

| | real | control |
|---|---|---|
| n | 30 | 30 |
| median generations (Kaplan-Meier) | 30.000 | not reached |
| censored (never reached) | 8/30 | 30/30 |

Log-rank chi-square = 35.701, p < 0.0001.

An exploratory (post-hoc, not pre-registered) analysis of *why* the shuffled networks fail is in [EXPLORATORY.md](EXPLORATORY.md).

### Figures

1. ![Held-out score by network](../docs-site/figures/primary_strip.png)
2. ![Mean fitness across generations](../docs-site/figures/fitness_curves.png)
3. ![Probe-score curves](../docs-site/figures/probe_curves.png)
4. ![Giant-fiber (DNp01) firing across one episode](../docs-site/figures/gf_habituation.png)

   Figure 4 was planned to show whether the giant fiber habituates under repeated looming. It cannot: the model has no adaptation mechanism (see Limitations), so the steady firing here says nothing about habituation in the real fly.

## Limitations

- This is a sub-circuit of 2,493 neurons on the path from the looming-detector inputs to the takeoff descending-neuron outputs -- not the whole *Drosophila* brain.
- Edge signs come from predicted, not measured, neurotransmitters (Eckstein et al. 2024 confidence-scored predictions).
- No neuromodulation: dopamine, serotonin and octopamine edges are excluded from the fast simulation entirely (see docs/DATA.md for the excluded-edge count), so no reward or arousal signalling is modelled.
- No learning happens inside the connectome -- weights, signs and wiring are frozen (PLAN L1). Only the 16-parameter game<->neuron interface (Step 5) was fitted.
- Synapse counts are used as a proxy for connection strength, not a measured physiological weight.
- The dt=0.2 ms simulation step (vs. the paper's 0.1 ms) carries a measured timing bias: <= 1.8% for 8 of 10 output neurons, and +11.9% for DNp06 L (docs/DATA.md).
- The left/right split of input seeds for above-gap/below-gap drive is an interface convention with no biological meaning -- left/right is really visual-field side.
- The interface (Step 5) was fitted specifically for this task; it is not a general-purpose readout of the connectome.
- The neuron model has no adaptation or short-term synaptic plasticity, so no neuron -- the giant fiber included -- can habituate. The giant fiber's sustained firing in figure 4 is a property of the model, not a prediction about the real fly, whose giant fiber fires single spikes to trigger an escape.

