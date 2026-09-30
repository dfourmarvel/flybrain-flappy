# flybrain-flappy

A sub-circuit of a real fruit-fly brain plays Flappy Bird. The brain's wiring is taken from the
Janelia/Google **MaleCNS v1.0** connectome and is never changed. Only a 16-parameter interface
between the game and the brain is fitted.

**Demo:** https://dfourmarvel.github.io/flybrain-flappy/ — watch a recorded run, then play the
same level yourself.

## What was done

- 2,493 neurons on the path from the fly's looming-detector visual neurons (LC4, LPLC2) to its
  takeoff/escape descending neurons (DNp01 giant fiber, DNp02, DNp04, DNp06, DNp11), simulated
  as leaky integrate-and-fire neurons using the model of Shiu et al. 2024 (*Nature*).
- The game drives the looming neurons; a weighted readout of the escape neurons decides each flap.
  Only those interface weights are fitted (CMA-ES). Connection weights and signs stay frozen.
- 30 runs on the real wiring, 30 on degree-preserving shuffles of it (same neurons, same number
  of connections per neuron, same neurotransmitters, different partners). Every run is scored on
  20 levels it never saw during fitting.
- Outcome measures were fixed in [docs/PLAN.md](docs/PLAN.md) before any run on a shuffled
  network started.

## Results

Full write-up: [docs/RESULTS.md](docs/RESULTS.md). Post-hoc explanation:
[docs/EXPLORATORY.md](docs/EXPLORATORY.md).

- Real wiring: median held-out score 18.1 (of 22); 28 of 30 runs beat the best shuffled run.
- Shuffled wiring: 29 of 30 runs score 0; the best scores 0.05.
- **What the brain is doing:** the interface already works out whether the gap is above or
  below the bird and delivers it as *which side's* looming neurons are driven. A one-line rule
  using that same signal, with no brain at all, scores 20.45 — better than 28 of 30 real runs.
  So the circuit is not solving the game. Its contribution is carrying the left/right signal to
  the escape neurons without mixing the two sides, which the real wiring does and every shuffle
  fails to do. The shuffles also run about 3x more active overall, and no side-preserving
  shuffle was tested, so the result is specific to this circuit, interface and task.

Limitations are listed in [docs/RESULTS.md](docs/RESULTS.md#limitations).

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install -e .
.venv/Scripts/python -m flybrain.fetch          # ~560 MB of connectome data
.venv/Scripts/python -m flybrain.subcircuit     # extract the 2,493-neuron circuit
.venv/Scripts/python -m flybrain.control --n 30 # the shuffled controls
.venv/Scripts/python -m flybrain.train --all    # ~15 hours on 6 cores
.venv/Scripts/python -m flybrain.analyse        # writes docs/RESULTS.md
```

The per-run results from the published sweep are committed under `runs/`, so
`python -m flybrain.analyse` reproduces the write-up without retraining. The extracted
sub-circuit (`data/derived/`) is committed so the tests run on a clone.

## Data

MaleCNS v1.0 connectome, Janelia Research Campus and Google, licensed CC-BY. Source, file
checksums and citation: [docs/DATA.md](docs/DATA.md). Neuron model parameters: Shiu et al. 2024,
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
