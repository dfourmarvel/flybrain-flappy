# Data (Step 1)

## Sources & licence/citation

- Bulk download bucket: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, also served over
  plain HTTPS at `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
  Public, no login.
- Project pages: https://male-cns.janelia.org/ and https://male-cns.janelia.org/download/
- **Licence: CC-BY**, stated verbatim on both pages: "The Male CNS dataset is licensed under CC-BY."
- Collaboration credit (from the project page): "This project is a collaboration between FlyEM (HHMI
  Janelia), the University of Cambridge (Dept. of Zoology), the MRC Laboratory of Molecular Biology,
  and Google Research."
- Paper: the site links a bioRxiv preprint and reports "2026-09-03 — MaleCNS paper published!" but the
  exact title/author list/DOI were not extracted from the page text in this pass —
  **UNVERIFIED — check https://male-cns.janelia.org/overview (BioRxiv Preprint link) for the precise
  citation before redistributing anything that needs a formal citation.**
- Neurotransmitter sign convention source: Eckstein et al. 2024 (per `docs/PLAN.md` Step 2), used
  downstream, not re-verified here.

## Files

Downloaded 2026-09-22 via `python -m flybrain.fetch` to `data/raw/` (gitignored).

| File | Size (bytes) | SHA-256 | Rows |
|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | 211,577 |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | 1,835,518 |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502,169,298 | `5c536423a62a688e59e7b441f9c04d6272c9a1f017e35814cf561f8c275d9e9e` | 25,568,639 |

Source URL for each: `<BASE_URL>/<file name>` where
`BASE_URL = https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.

All three numbers above (size, sha256, row count) are computed by `flybrain.fetch.summarise()`,
never hand-typed. Reproduce with `.venv/Scripts/python -m flybrain.fetch`.

## Schemas

### `body-annotations-male-cns-v1.0-minconf-0.5.feather` (211,577 rows)

| column | dtype |
|---|---|
| assignedOlHex1 | float64 |
| assignedOlHex2 | float64 |
| bodyId | int64 |
| flywireType | object |
| group | float64 |
| instance | object |
| somaSide | object |
| statusLabel | category |
| superclass | object |
| type | object |
| vfbId | object |
| hemibrainType | object |
| itoleeHl | object |
| supertype | object |
| birthtime | object |
| mancBodyid | float64 |
| mancGroup | float64 |
| mancType | object |
| subclass | object |
| synonyms | object |
| class | object |
| rootSide | object |
| somaNeuromere | object |
| trumanHl | object |
| dimorphism | object |
| matchingNotes | object |
| entryNerve | object |
| mancSerial | float64 |
| mcnsSerial | float64 |
| serialMotif | object |
| fruDsx | object |
| exitNerve | object |
| receptorType | object |
| somaLocation | object |
| tosomaLocation | object |
| status | object |

Note: `docs/PLAN.md` names `bodyId`, `type`, `instance`, `somaSide`, `superclass`, `class`, `status`
as the key columns — all present as listed above.

### `body-neurotransmitters-male-cns-v1.0.feather` (1,835,518 rows)

| column | dtype |
|---|---|
| body | int64 |
| cell_type | object |
| total_nt_predictions | int32 |
| predicted_nt_confidence | float64 |
| predicted_nt | object |
| ground_truth | object |
| celltype_total_nt_predictions | int32 |
| celltype_predicted_nt | object |
| celltype_predicted_nt_confidence | float64 |
| consensus_nt | object |

Note: one row per body (`body` is unique, 1,835,518 rows), including unannotated fragments —
which is why `consensus_nt` is `unclear` for 1,671,117 bodies overall. Every `body_pre` in the
weights file is present here. **Within the Step 2 sub-circuit (weight >= 5, 3 hops; 2,493 neurons)
coverage is near-complete:** consensus_nt = acetylcholine 1,630, GABA 642, glutamate 201,
unclear 13, dopamine 4, octopamine 3. By synapse weight: ACh 60.3%, GABA 30.8%, glutamate 7.4%,
unclear 0.7%, modulatory 0.8%. Median `predicted_nt_confidence` 0.94. Histamine (8,024 bodies
overall) does not occur in the sub-circuit; if it ever does, sign it inhibitory (fly histamine
receptors are chloride channels) and log the count. (Lead spike, 2026-09-22.)

### `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` (25,568,639 rows)

| column | dtype |
|---|---|
| body_pre | int64 |
| body_post | int64 |
| weight | int64 |
| type_pre | object |
| type_post | object |

Edge table: one row per (presynaptic body, postsynaptic body) pair with aggregate synapse `weight`.
Actual column names differ from the `pre`/`post` guess in `docs/PLAN.md` Step 1 — they are
`body_pre` / `body_post` / `weight`.

**Weight column stats** (from `flybrain.fetch.summarise()`):

| stat | value |
|---|---|
| min | 1 |
| median | 2.0 |
| max | 2591 |
| distinct `body_pre` | 163,663 |
| distinct `body_post` | 164,607 |

## Sanity checks

- Annotation file: 211,577 rows — order of magnitude matches the ~1.4x10^5 neuron estimate in
  `docs/PLAN.md` (this dataset has more entries than that rough estimate, which is fine; it is a
  full-CNS release, not just brain).
- Weights file: 25,568,639 edge rows — within the 10^7-10^8 synapse-row range named in the plan.
- All three files re-downloaded/verified idempotently: re-running `python -m flybrain.fetch` after
  all three exist prints `skip` for each (size match), no re-download.
- `git status` shows nothing under `data/` (gitignored, confirmed below).

## Sub-circuit (Step 2)

Seeds (from `body-annotations` `type` column, both hemispheres): input seeds `LC4` + `LPLC2` =
311 neurons; output seeds `DNp01`/`DNp02`/`DNp04`/`DNp06`/`DNp11` = 10 neurons.

Method: filter edges to `weight >= threshold`; build the directed graph over every body id
appearing in those edges; forward BFS from the input seeds gives `d_f`, backward BFS (on the
transposed graph) from the output seeds gives `d_b`; keep neurons with `d_f + d_b <= hops`; the
network is the induced subgraph (all threshold-passing edges with both endpoints kept).
Self-loops (101 rows / 54 at weight>=3 in the raw weights table) are dropped before graph
construction. None of them involve neurons in the chosen th5/hops3 network, so the drop
removes no connection from it (verified: 2,493 neurons / 94,702 edges with or without the drop).
All nine rows below reached all 10 output seeds. Produced by
`.venv/Scripts/python -m flybrain.subcircuit --sweep`:

| threshold | hops | neurons | edges | all 10 outputs reached |
|---|---|---|---|---|
| 3 | 2 | 804 | 41,488 | yes |
| 3 | 3 | 4,014 | 207,837 | yes |
| 3 | 4 | 52,395 | 3,643,894 | yes |
| 5 | 2 | 596 | 19,556 | yes |
| 5 | 3 | 2,493 | 94,702 | yes |
| 5 | 4 | 25,110 | 1,180,825 | yes |
| 10 | 2 | 451 | 6,458 | yes |
| 10 | 3 | 1,531 | 31,457 | yes |
| 10 | 4 | 8,759 | 244,337 | yes |

**Chosen parameters: threshold = 5, hops = 3 → 2,493 neurons, 94,702 edges.** Within the size gate
(300–8,000 neurons); no gate adjustment needed.

Signing (Eckstein et al. 2024, presynaptic neuron's `consensus_nt` sets the sign; raw values are
lower-case, e.g. `"gaba"`): acetylcholine → +1; gaba, glutamate, histamine → −1; dopamine,
serotonin, octopamine, unclear → excluded (neuron stays as a node, its outgoing edges are dropped
from the signed matrix and counted). No histamine or serotonin neurons occur in this sub-circuit.

Summary, produced by `.venv/Scripts/python -m flybrain.subcircuit` (default threshold=5, hops=3):

| metric | value |
|---|---|
| neurons | 2,493 |
| edges kept (induced subgraph, before NT exclusion) | 94,702 |
| edges included in signed matrix | 92,825 |
| edges excluded by neurotransmitter | 1,877 |
| % excitatory by edge count | 64.3% |
| % inhibitory by edge count | 35.7% |
| % excitatory by synapse weight | 61.2% |
| % inhibitory by synapse weight | 38.8% |
| input seeds in network | 311 |
| output seeds in network | 10 |
| matrix nnz | 92,825 |

`consensus_nt` within the 2,493 kept neurons: acetylcholine 1,630, gaba 642, glutamate 201,
unclear 13, dopamine 4, octopamine 3 (sums to 2,493).

Matrix convention: `W[pre, post] = sign(pre) * synapse_count`, `float32`, scipy CSR, shape
`(2493, 2493)`. Row/col order matches `neurons.parquet` row order exactly: input seeds first
(sorted by `bodyId`), then output seeds (sorted by `bodyId`), then interneurons (sorted by
`bodyId`). Step 3 applies `w_syn = 0.275 mV` itself; it is **not** baked into this matrix.

`neurons.parquet` columns: `idx`, `bodyId`, `type`, `somaSide`, `consensus_nt`, `sign` (+1/−1/0,
0 = excluded neurotransmitter), `role` (`input_seed` / `output_seed` / `interneuron`).

`data/derived/subcircuit.npz`: 234,597 bytes (0.22 MB, well under the 20 MB commit gate);
sha256 `2794848d4744f323ca5036f1b7ecdc9bc977ba0b6a51163caca90a78ea2c87c4`.

## LIF parameters & benchmark (Step 3)

_(to be filled in Step 3)_
