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

Note: one row per body (`body` is unique, 1,835,518 rows), including unannotated fragments �
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

_(to be filled in Step 2)_

## LIF parameters & benchmark (Step 3)

_(to be filled in Step 3)_
