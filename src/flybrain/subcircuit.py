"""Step 2 -- sub-circuit extraction.

Builds the frozen looming -> escape network the rest of the pipeline simulates.

Method (locked by the lead's 2026-09-22 spike, see docs/PLAN.md Step 2 / docs/DATA.md):
  1. Filter the weights table to edges with weight >= --threshold. The directed graph's
     node set is every body id appearing as body_pre or body_post in those edges.
  2. Forward BFS from the input-seed set (LC4 + LPLC2) gives each node a distance d_f.
     Backward BFS from the output-seed set (DNp01/02/04/06/11), i.e. forward BFS on the
     transposed graph, gives each node a distance d_b.
  3. Keep nodes with d_f + d_b <= --hops (both distances must be defined). The network is
     the induced subgraph: all threshold-passing edges with both endpoints kept.

Signing (Eckstein et al. 2024 predicted neurotransmitter, presynaptic neuron sets the sign):
  acetylcholine -> +1; GABA, glutamate, histamine -> -1;
  dopamine / serotonin / octopamine / unclear (or anything else unexpected) -> excluded:
  the neuron stays as a node, but its OUTGOING edges are dropped from the signed matrix
  and counted separately.

Matrix convention: W[pre, post] = sign(pre) * synapse_count, float32, scipy CSR,
shape (N, N). Row/col order matches neurons.parquet row order exactly. Step 3 applies
w_syn = 0.275 mV itself -- it is NOT baked into this matrix.

Self-loops (body_pre == body_post, 101 rows / 54 at weight>=3 in the raw weights table)
are dropped before graph construction -- an autapse is not part of the described BFS
method and the sub-circuit must contain none (tested).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DERIVED_DIR = Path(__file__).resolve().parents[2] / "data" / "derived"

ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"

INPUT_TYPES = ("LC4", "LPLC2")
OUTPUT_TYPES = ("DNp01", "DNp02", "DNp04", "DNp06", "DNp11")

# consensus_nt values are lower-case in the raw table (e.g. "gaba", not "GABA")
EXCITATORY_NT = {"acetylcholine"}
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}
# dopamine / serotonin / octopamine / unclear, and anything unrecognised, are excluded

MIN_LOAD_WEIGHT = 3  # cheapest sweep threshold: filter at load time to cut memory early

DEFAULT_THRESHOLD = 5
DEFAULT_HOPS = 3
SWEEP_THRESHOLDS = (3, 5, 10)
SWEEP_HOPS = (2, 3, 4)


def load_weights(min_weight: int = MIN_LOAD_WEIGHT) -> pd.DataFrame:
    """Load body_pre/body_post/weight only, filtered early, self-loops dropped."""
    df = pd.read_feather(RAW_DIR / WEIGHTS_FILE, columns=["body_pre", "body_post", "weight"])
    df = df[df["weight"] >= min_weight]
    df = df[df["body_pre"] != df["body_post"]]
    return df.reset_index(drop=True)


def load_annotations() -> pd.DataFrame:
    return pd.read_feather(RAW_DIR / ANNOTATIONS_FILE, columns=["bodyId", "type", "somaSide"])


def load_nt() -> pd.DataFrame:
    return pd.read_feather(RAW_DIR / NT_FILE, columns=["body", "consensus_nt", "predicted_nt_confidence"])


def seed_ids(annotations: pd.DataFrame, types: tuple[str, ...]) -> np.ndarray:
    return np.sort(annotations.loc[annotations["type"].isin(types), "bodyId"].unique())


def _bfs_multi_source(seed_local: np.ndarray, adj: sparse.csr_matrix, hops: int) -> np.ndarray:
    """Multi-source BFS on a boolean CSR adjacency (rows = source node). -1 = unreached."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int16)
    if seed_local.size == 0:
        return dist
    dist[seed_local] = 0
    visited = np.zeros(n, dtype=bool)
    visited[seed_local] = True
    frontier = seed_local
    for d in range(1, hops + 1):
        if frontier.size == 0:
            break
        nxt = np.unique(adj[frontier].indices)
        nxt = nxt[~visited[nxt]]
        if nxt.size == 0:
            break
        dist[nxt] = d
        visited[nxt] = True
        frontier = nxt
    return dist


class SubcircuitResult:
    def __init__(
        self,
        node_ids: np.ndarray,
        kept_mask: np.ndarray,
        d_f: np.ndarray,
        d_b: np.ndarray,
        edges_pre: np.ndarray,
        edges_post: np.ndarray,
        edges_weight: np.ndarray,
        input_seed_ids: np.ndarray,
        output_seed_ids: np.ndarray,
    ):
        self.node_ids = node_ids
        self.kept_mask = kept_mask
        self.d_f = d_f
        self.d_b = d_b
        self.edges_pre = edges_pre
        self.edges_post = edges_post
        self.edges_weight = edges_weight
        self.input_seed_ids = input_seed_ids
        self.output_seed_ids = output_seed_ids

    @property
    def kept_ids(self) -> np.ndarray:
        return self.node_ids[self.kept_mask]

    @property
    def n_neurons(self) -> int:
        return int(self.kept_mask.sum())

    @property
    def n_edges(self) -> int:
        return int(self.edges_pre.size)


def extract(
    weights: pd.DataFrame,
    input_seed_ids: np.ndarray,
    output_seed_ids: np.ndarray,
    threshold: int,
    hops: int,
) -> SubcircuitResult:
    """Apply the BFS/induced-subgraph method for one (threshold, hops) pair.

    weights must already be filtered to weight >= MIN_LOAD_WEIGHT with self-loops
    dropped (see load_weights); this function filters down to `threshold` itself.
    """
    sub = weights[weights["weight"] >= threshold]
    pre = sub["body_pre"].to_numpy()
    post = sub["body_post"].to_numpy()
    weight = sub["weight"].to_numpy()

    node_ids = np.union1d(pre, post)
    n = node_ids.size
    pre_idx = np.searchsorted(node_ids, pre)
    post_idx = np.searchsorted(node_ids, post)

    fwd = sparse.csr_matrix(
        (np.ones(pre_idx.size, dtype=np.int8), (pre_idx, post_idx)), shape=(n, n)
    )
    bwd = sparse.csr_matrix(
        (np.ones(pre_idx.size, dtype=np.int8), (post_idx, pre_idx)), shape=(n, n)
    )

    in_seed_local = np.searchsorted(node_ids, input_seed_ids)
    in_seed_local = in_seed_local[
        (in_seed_local < n) & (node_ids[np.clip(in_seed_local, 0, n - 1)] == input_seed_ids)
    ]
    out_seed_local = np.searchsorted(node_ids, output_seed_ids)
    out_seed_local = out_seed_local[
        (out_seed_local < n) & (node_ids[np.clip(out_seed_local, 0, n - 1)] == output_seed_ids)
    ]

    d_f = _bfs_multi_source(in_seed_local, fwd, hops)
    d_b = _bfs_multi_source(out_seed_local, bwd, hops)

    reachable = (d_f >= 0) & (d_b >= 0)
    total = np.where(reachable, d_f + d_b, hops + 1)
    kept_mask = reachable & (total <= hops)

    kept_ids = node_ids[kept_mask]
    edge_keep = np.isin(pre, kept_ids) & np.isin(post, kept_ids)

    return SubcircuitResult(
        node_ids=node_ids,
        kept_mask=kept_mask,
        d_f=d_f,
        d_b=d_b,
        edges_pre=pre[edge_keep],
        edges_post=post[edge_keep],
        edges_weight=weight[edge_keep],
        input_seed_ids=input_seed_ids,
        output_seed_ids=output_seed_ids,
    )


def sweep_table(weights: pd.DataFrame, input_seed_ids: np.ndarray, output_seed_ids: np.ndarray) -> list[dict]:
    rows = []
    for th in SWEEP_THRESHOLDS:
        for hp in SWEEP_HOPS:
            r = extract(weights, input_seed_ids, output_seed_ids, th, hp)
            n_reached_out = np.isin(output_seed_ids, r.kept_ids).sum()
            rows.append(
                {
                    "threshold": th,
                    "hops": hp,
                    "neurons": r.n_neurons,
                    "edges": r.n_edges,
                    "all_outputs_reached": bool(n_reached_out == output_seed_ids.size),
                }
            )
    return rows


def print_sweep_table(rows: list[dict]) -> None:
    print(f"{'threshold':>9} {'hops':>5} {'neurons':>9} {'edges':>9} {'all_outputs':>12}")
    for r in rows:
        print(f"{r['threshold']:>9} {r['hops']:>5} {r['neurons']:>9} {r['edges']:>9} {str(r['all_outputs_reached']):>12}")


def sign_and_matrix(
    result: SubcircuitResult, nt_lookup: dict[int, str]
) -> tuple[sparse.csr_matrix, np.ndarray, int, int]:
    """Build the signed CSR matrix over result.kept_ids (in that order).

    Returns (matrix, sign_per_neuron, n_edges_excluded_by_nt, n_edges_kept_in_matrix).
    sign_per_neuron: +1 / -1 / 0 (0 = excluded neurotransmitter -> outgoing edges dropped).
    """
    kept_ids = result.kept_ids
    n = kept_ids.size
    local_idx = {int(bid): i for i, bid in enumerate(kept_ids)}

    sign_per_neuron = np.zeros(n, dtype=np.int8)
    for i, bid in enumerate(kept_ids):
        nt = nt_lookup.get(int(bid))
        if nt in EXCITATORY_NT:
            sign_per_neuron[i] = 1
        elif nt in INHIBITORY_NT:
            sign_per_neuron[i] = -1
        else:
            sign_per_neuron[i] = 0  # dopamine/serotonin/octopamine/unclear/unrecognised

    pre_sign = sign_per_neuron[np.array([local_idx[int(b)] for b in result.edges_pre], dtype=np.int64)]
    include = pre_sign != 0
    n_excluded = int((~include).sum())
    n_included = int(include.sum())

    rows = np.array([local_idx[int(b)] for b in result.edges_pre[include]], dtype=np.int64)
    cols = np.array([local_idx[int(b)] for b in result.edges_post[include]], dtype=np.int64)
    data = (pre_sign[include].astype(np.float32)) * result.edges_weight[include].astype(np.float32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return matrix, sign_per_neuron, n_excluded, n_included


def build_neurons_frame(
    result: SubcircuitResult,
    sign_per_neuron: np.ndarray,
    annotations: pd.DataFrame,
    nt_lookup: dict[int, str],
) -> pd.DataFrame:
    kept_ids = result.kept_ids
    input_set = set(int(b) for b in result.input_seed_ids)
    output_set = set(int(b) for b in result.output_seed_ids)

    def role_of(bid: int) -> str:
        if bid in input_set:
            return "input_seed"
        if bid in output_set:
            return "output_seed"
        return "interneuron"

    ordered = sorted(
        kept_ids,
        key=lambda bid: (
            0 if int(bid) in input_set else (1 if int(bid) in output_set else 2),
            int(bid),
        ),
    )
    ordered = np.array(ordered, dtype=kept_ids.dtype)
    local_of_kept = {int(bid): i for i, bid in enumerate(kept_ids)}

    ann_lookup = annotations.set_index("bodyId")[["type", "somaSide"]]

    rows = []
    for idx, bid in enumerate(ordered):
        bid_i = int(bid)
        orig_local = local_of_kept[bid_i]
        typ = ann_lookup["type"].get(bid_i, None)
        soma = ann_lookup["somaSide"].get(bid_i, None)
        rows.append(
            {
                "idx": idx,
                "bodyId": bid_i,
                "type": typ,
                "somaSide": soma,
                "consensus_nt": nt_lookup.get(bid_i, "unclear"),
                "sign": int(sign_per_neuron[orig_local]),
                "role": role_of(bid_i),
            }
        )
    return pd.DataFrame(rows), ordered


def reorder_matrix(matrix: sparse.csr_matrix, kept_ids: np.ndarray, ordered_ids: np.ndarray) -> sparse.csr_matrix:
    """Permute matrix rows/cols from kept_ids order to ordered_ids order."""
    pos = {int(bid): i for i, bid in enumerate(kept_ids)}
    perm = np.array([pos[int(bid)] for bid in ordered_ids], dtype=np.int64)
    return matrix[perm][:, perm].tocsr()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2 -- sub-circuit extraction")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS)
    parser.add_argument("--sweep", action="store_true", help="print the 9-row threshold x hops table and exit")
    args = parser.parse_args()

    print("loading weights, annotations, neurotransmitters...")
    weights = load_weights()
    annotations = load_annotations()
    nt = load_nt()
    nt_lookup = dict(zip(nt["body"].to_numpy(), nt["consensus_nt"].to_numpy()))

    input_ids = seed_ids(annotations, INPUT_TYPES)
    output_ids = seed_ids(annotations, OUTPUT_TYPES)
    print(f"input seeds ({'+'.join(INPUT_TYPES)}): {input_ids.size}")
    print(f"output seeds ({'+'.join(OUTPUT_TYPES)}): {output_ids.size}")

    if args.sweep:
        rows = sweep_table(weights, input_ids, output_ids)
        print_sweep_table(rows)
        return

    result = extract(weights, input_ids, output_ids, args.threshold, args.hops)

    reached = np.isin(output_ids, result.kept_ids)
    if not reached.all():
        missing = output_ids[~reached]
        print(
            f"FATAL: {missing.size} output seed(s) unreachable from inputs at "
            f"threshold={args.threshold} hops={args.hops}: {missing.tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix, sign_per_neuron, n_excluded_edges, n_included_edges = sign_and_matrix(result, nt_lookup)
    neurons_df, ordered_ids = build_neurons_frame(result, sign_per_neuron, annotations, nt_lookup)
    matrix = reorder_matrix(matrix, result.kept_ids, ordered_ids)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(DERIVED_DIR / "subcircuit.npz", matrix)
    neurons_df.to_parquet(DERIVED_DIR / "neurons.parquet", index=False)

    n_neurons = int(neurons_df.shape[0])
    n_edges_total = result.n_edges
    n_input = int((neurons_df["role"] == "input_seed").sum())
    n_output = int((neurons_df["role"] == "output_seed").sum())

    pos_mask = matrix.data > 0
    neg_mask = matrix.data < 0
    n_pos_edges = int(pos_mask.sum())
    n_neg_edges = int(neg_mask.sum())
    pct_exc_by_count = 100.0 * n_pos_edges / n_included_edges if n_included_edges else 0.0
    pct_inh_by_count = 100.0 * n_neg_edges / n_included_edges if n_included_edges else 0.0

    total_w = float(np.abs(matrix.data).sum())
    exc_w = float(matrix.data[pos_mask].sum())
    inh_w = float(-matrix.data[neg_mask].sum())
    pct_exc_by_weight = 100.0 * exc_w / total_w if total_w else 0.0
    pct_inh_by_weight = 100.0 * inh_w / total_w if total_w else 0.0

    print("--- summary ---")
    print(f"neurons: {n_neurons}")
    print(f"edges kept (induced subgraph, before NT exclusion): {n_edges_total}")
    print(f"edges included in signed matrix: {n_included_edges}")
    print(f"edges excluded by neurotransmitter: {n_excluded_edges}")
    print(f"% excitatory by edge count: {pct_exc_by_count:.1f}%")
    print(f"% inhibitory by edge count: {pct_inh_by_count:.1f}%")
    print(f"% excitatory by synapse weight: {pct_exc_by_weight:.1f}%")
    print(f"% inhibitory by synapse weight: {pct_inh_by_weight:.1f}%")
    print(f"input seeds in network: {n_input}")
    print(f"output seeds in network: {n_output}")
    print(f"matrix nnz: {matrix.nnz}")


if __name__ == "__main__":
    main()
