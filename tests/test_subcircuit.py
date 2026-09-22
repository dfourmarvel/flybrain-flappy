"""Tests for Step 2 sub-circuit extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flybrain.subcircuit import (
    DEFAULT_HOPS,
    DEFAULT_THRESHOLD,
    DERIVED_DIR,
    _bfs_multi_source,
    extract,
)

NPZ_PATH = DERIVED_DIR / "subcircuit.npz"
PARQUET_PATH = DERIVED_DIR / "neurons.parquet"

DERIVED_MISSING_REASON = (
    "derived files not present -- run `.venv/Scripts/python -m flybrain.subcircuit` first"
)

# Reference numbers from the lead's 2026-09-22 spike (docs/PLAN.md / docs/DATA.md Step 2),
# reproduced by the default threshold=5, hops=3 run.
EXPECTED_N_NEURONS = 2493
EXPECTED_N_EDGES = 94702
EXPECTED_N_INPUT = 311
EXPECTED_N_OUTPUT = 10
# A GABAergic interneuron confirmed (by the lead spike script) to be in the th=5/hops=3
# sub-circuit with outgoing edges, so its row must be entirely non-positive.
GABAERGIC_BODY_ID = 10033


def _derived_available() -> bool:
    return NPZ_PATH.exists() and PARQUET_PATH.exists()


@pytest.fixture(scope="module")
def neurons() -> pd.DataFrame:
    if not _derived_available():
        pytest.skip(DERIVED_MISSING_REASON)
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def matrix() -> sparse.csr_matrix:
    if not _derived_available():
        pytest.skip(DERIVED_MISSING_REASON)
    return sparse.load_npz(NPZ_PATH)


class TestDerivedFiles:
    def test_shape_matches_parquet_length(self, matrix, neurons):
        n = len(neurons)
        assert matrix.shape == (n, n)

    def test_dtype_float32(self, matrix):
        assert matrix.dtype == np.float32

    def test_no_nan_or_inf(self, matrix):
        data = matrix.data
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

    def test_no_self_loops(self, matrix):
        coo = matrix.tocoo()
        assert not np.any(coo.row == coo.col)

    def test_idx_is_0_to_n_minus_1_in_documented_order(self, neurons):
        assert (neurons["idx"].to_numpy() == np.arange(len(neurons))).all()

        n_input = int((neurons["role"] == "input_seed").sum())
        n_output = int((neurons["role"] == "output_seed").sum())
        head = neurons.iloc[:n_input]
        mid = neurons.iloc[n_input : n_input + n_output]
        tail = neurons.iloc[n_input + n_output :]

        assert (head["role"] == "input_seed").all()
        assert (mid["role"] == "output_seed").all()
        assert (tail["role"] == "interneuron").all()
        assert head["bodyId"].is_monotonic_increasing
        assert mid["bodyId"].is_monotonic_increasing
        assert tail["bodyId"].is_monotonic_increasing

    def test_role_counts(self, neurons):
        assert int((neurons["role"] == "input_seed").sum()) == EXPECTED_N_INPUT
        assert int((neurons["role"] == "output_seed").sum()) == EXPECTED_N_OUTPUT

    def test_neuron_count_matches_reference(self, neurons):
        assert len(neurons) == EXPECTED_N_NEURONS

    def test_nnz_consistent_with_induced_edges_minus_nt_exclusion(self, matrix, neurons):
        # 94,702 induced-subgraph edges minus however many were dropped because their
        # presynaptic neuron's sign is 0 (excluded neurotransmitter). Both numbers are
        # reported so a mismatch is legible, not just a bare assertion failure.
        n_excluded_neurons = int((neurons["sign"] == 0).sum())
        assert n_excluded_neurons > 0
        # nnz must be <= the induced edge count, and the gap must be explainable by
        # excluded-neuron out-edges only (checked precisely in test_row_sign_consistency
        # and test_excluded_neuron_rows_are_zero below).
        assert matrix.nnz < EXPECTED_N_EDGES
        assert matrix.nnz == 92825, (
            f"nnz={matrix.nnz}, induced edges={EXPECTED_N_EDGES}, "
            f"edges excluded by NT={EXPECTED_N_EDGES - matrix.nnz}"
        )

    def test_row_sign_consistency(self, matrix, neurons):
        """Every nonzero in row i has sign equal to neurons.sign[i]."""
        coo = matrix.tocoo()
        signs = neurons["sign"].to_numpy()
        expected_sign = np.where(signs[coo.row] > 0, 1.0, np.where(signs[coo.row] < 0, -1.0, 0.0))
        actual_sign = np.sign(coo.data)
        assert np.array_equal(actual_sign, expected_sign)

    def test_excluded_neuron_rows_are_all_zero(self, matrix, neurons):
        excluded_idx = neurons.loc[neurons["sign"] == 0, "idx"].to_numpy()
        sub = matrix[excluded_idx]
        assert sub.nnz == 0

    def test_rows_match_raw_edges_by_body_id(self, matrix, neurons):
        """Each matrix row/col is the neuron its parquet row names: spot-check against raw edges."""
        raw = Path(__file__).resolve().parents[1] / "data" / "raw" / (
            "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather")
        if not raw.exists():
            pytest.skip("raw weights file not downloaded")
        ids = neurons["bodyId"].to_numpy()
        pos = {b: i for i, b in enumerate(ids)}
        w = pd.read_feather(raw)
        w = w[w.body_pre.isin(pos) & w.body_post.isin(pos) & (w.weight >= 5)]
        sample = w.sample(n=500, random_state=0)
        sign = neurons["sign"].to_numpy()
        for pre, post, wt in sample[["body_pre", "body_post", "weight"]].itertuples(index=False):
            i, j = pos[pre], pos[post]
            assert matrix[i, j] == sign[i] * wt

    def test_gabaergic_neuron_has_only_negative_outgoing_weights(self, matrix, neurons):
        rows = neurons.loc[neurons["bodyId"] == GABAERGIC_BODY_ID, "idx"]
        if rows.empty:
            pytest.skip(f"body {GABAERGIC_BODY_ID} not present in this run's sub-circuit")
        idx = int(rows.iloc[0])
        assert neurons.loc[neurons["idx"] == idx, "consensus_nt"].iloc[0] == "gaba"
        row = matrix.getrow(idx)
        assert row.nnz > 0
        assert (row.data < 0).all()


class TestReproducesReferenceSweep:
    """Re-run the default (threshold=5, hops=3) extraction from raw data and check it
    matches the numbers the derived files were built from."""

    def test_default_params_match_reference(self):
        from flybrain import subcircuit as sc

        raw_dir = sc.RAW_DIR
        needed = [sc.ANNOTATIONS_FILE, sc.NT_FILE, sc.WEIGHTS_FILE]
        if not all((raw_dir / f).exists() for f in needed):
            pytest.skip("raw data files not present in data/raw/")

        weights = sc.load_weights()
        annotations = sc.load_annotations()
        input_ids = sc.seed_ids(annotations, sc.INPUT_TYPES)
        output_ids = sc.seed_ids(annotations, sc.OUTPUT_TYPES)

        assert input_ids.size == EXPECTED_N_INPUT
        assert output_ids.size == EXPECTED_N_OUTPUT

        result = extract(weights, input_ids, output_ids, DEFAULT_THRESHOLD, DEFAULT_HOPS)
        assert result.n_neurons == EXPECTED_N_NEURONS
        assert result.n_edges == EXPECTED_N_EDGES
        assert np.isin(output_ids, result.kept_ids).all()


class TestBfsOnTinyGraph:
    """Independent hand-built graph with a known answer, no real data involved.

    Graph (edges A->B):
      0 -> 1 -> 2 -> 3 -> 4
                2 -> 5
      6 -> 5
      7 isolated

    Input seed: {0}. Output seed: {4}.
    Forward distances from 0:  0:0 1:1 2:2 3:3 4:4 5:3 6:-1 7:-1
    Backward distances to 4 (i.e. forward BFS on reversed edges from 4):
      4:0 3:1 2:2 1:3 0:4 5:-1 6:-1 7:-1
    With hops=4: kept = nodes where d_f>=0 and d_b>=0 and d_f+d_b<=4
      0: 0+4=4 keep; 1: 1+3=4 keep; 2: 2+2=4 keep; 3: 3+1=4 keep; 4: 4+0=4 keep
      5: d_b=-1 (5 cannot reach 4) -> dropped; 6,7: d_f=-1 -> dropped
    Expected kept set: {0,1,2,3,4}
    """

    @staticmethod
    def _build_adj(edges, n):
        rows = np.array([e[0] for e in edges], dtype=np.int64)
        cols = np.array([e[1] for e in edges], dtype=np.int64)
        data = np.ones(len(edges), dtype=np.int8)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def test_bfs_multi_source_known_distances(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (6, 5)]
        n = 8
        fwd = self._build_adj(edges, n)
        bwd = self._build_adj([(b, a) for a, b in edges], n)

        d_f = _bfs_multi_source(np.array([0]), fwd, hops=4)
        d_b = _bfs_multi_source(np.array([4]), bwd, hops=4)

        assert d_f.tolist() == [0, 1, 2, 3, 4, 3, -1, -1]
        assert d_b.tolist() == [4, 3, 2, 1, 0, -1, -1, -1]

        reachable = (d_f >= 0) & (d_b >= 0)
        total = np.where(reachable, d_f + d_b, 5)
        kept = np.where(reachable & (total <= 4))[0]
        assert kept.tolist() == [0, 1, 2, 3, 4]

    def test_extract_on_tiny_graph_induced_subgraph(self):
        """extract() end to end on the tiny graph: kept edges are ALL threshold-passing
        edges among kept nodes (induced subgraph), not just BFS-path edges."""
        weights = pd.DataFrame(
            {
                "body_pre": [0, 1, 2, 3, 2, 6, 0],
                "body_post": [1, 2, 3, 4, 5, 5, 3],  # extra edge 0->3: a chord, must be kept
                "weight": [10, 10, 10, 10, 10, 10, 10],
            }
        )
        input_ids = np.array([0])
        output_ids = np.array([4])

        result = extract(weights, input_ids, output_ids, threshold=1, hops=4)

        assert sorted(result.kept_ids.tolist()) == [0, 1, 2, 3, 4]
        # induced subgraph: every edge with both endpoints in {0,1,2,3,4} is kept,
        # including the chord 0->3, but not 2->5 or 6->5 (5,6 not kept).
        kept_edges = set(zip(result.edges_pre.tolist(), result.edges_post.tolist()))
        assert kept_edges == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)}
