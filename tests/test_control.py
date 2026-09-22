"""Tests for degree-preserving network shuffle."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, coo_matrix, load_npz
from pathlib import Path

from flybrain.control import shuffle_network


@pytest.fixture
def subcircuit():
    """Load the real subcircuit."""
    project_root = Path(__file__).parent.parent
    subcircuit_path = project_root / "data" / "derived" / "subcircuit.npz"
    if not subcircuit_path.exists():
        pytest.skip(f"{subcircuit_path} not found")
    return load_npz(subcircuit_path).astype(np.float32)


@pytest.fixture
def tiny_network():
    """Hand-built tiny network: 6 nodes, directed edges."""
    # 6 nodes: create a small network with known structure.
    # Node 0 -> 1, 2, 3 (weights +1, +1, +1)
    # Node 1 -> 2, 4 (weights -1, -1)
    # Node 2 -> 5 (weight +1)
    # Node 3 -> 1, 4 (weights -1, -1)
    # Node 4 -> 2, 5 (weights +1, +1)
    # Node 5 -> 0 (weight -1)
    # Total: 11 edges
    row = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5])
    col = np.array([1, 2, 3, 2, 4, 5, 1, 4, 2, 5, 0])
    data = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    return csr_matrix((data, (row, col)), shape=(6, 6), dtype=np.float32)


class TestDegreePreservation:
    """Test that in-degree and out-degree (edge counts) are preserved.

    Note: degree = edge count, not weighted sum. In-strength (weighted in-degree)
    is not preserved by the algorithm.
    """

    def test_in_degree_preserved_tiny(self, tiny_network):
        """In-degree sequence (edge count) preserved on tiny network."""
        W = tiny_network
        # In-degree = number of incoming edges per node
        in_deg_orig = np.array(W.getnnz(axis=0))

        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)
        in_deg_shuffled = np.array(W_shuffled.getnnz(axis=0))

        np.testing.assert_array_equal(in_deg_shuffled, in_deg_orig)

    def test_out_degree_preserved_tiny(self, tiny_network):
        """Out-degree sequence (edge count) preserved on tiny network."""
        W = tiny_network
        # Out-degree = number of outgoing edges per node
        out_deg_orig = np.array(W.getnnz(axis=1))

        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)
        out_deg_shuffled = np.array(W_shuffled.getnnz(axis=1))

        np.testing.assert_array_equal(out_deg_shuffled, out_deg_orig)

    def test_in_degree_preserved_full(self, subcircuit):
        """In-degree sequence (edge count) preserved on real subcircuit (small budget)."""
        W = subcircuit
        in_deg_orig = np.array(W.getnnz(axis=0))

        # Small budget to keep test fast: 100 swaps (~0.1% of edges).
        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)
        in_deg_shuffled = np.array(W_shuffled.getnnz(axis=0))

        np.testing.assert_array_equal(in_deg_shuffled, in_deg_orig)

    def test_out_degree_preserved_full(self, subcircuit):
        """Out-degree sequence (edge count) preserved on real subcircuit (small budget)."""
        W = subcircuit
        out_deg_orig = np.array(W.getnnz(axis=1))

        # Small budget to keep test fast: 100 swaps (~0.1% of edges).
        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)
        out_deg_shuffled = np.array(W_shuffled.getnnz(axis=1))

        np.testing.assert_array_equal(out_deg_shuffled, out_deg_orig)


class TestWeightMultisets:
    """Test that per-row weight multisets are preserved."""

    def test_per_row_weights_preserved_tiny(self, tiny_network):
        """Per-row sorted weight lists are identical."""
        W = tiny_network
        row_weights_orig = []
        for i in range(W.shape[0]):
            row_data = W.getrow(i).data
            row_weights_orig.append(np.sort(row_data))

        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)
        row_weights_shuffled = []
        for i in range(W_shuffled.shape[0]):
            row_data = W_shuffled.getrow(i).data
            row_weights_shuffled.append(np.sort(row_data))

        for i, (orig, shuffled) in enumerate(zip(row_weights_orig, row_weights_shuffled)):
            np.testing.assert_array_equal(shuffled, orig, err_msg=f"Row {i} weights differ")

    def test_per_row_weights_preserved_full(self, subcircuit):
        """Per-row sorted weight lists are identical (full circuit, small budget)."""
        W = subcircuit
        row_weights_orig = []
        for i in range(W.shape[0]):
            row_data = W.getrow(i).data
            row_weights_orig.append(np.sort(row_data))

        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)
        row_weights_shuffled = []
        for i in range(W_shuffled.shape[0]):
            row_data = W_shuffled.getrow(i).data
            row_weights_shuffled.append(np.sort(row_data))

        for i, (orig, shuffled) in enumerate(zip(row_weights_orig, row_weights_shuffled)):
            np.testing.assert_array_almost_equal(
                shuffled, orig, decimal=5, err_msg=f"Row {i} weights differ"
            )


class TestStructuralProperties:
    """Test that no self-loops, duplicates, and nnz is preserved."""

    def test_no_self_loops_tiny(self, tiny_network):
        """No self-loops created."""
        W = tiny_network
        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)

        # Check diagonal is zero.
        diag = W_shuffled.diagonal()
        assert np.all(diag == 0), "Self-loops detected"

    def test_no_self_loops_full(self, subcircuit):
        """No self-loops on real circuit."""
        W = subcircuit
        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)

        diag = W_shuffled.diagonal()
        assert np.all(diag == 0), "Self-loops detected"

    def test_no_duplicates_tiny(self, tiny_network):
        """No duplicate entries in COO form."""
        W = tiny_network
        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)

        # Convert to COO and check for duplicates.
        W_coo = W_shuffled.tocoo()
        # sum_duplicates() modifies in-place; check nnz before and after.
        nnz_before = W_coo.nnz
        W_coo.sum_duplicates()
        nnz_after = W_coo.nnz

        assert nnz_before == nnz_after, f"Duplicates detected: {nnz_before} != {nnz_after}"

    def test_no_duplicates_full(self, subcircuit):
        """No duplicate entries in real circuit."""
        W = subcircuit
        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)

        W_coo = W_shuffled.tocoo()
        nnz_before = W_coo.nnz
        W_coo.sum_duplicates()
        nnz_after = W_coo.nnz

        assert nnz_before == nnz_after, f"Duplicates detected: {nnz_before} != {nnz_after}"

    def test_nnz_preserved_tiny(self, tiny_network):
        """Edge count unchanged."""
        W = tiny_network
        nnz_orig = W.nnz

        W_shuffled, _ = shuffle_network(W, n_swaps=200, seed=42)
        nnz_shuffled = W_shuffled.nnz

        assert nnz_shuffled == nnz_orig

    def test_nnz_preserved_full(self, subcircuit):
        """Edge count unchanged."""
        W = subcircuit
        nnz_orig = W.nnz

        W_shuffled, _ = shuffle_network(W, n_swaps=1000, seed=42)
        nnz_shuffled = W_shuffled.nnz

        assert nnz_shuffled == nnz_orig


class TestEdgeChanges:
    """Test that edges actually change."""

    def test_edges_changed_tiny(self, tiny_network):
        """Edges change after shuffle (tiny network, small budget)."""
        W = tiny_network
        W_coo_orig = W.tocoo()
        orig_edges = set(zip(W_coo_orig.row, W_coo_orig.col))

        W_shuffled, stats = shuffle_network(W, n_swaps=200, seed=42)
        W_coo_shuffled = W_shuffled.tocoo()
        shuffled_edges = set(zip(W_coo_shuffled.row, W_coo_shuffled.col))

        # With 200 swaps on 11 edges, expect significant change.
        n_changed = len(orig_edges.symmetric_difference(shuffled_edges)) / 2
        frac_changed = n_changed / len(orig_edges)
        print(f"Tiny network: {frac_changed:.1%} edges changed")
        # Expect at least some change with 200 swaps.
        assert frac_changed > 0, "No edges changed"

    def test_edges_changed_full(self, subcircuit):
        """A saved full-budget control is fully mixed: the fraction of edges changed matches the
        analytic expectation for a random graph with the same degree sequence (within 1 point).

        The expectation is not ~100%: a random graph with these degrees keeps each original edge
        i->j with probability ~k_out(i)*k_in(j)/E, which is large around hub neurons. For this
        circuit that is ~93.5% changed, so a fixed 95% threshold would be unreachable.
        """
        path = Path(__file__).resolve().parents[1] / "data" / "derived" / "controls" / "control_00.npz"
        if not path.exists():
            pytest.skip("controls not generated; run python -m flybrain.control --n 30")
        W = subcircuit.tocoo()
        C = load_npz(path).tocoo()
        E = W.nnz
        orig = set(zip(W.row.tolist(), W.col.tolist()))
        shuf = set(zip(C.row.tolist(), C.col.tolist()))
        changed = 1.0 - len(orig & shuf) / E
        kout = np.bincount(W.row, minlength=W.shape[0])
        kin = np.bincount(W.col, minlength=W.shape[0])
        expected = 1.0 - np.minimum(1.0, kout[W.row] * kin[W.col] / E).mean()
        print(f"changed {changed:.4f}, expected {expected:.4f}")
        assert abs(changed - expected) < 0.01

    def test_same_seed_identical_output(self, tiny_network):
        """Same seed produces identical output."""
        W = tiny_network
        W_shuf1, _ = shuffle_network(W, n_swaps=200, seed=42)
        W_shuf2, _ = shuffle_network(W, n_swaps=200, seed=42)

        # Check that the matrices are identical.
        diff = (W_shuf1 - W_shuf2).nnz
        assert diff == 0, f"Same seed produced different outputs (diff nnz={diff})"

    def test_different_seeds_different_output(self, tiny_network):
        """Different seeds produce different outputs."""
        W = tiny_network
        W_shuf1, _ = shuffle_network(W, n_swaps=200, seed=42)
        W_shuf2, _ = shuffle_network(W, n_swaps=200, seed=43)

        # Expect them to differ (with very high probability).
        diff = (W_shuf1 - W_shuf2).nnz
        # They have the same number of edges, so we check if the edge sets differ.
        W1_coo = W_shuf1.tocoo()
        W2_coo = W_shuf2.tocoo()
        edges1 = set(zip(W1_coo.row, W1_coo.col))
        edges2 = set(zip(W2_coo.row, W2_coo.col))
        n_same = len(edges1 & edges2)

        # With different seeds, expect significant difference.
        frac_same = n_same / len(edges1)
        assert frac_same < 1.0, "Different seeds produced identical outputs"


class TestZeroRows:
    """Test that all-zero rows remain all-zero."""

    def test_zero_rows_preserved_tiny(self, tiny_network):
        """All-zero rows of the network remain all-zero."""
        # Create a network with an isolated node (row with no outgoing edges).
        row = np.array([0, 0, 0, 1, 1, 2])
        col = np.array([1, 2, 3, 2, 4, 5])
        data = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        W = csr_matrix((data, (row, col)), shape=(7, 7), dtype=np.float32)

        # Rows with no outgoing edges (by edge count).
        zero_rows = np.where(np.array(W.getnnz(axis=1)) == 0)[0]

        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)
        zero_rows_shuffled = np.where(np.array(W_shuffled.getnnz(axis=1)) == 0)[0]

        np.testing.assert_array_equal(zero_rows_shuffled, zero_rows)

    def test_zero_rows_preserved_full(self, subcircuit):
        """All-zero rows of the real circuit remain all-zero."""
        W = subcircuit
        zero_rows = np.where(np.array(W.sum(axis=1)).flatten() == 0)[0]

        if len(zero_rows) == 0:
            pytest.skip("No zero rows in subcircuit")

        W_shuffled, _ = shuffle_network(W, n_swaps=100, seed=42)
        zero_rows_shuffled = np.where(np.array(W_shuffled.sum(axis=1)).flatten() == 0)[0]

        np.testing.assert_array_equal(zero_rows_shuffled, zero_rows)


class TestStats:
    """Test that statistics are computed and returned."""

    def test_stats_returned(self, tiny_network):
        """Stats dict is returned with required keys."""
        W = tiny_network
        _, stats = shuffle_network(W, n_swaps=200, seed=42)

        assert "swaps_attempted" in stats
        assert "swaps_accepted" in stats
        assert "edges_changed" in stats
        assert "seconds" in stats
        assert stats["swaps_accepted"] >= 200
        assert stats["seconds"] >= 0  # May be < 1ms on fast machines
