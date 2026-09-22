"""Degree-preserving network shuffle: rewire edges while preserving in-degree, out-degree, and weight multisets.

This module generates shuffled controls of the sub-circuit connectome by swapping edge targets
while preserving:
- Each neuron's in-degree and out-degree (exactly)
- Each neuron's outgoing weight multiset (out-degree and outgoing weight strength)
- Neurotransmitter composition (excitatory/inhibitory proportions)
- Input-seed and output-seed role labels (same positions)

Does NOT preserve: in-strength (the sum of incoming weights per neuron).

Algorithm: double-edge-swap using at least 10 × n_edges successful swaps, with rejection
sampling for self-loops and duplicates. Accelerated with numba.njit.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import numba
from scipy.sparse import csr_matrix, coo_matrix, load_npz, save_npz


@numba.njit(cache=True)
def _mix64(x):
    # splitmix64 finaliser (same construction as flybrain.lif), well mixed in every bit
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@numba.njit(cache=True)
def _edge_swap_kernel(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    n_edges: int,
    target_swaps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Numba-accelerated double-edge swap kernel.

    Args:
        src, dst, w: edge arrays (modified in-place).
        n_edges: number of edges.
        target_swaps: minimum successful swaps.
        seed: random seed.

    Returns:
        src, dst, w (modified), swaps_attempted, swaps_accepted.
    """
    # Counter-based RNG: draw k is mix64(mix64(seed) + k). Replaces an LCG whose low bits
    # (used by the modulo below) are weak and whose nearby seeds started correlated.
    seed_h = _mix64(np.uint64(seed))
    counter = np.uint64(0)

    # Build edge set as a sorted array for fast lookup.
    # Encode (s, d) as s * (max_node + 1) + d.
    max_node = np.max(src)
    encode_mult = max_node + 1

    edge_set = set()
    for i in range(n_edges):
        code = int(src[i]) * encode_mult + int(dst[i])
        edge_set.add(code)

    swaps_attempted = 0
    swaps_accepted = 0

    while swaps_accepted < target_swaps:
        # Pick two distinct edges uniformly at random.
        e1_idx = int(_mix64(seed_h + counter) % np.uint64(n_edges))
        e2_idx = int(_mix64(seed_h + counter + np.uint64(1)) % np.uint64(n_edges))
        counter += np.uint64(2)
        if e1_idx == e2_idx:
            continue

        swaps_attempted += 1

        a = int(src[e1_idx])
        b = int(dst[e1_idx])
        w1 = w[e1_idx]

        c = int(src[e2_idx])
        d = int(dst[e2_idx])
        w2 = w[e2_idx]

        # Propose swap: (a→b, w1), (c→d, w2) → (a→d, w1), (c→b, w2).
        if a == d or c == b:
            continue  # Would create self-loop.

        # Check for duplicates using encoded pairs.
        code_ad = a * encode_mult + d
        code_cb = c * encode_mult + b
        if code_ad in edge_set or code_cb in edge_set:
            continue  # Would create duplicate.

        # Accept the swap.
        code_ab = a * encode_mult + b
        code_cd = c * encode_mult + d
        edge_set.discard(code_ab)
        edge_set.discard(code_cd)
        edge_set.add(code_ad)
        edge_set.add(code_cb)

        dst[e1_idx] = d
        w[e1_idx] = w1
        dst[e2_idx] = b
        w[e2_idx] = w2

        swaps_accepted += 1

    return src, dst, w, swaps_attempted, swaps_accepted


def shuffle_network(W: csr_matrix, n_swaps: int, seed: int) -> tuple[csr_matrix, dict]:
    """Shuffle a weighted directed network via double-edge swap.

    Preserves in-degree, out-degree, and per-row weight multiset.

    Args:
        W: CSR matrix, shape (N, N), float32. W[i, j] = weight from i to j.
        n_swaps: minimum number of successful swaps to perform.
        seed: random seed (passed to default_rng).

    Returns:
        W_shuffled: CSR matrix, same shape and dtype as W.
        stats: dict with 'swaps_attempted', 'swaps_accepted', 'edges_changed', 'seconds'.
    """
    start_time = time.time()

    # Convert to COO for manipulation; record original edge set.
    W_coo = W.tocoo(copy=True)
    src_orig = W_coo.row.copy()
    dst_orig = W_coo.col.copy()
    w_orig = W_coo.data.copy()
    n_edges = len(src_orig)

    # Copy to mutable arrays for numba kernel.
    src = src_orig.copy().astype(np.int64)
    dst = dst_orig.copy().astype(np.int64)
    w = w_orig.astype(np.float32)

    target_swaps = max(n_swaps, 10 * n_edges)

    # Call numba kernel.
    src, dst, w, swaps_attempted, swaps_accepted = _edge_swap_kernel(
        src, dst, w, n_edges, target_swaps, seed
    )

    elapsed = time.time() - start_time

    # Reconstruct CSR matrix.
    W_shuffled = coo_matrix((w, (src, dst)), shape=W.shape, dtype=W.dtype)
    W_shuffled = W_shuffled.tocsr()

    # Count edges that changed.
    orig_edges = set(zip(src_orig, dst_orig))
    edges_changed = 0
    for s, d in zip(src, dst):
        if (int(s), int(d)) not in orig_edges:
            edges_changed += 1

    stats = {
        "swaps_attempted": swaps_attempted,
        "swaps_accepted": swaps_accepted,
        "edges_changed": edges_changed,
        "seconds": elapsed,
    }

    return W_shuffled, stats


def main():
    """Generate 30 degree-preserving shuffled control networks."""
    parser = argparse.ArgumentParser(
        description="Generate degree-preserving shuffled control networks."
    )
    parser.add_argument(
        "-n",
        "--num_controls",
        type=int,
        default=30,
        help="Number of controls to generate (default: 30).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    subcircuit_path = project_root / "data" / "derived" / "subcircuit.npz"
    controls_dir = project_root / "data" / "derived" / "controls"

    # Load subcircuit.
    if not subcircuit_path.exists():
        print(f"Error: {subcircuit_path} not found.", file=sys.stderr)
        sys.exit(1)

    W = load_npz(subcircuit_path).astype(np.float32)
    print(f"Loaded subcircuit: shape {W.shape}, {W.nnz} edges")

    # Ensure controls directory exists.
    controls_dir.mkdir(parents=True, exist_ok=True)

    # Generate controls.
    total_start = time.time()
    all_stats = []

    for i in range(args.num_controls):
        seed = 1000 + i
        print(f"Generating control {i:02d} (seed {seed})...", end=" ", flush=True)

        W_control, stats = shuffle_network(W, n_swaps=10 * W.nnz, seed=seed)

        # Save control.
        control_path = controls_dir / f"control_{i:02d}.npz"
        save_npz(control_path, W_control)

        all_stats.append(stats)

        frac_changed = stats["edges_changed"] / W.nnz
        print(
            f"done. Accepted {stats['swaps_accepted']:6d} / {stats['swaps_attempted']:6d} "
            f"swaps; {frac_changed:.1%} edges changed in {stats['seconds']:.1f}s"
        )

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Average time per control: {total_elapsed / args.num_controls:.1f}s")


if __name__ == "__main__":
    main()
