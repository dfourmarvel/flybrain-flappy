"""Phase 2, P1: export the frozen sub-circuit and one run's fitted interface for the browser's
live mode (docs/PLAN.md, Phase 2).

    python -m flybrain.export_web --run runs/real/seed_17

Writes docs-site/live/: three little-endian binary arrays holding the CSR wiring, and
model.json with the model constants, neuron roles, fitted interface, game constants and
reference numbers from the Python simulator. The reference numbers are what the browser
engine is tested against (docs-site/live.test.mjs), so its equivalence is measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from flybrain import interface, lif
from flybrain.game import GameConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs-site" / "live"

REF_RATE_HZ = 150.0
REF_TRIALS = 40
REF_SEED0 = 20_000
REF_STEPS = 5000  # 1 simulated second at dt = 0.2 ms
MAX_TOTAL_BYTES = 2 * 1024 * 1024


def _roles(neurons) -> dict:
    role = neurons["role"].to_numpy()
    side = neurons["somaSide"].to_numpy()
    ntype = neurons["type"].to_numpy()
    idx = np.arange(len(neurons))
    output = idx[role == "output_seed"]  # readout order == network row order (interface.py)
    return {
        "input_L": idx[(role == "input_seed") & (side == "L")].tolist(),
        "input_R": idx[(role == "input_seed") & (side == "R")].tolist(),
        "lc4": idx[ntype == "LC4"].tolist(),
        "output": output.tolist(),
        "output_labels": [f"{ntype[i]}_{side[i]}" for i in output],
        "dnp01": idx[ntype == "DNp01"].tolist(),
        "inhibitory": idx[neurons["sign"].to_numpy() < 0].tolist(),
    }


def _reference(W, neurons, roles) -> dict:
    """Open-loop reference from the Python simulator: LC4 driven at 150 Hz for 1 s."""
    n = W.shape[0]
    sim = lif.Simulator(W, n_candidates=REF_TRIALS, dt=lif.DT_DEFAULT,
                        seeds=np.arange(REF_SEED0, REF_SEED0 + REF_TRIALS, dtype=np.uint64))
    rates = np.zeros((REF_TRIALS, n), np.float32)
    rates[:, roles["lc4"]] = REF_RATE_HZ
    counts = sim.run(REF_STEPS, rates).astype(np.float64)  # 1 s window, so counts == Hz
    non_input = (neurons["role"] != "input_seed").to_numpy()
    return {
        "protocol": (f"LC4 input seeds at {REF_RATE_HZ:g} Hz, 1 simulated second, {REF_TRIALS} "
                     f"trials (seeds {REF_SEED0}-{REF_SEED0 + REF_TRIALS - 1}), dt "
                     f"{lif.DT_DEFAULT} ms, Python flybrain.lif.Simulator"),
        "output_rates_hz": counts[:, roles["output"]].mean(0).round(4).tolist(),
        "network_rate_hz": round(float(counts[:, non_input].mean()), 4),
    }


def export(run_dir: Path, out_dir: Path = OUT_DIR) -> dict:
    W, neurons = lif.load_network()
    W = W.tocsr()
    W.sort_indices()
    n = W.shape[0]
    assert n < 65536, "indices are stored as Uint16"
    counts = W.data.astype(np.int64)
    assert np.all(counts == np.round(counts)), "weights must be whole signed synapse counts"
    assert np.abs(counts).max() < 32768, "weights are stored as Int16"

    out_dir.mkdir(parents=True, exist_ok=True)
    W.indptr.astype("<i4").tofile(out_dir / "csr_indptr.bin")
    W.indices.astype("<u2").tofile(out_dir / "csr_indices.bin")
    counts.astype("<i2").tofile(out_dir / "csr_weights.bin")

    best_x = np.load(run_dir / "best_x.npy")
    p = interface.from_vector(best_x)
    heldout = json.loads((run_dir / "heldout.json").read_text(encoding="utf-8"))
    roles = _roles(neurons)

    model = {
        "schema_version": 1,
        "n_neurons": int(n),
        "nnz": int(W.nnz),
        "constants": {
            "v_0": lif.V_0, "v_rst": lif.V_RST, "v_th": lif.V_TH, "t_mbr": lif.T_MBR,
            "tau": lif.TAU, "t_rfc": lif.T_RFC, "t_dly": lif.T_DLY, "w_syn": lif.W_SYN,
            "f_poi": lif.F_POI, "dt": lif.DT_DEFAULT,
        },
        "roles": roles,
        "interface": {
            "G": float(p.G), "lam": float(p.lam), "kappa": float(p.kappa), "r0": float(p.r0),
            "w": [float(v) for v in p.w], "tau_ms": float(p.tau_ms), "bias": float(p.bias),
            "source_run": f"{run_dir.parent.name}/{run_dir.name}",
            "source_heldout_mean": float(heldout["mean"]),
        },
        "game": asdict(GameConfig()),
        "reference": _reference(W, neurons, roles),
        "files": {
            "indptr": "csr_indptr.bin (Int32 LE, n_neurons + 1)",
            "indices": "csr_indices.bin (Uint16 LE, nnz; post-synaptic neuron)",
            "weights": "csr_weights.bin (Int16 LE, nnz; sign x synapse count, w_syn not applied)",
        },
    }
    (out_dir / "model.json").write_bytes(json.dumps(model, indent=1).encode("utf-8"))

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    if total > MAX_TOTAL_BYTES:
        raise SystemExit(f"export is {total} bytes, over the {MAX_TOTAL_BYTES}-byte budget")
    return {"bytes": total, "model": model}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export the live-mode model for the browser.")
    parser.add_argument("--run", default=str(PROJECT_ROOT / "runs" / "real" / "seed_17"))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    result = export(Path(args.run), Path(args.out))
    ref = result["model"]["reference"]
    print(f"wrote {args.out} ({result['bytes']:,} bytes)")
    print(f"reference output rates (Hz): {ref['output_rates_hz']}")
    print(f"reference network rate (Hz): {ref['network_rate_hz']}")


if __name__ == "__main__":
    main()
