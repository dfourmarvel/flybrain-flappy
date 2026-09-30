"""Phase 2, P1: the browser export must be an exact copy of the frozen circuit and fitted run."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from flybrain import export_web, interface, lif

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "real" / "seed_17"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if not (RUN / "best_x.npy").exists():
        pytest.skip("runs/real/seed_17 not present")
    out = tmp_path_factory.mktemp("live")
    result = export_web.export(RUN, out)
    return out, result


def test_csr_round_trips_exactly(exported):
    out, result = exported
    n = result["model"]["n_neurons"]
    indptr = np.fromfile(out / "csr_indptr.bin", dtype="<i4")
    indices = np.fromfile(out / "csr_indices.bin", dtype="<u2")
    weights = np.fromfile(out / "csr_weights.bin", dtype="<i2")
    rebuilt = sparse.csr_matrix((weights.astype(np.float32), indices, indptr), shape=(n, n))
    W, _ = lif.load_network()
    assert (rebuilt != W.tocsr()).nnz == 0
    assert result["model"]["nnz"] == W.nnz == weights.size


def test_roles_match_the_neuron_table(exported):
    _, result = exported
    roles = result["model"]["roles"]
    _, neurons = lif.load_network()
    assert len(roles["input_L"]) + len(roles["input_R"]) == int((neurons.role == "input_seed").sum())
    assert roles["output"] == neurons.index[neurons.role == "output_seed"].tolist()
    assert len(roles["dnp01"]) == 2
    assert all(neurons.sign.iloc[i] < 0 for i in roles["inhibitory"])
    assert len(roles["inhibitory"]) == int((neurons.sign < 0).sum())


def test_interface_matches_the_fitted_run(exported):
    _, result = exported
    p = interface.from_vector(np.load(RUN / "best_x.npy"))
    got = result["model"]["interface"]
    assert got["G"] == pytest.approx(p.G) and got["bias"] == pytest.approx(p.bias)
    assert got["w"] == pytest.approx(list(p.w))
    assert got["source_run"] == "real/seed_17"


def test_model_json_is_valid_and_small(exported):
    out, result = exported
    json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert result["bytes"] < export_web.MAX_TOTAL_BYTES
    ref = result["model"]["reference"]
    assert len(ref["output_rates_hz"]) == 10 and ref["network_rate_hz"] > 0
