"""Step 1 — download the MaleCNS v1.0 flat-connectome files and summarise them.

Public bulk download, no login required. See docs/DATA.md for licence/citation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
)

# name -> expected size in bytes (verified 2026-09-22, lead spike; see docs/PLAN.md Step 1)
FILES: dict[str, int] = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 14_483_314,
    "body-neurotransmitters-male-cns-v1.0.feather": 43_282_834,
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather": 502_169_298,
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _remote_size(name: str) -> int:
    """HEAD the object to confirm its size before trusting the local skip check."""
    resp = requests.head(BASE_URL + name, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def download_one(name: str, expected_size: int) -> Path:
    remote = _remote_size(name)
    if remote != expected_size:
        print(f"note  {name}: remote is {remote:,} bytes, pinned {expected_size:,}; using remote")
        expected_size = remote
    dest = RAW_DIR / name
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"skip  {name} (already {expected_size:,} bytes)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    url = BASE_URL + name
    print(f"fetch {name} <- {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        downloaded = 0
        chunk_size = 1024 * 1024
        report_every = 50 * chunk_size
        next_report = report_every
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  {name}: {downloaded:,} / {expected_size:,} bytes")
                    next_report += report_every
    got = tmp.stat().st_size
    if got != expected_size:
        raise RuntimeError(f"{name}: downloaded {got:,} bytes, expected {expected_size:,}")
    tmp.replace(dest)  # replace, not rename: rename fails on Windows if dest exists
    print(f"done  {name}: {dest.stat().st_size:,} bytes")
    return dest


def download_all() -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [download_one(name, size) for name, size in FILES.items()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarise() -> dict:
    """Load each downloaded file with pandas and compute the numbers docs/DATA.md
    reports. Printed here and returned so DATA.md is never hand-typed."""
    out: dict = {}

    for name in FILES:
        path = RAW_DIR / name
        size = path.stat().st_size
        digest = sha256_of(path)
        df = pd.read_feather(path)
        cols = [(c, str(df[c].dtype)) for c in df.columns]
        info = {
            "size": size,
            "sha256": digest,
            "rows": len(df),
            "columns": cols,
        }
        out[name] = info
        print(f"\n{name}")
        print(f"  size   : {size:,} bytes")
        print(f"  sha256 : {digest}")
        print(f"  rows   : {len(df):,}")
        print(f"  cols   : {cols}")

    weights_name = "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"
    wdf = pd.read_feather(RAW_DIR / weights_name)
    weight_col = "weight" if "weight" in wdf.columns else None
    if weight_col is None:
        # SPEC-GAP: exact weight column name not pre-verified; fall back to the
        # first numeric column that isn't a body id.
        for c in wdf.columns:
            if c not in ("pre", "post", "bodyId_pre", "bodyId_post") and pd.api.types.is_numeric_dtype(wdf[c]):
                weight_col = c
                break

    pre_col = next(c for c in ("pre", "bodyId_pre", "body_pre") if c in wdf.columns)
    post_col = next(c for c in ("post", "bodyId_post", "body_post") if c in wdf.columns)

    weight_stats = {
        "weight_col": weight_col,
        "min": wdf[weight_col].min() if weight_col else None,
        "median": wdf[weight_col].median() if weight_col else None,
        "max": wdf[weight_col].max() if weight_col else None,
        "n_distinct_pre": wdf[pre_col].nunique() if pre_col in wdf.columns else None,
        "n_distinct_post": wdf[post_col].nunique() if post_col in wdf.columns else None,
    }
    out["weights_stats"] = weight_stats
    print(f"\nweights stats: {weight_stats}")

    return out


def main() -> None:
    download_all()
    summarise()


if __name__ == "__main__":
    sys.exit(main())
