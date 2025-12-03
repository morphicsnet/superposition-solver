from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile, ZIP_DEFLATED


def _add_if_exists(pairs: List[Tuple[str, str]], abs_path: str, rel_path: str) -> None:
    if abs_path and os.path.exists(abs_path) and os.path.isfile(abs_path):
        pairs.append((os.path.abspath(abs_path), rel_path))


def collect_artifacts(run_dirs: Dict[str, Optional[str]]) -> List[Tuple[str, str]]:
    """
    Collect key artifacts across selected run directories.

    Args:
      run_dirs: mapping like
        {
          "baseline": "/abs/path/to/run1" | None,
          "ensemble": "/abs/path/to/run2" | None,
          "spike_hypergraph": "/abs/path/to/run3" | None,
          "causal": "/abs/path/to/run4" | None,
        }

    Returns:
      List of (abs_path, rel_path_in_zip)
    """
    artifacts: List[Tuple[str, str]] = []

    # Per-category expected files
    plan: Dict[str, List[str]] = {
        "baseline": [
            "metrics.json",
            "config.yaml",
            "poly_hist.png",
            "probs.npy",
            "poly_counts.npy",
            "entropy.npy",
        ],
        "ensemble": [
            "metrics_single.json",
            "metrics_intersection.json",
            "compare.json",
            "config.yaml",
            "poly_hist_single.png",
            "poly_hist_intersection.png",
            "poly_hist_dual.png",
            "probs_single.npy",
            "poly_counts_single.npy",
            "entropy_single.npy",
            "probs_intersection.npy",
            "poly_counts_intersection.npy",
            "entropy_intersection.npy",
        ],
        "spike_hypergraph": [
            "metrics_hyperedges.json",
            "config.yaml",
            "poly_hist_hyperedges.png",
            # HIF (demo3)
            "hypergraph.hif.json",
        ],
        "causal": [
            "stii_values.json",
            "acdc_minimal_circuit.json",
            "fairness_report.json",
            "config.yaml",
            # HIF (demo4)
            "hypergraph_stii.hif.json",
        ],
    }

    for key, run_dir in (run_dirs or {}).items():
        if not run_dir:
            continue
        if not os.path.isdir(run_dir):
            continue
        # Standard files
        for name in plan.get(key, []):
            abs_path = os.path.join(run_dir, name)
            rel_path = os.path.join(key, os.path.basename(run_dir), name)
            _add_if_exists(artifacts, abs_path, rel_path)

        # Also include any generic HIF naming overlap if present
        for name in ("hypergraph.hif.json", "hypergraph_stii.hif.json"):
            abs_path = os.path.join(run_dir, name)
            rel_path = os.path.join(key, os.path.basename(run_dir), name)
            _add_if_exists(artifacts, abs_path, rel_path)

    return artifacts


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    artifacts: List[Tuple[str, str]],
    out_path: str,
    extra: Optional[dict] = None,
) -> None:
    """
    Write a JSON manifest describing artifacts included in a bundle.

    Schema:
      {
        "created_at": ISO8601,
        "artifacts": [
          {"rel_path": "...", "sha256": "...", "size": int},
          ...
        ],
        "extra": {...}
      }
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = []
    for abs_path, rel_path in artifacts:
        try:
            size = os.path.getsize(abs_path)
            sha = _sha256_file(abs_path)
        except Exception:
            size, sha = 0, ""
        items.append(
            {
                "rel_path": rel_path,
                "sha256": sha,
                "size": int(size),
            }
        )

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "artifacts": items,
        "extra": extra or {},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_zip(artifacts: List[Tuple[str, str]], zip_path: str) -> None:
    """
    Create a zip file at zip_path containing all artifacts at their rel paths.
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for abs_path, rel_path in artifacts:
            # Normalize rel path separator
            arcname = rel_path.replace("\\", "/").lstrip("/")
            try:
                zf.write(abs_path, arcname=arcname)
            except FileNotFoundError:
                # Skip missing files (defensive)
                continue