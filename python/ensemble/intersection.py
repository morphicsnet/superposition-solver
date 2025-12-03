from __future__ import annotations

import os
from typing import List

import numpy as np


def _require_py_nsi():
    try:
        from py_nsi import PySimpleSaeEncoder, PyEnsemble  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "py_nsi is not importable. Build the local Rust wheel first:\n"
            "  cd py_nsi && maturin develop --release\n"
            "Then re-run the demo."
        ) from e
    return PySimpleSaeEncoder, PyEnsemble


def _resolve_input_dim() -> int:
    """
    Determine encoder input dimension. The orchestrator should set:
        os.environ['PY_NSI_INPUT_DIM'] = str(acts.shape[1])
    before calling build_pyensemble(). This keeps the required signature while
    ensuring correctness w.r.t. the model's activation size.
    """
    val = os.environ.get("PY_NSI_INPUT_DIM", "").strip()
    if not val.isdigit():
        raise RuntimeError(
            "PY_NSI_INPUT_DIM environment variable is not set or invalid. "
            "Set it to the activation dimension (acts.shape[1]) before calling build_pyensemble()."
        )
    return int(val)


def build_pyensemble(feature_dim: int, top_k: int, seeds: List[int]) -> object:
    """
    Construct a PyEnsemble of PySimpleSaeEncoder, one per seed.

    Args:
        feature_dim: out_dim per encoder
        top_k: k nonzeros per encoder output (enforced inside encoder)
        seeds: list of integer seeds (diversity across encoders)

    Returns:
        A py_nsi.PyEnsemble instance
    """
    PySimpleSaeEncoder, PyEnsemble = _require_py_nsi()
    in_dim = _resolve_input_dim()

    encs = []
    for s in seeds:
        encs.append(PySimpleSaeEncoder(int(in_dim), int(feature_dim), int(top_k), int(s)))
    return PyEnsemble(encs)


def encode_all_and_intersect(ensemble, acts: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each activation vector x in acts, call:
      outs = PyEnsemble.encode_all(x)
      mask = PyEnsemble.intersect(outs, threshold)
    Collect masks into a boolean array of shape [N, H].

    Args:
        ensemble: py_nsi.PyEnsemble as returned by build_pyensemble()
        acts: np.ndarray [N, D] float activations
        threshold: float threshold for intersection (>)

    Returns:
        masks: np.ndarray [N, H] of dtype=bool
    """
    if not isinstance(acts, np.ndarray) or acts.ndim != 2:
        raise ValueError("acts must be a 2D numpy array [N, D]")

    acts_f32 = acts.astype(np.float32, copy=False)
    N = acts_f32.shape[0]
    out_rows: List[np.ndarray] = []

    # Probe first row to determine H
    if N == 0:
        return np.zeros((0, 0), dtype=bool)

    first_outs = ensemble.encode_all(acts_f32[0].tolist())
    first_mask = ensemble.intersect(first_outs, float(threshold))
    H = len(first_mask)
    out_rows.append(np.asarray(first_mask, dtype=bool))

    # Remaining rows
    for i in range(1, N):
        outs = ensemble.encode_all(acts_f32[i].tolist())
        mask = ensemble.intersect(outs, float(threshold))
        if len(mask) != H:
            raise RuntimeError(f"Inconsistent intersection length: got {len(mask)} vs expected {H}")
        out_rows.append(np.asarray(mask, dtype=bool))

    return np.stack(out_rows, axis=0)