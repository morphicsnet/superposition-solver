from __future__ import annotations

from typing import List, Optional

import numpy as np


def _require_py_nsi():
    try:
        # Imported lazily to allow repo to run without the wheel until needed
        from py_nsi import PySpike  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "py_nsi is not importable. Build the local Rust wheel first:\n"
            "  cd py_nsi && maturin develop --release\n"
            "Then re-run the demo."
        ) from e
    return PySpike


def activation_to_spike_time_py(
    activation: float,
    t_start: float,
    delta_t: float,
    min_sigmoid: float,
) -> float | None:
    """
    Python mirror of Rust latency-phase mapping, see
    [activation_to_spike_time()](nsi_core/src/encoding.rs:1).

    s = 1 / (1 + exp(-activation))
    if s < min_sigmoid: return None
    else: return t_start + (1 - s) * max(delta_t, 0)

    Notes:
    - Higher activations map to earlier spike times within [t_start, t_start + delta_t].
    - min_sigmoid drops very low-sigmoid activations.
    """
    # Numerically-stable enough for typical ranges
    s = 1.0 / (1.0 + float(np.exp(-float(activation))))
    if s < float(min_sigmoid):
        return None
    t = float(t_start) + (1.0 - s) * max(float(delta_t), 0.0)
    return t


def encode_spikes_for_sample(
    ensemble,
    activation_vector: "np.ndarray",
    t_start: float,
    delta_t: float,
    min_sigmoid: float,
) -> List:
    """
    For a single sample:
      - Get per-encoder outputs via [PyEnsemble.encode_all()](py_nsi/src/lib.rs:1)
      - For each encoder e and neuron j with activation v, compute spike time
        using [activation_to_spike_time_py()](python/encoders/spike.py:1)
      - Create [PySpike](py_nsi/src/lib.rs:1) (ensemble_id=e, neuron_id=j, t=time)
      - Return spikes sorted by (t, node-id) for determinism

    Determinism:
      - Stable sorting by time then canonical node id ((e << 32) | j)

    Filtering:
      - Only v > 0.0 are considered to avoid emitting spikes for zero/negative activations.
      - Additionally dropped by min_sigmoid in latency-phase code.
    """
    PySpike = _require_py_nsi()

    if not isinstance(activation_vector, np.ndarray) or activation_vector.ndim != 1:
        raise ValueError("activation_vector must be a 1D numpy array [D]")

    # Compute all encoder outputs for this sample
    outs_per_encoder = ensemble.encode_all(activation_vector.astype(np.float32, copy=False).tolist())
    spikes: List = []

    for e_idx, vec in enumerate(outs_per_encoder):
        # vec is a Python list[float] for encoder e_idx
        for j, v in enumerate(vec):
            # Heuristic: only positive encoder activations can spike
            if float(v) <= 0.0:
                continue
            t = activation_to_spike_time_py(float(v), float(t_start), float(delta_t), float(min_sigmoid))
            if t is None:
                continue
            sp = PySpike(int(e_idx), int(j), float(t))
            # Sort deterministically by (t, node id)
            node_id = ((int(e_idx) & 0xFFFF) << 32) | (int(j) & 0xFFFFFFFF)
            spikes.append((float(t), int(node_id), sp))

    # Deterministic order
    spikes.sort(key=lambda x: (x[0], x[1]))
    return [sp for (_, __, sp) in spikes]


def encode_spikes_batch(
    ensemble,
    acts: "np.ndarray",
    t_start: float,
    delta_t: float,
    min_sigmoid: float,
) -> List[List]:
    """
    Encode a batch of activation vectors into spike lists, calling
    [encode_spikes_for_sample()](python/encoders/spike.py:1) per row.

    Args:
        ensemble: [PyEnsemble](py_nsi/src/lib.rs:1) instance
        acts: np.ndarray [N, D] float activations
        t_start, delta_t, min_sigmoid: spike timing params

    Returns:
        List[List[PySpike]]: spikes per sample (sorted deterministically)
    """
    if not isinstance(acts, np.ndarray) or acts.ndim != 2:
        raise ValueError("acts must be a 2D numpy array [N, D]")

    acts_f32 = acts.astype(np.float32, copy=False)
    out: List[List] = []
    for i in range(acts_f32.shape[0]):
        out.append(
            encode_spikes_for_sample(
                ensemble=ensemble,
                activation_vector=acts_f32[i],
                t_start=t_start,
                delta_t=delta_t,
                min_sigmoid=min_sigmoid,
            )
        )
    return out