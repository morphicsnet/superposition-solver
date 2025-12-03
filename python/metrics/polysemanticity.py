from __future__ import annotations

import numpy as np


def concept_probs(features: np.ndarray, labels: np.ndarray, num_concepts: int, active_threshold: float) -> np.ndarray:
    """
    Compute P(C_k | f_j active) for each feature j and concept k.
    A feature is 'active' on a sample if feature value > active_threshold.

    Args:
        features: [N, H] feature activations (after top-k, non-negative recommended)
        labels: [N] integer labels in [0, num_concepts-1]
        num_concepts: number of labeled concepts (m)
        active_threshold: threshold for 'active'

    Returns:
        prob: [H, m] where prob[j, k] = P(C_k | f_j active)
              If feature j is never active, returns uniform distribution over concepts for that j.
    """
    assert features.ndim == 2, "features must be [N, H]"
    assert labels.ndim == 1 and features.shape[0] == labels.shape[0], "labels shape mismatch"
    N, H = features.shape
    m = int(num_concepts)

    # Active mask: [N, H]
    active = features > active_threshold

    # One-hot labels: [N, m]
    onehot = np.eye(m, dtype=np.float32)[labels.astype(int)]

    # Counts per feature per concept: [H, m]
    counts = active.T.astype(np.float32) @ onehot  # sum over N
    # Denominator: number of actives per feature: [H, 1]
    denom = active.sum(axis=0).astype(np.float32).reshape(H, 1)

    prob = np.zeros((H, m), dtype=np.float32)
    nonzero = denom.squeeze(-1) > 0
    prob[nonzero] = counts[nonzero] / denom[nonzero]
    # Fallback to uniform if never active
    if np.any(~nonzero):
        prob[~nonzero] = 1.0 / m
    return prob


def poly_count(prob: np.ndarray, eps: float) -> np.ndarray:
    """
    Count number of concepts per feature with mass > eps.
    prob: [H, m]
    returns: [H] integer counts
    """
    assert prob.ndim == 2, "prob must be [H, m]"
    return (prob > eps).sum(axis=1).astype(np.int32)


def entropy(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Entropy per feature (base-2): H(f) = -sum_k p_k log2(p_k)
    prob: [H, m]
    returns: [H]
    """
    p = np.clip(prob, eps, 1.0)
    return (- (p * (np.log(p) / np.log(2.0))).sum(axis=1)).astype(np.float32)


def summarize_polysemanticity(prob: np.ndarray, eps: float) -> dict:
    """
    Summary dictionary:
      - median_poly
      - p90_poly
      - monosemantic_rate (poly == 1)
    """
    pc = poly_count(prob, eps).astype(np.float32)
    H = float(len(pc)) if len(pc) > 0 else 1.0
    summary = {
        "median_poly": float(np.median(pc)) if len(pc) > 0 else 0.0,
        "p90_poly": float(np.percentile(pc, 90.0)) if len(pc) > 0 else 0.0,
        "monosemantic_rate": float((pc == 1).sum() / H),
    }
    return summary