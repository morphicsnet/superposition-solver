from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def enumerate_subsets(m: int, max_order_k: int) -> List[int]:
    """
    Enumerate non-empty subset bitmasks over m elements, constrained to order <= max_order_k.
    Returns masks sorted by (subset_size, mask).
    """
    if m <= 0:
        return []
    k = int(max(1, min(m, int(max_order_k))))
    masks: List[int] = []
    full = 1 << m
    for mask in range(1, full):
        if mask.bit_count() <= k:
            masks.append(mask)
    masks.sort(key=lambda b: (b.bit_count(), b))
    return masks


def masked_predictions_logreg(model, X: np.ndarray, node_cols: List[int], subset_mask: int) -> np.ndarray:
    """
    Given a subset (as bitmask over node_cols), zero those columns in X and return
    the predicted probability of the positive class from a trained sklearn LogisticRegression.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array [N, U]")
    if subset_mask == 0:
        proba = model.predict_proba(X)  # type: ignore[attr-defined]
        return np.asarray(proba)[:, 1]
    cols_to_zero: List[int] = []
    for i, c in enumerate(node_cols):
        if (subset_mask >> i) & 1:
            cols_to_zero.append(int(c))
    if len(cols_to_zero) == 0:
        proba = model.predict_proba(X)  # type: ignore[attr-defined]
        return np.asarray(proba)[:, 1]
    X_mod = X.copy()
    X_mod[:, cols_to_zero] = 0.0
    proba = model.predict_proba(X_mod)  # type: ignore[attr-defined]
    return np.asarray(proba)[:, 1]


def compute_stii_for_hyperedge(
    store,
    edge_key: Tuple[int, ...],
    node_to_col: Dict[int, int],
    X_base: np.ndarray,
    y: np.ndarray,
    logreg_model,
    max_order_k: int,
) -> float:
    """
    Compute Shapley–Taylor Interaction Index for a given hyperedge.
    
    Steps:
      - Build node_cols by mapping node ids in edge_key to column indices in X_base (nodes_by_sample).
      - Compute baseline predicted probabilities on X_base using the provided logistic regression.
      - For each subset up to order k, zero the subset's columns and measure mean delta in prob:
            delta = mean(pred_base - pred_masked)
      - Send list of (subset_size, delta) to [compute_stii](nsi_core/src/metrics.rs:1) via PyHypergraphStore.
      - Returns the stii_value as computed by the Rust backend (also updates edge's stii weight inside the store).
    """
    if not isinstance(X_base, np.ndarray) or X_base.ndim != 2:
        raise ValueError("X_base must be [N, U]")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X_base.shape[0]:
        raise ValueError("y must be [N] aligned with X_base")

    # Map hyperedge node ids -> node feature columns
    node_cols: List[int] = []
    for nid in edge_key:
        if int(nid) not in node_to_col:
            # Node not present in node feature matrix; skip STII for this edge.
            return 0.0
        node_cols.append(int(node_to_col[int(nid)]))
    m = len(node_cols)
    if m == 0:
        return 0.0

    # Baseline predictions
    base_probs = np.asarray(logreg_model.predict_proba(X_base))[:, 1]  # type: ignore[attr-defined]
    subsets = enumerate_subsets(m, 1)
    
    deltas: List[Tuple[int, float]] = []
    for mask in subsets:
        masked_probs = masked_predictions_logreg(logreg_model, X_base, node_cols, mask)
        delta = float(np.mean(base_probs - masked_probs))
        subset_size = int(mask.bit_count())
        deltas.append((subset_size, delta))

    # Compute STII via Rust-backed store and return value
    node_ids_sorted = sorted([int(n) for n in edge_key])
    stii_value = float(store.compute_stii(node_ids_sorted, [(int(mask), float(d)) for (mask, d) in deltas]))
    return stii_value