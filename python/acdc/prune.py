from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _train_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", random_state=int(seed), max_iter=1000)
    clf.fit(X, y)
    return clf


def _accuracy(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = clf.predict(X)
    return float(accuracy_score(y, y_pred))


def acdc_minimal_circuit(
    edge_keys: List[Tuple[int, ...]],
    stii: Dict[Tuple[int, ...], float],
    X_edge: np.ndarray,
    y: np.ndarray,
    tolerance_drop: float,
    max_edges: int,
    seed: int = 0,
) -> Dict:
    """
    Greedy ACDC-style pruning over hyperedge features.

    Strategy:
      - Train a LogisticRegression on a train split of X_edge to predict y.
      - Evaluate base accuracy on held-out test split.
      - Starting from 'kept' = all edges, iteratively propose removing one edge:
          * For each candidate edge e in kept, zero that column (along with already-removed columns),
            evaluate test accuracy drop relative to current accuracy.
          * Prefer the edge with the smallest drop; use STII ascending as a secondary tiebreaker.
          * Accept removal if drop <= tolerance_drop.
      - Stop when no removal within tolerance, or when len(kept) <= max_edges.

    Returns:
      {
        "kept_edges": list[tuple[int,...]],
        "removed_edges": list[tuple[int,...]],
        "base_acc": float,
        "final_acc": float
      }
    """
    if not isinstance(X_edge, np.ndarray) or X_edge.ndim != 2:
        raise ValueError("X_edge must be [N, E]")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X_edge.shape[0]:
        raise ValueError("y must be [N] aligned with X_edge")
    N, E = X_edge.shape
    if E == 0 or len(edge_keys) == 0:
        return {
            "kept_edges": [],
            "removed_edges": [],
            "base_acc": 0.0,
            "final_acc": 0.0,
        }

    # Ensure column order matches edge_keys
    if len(edge_keys) != E:
        raise ValueError(f"edge_keys length {len(edge_keys)} must equal X_edge.shape[1] {E}")

    X = X_edge.astype(np.float32, copy=False)
    y_i32 = y.astype(np.int32, copy=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_i32, test_size=0.2, random_state=int(seed), stratify=y_i32
    )
    clf = _train_logreg(X_train, y_train, seed=seed)
    base_acc = _accuracy(clf, X_test, y_test)

    kept: List[Tuple[int, ...]] = list(edge_keys)
    removed: List[Tuple[int, ...]] = []
    key_to_col: Dict[Tuple[int, ...], int] = {k: i for i, k in enumerate(edge_keys)}

    # Current mask of removed columns (applied on top of X_test for evaluation)
    removed_cols: set[int] = set()
    current_acc = base_acc

    while True:
        if len(kept) <= int(max_edges):
            break

        # Evaluate each candidate removal and pick smallest drop; tie-break by lower STII
        best_key = None
        best_acc = -1.0
        best_drop = float("inf")
        # Build a base-masked version reflecting already removed cols
        if len(removed_cols) > 0:
            X_eval_base = X_test.copy()
            X_eval_base[:, list(removed_cols)] = 0.0
        else:
            X_eval_base = X_test

        for k in kept:
            col = key_to_col[k]
            if col in removed_cols:
                continue
            X_eval = X_eval_base.copy()
            X_eval[:, col] = 0.0
            acc_k = _accuracy(clf, X_eval, y_test)
            drop = current_acc - acc_k
            # Secondary key: STII ascending (missing -> +inf to de-prioritize)
            stii_val = float(stii.get(k, float("inf")))
            rank_tuple = (drop, stii_val, col)  # smaller is better
            if drop < best_drop or (
                abs(drop - best_drop) <= 1e-12 and (stii_val, col) < (float(stii.get(best_key, float("inf"))) if best_key is not None else float("inf"), key_to_col.get(best_key, 10**9) if best_key is not None else 10**9)  # type: ignore
            ):
                best_key = k
                best_acc = acc_k
                best_drop = drop

        if best_key is None:
            break

        # Accept removal only if within tolerance
        if best_drop <= float(tolerance_drop):
            removed.append(best_key)
            kept.remove(best_key)
            removed_cols.add(key_to_col[best_key])
            current_acc = best_acc
        else:
            # No acceptable removal; stop
            break

    result = {
        "kept_edges": kept,
        "removed_edges": removed,
        "base_acc": float(base_acc),
        "final_acc": float(current_acc),
    }
    return result