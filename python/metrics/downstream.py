from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def evaluate_logreg(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> Dict:
    """
    Train/test split (80/20 stratified), LogisticRegression(solver="liblinear"),
    return {"accuracy": float} on the held-out test set.

    Args:
        features: np.ndarray [N, H]
        labels: np.ndarray [N]
        seed: int random_state

    Returns:
        dict: {"accuracy": float}
    """
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        raise ValueError("features must be a 2D numpy array [N, H]")
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("labels must be a 1D numpy array [N]")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same #rows")

    X = features.astype(np.float32, copy=False)
    y = labels.astype(np.int32, copy=False)

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(seed), stratify=y
    )

    # Small, robust classifier for tiny datasets
    clf = LogisticRegression(solver="liblinear", random_state=int(seed), max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    return {"accuracy": acc}