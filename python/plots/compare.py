from __future__ import annotations

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_dual_hist(values_a: np.ndarray, values_b: np.ndarray, bins: int, labels: Tuple[str, str], title: str, path: str) -> None:
    """
    Overlaid histograms comparison, with legend and grid.
    - X-axis: poly(f) count (or provided values)
    - Y-axis: frequency
    """
    a = np.asarray(values_a).ravel()
    b = np.asarray(values_b).ravel()

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(a, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.70, label=labels[0])
    plt.hist(b, bins=bins, color="#F58518", edgecolor="white", alpha=0.55, label=labels[1])
    plt.title(title)
    plt.xlabel("poly(f): # of concepts with P(C_k | f_active) > eps")
    plt.ylabel("frequency")
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()