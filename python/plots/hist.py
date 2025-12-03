from __future__ import annotations

from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(values: np.ndarray, bins: int, title: str, path: str) -> None:
    """
    Save a histogram plot to 'path'.
    - X-axis: polysemanticity (poly(f) count)
    - Y-axis: frequency
    """
    values = np.asarray(values)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    plt.title(title)
    plt.xlabel("poly(f): # of concepts with P(C_k | f_active) > eps")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()