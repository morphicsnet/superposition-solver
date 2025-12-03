from __future__ import annotations

from typing import List, Tuple
import numpy as np


def generate_loans_dataset(n_samples: int, seed: int, bias_strength: float, noise: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Generate a toy loans dataset with a gender-protected attribute and injected bias.

    Returns (texts, labels, protected) where:
      - texts: list[str] describing features and a gender token
      - labels: np.ndarray[int] loan_denied (1) vs approved (0)
      - protected: np.ndarray[int] gender (0=male, 1=female)

    Design:
      - Base approval rule depends on income, credit score, and debt ratio.
      - A gender bias term is applied: for female (1), with probability=bias_strength,
        the score is shifted downward.
      - A final label flip noise is applied with probability=noise.

    Determinism: uses numpy.random.Generator(seed) only.

    Notes:
      - Text vocabulary is small but varied; each record includes a clear gender token:
        "gender: male" / "gender: female".
    """
    rng = np.random.default_rng(int(seed))
    n = int(n_samples)
    if n <= 0:
        return [], np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    # Protected attribute: gender (balanced-ish)
    genders = rng.integers(low=0, high=2, size=n, dtype=np.int32)  # 0=male, 1=female

    # Continuous features
    # income in thousands, credit score, debt_ratio in [0,1]
    income_k = np.clip(rng.normal(loc=60.0, scale=20.0, size=n), 15.0, 200.0)  # $k
    credit = np.clip(rng.normal(loc=650.0, scale=100.0, size=n), 300.0, 850.0)
    debt_ratio = np.clip(rng.beta(2.0, 5.0, size=n), 0.0, 1.0)

    # Base score (higher means more likely to be approved)
    # Normalize features into roughly comparable ranges
    inc_n = income_k / 120.0
    crd_n = credit / 850.0
    dr_n = debt_ratio  # already in [0,1]

    score = 0.6 * inc_n + 0.9 * crd_n - 0.8 * dr_n - 0.6  # threshold ~ 0
    # Inject group bias: for female (1), with prob=bias_strength, shift down the score
    if bias_strength > 0.0:
        biased = (genders == 1) & (rng.random(n) < float(bias_strength))
        # Shift chosen so impact is visible but not overwhelming
        score[biased] -= 0.35

    # Base decision: approve if score >= 0, else deny
    labels = (score < 0.0).astype(np.int32)  # 1 = denied, 0 = approved

    # Flip a fraction via noise
    if noise > 0.0:
        flip = rng.random(n) < float(noise)
        labels[flip] = 1 - labels[flip]

    # Text rendering with a small vocabulary variety
    gender_tokens = np.where(genders == 0, "gender: male", "gender: female")
    inc_words = np.array(["income", "earnings", "salary"])
    crd_words = np.array(["credit_score", "fico", "credit"])
    dr_words = np.array(["debt_ratio", "dti", "debt_to_income"])

    inc_sel = rng.integers(0, len(inc_words), size=n)
    crd_sel = rng.integers(0, len(crd_words), size=n)
    dr_sel = rng.integers(0, len(dr_words), size=n)

    texts: List[str] = []
    for i in range(n):
        # Mix order slightly to diversify prompts
        if i % 2 == 0:
            s = f"{gender_tokens[i]}; {inc_words[inc_sel[i]]}={income_k[i]:.0f}k; {crd_words[crd_sel[i]]}={credit[i]:.0f}; {dr_words[dr_sel[i]]}={debt_ratio[i]:.2f}"
        else:
            s = f"{inc_words[inc_sel[i]]}={income_k[i]:.0f}k; {dr_words[dr_sel[i]]}={debt_ratio[i]:.2f}; {gender_tokens[i]}; {crd_words[crd_sel[i]]}={credit[i]:.0f}"
        texts.append(s)

    return texts, labels, genders