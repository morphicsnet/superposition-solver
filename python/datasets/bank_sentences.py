from typing import List, Tuple
import numpy as np


def generate_bank_dataset(n_per_class: int, seed: int) -> Tuple[List[str], List[int]]:
    """
    Generate a balanced dataset of sentences for two 'bank' concepts:
      - Class 0: 'river bank' contexts (boats, shoreline, current)
      - Class 1: 'finance bank' contexts (loans, interest rates, accounts)

    Deterministic via numpy RNG.

    Args:
        n_per_class: Number of samples per class.
        seed: RNG seed.

    Returns:
        texts: List of sentences.
        labels: List of 0/1 labels aligned with texts.
    """
    rng = np.random.default_rng(seed)

    # Synonym pools and templates for river bank
    river_water = ["river", "stream", "waterway"]
    shoreline_adj = ["rocky", "sandy", "muddy", "grassy", "steep"]
    current_syn = ["current", "flow", "stream"]
    boats = ["kayak", "canoe", "boat", "raft", "ferry"]
    wildlife = ["herons", "ducks", "fish", "otters", "turtles"]
    activities = ["fishing", "picnicking", "skipping stones", "camping", "resting"]
    river_verbs = ["drifted", "floated", "moored", "landed", "glided"]

    river_templates = [
        "The {boat} {verb} near the {adj} bank of the {water}.",
        "Children played on the {adj} bank while the {water} {current} moved by.",
        "We sat by the bank of the {water}, {activity} and watching the {wildlife}.",
        "Waves lapped against the {adj} river bank as a {boat} passed.",
        "The {water} {current} carved the bank over years, leaving {wildlife} in the shallows.",
        "A trail follows the bank of the {water}, perfect for {activity}.",
    ]

    # Synonym pools and templates for finance bank
    loans = ["loan", "mortgage", "credit line", "student loan"]
    interest = ["interest", "APR", "rate"]
    accounts = ["checking account", "savings account", "business account"]
    money_verbs = ["deposited", "withdrew", "transferred", "invested"]
    finance_adj = ["local", "private", "national", "community"]
    fees = ["fees", "charges", "minimums", "penalties"]

    finance_templates = [
        "She applied for a {loan} at the {adj} bank downtown.",
        "The bank raised {interest} rates after the meeting.",
        "He checked his {account} balance at the bank branch.",
        "They {money_verb} funds through the bank's mobile app.",
        "A {adj} bank lowered {interest} on {loan}s to attract customers.",
        "The bank removed monthly {fees} on new {account}s.",
    ]

    def sample_from(pool: List[str]) -> str:
        return pool[rng.integers(0, len(pool))]

    def build_river_sentence() -> str:
        t = sample_from(river_templates)
        return t.format(
            boat=sample_from(boats),
            verb=sample_from(river_verbs),
            adj=sample_from(shoreline_adj),
            water=sample_from(river_water),
            current=sample_from(current_syn),
            activity=sample_from(activities),
            wildlife=sample_from(wildlife),
        )

    def build_finance_sentence() -> str:
        t = sample_from(finance_templates)
        return t.format(
            loan=sample_from(loans),
            adj=sample_from(finance_adj),
            interest=sample_from(interest),
            account=sample_from(accounts),
            money_verb=sample_from(money_verbs),
            fees=sample_from(fees),
        )

    river_texts = [build_river_sentence() for _ in range(n_per_class)]
    finance_texts = [build_finance_sentence() for _ in range(n_per_class)]

    texts = river_texts + finance_texts
    labels = [0] * n_per_class + [1] * n_per_class

    # Shuffle deterministically while maintaining alignment
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    return texts, labels