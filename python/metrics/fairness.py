from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def gender_concept_probs(nodes_by_sample: np.ndarray, genders: np.ndarray) -> np.ndarray:
    """
    Compute P(gender | node_active) for gender in {0,1}.
    Args:
      nodes_by_sample: [N, U] bool/float where True/1 indicates node fired at least once for the sample
      genders: [N] int (0=male, 1=female)
    Returns:
      probs: [U, 2] float32 where probs[u, g] = P(gender=g | node u active)
             If node u never active, returns [0.5, 0.5].
    """
    if not isinstance(nodes_by_sample, np.ndarray) or nodes_by_sample.ndim != 2:
        raise ValueError("nodes_by_sample must be a 2D numpy array [N, U]")
    if not isinstance(genders, np.ndarray) or genders.ndim != 1:
        raise ValueError("genders must be a 1D numpy array [N]")
    if nodes_by_sample.shape[0] != genders.shape[0]:
        raise ValueError("nodes_by_sample and genders must have the same #rows")

    N, U = nodes_by_sample.shape
    active = nodes_by_sample.astype(bool, copy=False)
    g = genders.astype(np.int32, copy=False)
    probs = np.zeros((U, 2), dtype=np.float32)

    # Count for each node: among active rows, distribution over genders
    for u in range(U):
        mask = active[:, u]
        k = int(mask.sum())
        if k <= 0:
            probs[u, :] = 0.5  # uniform when never active
            continue
        # P(g=0 | active), P(g=1 | active)
        g_act = g[mask]
        p1 = float((g_act == 1).sum()) / float(k)
        probs[u, 1] = p1
        probs[u, 0] = 1.0 - p1
    return probs


def report_bias_presence(
    minimal_edges: List[Tuple[int, ...]],
    edge_to_nodes: Dict[Tuple[int, ...], List[int]],
    node_gender_probs: np.ndarray,
    node_keys: List[Tuple[int]],
    threshold: float = 0.6,
) -> Dict:
    """
    Determine if nodes with high gender association are present in the minimal circuit.

    Args:
      minimal_edges: list of hyperedge keys (tuples of node_ids)
      edge_to_nodes: dict mapping edge_key -> list[node_id]
      node_gender_probs: [U, 2] P(g | node_active)
      node_keys: list[(node_id,)] aligned to columns in node_gender_probs
      threshold: float; consider a node 'gender-associated' if max P >= threshold

    Returns:
      dict with counts and example nodes in minimal circuit.
    """
    if not isinstance(node_gender_probs, np.ndarray) or node_gender_probs.ndim != 2:
        raise ValueError("node_gender_probs must be [U, 2]")
    if len(node_keys) != node_gender_probs.shape[0]:
        raise ValueError("node_keys length must match node_gender_probs rows")

    # Map node_id -> column index
    node_to_col: Dict[int, int] = {int(nid): i for i, (nid,) in enumerate(node_keys)}

    # Identify gender-associated nodes by threshold
    max_p = node_gender_probs.max(axis=1)
    assoc_cols = np.where(max_p >= float(threshold))[0]
    assoc_node_ids: List[int] = [int(node_keys[idx][0]) for idx in assoc_cols]
    assoc_set = set(assoc_node_ids)

    # Collect nodes present in minimal circuit
    minimal_nodes: List[int] = []
    for ek in minimal_edges:
        for nid in edge_to_nodes.get(ek, list(ek)):
            minimal_nodes.append(int(nid))
    minimal_nodes_set = set(minimal_nodes)

    # Intersect to find gender-associated nodes in minimal circuit
    in_minimal_assoc = [nid for nid in assoc_node_ids if nid in minimal_nodes_set]
    count_in_minimal = len(in_minimal_assoc)

    # Build examples sorted by association strength (max P)
    examples: List[Dict] = []
    for nid in assoc_node_ids:
        col = node_to_col[nid]
        p_male, p_female = float(node_gender_probs[col, 0]), float(node_gender_probs[col, 1])
        in_min = nid in minimal_nodes_set
        # list minimal edges containing nid (up to 5)
        containing = []
        for ek in minimal_edges:
            nodes = edge_to_nodes.get(ek, list(ek))
            if nid in nodes:
                containing.append([int(x) for x in ek])
                if len(containing) >= 5:
                    break
        examples.append(
            {
                "node_id": int(nid),
                "p_male": p_male,
                "p_female": p_female,
                "in_minimal": bool(in_min),
                "example_edges": containing,
            }
        )
    examples.sort(key=lambda d: max(d["p_male"], d["p_female"]), reverse=True)
    examples = examples[:20]

    report = {
        "threshold": float(threshold),
        "num_biased_nodes": int(len(assoc_node_ids)),
        "num_minimal_edges": int(len(minimal_edges)),
        "biased_nodes_in_minimal_count": int(count_in_minimal),
        "biased_nodes_in_minimal_ratio": float(count_in_minimal / max(len(assoc_node_ids), 1)),
        "any_biased_node_in_minimal": bool(count_in_minimal > 0),
        "examples": examples,
    }
    return report