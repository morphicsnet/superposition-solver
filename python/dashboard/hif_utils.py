from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------- Robust IO helpers ----------


def load_json(path: str) -> dict:
    """
    Best-effort JSON loader.
    Returns {} if the file does not exist or cannot be parsed.
    """
    try:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def load_hif(path: str) -> dict:
    """
    Load a Hypergraph Interchange Format (HIF) JSON if present, else {}.
    The HIF schema may vary slightly across demos; downstream functions should be defensive.
    """
    return load_json(path)


# ---------- HIF summarization ----------


def _edge_nodes_from_item(item: Any) -> Tuple[List[int], Optional[float], Optional[float]]:
    """
    Extract member node ids and optional attributes (stii_weight, observation_count)
    from a single hyperedge representation.

    Returns:
        (nodes, stii_weight, observation_count)
    """
    nodes: List[int] = []
    stii: Optional[float] = None
    obs: Optional[float] = None

    if isinstance(item, dict):
        # Try common member keys
        for k in ("nodes", "members", "incidences", "edge", "node_ids"):
            if k in item and isinstance(item[k], (list, tuple)):
                nodes = [int(x) for x in item[k]]
                break

        # Attributes for weights and counts (several possible keys)
        # Direct keys
        for k in ("stii", "stii_weight", "stiiValue", "weight"):
            if k in item and isinstance(item[k], (int, float)):
                stii = float(item[k])
                break
        for k in ("count", "observation_count", "observations", "frequency"):
            if k in item and isinstance(item[k], (int, float)):
                obs = float(item[k])
                break

        # Nested attrs
        attrs = item.get("attrs") if isinstance(item.get("attrs"), dict) else {}
        if stii is None:
            for k in ("stii", "stii_weight"):
                if k in attrs and isinstance(attrs[k], (int, float)):
                    stii = float(attrs[k])
                    break
        if obs is None:
            for k in ("count", "observation_count"):
                if k in attrs and isinstance(attrs[k], (int, float)):
                    obs = float(attrs[k])
                    break

    elif isinstance(item, (list, tuple)):
        # Plain list of member ids
        nodes = [int(x) for x in item]
    else:
        # Unknown representation; return empty
        nodes = []

    return nodes, stii, obs


def _collect_edges(hif: dict) -> List[Tuple[List[int], Optional[float], Optional[float]]]:
    """
    Extract a list of hyperedges from a HIF dict in a schema-agnostic way.
    Each element is (member_nodes, stii_weight?, observation_count?).
    """
    if not isinstance(hif, dict):
        return []

    candidate_lists: List[Iterable] = []
    # Common top-level containers
    for key in ("edges", "hyperedges", "E"):
        if key in hif and isinstance(hif[key], list):
            candidate_lists.append(hif[key])

    # Nested common containers
    if "data" in hif and isinstance(hif["data"], dict):
        for key in ("edges", "hyperedges", "E"):
            if key in hif["data"] and isinstance(hif["data"][key], list):
                candidate_lists.append(hif["data"][key])

    # Fallback: islands - treat each island as a set of nodes
    if not candidate_lists and "islands" in hif and isinstance(hif["islands"], list):
        candidate_lists.append(hif["islands"])

    edges: List[Tuple[List[int], Optional[float], Optional[float]]] = []
    for lst in candidate_lists:
        for item in lst:
            nodes, stii, obs = _edge_nodes_from_item(item)
            if nodes:
                edges.append((nodes, stii, obs))

    # Deduplicate by canonical tuple of sorted members
    seen: set = set()
    deduped: List[Tuple[List[int], Optional[float], Optional[float]]] = []
    for nodes, stii, obs in edges:
        key = tuple(sorted(nodes))
        if key not in seen:
            seen.add(key)
            deduped.append((list(key), stii, obs))

    return deduped


def summarize_hif(hif: dict) -> dict:
    """
    Summarize a HIF dict.
    Returns:
      {
        "num_nodes": int,
        "num_edges": int,
        "edge_size_hist": {size: count, ...},
        "stii": {
          "min": float | None,
          "max": float | None,
          "mean": float | None,
          "values": list[float],  # histogram-ready
        }
      }
    """
    edges = _collect_edges(hif)
    # Edge sizes and unique node ids
    size_counts: Dict[int, int] = {}
    node_ids: set[int] = set()
    stii_values: List[float] = []

    for nodes, stii, _obs in edges:
        for nid in nodes:
            node_ids.add(int(nid))
        size = int(len(nodes))
        size_counts[size] = size_counts.get(size, 0) + 1
        if isinstance(stii, (int, float)):
            stii_values.append(float(stii))

    stii_stats = {
        "min": float(np.min(stii_values)) if stii_values else None,
        "max": float(np.max(stii_values)) if stii_values else None,
        "mean": float(np.mean(stii_values)) if stii_values else None,
        "values": stii_values,  # ready for hist
    }

    return {
        "num_nodes": int(len(node_ids)),
        "num_edges": int(len(edges)),
        "edge_size_hist": {int(k): int(v) for k, v in sorted(size_counts.items())},
        "stii": stii_stats,
    }


# ---------- Metrics aggregation across demos ----------


def _safe_load(path: str) -> Optional[dict]:
    d = load_json(path)
    return d if d else None


def load_metrics(run_dir: str) -> dict:
    """
    Best-effort loader for known metrics/artifacts emitted by Demos 1–4.

    Files (if present):
      - baseline: metrics.json
      - ensemble: metrics_single.json, metrics_intersection.json, compare.json
      - hypergraph (demo3): metrics_hyperedges.json
      - causal (demo4): stii_values.json, acdc_minimal_circuit.json, fairness_report.json
      - HIF JSON: hypergraph_stii.hif.json (demo4) or hypergraph.hif.json (demo3)
      - arrays/plots (optional): *.npy, *.png from upstream demos

    Returns a dict with keys when present:
      {
        "baseline": {...},
        "ensemble_single": {...},
        "ensemble_intersection": {...},
        "ensemble_compare": {...},
        "hypergraph": {...},
        "causal_stii": {...},
        "acdc": {...},
        "fairness": {...},
        "hif_path": str | None,
        "arrays": { "poly_counts": paths, ... },   # optional paths for convenience
        "plots": { "poly_hist": paths, ... }       # optional paths for convenience
      }
    """
    out: Dict[str, Any] = {}
    if not run_dir or not os.path.isdir(run_dir):
        return out

    def P(name: str) -> str:
        return os.path.join(run_dir, name)

    # Baseline (demo1)
    b = _safe_load(P("metrics.json"))
    if b:
        out["baseline"] = b

    # Ensemble (demo2)
    e1 = _safe_load(P("metrics_single.json"))
    if e1:
        out["ensemble_single"] = e1
    e2 = _safe_load(P("metrics_intersection.json"))
    if e2:
        out["ensemble_intersection"] = e2
    ec = _safe_load(P("compare.json"))
    if ec:
        out["ensemble_compare"] = ec

    # Hypergraph (demo3)
    h = _safe_load(P("metrics_hyperedges.json"))
    if h:
        out["hypergraph"] = h

    # Causal (demo4)
    stii = _safe_load(P("stii_values.json"))
    if stii:
        out["causal_stii"] = stii
    acdc = _safe_load(P("acdc_minimal_circuit.json"))
    if acdc:
        out["acdc"] = acdc
    fair = _safe_load(P("fairness_report.json"))
    if fair:
        out["fairness"] = fair

    # HIF paths (demo4 first, then demo3)
    hif4 = P("hypergraph_stii.hif.json")
    hif3 = P("hypergraph.hif.json")
    out["hif_path"] = hif4 if os.path.exists(hif4) else (hif3 if os.path.exists(hif3) else None)

    # Optional array/plot paths (if present)
    arrays: Dict[str, str] = {}
    plots: Dict[str, str] = {}
    # Demo1 arrays/plots
    for name in ("probs.npy", "poly_counts.npy", "entropy.npy"):
        p = P(name)
        if os.path.exists(p):
            arrays[name] = p
    if os.path.exists(P("poly_hist.png")):
        plots["poly_hist_baseline"] = P("poly_hist.png")

    # Demo2 arrays/plots
    for name in ("probs_single.npy", "poly_counts_single.npy", "entropy_single.npy",
                 "probs_intersection.npy", "poly_counts_intersection.npy", "entropy_intersection.npy"):
        p = P(name)
        if os.path.exists(p):
            arrays[name] = p
    for name in ("poly_hist_single.png", "poly_hist_intersection.png", "poly_hist_dual.png"):
        p = P(name)
        if os.path.exists(p):
            plots[name] = p

    # Demo3 arrays/plots
    if os.path.exists(P("features_hyperedges.npy")):
        arrays["features_hyperedges.npy"] = P("features_hyperedges.npy")
    if os.path.exists(P("poly_hist_hyperedges.png")):
        plots["poly_hist_hyperedges"] = P("poly_hist_hyperedges.png")

    out["arrays"] = arrays
    out["plots"] = plots
    return out