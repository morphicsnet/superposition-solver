from __future__ import annotations

from typing import List, Tuple, Optional
import math

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# For schema-agnostic edge extraction when given a HIF dict
try:
    from python.dashboard.hif_utils import _collect_edges  # type: ignore
except Exception:  # pragma: no cover
    _collect_edges = None  # type: ignore


def hist(values, bins: int, title: str, xlabel: str) -> "matplotlib.figure.Figure":
    """
    Simple histogram returning a Matplotlib Figure for embedding (e.g., Streamlit).
    """
    arr = np.asarray(values).ravel()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if arr.size == 0 or not np.isfinite(arr).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    ax.hist(arr, bins=int(bins), color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def stii_bar(top_items: List[Tuple[str, float]], title: str) -> "matplotlib.figure.Figure":
    """
    Bar chart for top STII items.
    top_items: list of (label, value) sorted by value desc preferred.
    """
    labels = [str(k) for k, _ in top_items]
    vals = [float(v) for _, v in top_items]
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.35 * len(labels) + 1.5)))
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No STII entries", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    ax.barh(y_pos, vals, color="#2E8540", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("STII weight")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def _choose_top_edges(hif: dict, top_k_edges: int) -> List[Tuple[Tuple[int, ...], float, float]]:
    """
    Produce a list of (edge_key, stii_weight, observation_count) limited to top_k_edges,
    preferring sort by stii_weight (desc), then observation_count (desc), then size (desc).
    """
    edges_raw = []
    if _collect_edges is not None:
        for nodes, stii, obs in _collect_edges(hif):
            ek = tuple(sorted(int(n) for n in nodes))
            s = float(stii) if isinstance(stii, (int, float)) else float("nan")
            c = float(obs) if isinstance(obs, (int, float)) else float("nan")
            edges_raw.append((ek, s, c))
    else:
        # Fallback: try common HIF forms
        for item in (hif.get("edges") or hif.get("hyperedges") or []):
            try:
                nodes = item.get("nodes") or item.get("members") or item.get("edge") or []
                ek = tuple(sorted(int(n) for n in nodes))
                s = float(item.get("stii", float("nan")))
                c = float(item.get("observation_count", float("nan")))
                if ek:
                    edges_raw.append((ek, s, c))
            except Exception:
                continue

    def sort_key(t):
        ek, s, c = t
        # Use -inf for missing to push down
        s_key = s if math.isfinite(s) else -float("inf")
        c_key = c if math.isfinite(c) else -float("inf")
        return (s_key, c_key, len(ek))

    edges_raw.sort(key=sort_key, reverse=True)
    return edges_raw[: max(0, int(top_k_edges))]


def hypergraph_small_graph(hif: dict, top_k_edges: int = 25) -> "matplotlib.figure.Figure":
    """
    Build a small bipartite visualization:
      - Left partition: hyperedge nodes labeled as e:<rank> (or e:<size>)
      - Right partition: member node ids labeled as n:<id>

    Edges chosen by top STII weight where available, else by observation_count, else by size.
    """
    chosen = _choose_top_edges(hif, top_k_edges=top_k_edges)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    if not chosen:
        ax.text(0.5, 0.5, "No hyperedges available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    G = nx.Graph()
    left_nodes = []
    right_nodes = set()

    # Add bipartite nodes and edges
    for rank, (ek, s, c) in enumerate(chosen, start=1):
        e_label = f"e:{rank}"
        left_nodes.append(e_label)
        G.add_node(e_label, bipartite=0, size=len(ek), stii=s, count=c)
        for nid in ek:
            n_label = f"n:{int(nid)}"
            right_nodes.add(n_label)
            G.add_node(n_label, bipartite=1)
            G.add_edge(e_label, n_label)

    right_nodes = sorted(right_nodes)

    # Layout: bipartite layout then small jitter
    pos = {}
    try:
        pos = nx.bipartite_layout(G, nodes=left_nodes, align="vertical", scale=1.0)
    except Exception:
        # Fallback to spring layout with fixed seeds
        pos = nx.spring_layout(G, seed=42, k=0.8 / math.sqrt(G.number_of_nodes()))

    # Draw
    ax.axis("off")
    # Left partition
    nx.draw_networkx_nodes(G, pos, nodelist=left_nodes, node_color="#4C78A8", node_size=400, alpha=0.9, ax=ax)
    # Right partition
    nx.draw_networkx_nodes(G, pos, nodelist=right_nodes, node_color="#F58518", node_size=250, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color="#777", alpha=0.6, ax=ax)

    # Labels: edges show size or short STII
    edge_labels = {}
    for n in left_nodes:
        data = G.nodes[n]
        lbl = f"{n} | k={data.get('size', '?')}"
        s = data.get("stii", None)
        if isinstance(s, (int, float)) and math.isfinite(s):
            lbl += f" | s={s:.2f}"
        edge_labels[n] = lbl

    nx.draw_networkx_labels(G, pos, labels={**{n: n for n in right_nodes}, **edge_labels}, font_size=8, ax=ax)
    ax.set_title("Hypergraph bipartite view (top edges)")
    fig.tight_layout()
    return fig