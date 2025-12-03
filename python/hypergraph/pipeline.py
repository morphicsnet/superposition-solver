from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def _require_py_nsi():
    try:
        from py_nsi import PyGse, PyHypergraphStore  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "py_nsi is not importable. Build the local Rust wheel first:\n"
            "  cd py_nsi && maturin develop --release\n"
            "Then re-run the demo."
        ) from e
    return PyGse, PyHypergraphStore


def node_id_u64(ensemble_id: int, neuron_id: int) -> int:
    """
    Canonical 64-bit node id, matching [Spike.node_id()](nsi_core/src/encoding.rs:1):
        (ensemble_id << 32) | neuron_id
    """
    return ((int(ensemble_id) & 0xFFFF) << 32) | (int(neuron_id) & 0xFFFFFFFF)


def build_hypergraph_with_nodes(
    ensemble,
    acts: "np.ndarray",
    labels: "np.ndarray",
    t_start: float,
    delta_t: float,
    min_sigmoid: float,
    gse_window: float,
):
    """
    Build temporal-coincidence hypergraph (as in [build_hypergraph()](python/hypergraph/pipeline.py:28)) and, in addition,
    construct node-level features derived from spike events.

    Returns:
      (store,
       features_by_sample [N, E] bool,
       edge_keys list[tuple[u64]],
       nodes_by_sample [N, U] bool,
       node_keys list[tuple[u64]]  # length-1 tuples, aligned to columns in nodes_by_sample
      )

    Determinism:
      - Per-sample GSE instance (no cross-sample islands).
      - Stable sorting by node id within islands and for final node_keys/edge_keys.
    """
    # Lazy import to provide a clear error if wheel not built
    PyGse, PyHypergraphStore = _require_py_nsi()

    # Validate inputs
    if not isinstance(acts, np.ndarray) or acts.ndim != 2:
        raise ValueError("acts must be a 2D numpy array [N, D]")
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("labels must be a 1D numpy array [N]")
    if acts.shape[0] != labels.shape[0]:
        raise ValueError("acts and labels must have the same number of rows")

    N = int(acts.shape[0])
    acts_f32 = acts.astype(np.float32, copy=False)

    # Spike encoding (deterministic ordering ensured inside)
    from python.encoders.spike import encode_spikes_batch  # [encode_spikes_batch()](python/encoders/spike.py:97)

    # Will hold the final hypergraph
    store = PyHypergraphStore()

    # Edge bookkeeping
    edge_keys: List[Tuple[int, ...]] = []
    edge_index: Dict[Tuple[int, ...], int] = {}
    active_cols_per_sample: List[Set[int]] = [set() for _ in range(N)]

    # Node bookkeeping
    node_sets_per_sample: List[Set[int]] = [set() for _ in range(N)]
    all_nodes: Set[int] = set()

    # Encode spikes per sample once (deterministic order)
    spikes_per_sample: List[List] = encode_spikes_batch(
        ensemble=ensemble,
        acts=acts_f32,
        t_start=float(t_start),
        delta_t=float(delta_t),
        min_sigmoid=float(min_sigmoid),
    )

    # Collect node activity per sample directly from spikes
    for i in range(N):
        for sp in spikes_per_sample[i]:
            nid = int(sp.node_id())
            node_sets_per_sample[i].add(nid)
            all_nodes.add(nid)

    # Process samples in fixed order for GSE/hyperedges
    for i in range(N):
        # Reset GSE per sample to avoid cross-sample islands
        gse = PyGse(float(gse_window))

        # Ingest spikes; each returns zero or more islands
        sample_spikes = spikes_per_sample[i]
        for sp in sample_spikes:
            islands = gse.ingest(sp)  # List[List[PySpike]]

            # Determinism: sort islands by key (sorted node-ids)
            sortable_islands: List[Tuple[Tuple[int, ...], List]] = []
            for isl in islands:
                # Sort spikes within island by node id
                isl_sorted = sorted(isl, key=lambda s: int(s.node_id()))
                # Compute canonical key (sorted unique node ids); only hyperedges (size >= 2)
                node_ids = tuple(sorted({int(s.node_id()) for s in isl_sorted}))
                if len(node_ids) < 2:
                    continue
                sortable_islands.append((node_ids, isl_sorted))

            sortable_islands.sort(key=lambda x: x[0])

            for node_ids, isl_sorted in sortable_islands:
                # Add island to store (already sorted for determinism)
                store.add_island(isl_sorted)

                # Track boolean feature for this sample (hyperedge fired at least once)
                if node_ids not in edge_index:
                    edge_index[node_ids] = len(edge_keys)
                    edge_keys.append(node_ids)
                col = edge_index[node_ids]
                active_cols_per_sample[i].add(col)

    # Finalize boolean design matrix for edges [N, E]
    E = len(edge_keys)
    features = np.zeros((N, E), dtype=bool)
    for i, cols in enumerate(active_cols_per_sample):
        if cols:
            idx = np.fromiter(cols, dtype=np.int64, count=len(cols))
            features[i, idx] = True

    # Finalize node-level design matrix [N, U] with stable order by node id
    node_keys: List[Tuple[int]] = [(nid,) for nid in sorted(all_nodes)]
    node_index: Dict[int, int] = {nid: j for j, (nid,) in enumerate(node_keys)}
    U = len(node_keys)
    nodes_by_sample = np.zeros((N, U), dtype=bool)
    for i, nset in enumerate(node_sets_per_sample):
        if nset:
            idx = np.fromiter((node_index[n] for n in sorted(nset)), dtype=np.int64, count=len(nset))
            nodes_by_sample[i, idx] = True

    return store, features, edge_keys, nodes_by_sample, node_keys


def build_hypergraph(
    ensemble,
    acts: "np.ndarray",
    labels: "np.ndarray",
    t_start: float,
    delta_t: float,
    min_sigmoid: float,
    gse_window: float,
) -> Tuple[object, np.ndarray, List[Tuple[int, ...]]]:
    """
    Build a temporal-coincidence hypergraph via GSE and aggregate into a store.

    Steps:
      1) Encode spikes per sample using latency-phase code in
         [encode_spikes_batch()](python/encoders/spike.py:1).
      2) For each sample, instantiate [PyGse](py_nsi/src/lib.rs:1) with window and ingest
         spikes in deterministic order (already sorted by encoder).
      3) Each ingest returns 0+ temporal islands (List[PySpike]). For determinism:
         - sort spikes inside each island by their canonical node id
         - sort islands by their (sorted node id tuple) before adding
      4) Add each island to [PyHypergraphStore.add_island](py_nsi/src/lib.rs:1).
      5) While processing, record a boolean feature per unique island key
         (sorted unique node id tuple, size ≥ 2) indicating the hyperedge fired at
         least once in that sample.

    Returns:
      (store, features_by_sample [N, E] bool, edge_keys list[tuple[u64]])
    """
    # Lazy import to provide a clear error if wheel not built
    PyGse, PyHypergraphStore = _require_py_nsi()

    # Validate inputs
    if not isinstance(acts, np.ndarray) or acts.ndim != 2:
        raise ValueError("acts must be a 2D numpy array [N, D]")
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("labels must be a 1D numpy array [N]")
    if acts.shape[0] != labels.shape[0]:
        raise ValueError("acts and labels must have the same number of rows")

    N = int(acts.shape[0])
    acts_f32 = acts.astype(np.float32, copy=False)

    # Spike encoding (deterministic ordering ensured inside)
    from python.encoders.spike import encode_spikes_batch  # local import to keep module graph tidy

    # Will hold the final hypergraph
    store = PyHypergraphStore()

    # Edge bookkeeping
    edge_keys: List[Tuple[int, ...]] = []
    edge_index: Dict[Tuple[int, ...], int] = {}
    active_cols_per_sample: List[Set[int]] = [set() for _ in range(N)]

    # Encode spikes per sample once (deterministic order)
    spikes_per_sample: List[List] = encode_spikes_batch(
        ensemble=ensemble,
        acts=acts_f32,
        t_start=float(t_start),
        delta_t=float(delta_t),
        min_sigmoid=float(min_sigmoid),
    )

    # Process samples in fixed order
    for i in range(N):
        # Reset GSE per sample to avoid cross-sample islands
        gse = PyGse(float(gse_window))

        # Ingest spikes; each returns zero or more islands
        sample_spikes = spikes_per_sample[i]
        for sp in sample_spikes:
            islands = gse.ingest(sp)  # List[List[PySpike]]

            # Determinism: sort islands by key (sorted node-ids)
            sortable_islands: List[Tuple[Tuple[int, ...], List]] = []
            for isl in islands:
                # Sort spikes within island by node id
                isl_sorted = sorted(isl, key=lambda s: int(s.node_id()))
                # Compute canonical key (sorted unique node ids)
                node_ids = tuple(sorted({int(s.node_id()) for s in isl_sorted}))
                if len(node_ids) < 2:
                    # Only consider hyperedges of size ≥ 2
                    continue
                sortable_islands.append((node_ids, isl_sorted))

            sortable_islands.sort(key=lambda x: x[0])

            for node_ids, isl_sorted in sortable_islands:
                # Add island to store (already sorted for determinism)
                store.add_island(isl_sorted)

                # Track boolean feature for this sample
                if node_ids not in edge_index:
                    edge_index[node_ids] = len(edge_keys)
                    edge_keys.append(node_ids)
                col = edge_index[node_ids]
                active_cols_per_sample[i].add(col)

    # Finalize boolean design matrix [N, E]
    E = len(edge_keys)
    features = np.zeros((N, E), dtype=bool)
    for i, cols in enumerate(active_cols_per_sample):
        if cols:
            idx = np.fromiter(cols, dtype=np.int64, count=len(cols))
            features[i, idx] = True

    return store, features, edge_keys