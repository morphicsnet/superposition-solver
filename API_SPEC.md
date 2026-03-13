# API Spec: Python Bindings and Data Contracts

This document specifies the Python APIs exposed by `py_nsi` (pyo3 bindings) and their direct Rust counterparts in `nsi_core`. The bindings are intentionally minimal and deterministic; higher-level orchestration lives in `python/`.

## Conventions
- Module name: `py_nsi`
- Data shapes use Python-native containers for portability (lists and dicts)
- All numeric inputs are castable to `float`/`int` in Python and to `f32`/`u64` in Rust
- Determinism requires fixed seeds for encoder construction

## Core Types

### PySimpleSaeEncoder
- Rust: `nsi_core::SimpleSaeEncoder`
- Constructor:
  - `PySimpleSaeEncoder(in_dim: int, out_dim: int, top_k: int, seed: int)`
- Purpose: deterministic SAE-like encoder with top-k sparsity

### PyEnsemble
- Rust: `nsi_core::Ensemble<SimpleSaeEncoder>`
- Constructor:
  - `PyEnsemble(encoders: list[PySimpleSaeEncoder])`
- Methods:
  - `encode_all(activations: list[float]) -> list[list[float]]`
    - Input: a single activation vector (no batch)
    - Output: list of per-encoder outputs
  - `intersect(outputs: list[list[float]], threshold: float) -> list[bool]`
    - Computes a feature-wise intersection mask across encoder outputs

### PySpike
- Rust: `nsi_core::Spike`
- Fields (get/set):
  - `ensemble_id: int` (u16)
  - `neuron_id: int` (u32)
  - `t: float`
- Methods:
  - `node_id() -> int` (u64, encodes ensemble and neuron ids)

### PyGse
- Rust: `nsi_core::Gse`
- Constructor:
  - `PyGse(window: float)`
- Methods:
  - `ingest(spike: PySpike) -> list[list[PySpike]]`
    - Returns zero or more temporal coincidence islands

### PyHypergraphStore
- Rust: `nsi_core::HypergraphStore`
- Constructor:
  - `PyHypergraphStore()`
- Methods:
  - `add_island(island: list[PySpike]) -> None`
  - `edges() -> list[dict]`
    - Dict shape: `{ "key": [u64, ...], "observation_count": int, "stii_weight": float }`
  - `compute_stii(node_ids: list[int], deltas: list[tuple[int, float]]) -> float`
    - `node_ids` are canonicalized (sorted, deduped)
    - `deltas` are `(subset_size, delta_value)` pairs
  - `export_hif(path: str) -> None`
    - Writes a minimal HIF JSON (see below)

## Minimal HIF Export Shape
The Rust exporter in `nsi_core/src/hypergraph.rs` writes a minimal, deterministic JSON structure:

```json
{
  "network-type": "hypergraph",
  "nodes": [{"id": 123}, {"id": 456}],
  "edges": [
    {"id": 0, "key": [123, 456], "observation_count": 3, "stii_weight": 0.42}
  ],
  "incidences": [
    {"edge": 0, "nodes": [123, 456]}
  ]
}
```

- Node ids are `u64` values returned by `PySpike.node_id()`.
- `edges[*].key` is the canonical list of node ids for that hyperedge.
- `incidences` mirror edge membership for downstream tooling.

## Higher-Level Python Orchestration
The demos use lightweight wrappers in `python/` to load YAML configs and wire the bindings:
- `python/ensemble/intersection.py` builds `PySimpleSaeEncoder` and `PyEnsemble` from YAML, and expects `PY_NSI_INPUT_DIM` to be set before construction.
- `python/encoders/spike.py` constructs `PySpike` instances and streams them into `PyGse`.
- `python/hypergraph/pipeline.py` manages GSE ingestion and `PyHypergraphStore` exports.

## Error Semantics
- `PySimpleSaeEncoder` raises `ValueError` if `in_dim`/`out_dim` are zero or `top_k > out_dim`.
- `PyHypergraphStore.compute_stii` raises `ValueError` if fewer than two distinct node ids are provided.
- `export_hif` raises `IOError` on write failures.

## Minimal Usage Sketch
```python
from py_nsi import PySimpleSaeEncoder, PyEnsemble, PySpike, PyGse, PyHypergraphStore

enc = PySimpleSaeEncoder(256, 256, 16, 1337)
ens = PyEnsemble([enc])
outs = ens.encode_all([0.1, 0.2, -0.1, 0.3])
mask = ens.intersect(outs, 0.5)

spike = PySpike(0, 12, 0.05)
spike2 = PySpike(0, 13, 0.07)
gse = PyGse(0.1)
store = PyHypergraphStore()
for island in gse.ingest(spike):
    store.add_island(island)

stii = store.compute_stii([spike.node_id(), spike2.node_id()], [(2, 1.0), (2, 0.0)])
store.export_hif("outputs/spike_hypergraph/<run_tag>/hypergraph.hif.json")
```
