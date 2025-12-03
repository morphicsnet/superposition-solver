# API Spec: Python-Facing Interfaces and Data Contracts

This specification defines the Python APIs exposed via pyo3 bindings in [py_nsi/src/lib.rs](py_nsi/src/lib.rs). Each method maps directly to Rust implementations in [nsi_core/](nsi_core/) and emits deterministic, versioned artifacts as defined in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Where applicable, we reference constructs like [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1), [pub trait Encoder](nsi_core/src/ensemble.rs:1), [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1), [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1), [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1), [pub fn add_island](nsi_core/src/hypergraph.rs:1), [pub fn export_hif](nsi_core/src/hypergraph.rs:1), and [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1).

## Conventions

- Module name: `py_nsi`
- Types use Python-native containers for portability; numpy arrays are accepted for numeric tensors.
- All time/threshold parameters are floats; seeds and IDs are integers.
- Determinism requires fixed seeds and frozen configs under [configs/](configs/).

## Core Types

- Spike (Python)
  - Tuple layout: `(ensemble_id: int, neuron_id: int, t: float)`
  - Matches [#[derive(Clone, Debug)] pub struct Spike](nsi_core/src/encoding.rs:1)
  - Created by the SpikeEncoder or directly constructed by advanced users.

- HyperedgeKey (Python)
  - String key using canonical, sorted incidences (e.g., `E0:N12|E2:N7->OUT:tok_42`)
  - Mirrors [#[derive(Clone, Debug, Hash, Eq, PartialEq)] pub struct HyperedgeKey](nsi_core/src/hypergraph.rs:1)
  - Returned by helpers when available; otherwise pass the canonical key string.

## Classes and Methods

### EnsembleEncoder (wraps [#[pyclass] struct PyEnsemble](py_nsi/src/lib.rs:1))

- Purpose
  - Manage an ensemble of encoders implemented as [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1) with E = [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1).
- Methods
  - from_config(path: str) -> EnsembleEncoder
    - Loads [configs/ensemble.yaml](configs/ensemble.yaml); constructs K encoders with seeds and sparsity settings.
    - Errors: FileNotFoundError, ValueError (invalid schema), ValueError (seeds length < ensemble_size).
  - encode_all(activations: np.ndarray) -> list[list[float]]
    - Inputs: `activations` shape [F] or [B, F], dtype float32/float64.
    - Output: for [F], returns list-of-list per encoder (K x F’ after sparsity); for [B, F], returns list (batch) of K x F’ lists.
    - Maps to [pub fn encode_all](nsi_core/src/ensemble.rs:1).
  - intersect(outputs: list[list[float]], threshold: float) -> list[bool]
    - Compute boolean mask for features active under ensemble agreement.
    - threshold ∈ (0,1]; config default in [configs/ensemble.yaml](configs/ensemble.yaml).
    - Maps to [pub fn intersect_features](nsi_core/src/ensemble.rs:1).
- Notes
  - Encoders are deterministic given seeds; ensure seeds are recorded (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

### SpikeEncoder

- Purpose
  - Convert dense activations into spike events using latency–phase coding.
- Methods
  - from_config(path: str) -> SpikeEncoder
    - Loads [configs/spike.yaml](configs/spike.yaml).
  - encode_batch(activations: np.ndarray, meta: dict) -> list[Spike]
    - Inputs: `activations` shape [F] or [B, F], dtype float32/float64.
    - For each activation x, compute spike time via [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1) with `(t_start, delta_t)` and drop if sigmoid(x) < `min_sigmoid`.
    - Output: list of [Spike](nsi_core/src/encoding.rs:1) tuples `(ensemble_id, neuron_id, t)`.
    - `meta` may include `ensemble_id` (int) to annotate spikes when processing per-encoder.
- Notes
  - Use the same seeds/ordering as EnsembleEncoder to keep neuron_id alignment stable.

### GraphStreamingEngine (wraps [#[pyclass] struct PyGse](py_nsi/src/lib.rs:1))

- Purpose
  - Detect temporal coincidences across ensembles using a sliding window (GSE).
- Constructor
  - GraphStreamingEngine(window: float, cross_ensemble_required: bool = True)
    - `window` maps to [impl Gse { pub fn new }](nsi_core/src/hypergraph.rs:1).
- Methods
  - ingest(spike: Spike) -> list[list[Spike]]
    - Add a spike, return zero or more candidate “islands” (each island is a list of Spike) that meet coincidence criteria.
    - Maps to [pub fn ingest](nsi_core/src/hypergraph.rs:1).
- Notes
  - Enforce cross-ensemble coincidence if `cross_ensemble_required=True`; must match [configs/spike.yaml](configs/spike.yaml).

### HypergraphStore (wraps [#[pyclass] struct PyHypergraphStore](py_nsi/src/lib.rs:1))

- Purpose
  - Persist islands as hyperedges, iterate edges, compute STII, and export HIF.
- Methods
  - add_island(island: list[Spike]) -> None
    - Canonicalizes spikes and inserts/updates a hyperedge.
    - Maps to [pub fn add_island](nsi_core/src/hypergraph.rs:1).
  - edges() -> Iterable[dict]
    - Yields edge dictionaries with fields: `key: str`, `nodes: list[str]`, `order: int`, `count: int`, optionally `stii: float`.
    - Backed by [pub struct Hyperedge](nsi_core/src/hypergraph.rs:1).
  - export_hif(path: str) -> None
    - Serialize HIF JSON with deterministic ordering.
    - Maps to [pub fn export_hif](nsi_core/src/hypergraph.rs:1).
  - compute_stii(key: str, deltas: list[tuple[int, float]]) -> dict
    - `key` is the canonical HyperedgeKey string; `deltas` are `(mask_id, delta_metric)` pairs defined by the notebook caller.
    - Returns a dict with STII summary (value, ci, samples).
    - Maps to [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1).

## Shapes and dtypes (per method)

- [PyEnsemble.encode_all](py_nsi/src/lib.rs:1)
  - Input: activations [F] or [B,F], float32/float64
  - Output: list[list[float]] shaped as K×F′ (or batch list of K×F′)
- [PyEnsemble.intersect](py_nsi/src/lib.rs:1)
  - Input: outputs: list[list[float]] (K×F′), threshold: float∈(0,1]
  - Output: list[bool] mask of length F′
- [PyGse.ingest](py_nsi/src/lib.rs:1)
  - Input: Spike tuple (ensemble_id:int, neuron_id:int, t:float)
  - Output: list[island] where island = list[Spike]
- [PyHypergraphStore.export_hif](py_nsi/src/lib.rs:1)
  - Input: path: str
  - Output: None (writes deterministic JSON)
- STII call-through → [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)
  - Input: key: str, deltas: list[tuple[int,float]]
  - Output: dict {value: float, ci: [lo, hi], samples: int}

## Data Shapes, DTypes, and Thresholds

- Activations
  - Shape: [F] or [B, F]; dtype: float32 preferred (float64 accepted).
  - Values should be pre-activation or chosen layer outputs per [configs/ensemble.yaml](configs/ensemble.yaml). Document the layer in artifacts.
- Ensemble intersection
  - `threshold` ∈ (0, 1]; default 0.5. Must be recorded.
- Spike encoding
  - `t_start` (float), `delta_t` (float), `min_sigmoid` (float ∈ [0,1)), `gse_window` (float), `cross_ensemble_required` (bool True).
- STII
  - [configs/stii.yaml](configs/stii.yaml): `max_order_k ∈ {2,3}`, `subset_sampling.enabled`, `per_edge_samples ≥ 16`, `bootstrap ≥ 0`.

## Error Semantics

- Shape errors
  - TypeError / ValueError: mismatched array rank or dtype.
- Config errors
  - ValueError: ensemble_size vs seeds length; threshold bounds; gse_window > delta_t.
- I/O errors
  - OSError / IOError: export path write failures; missing config files.
- Determinism guard
  - Warning: missing seeds or non-unique seeds; notebooks must abort or log and continue per policy.

## Backward Compatibility

- Encoder abstraction
  - The Python `EnsembleEncoder` interface remains stable if [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1) is replaced by future SNN encoders; configs may introduce `encoder_type` variants while preserving method signatures.
- HIF schema
  - Keys, nodes, and edge fields remain stable; additional metadata fields (e.g., reliability, timestamps) may be additive.

## Minimal Usage Sketch

```python
# Ensemble → Intersect
from py_nsi import PyEnsemble, PyGse, PyHypergraphStore  # [#[pymodule] fn py_nsi](py_nsi/src/lib.rs:1)

ens = PyEnsemble.from_config("configs/ensemble.yaml")           # wraps [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1)
outs = ens.encode_all(activations)                              # [PyEnsemble.encode_all](py_nsi/src/lib.rs:1)
mask = ens.intersect(outs, threshold=0.5)                       # [PyEnsemble.intersect](py_nsi/src/lib.rs:1)

# Spikes → GSE → Hypergraph
gse = PyGse(window=0.1)                                         # [PyGse](py_nsi/src/lib.rs:1) → [impl Gse { pub fn new }](nsi_core/src/hypergraph.rs:1)
store = PyHypergraphStore()                                     # [PyHypergraphStore](py_nsi/src/lib.rs:1)
for spike in spikes:                                            # spikes from [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1)
    for island in gse.ingest(spike):                            # [PyGse.ingest](py_nsi/src/lib.rs:1)
        store.add_island(island)                                # [PyHypergraphStore.add_island](py_nsi/src/lib.rs:1)

# STII → HIF
res = store.compute_stii(key, deltas)                           # [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)
store.export_hif("outputs/<stamp>/hif/hypergraph.json")         # [PyHypergraphStore.export_hif](py_nsi/src/lib.rs:1)
```

All APIs are intentionally minimal to keep the notebooks canonical and the Rust core stable. Future extensions (SNN-native encoders, reliability scores, motif mining) will be additive and will not break existing signatures.
