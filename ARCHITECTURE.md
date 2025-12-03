# Architecture: Rust Core, Python Bindings, and Reproducible Demos

This document specifies the implementable architecture for the Rust core and Python bindings used by the demo corridor. It translates the blueprint in [shg.md](shg.md) into crate layout, explicit APIs, and build strategy. Code constructs and files are referenced precisely for immediate implementation in Code mode.

## Repository Layout (Workspace)

```
.
├─ nsi_core/                 # Rust library crate (NSI software core)
│  ├─ Cargo.toml
│  └─ src/
│     ├─ lib.rs
│     ├─ ensemble.rs        # Encoder trait, Ensemble, SimpleSaeEncoder
│     ├─ encoding.rs        # Activation→spike mapping, Spike type
│     ├─ hypergraph.rs      # GSE, HypergraphStore, HIF export
│     └─ metrics.rs         # STII placeholder, poly/entropy utilities
├─ py_nsi/                   # pyo3 crate (Python bindings) built via maturin
│  ├─ Cargo.toml
│  ├─ pyproject.toml
│  └─ src/
│     └─ lib.rs             # PyEnsemble, PyGse, PyHypergraphStore
├─ configs/                  # Config-first orchestration (YAML)
│  ├─ ensemble.yaml
│  ├─ spike.yaml
│  └─ stii.yaml
├─ notebooks/                # Canonical demo notebooks (0–5)
│  ├─ 01_baseline_sae_polysemanticity.ipynb
│  ├─ 02_ensemble_intersection_reduction.ipynb
│  ├─ 03_spike_temporal_hypergraph.ipynb
│  ├─ 04_stii_causal_circuits_fairness.ipynb
│  └─ 05_investor_dashboard.ipynb
├─ python/                   # Optional Python utilities (plotting, I/O glue)
└─ outputs/                  # Versioned, deterministic artifacts per run
```

## Rust Module and API Surface (nsi_core)

- Ensemble and encoders ([nsi_core/src/ensemble.rs](nsi_core/src/ensemble.rs))
  - [pub trait Encoder](nsi_core/src/ensemble.rs:1)
    - [fn encode(&self, activations: &[f32]) -> Vec<f32>](nsi_core/src/ensemble.rs:1)
  - [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1)
    - [pub fn encode_all(&self, activations: &[f32]) -> Vec<Vec<f32>>](nsi_core/src/ensemble.rs:1)
    - [pub fn intersect_features(&self, outputs: &[Vec<f32>], threshold: f32) -> Vec<bool>](nsi_core/src/ensemble.rs:1)
  - [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1)
    - Minimal linear + ReLU + top-k sparsifier
    - Weights held in Rust; constructor accepts optional seed to ensure determinism
    - Top-k controlled via config (see [CONFIG_SPECS.md](CONFIG_SPECS.md))

- Spike encoding ([nsi_core/src/encoding.rs](nsi_core/src/encoding.rs))
  - [pub fn activation_to_spike_time(activation: f32, t_start: f32, delta_t: f32) -> Option<f32>](nsi_core/src/encoding.rs:1)
    - Latency-phase code; None indicates “no spike”
  - [#[derive(Clone, Debug)] pub struct Spike](nsi_core/src/encoding.rs:1)
    - Fields: ensemble_id: u32, neuron_id: u32, t: f32

- Hypergraph construction and export ([nsi_core/src/hypergraph.rs](nsi_core/src/hypergraph.rs))
  - GSE
    - [pub struct Gse](nsi_core/src/hypergraph.rs:1)
    - [impl Gse { pub fn new(window: f32) -> Self }](nsi_core/src/hypergraph.rs:1)
    - [pub fn ingest(&mut self, spike: Spike) -> Vec<Vec<Spike>>](nsi_core/src/hypergraph.rs:1)  // sliding window + cross-ensemble coincidence; returns candidate “islands”
  - Hypergraph store
    - [#[derive(Clone, Debug, Hash, Eq, PartialEq)] pub struct HyperedgeKey](nsi_core/src/hypergraph.rs:1)
    - [pub struct Hyperedge](nsi_core/src/hypergraph.rs:1)
    - [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1)
    - Methods:
      - [pub fn add_island(&mut self, island: &[Spike])](nsi_core/src/hypergraph.rs:1)
      - [pub fn edges(&self) -> impl Iterator<Item = &Hyperedge>](nsi_core/src/hypergraph.rs:1)
      - [pub fn export_hif<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()>](nsi_core/src/hypergraph.rs:1)

- Metrics and utilities ([nsi_core/src/metrics.rs](nsi_core/src/metrics.rs))
  - Types:
    - [pub struct StiiResult](nsi_core/src/metrics.rs:1)
  - Methods:
    - [impl HypergraphStore { pub fn compute_stii(&mut self, key: &HyperedgeKey, deltas: &[(u64, f32)]) -> StiiResult }](nsi_core/src/metrics.rs:1)  // placeholder stable aggregation; exact STII can iterate/cumulate
  - Polysemanticity utilities:
    - [pub fn poly_count(prob: &[f32], eps: f32) -> usize](nsi_core/src/metrics.rs:1)
    - [pub fn entropy(prob: &[f32]) -> f32](nsi_core/src/metrics.rs:1)

### Design Notes

- Config-first: All operational params flow from YAML under [configs/](configs/).
- Determinism: Every constructor accepts optional seed; record seeds + git hashes in outputs (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
- Performance: Zero-copy slices, pre-allocation for top-k, ring buffer for GSE window, and streaming HIF writer.

## Python pyo3 Bindings (py_nsi)

- [py_nsi/src/lib.rs](py_nsi/src/lib.rs)
  - [#[pyclass] struct PyEnsemble](py_nsi/src/lib.rs:1) wrapping `Ensemble<SimpleSaeEncoder>`
    - [#[pymethods] fn encode_all](py_nsi/src/lib.rs:1)
    - [fn intersect](py_nsi/src/lib.rs:1)
  - [#[pyclass] struct PyGse](py_nsi/src/lib.rs:1)
    - Minimal constructor from window
    - ingest(spike) -> list of islands
  - [#[pyclass] struct PyHypergraphStore](py_nsi/src/lib.rs:1)
    - add_island, edges, export_hif
- Packaging
  - maturin for build/release wheels
  - Universal2 wheels on macOS; CPython 3.10–3.12

## Build surfaces

- Rust edition and workspace
  - Rust edition: 2021 (workspace root [Cargo.toml](Cargo.toml))
  - Core crate: [nsi_core/Cargo.toml](nsi_core/Cargo.toml)
- Python bindings (pyo3 + maturin)
  - ABI3 wheels (CPython ≥3.10). Configure abi3-py310 in [py_nsi/Cargo.toml](py_nsi/Cargo.toml); crate-type = "cdylib".
  - Build via `maturin develop -m` [py_nsi/Cargo.toml](py_nsi/Cargo.toml) for local dev; `maturin build` for distribution.
- Feature flags
  - Optional `metrics_stii_exact` feature to enable heavier STII paths in [nsi_core/src/metrics.rs](nsi_core/src/metrics.rs).

## Build Strategy

- Rust
  - Use stable toolchain; optional features for “metrics_stii_exact” to toggle heavier computation paths
- Python
  - maturin develop for local dev; maturin build for distribution
  - Wheel targets: macOS universal2, Linux manylinux2014, Windows win_amd64 (as feasible)
- CI hooks (future)
  - Validate that [py_nsi](py_nsi/) imports and core methods run smoke tests

## Minimal Data/Model Defaults

- Models
  - distilgpt2 (Hugging Face) or 1–2 layer toy Transformer
- Datasets
  - Ambiguity: “bank” river/finance labeled sentences
  - Toy fairness: protected attribute labels for a simple decision task
- Small, fast, reproducible; CPU acceptable

## Error Handling and Determinism

- All public constructors log:
  - RNG seeds, top-k values, thresholds, window sizes
- All I/O operations return `Result` and bubble errors with context
- Deterministic ordering in:
  - Feature intersections (stable sorting before hashing)
  - Hyperedge keys (sorted incidence lists)
  - HIF serialization (sorted maps, stable float formatting)

## Mapping to Hardware (NSI-CP)
 
- GSE
  - Software: [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub fn ingest](nsi_core/src/hypergraph.rs:1)
  - Hardware: spike-native temporal coincidence window with cross-ensemble constraint
- GMF
  - Software: [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1), [pub fn add_island](nsi_core/src/hypergraph.rs:1), [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
  - Hardware: PIM-backed hypergraph state and HIF stream-out
- PTA (STII/ACDC)
  - Software: [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1) as a stand-in
- FSM
  - Software: post-verified motif extraction in notebooks; hardware: canonical labeling + CAM counting

## Data flows (ASCII)

```
Activations
  ↓
[pub trait Encoder](nsi_core/src/ensemble.rs:1)
  ↓ encode → [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1) / [PyEnsemble.encode_all](py_nsi/src/lib.rs:1)
  ↓ intersect → [pub fn intersect_features](nsi_core/src/ensemble.rs:1) / [PyEnsemble.intersect](py_nsi/src/lib.rs:1)
  ↓
[pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1) → [pub struct Spike](nsi_core/src/encoding.rs:1)
  ↓
[GSE ingest](nsi_core/src/hypergraph.rs:1) / [PyGse.ingest](py_nsi/src/lib.rs:1) → islands
  ↓
[pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1)
  ↓ add → [pub fn add_island](nsi_core/src/hypergraph.rs:1) / [PyHypergraphStore.add_island](py_nsi/src/lib.rs:1)
  ↓ compute → [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)
  ↓ prune → ACDC (software)
  ↓ export → [pub fn export_hif](nsi_core/src/hypergraph.rs:1) / [PyHypergraphStore.export_hif](py_nsi/src/lib.rs:1)
  ↓
HIF (auditable circuits) → dashboard + hardware mapping (NSI-CP: GSE/GMF/PTA/FSM)
```
