# Architecture: Rust Core, Python Bindings, and Demo Orchestration

This document summarizes the implementable architecture for the Rust core (`nsi_core`), the Python bindings (`py_nsi`), and the demo orchestration in `python/` and `notebooks/`.

## Repository Layout

```
.
├─ nsi_core/                 # Rust library crate (encoders, spikes, hypergraph, metrics)
├─ py_nsi/                   # pyo3 bindings (PySimpleSaeEncoder, PyEnsemble, PySpike, PyGse, PyHypergraphStore)
├─ configs/                  # YAML configs used by demos
├─ notebooks/                # Canonical demo notebooks (01 to 05)
├─ python/                   # Demo scripts + helpers
└─ outputs/                  # Stamped artifacts per run
```

## Rust Core (nsi_core)

### Encoders and ensembles
- `nsi_core/src/ensemble.rs`
  - `pub trait Encoder`
  - `pub struct Ensemble<E: Encoder>`
  - `pub struct SimpleSaeEncoder`
    - Constructor: `SimpleSaeEncoder::new(in_dim, out_dim, top_k, seed)`
  - `pub fn encode_all(&self, activations: &[f32]) -> Vec<Vec<f32>>`
  - `pub fn intersect_features(&self, outputs: &[Vec<f32>], threshold: f32) -> Vec<bool>`

### Spike encoding
- `nsi_core/src/encoding.rs`
  - `pub fn activation_to_spike_time(...) -> Option<f32>`
  - `pub struct Spike { ensemble_id: u16, neuron_id: u32, t: f32 }`

### Hypergraph construction and export
- `nsi_core/src/hypergraph.rs`
  - `pub struct Gse` and `pub fn ingest(&mut self, spike: Spike) -> Vec<Vec<Spike>>`
  - `pub struct HypergraphStore` with `add_island`, `edges`, and `export_hif`
  - `export_hif` writes a minimal JSON:
    - `network-type`, `nodes` (u64 ids), `edges` (key + observation_count + stii_weight), `incidences`

### Metrics
- `nsi_core/src/metrics.rs`
  - `pub fn poly_count(prob: &[f32], eps: f32) -> usize`
  - `pub fn entropy(prob: &[f32]) -> f32`
  - `impl HypergraphStore { pub fn compute_stii(...) -> StiiResult }` (placeholder aggregation)

## Python Bindings (py_nsi)

Bindings in `py_nsi/src/lib.rs` expose a minimal, deterministic surface:
- `PySimpleSaeEncoder` (wraps `SimpleSaeEncoder`)
- `PyEnsemble` (wraps `Ensemble<SimpleSaeEncoder>`)
- `PySpike` (wraps `Spike` and provides `node_id()`)
- `PyGse` (wraps `Gse`)
- `PyHypergraphStore` (wraps `HypergraphStore`, exports minimal HIF)

## Demo Orchestration

- YAML configs live under `configs/` and are loaded by helper utilities in `python/`.
- `python/ensemble/intersection.py` constructs `PySimpleSaeEncoder`/`PyEnsemble` from YAML; it expects `PY_NSI_INPUT_DIM` to be set to the activation dimension.
- `python/encoders/spike.py` and `python/hypergraph/pipeline.py` build spikes, run GSE, and export HIF artifacts.

## Build Surfaces

- Rust workspace root: `Cargo.toml` (members: `nsi_core`, `py_nsi`)
- Python bindings (maturin):
  - `maturin develop -m py_nsi/Cargo.toml`
  - ABI targets: CPython 3.10 to 3.12

## Design Notes

- Config-first reproducibility: each demo writes its YAML to `outputs/<demo>/<run_tag>/config.yaml`.
- Determinism: all encoders are seeded; output artifacts are intended to be byte-stable across runs with identical seeds.
- Performance: simple, readable implementations; this repo favors clarity over full-scale optimization.
