# Config Specs: Config-First Reproducibility Schema

This document defines the YAML schemas used to drive every demo. All parameters must be explicitly recorded, with seeds and git hashes embedded into the run artifacts under [outputs/](outputs/). The configs map directly onto Rust APIs in [nsi_core/](nsi_core/) and Python bindings in [py_nsi/](py_nsi/), notably [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1), [pub struct Gse](nsi_core/src/hypergraph.rs:1), and [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1). Python convenience wrappers in notebooks correspond to [#[pyclass] struct PyEnsemble](py_nsi/src/lib.rs:1), [#[pyclass] struct PyGse](py_nsi/src/lib.rs:1), and [#[pyclass] struct PyHypergraphStore](py_nsi/src/lib.rs:1).

## Directory

- [configs/ensemble.yaml](configs/ensemble.yaml)
- [configs/spike.yaml](configs/spike.yaml)
- [configs/stii.yaml](configs/stii.yaml)

Each demo notebook loads one or more YAMLs and writes back a frozen copy into outputs/<run_stamp>/configs/ for exact provenance.

### Ready-to-run demo configs

- Demo 1: [configs/demo1_baseline.yaml](configs/demo1_baseline.yaml)
- Demo 2: [configs/demo2_ensemble.yaml](configs/demo2_ensemble.yaml)
- Demo 3: [configs/demo3_spike_hypergraph.yaml](configs/demo3_spike_hypergraph.yaml)
- Demo 4: [configs/demo4_causal.yaml](configs/demo4_causal.yaml)
- Demo 5: [configs/demo5_dashboard.yaml](configs/demo5_dashboard.yaml)

Default threshold impact (guidance)
- `intersect_threshold` (Demo 2): higher → fewer, purer features; lower → more features, higher polysemanticity risk.
- `min_sigmoid` (Demo 3): higher → fewer spikes (stricter); affects island counts and HIF density exported by [pub fn export_hif](nsi_core/src/hypergraph.rs:1).
- `max_order_k` (Demo 4): larger increases interaction depth for [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1) at higher compute cost.

---

## Schema: configs/ensemble.yaml

Fields

- model_layer: int
  - Transformer layer index (0-based). Keep small (e.g., last MLP pre-activation of distilgpt2).
- feature_dim: int
  - Output feature dimension per encoder before sparsity.
- ensemble_size: int
  - Number K of encoders in the orthogonal ensemble (Demo 2+).
- encoder_type: enum [simple_sae]
  - Initial implementation maps to [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1).
- sparsity:
  - top_k: int
    - Enforced by SimpleSaeEncoder top-k selection.
  - threshold: float
    - Optional additional magnitude threshold post-ReLU.
- seeds: list[int]
  - Length must be ≥ ensemble_size. If longer, extra seeds ignored; if shorter, error.
- intersect_threshold: float
  - Threshold for [pub fn intersect_features](nsi_core/src/ensemble.rs:1) when merging encoder outputs.

Constraints and Defaults

- 1 ≤ ensemble_size ≤ 16 (default: 4)
- feature_dim ∈ {128, 256, 512} (default: 256)
- sparsity.top_k ≥ 1 and ≤ feature_dim/8 (default: 16)
- sparsity.threshold ∈ [0.0, 1.0] (default: 0.0)
- intersect_threshold ∈ (0.0, 1.0] (default: 0.5)
- seeds must be unique to promote diversity

Example (annotated)

```yaml
# configs/ensemble.yaml
model_layer: 9              # distilgpt2 final MLP block (0-based)
feature_dim: 256            # per-encoder latent size before top-k
ensemble_size: 4            # K encoders
encoder_type: simple_sae    # maps to SimpleSaeEncoder
sparsity:
  top_k: 16                 # keep top-16 features
  threshold: 0.0            # no extra threshold beyond top-k
seeds: [1337, 2025, 4242, 9001]
intersect_threshold: 0.5    # require ≥50% agreement for a feature to be "on"
```

---

## Schema: configs/spike.yaml

Fields

- t_start: float
  - Start time of the encoding window passed to [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1).
- delta_t: float
  - Window duration; larger increases time resolution range.
- min_sigmoid: float
  - If sigmoid(activation) < min_sigmoid, drop spike (None).
- gse_window: float
  - Sliding coincidence window used by [impl Gse { pub fn new }](nsi_core/src/hypergraph.rs:1) and [pub fn ingest](nsi_core/src/hypergraph.rs:1).
- cross_ensemble_required: bool
  - Enforce cross-ensemble coincidence filter in GSE; should be true for superposition elimination.

Constraints and Defaults

- delta_t > 0 (default: 1.0)
- 0.0 ≤ min_sigmoid < 1.0 (default: 0.05)
- gse_window ∈ (0, delta_t] (default: delta_t/10)
- cross_ensemble_required = true (default and strongly recommended)

Example (annotated)

```yaml
# configs/spike.yaml
t_start: 0.0               # beginning of latency-phase window
delta_t: 1.0               # 1.0 time units (arbitrary units)
min_sigmoid: 0.05          # prune very low-activation spikes
gse_window: 0.1            # spikes within 0.1 are considered coincident
cross_ensemble_required: true
```

---

## Schema: configs/stii.yaml

Fields

- max_order_k: int
  - Interaction order cap (2 or 3 recommended) for approximate STII in software.
- subset_sampling:
  - enabled: bool
  - per_edge_samples: int
    - Number of sampled perturbation subsets per hyperedge when true.
- stability_checks:
  - bootstrap: int
    - Number of bootstrap resamples for stability intervals.

Constraints and Defaults

- max_order_k ∈ {2, 3} (default: 2)
- If subset_sampling.enabled, per_edge_samples ≥ 16 (default: 64)
- bootstrap ≥ 0 (default: 100)

Example (annotated)

```yaml
# configs/stii.yaml
max_order_k: 2
subset_sampling:
  enabled: true
  per_edge_samples: 64
stability_checks:
  bootstrap: 100
```

---

## Cross-File Constraints

- The GSE coincidence window gse_window must be ≤ delta_t in [configs/spike.yaml](configs/spike.yaml).
- The ensemble seeds list length must be ≥ ensemble_size in [configs/ensemble.yaml](configs/ensemble.yaml).
- Intersections applied in Demo 2 must record intersect_threshold and all seeds used; notebooks should echo these into outputs/<run_stamp>/configs/.

---

## Versioning and Logging

Every run must create a stamped directory:

```
outputs/<YYYYmmdd-HHMMSS>/
  configs/
    ensemble.yaml          # frozen copy
    spike.yaml
    stii.yaml
  logs/
    seed_and_git.txt       # rng seeds, git rev-parse HEAD, dirty flag
  hif/
    hypergraph.json        # via [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
  metrics/
    sae_poly.json          # Demo 1
    ensemble_intersection.json  # Demo 2
    stii_edges.json        # Demo 4
  circuits/
    minimal_circuits.json  # Demo 4 ACDC output
  plots/
    *.png
  reports/
    investor_dashboard.html
```

seed_and_git.txt content must include:

- RNG seeds (all) from [configs/ensemble.yaml](configs/ensemble.yaml)
- Git commit hash (rev-parse HEAD)
- Dirty working tree flag
- Platform and Python/Rust versions
- JSON pointer paths to each artifact

---

## Determinism Notes

- Encoders: seed each [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1) with seeds[i].
- Intersections: stable sort features before [pub fn intersect_features](nsi_core/src/ensemble.rs:1).
- Spikes: [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1) must be a pure function of (activation, t_start, delta_t, min_sigmoid).
- Hypergraph: canonicalize [#[derive(Clone, Debug, Hash, Eq, PartialEq)] pub struct HyperedgeKey](nsi_core/src/hypergraph.rs:1) by sorted incidences; serialize via [pub fn export_hif](nsi_core/src/hypergraph.rs:1) with deterministic ordering.

---

## Usage Flow (Notebook-Level)

- Demo 1 loads [configs/ensemble.yaml](configs/ensemble.yaml); computes [pub fn poly_count](nsi_core/src/metrics.rs:1) and [pub fn entropy](nsi_core/src/metrics.rs:1).
- Demo 2 uses same plus intersect_threshold; calls [pub fn intersect_features](nsi_core/src/ensemble.rs:1).
- Demo 3 loads [configs/spike.yaml](configs/spike.yaml); applies [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1); streams to [pub struct Gse](nsi_core/src/hypergraph.rs:1); persists via [pub fn add_island](nsi_core/src/hypergraph.rs:1) and [pub fn export_hif](nsi_core/src/hypergraph.rs:1).
- Demo 4 loads [configs/stii.yaml](configs/stii.yaml); computes [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1); runs ACDC on exported HIF.

All thresholds and seeds must be frozen by copying the loaded YAMLs into outputs/<run_stamp>/configs/ before any computation.