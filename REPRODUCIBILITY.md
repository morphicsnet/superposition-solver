# Reproducibility: End-to-End Runbook and Artifact Expectations

This runbook specifies deterministic builds, execution of the five demos via notebooks, and the exact artifact packaging. All runs must record RNG seeds, git commit, and config copies under [outputs/](outputs/). Exposed APIs and file emitters are referenced to [nsi_core/](nsi_core/) and [py_nsi/](py_nsi/) constructs such as [pub fn export_hif](nsi_core/src/hypergraph.rs:1) and Python wrappers in [py_nsi/src/lib.rs](py_nsi/src/lib.rs:1).

## Environment Requirements

- OS: macOS (x86_64/arm64), Linux (x86_64), Windows (amd64)
- Rust: stable (latest), with cargo installed
- Python: CPython 3.10–3.12 (recommended: 3.11)
- Build: maturin ≥ 1.6, pip ≥ 23, setuptools ≥ 68
- Optional: CUDA or Metal not required (CPU-only acceptable for tiny demos)
- Jupyter: jupyterlab or notebook for interactive runs

Recommended setup

```bash
# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Base tooling
pip install --upgrade pip setuptools wheel maturin jupyterlab

# Rust toolchain (if needed)
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build Steps (py_nsi)

The Python bindings in [py_nsi/](py_nsi/) wrap the Rust core [nsi_core/](nsi_core/).

- Development (editable install)

```bash
# Build and install locally in the active venv
maturin develop -m py_nsi/Cargo.toml
python -c "import py_nsi; print('py_nsi OK')"
```

- Release wheels

```bash
# macOS universal2 example
export MACOSX_DEPLOYMENT_TARGET=11.0
maturin build -m py_nsi/Cargo.toml --release --universal2 --interpreter python3.10 python3.11 python3.12
# Manylinux / Windows builds can be added similarly
```

- Notes
  - Ensure Cargo.lock is committed after first successful build.
  - Wheels should target CPython 3.10–3.12; macOS universal2 for arm64/x86_64.
  - pyo3 classes and methods defined in [py_nsi/src/lib.rs](py_nsi/src/lib.rs:1).

## Minimal Commands

- Build bindings (required for Demos 2–4)
  - `maturin develop -m` [py_nsi/Cargo.toml](py_nsi/Cargo.toml)
- Demo 1 (Baseline)
  - `pip install -r` [python/requirements-demo1.txt](python/requirements-demo1.txt)
  - `python` [python/demo1_baseline.py](python/demo1_baseline.py)
- Demo 2 (Ensemble + Intersections)
  - `pip install -r` [python/requirements-demo2.txt](python/requirements-demo2.txt)
  - `python` [python/demo2_ensemble.py](python/demo2_ensemble.py)
- Demo 3 (Spike/Hypergraph)
  - `pip install -r` [python/requirements-demo3.txt](python/requirements-demo3.txt)
  - `python` [python/demo3_spike_hypergraph.py](python/demo3_spike_hypergraph.py)
- Demo 4 (STII + ACDC)
  - `pip install -r` [python/requirements-demo4.txt](python/requirements-demo4.txt)
  - `python` [python/demo4_causal.py](python/demo4_causal.py)
- Demo 5 (Dashboard)
  - `pip install -r` [python/requirements-demo5.txt](python/requirements-demo5.txt)
  - `streamlit run` [python/dashboard/app.py](python/dashboard/app.py)

Notes
- py_nsi is only required to regenerate Demos 2–4; Demo 5 reads artifacts under [outputs/](outputs/).

## Running the Demos (Notebooks)

Canonical demos reside in [notebooks/](notebooks/). Run in order; each notebook writes frozen configs and metrics to a new stamped output directory.

- [notebooks/01_baseline_sae_polysemanticity.ipynb](notebooks/01_baseline_sae_polysemanticity.ipynb)
- [notebooks/02_ensemble_intersection_reduction.ipynb](notebooks/02_ensemble_intersection_reduction.ipynb)
- [notebooks/03_spike_temporal_hypergraph.ipynb](notebooks/03_spike_temporal_hypergraph.ipynb)
- [notebooks/04_stii_causal_circuits_fairness.ipynb](notebooks/04_stii_causal_circuits_fairness.ipynb)
- [notebooks/05_investor_dashboard.ipynb](notebooks/05_investor_dashboard.ipynb)

Suggested invocation (interactive)

```bash
jupyter lab  # open and run each notebook top-to-bottom
```

Optional batch execution (papermill)

```bash
pip install papermill
papermill notebooks/01_baseline_sae_polysemanticity.ipynb outputs/01_baseline.ipynb
```

## Artifact Tree and Stamping

Each run must produce a unique timestamped folder; all configs are copied-in before computation.

```
outputs/<YYYYmmdd-HHMMSS>/
  configs/
    ensemble.yaml                   # frozen copies loaded by notebooks
    spike.yaml
    stii.yaml
  logs/
    seed_and_git.txt                # RNG seeds, git rev, dirty flag, platform
  metrics/
    sae_poly.json                   # Demo 1
    ensemble_intersection.json      # Demo 2
    stii_edges.json                 # Demo 4
  plots/
    poly_hist_sae.png               # Demo 1
    poly_hist_intersection.png      # Demo 2
    island_size_cdf.png             # Demo 3
    stii_rank_curves.png            # Demo 4
  hif/
    hypergraph.json                 # via [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
  circuits/
    minimal_circuits.json           # ACDC results (Demo 4)
  reports/
    investor_dashboard.html         # Demo 5
```

seed_and_git.txt must include:

- Seeds from [configs/ensemble.yaml](configs/ensemble.yaml)
- Git commit (rev-parse HEAD) and dirty flag
- Rust/Cargo and Python/pip versions
- Absolute paths of the run artifacts (for relocation audits)

## Reproducibility bundle

Use [collect_artifacts](python/repro/bundle.py:1), [write_manifest](python/repro/bundle.py:1), and [make_zip](python/repro/bundle.py:1) to package a stamped run with SHA-256 integrity.

Expected outputs
- manifest.json: list of artifacts with fields: path, sha256, bytes, created_at
- bundle.zip: ZIP containing all artifacts plus manifest.json and copied configs

Example
```python
from python.repro.bundle import collect_artifacts, write_manifest, make_zip  # see [python/repro/bundle.py](python/repro/bundle.py:1)

art_root = "outputs/20250101-120000"
arts = collect_artifacts(art_root)
manifest_path = write_manifest(art_root, arts)  # writes manifest.json with SHA-256 for each file
zip_path = make_zip(art_root, arts, manifest_path)
print("Bundle:", zip_path)
```

Record provenance (seeds, versions, git)
```bash
# Seeds are read from frozen copies under outputs/<stamp>/configs/*.yaml
echo "git=$(git rev-parse HEAD)" >> outputs/<stamp>/logs/seed_and_git.txt
echo "python=$(python -V)" >> outputs/<stamp>/logs/seed_and_git.txt
echo "rustc=$(rustc --version)" >> outputs/<stamp>/logs/seed_and_git.txt
```

## HIF JSON Example Snippet

Exported by [pub fn export_hif](nsi_core/src/hypergraph.rs:1). Structure-only example (see [shg.md](shg.md) §8.1):

```json
{
  "network-type": "directed-hypergraph",
  "nodes": [
    {"id": "E0:N12", "type": "feature"},
    {"id": "E2:N7",  "type": "feature"},
    {"id": "OUT:tok_42", "type": "output"}
  ],
  "edges": [
    {
      "key": "e:E0:N12|E2:N7->OUT:tok_42",
      "nodes": ["E0:N12", "E2:N7", "OUT:tok_42"],
      "order": 3,
      "stii": 0.31,
      "count": 7
    }
  ],
  "incidences": []
}
```

- Compatibility
  - HyperNetX: load via HNX hypergraph constructors; verify node/edge incidence consistency.
  - Cytoscape/Cytoscape.js: edges may be flattened to multipart relationships; retain edge key and order in data attributes.

## Determinism and Validation

- Deterministic encoding
  - Ensemble seeds from [configs/ensemble.yaml](configs/ensemble.yaml) applied to each encoder in [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1).
  - Intersection is stable via [pub fn intersect_features](nsi_core/src/ensemble.rs:1) on consistently ordered outputs.
- Deterministic spike mapping
  - [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1) is pure; drop if below min_sigmoid in [configs/spike.yaml](configs/spike.yaml).
- Deterministic hypergraph keys
  - Canonicalized [#[derive(Clone, Debug, Hash, Eq, PartialEq)] pub struct HyperedgeKey](nsi_core/src/hypergraph.rs:1) sorts incidences before hashing; JSON writer preserves order.
- Cross-run check
  - Re-run Demo 3 twice with identical seeds; verify `diff outputs/*/hif/hypergraph.json` is empty.

## Failure Modes and Troubleshooting

- Build import errors
  - Ensure maturin develop ran against the active interpreter; confirm `python -c "import py_nsi"`.
- Non-deterministic outputs
  - Confirm seeds, intersect_threshold, gse_window are frozen; ensure no parallel nondeterministic RNGs in Python probes.
- HIF loading errors
  - Validate JSON schema fields; ensure edges.nodes set includes all incident node ids.

## Traceability to APIs

- Export: [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
- GSE: [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub fn ingest](nsi_core/src/hypergraph.rs:1)
- Spikes: [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1), [pub struct Spike](nsi_core/src/encoding.rs:1)
- STII: [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)
