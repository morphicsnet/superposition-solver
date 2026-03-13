# Reproducibility: Runbook and Artifact Expectations

This runbook covers deterministic builds, demo execution, and artifact packaging. Demo scripts write a frozen copy of the active config to `config.yaml` inside each run directory. Artifacts are grouped by demo under `outputs/<demo>/<run_tag>/`.

## Environment Requirements
- OS: macOS (arm64/x86_64), Linux (x86_64), Windows (amd64)
- Rust: stable toolchain
- Python: CPython 3.10 to 3.12
- Build tooling: `maturin`, `pip`, `setuptools`, `wheel`
- Jupyter: `jupyterlab` or `notebook` for interactive runs

## Build Steps (py_nsi)
The Python bindings in `py_nsi/` wrap the Rust core `nsi_core/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel maturin
maturin develop --release -m py_nsi/Cargo.toml
python -c "import py_nsi; print('py_nsi OK')"
```

## Minimal Commands (Scripts)
- Demo 1 (Baseline):
  - `pip install -r python/requirements-demo1.txt`
  - `python python/demo1_baseline.py`
- Demo 2 (Ensemble + Intersections):
  - `pip install -r python/requirements-demo2.txt`
  - `python python/demo2_ensemble.py`
- Demo 3 (Spike + Hypergraph):
  - `pip install -r python/requirements-demo3.txt`
  - `python python/demo3_spike_hypergraph.py`
- Demo 4 (STII + ACDC):
  - `pip install -r python/requirements-demo4.txt`
  - `python python/demo4_causal.py`
- Demo 5 (Dashboard summary):
  - `pip install -r python/requirements-demo5.txt`
  - `python python/demo5_dashboard.py`

Notes
- `py_nsi` is required for Demos 2 to 4.
- Demo 5 reads artifacts from prior runs and writes a summary to `outputs/investor/<run_tag>/`.

## Running the Notebooks
Canonical notebooks live under `notebooks/`.

```bash
jupyter lab
```

## Artifact Layout
Each demo writes to a run directory created by `python/utils/artifacts.py:create_run_dir`:

```
outputs/<demo>/<run_tag>/
  config.yaml
  ... demo-specific artifacts ...
```

### Demo 1 (Baseline)
- `metrics.json`
- `poly_hist.png`
- `probs.npy`, `poly_counts.npy`, `entropy.npy`

### Demo 2 (Ensemble)
- `metrics_single.json`, `metrics_intersection.json`, `compare.json`
- `poly_hist_single.png`, `poly_hist_intersection.png`, `poly_hist_dual.png`
- `probs_single.npy`, `poly_counts_single.npy`, `entropy_single.npy`
- `probs_intersection.npy`, `poly_counts_intersection.npy`, `entropy_intersection.npy`

### Demo 3 (Spike + Hypergraph)
- `metrics_hyperedges.json`
- `poly_hist_hyperedges.png`
- `hypergraph.hif.json`
- `edge_keys.json`, `features_hyperedges.npy`

### Demo 4 (Causal)
- `stii_values.json`
- `acdc_minimal_circuit.json`
- `fairness_report.json`
- `hypergraph_stii.hif.json`

### Demo 5 (Dashboard summary)
- `dashboard_metrics.json`
- `dashboard_summary.md`
- PNG figures referenced in `dashboard_metrics.json`

## Reproducibility Bundle
Use `python/repro/bundle.py` to collect key artifacts and zip them with checksums.

Example:
```python
from python.repro.bundle import collect_artifacts, write_manifest, make_zip

run_dirs = {
    "baseline": "outputs/baseline/<run_tag>",
    "ensemble": "outputs/ensemble/<run_tag>",
    "spike_hypergraph": "outputs/spike_hypergraph/<run_tag>",
    "causal": "outputs/causal/<run_tag>",
}
artifacts = collect_artifacts(run_dirs)
write_manifest(artifacts, "outputs/bundle/manifest.json")
make_zip(artifacts, "outputs/bundle/bundle.zip")
```

## Minimal HIF JSON Example
Exported by `nsi_core::HypergraphStore::export_hif`.

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

## Determinism Notes
- Encoder determinism comes from fixed seeds passed to `PySimpleSaeEncoder`.
- Intersection is deterministic for a fixed encoder ensemble and threshold.
- HIF export is deterministic for a fixed hypergraph store and node ordering.
