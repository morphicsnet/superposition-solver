# NSI Demo Corridor: From Superposition to Causal, Auditable Circuits

What’s broken → Fix → Time/Topology → Causation → Product/IP: Today’s models are opaque; “interpretability” often stops at linear additivity and correlation. We fix this by keeping concepts small and reproducible, then adding time (spikes) and topology (hypergraphs) to observe interactions, not magnitudes. This drives causal verification (STII/ACDC) and exports auditable circuits (HIF), mapping directly onto NSI-CP hardware modules (GSE, GMF, PTA, FSM) as outlined in [shg.md](shg.md) and implemented in software via [pub trait Encoder](nsi_core/src/ensemble.rs:1), [pub struct Gse](nsi_core/src/hypergraph.rs:1), and [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1).

## Quickstart

- Prerequisites
  - Rust stable (edition 2021)
  - Python 3.10–3.12, pip, virtualenv/venv
  - maturin for Python bindings
- Build Python bindings (pyo3 via maturin)
  - Uses [py_nsi/Cargo.toml](py_nsi/Cargo.toml)
  - Commands:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install --upgrade pip maturin
    maturin develop -m py_nsi/Cargo.toml
    python -c "import py_nsi; print('py_nsi OK')"
    ```
- Run demos (scripts)
  - Baseline: python [python/demo1_baseline.py](python/demo1_baseline.py)
  - Ensemble: python [python/demo2_ensemble.py](python/demo2_ensemble.py)
  - Spike/Hypergraph: python [python/demo3_spike_hypergraph.py](python/demo3_spike_hypergraph.py)
  - STII+ACDC: python [python/demo4_causal.py](python/demo4_causal.py)
  - Dashboard (Streamlit): streamlit run [python/dashboard/app.py](python/dashboard/app.py)
- Jupyter notebooks (canonical, 01–05)
  - [notebooks/01_baseline_sae_polysemanticity.ipynb](notebooks/01_baseline_sae_polysemanticity.ipynb)
  - [notebooks/02_ensemble_intersection_reduction.ipynb](notebooks/02_ensemble_intersection_reduction.ipynb)
  - [notebooks/03_spike_temporal_hypergraph.ipynb](notebooks/03_spike_temporal_hypergraph.ipynb)
  - [notebooks/04_stii_causal_circuits_fairness.ipynb](notebooks/04_stii_causal_circuits_fairness.ipynb)
  - [notebooks/05_investor_dashboard.ipynb](notebooks/05_investor_dashboard.ipynb)

Important
- py_nsi is required to regenerate Demos 2–4. Demo 5 (dashboard) reads artifacts from [outputs/](outputs/).

## Repository map

- Core Rust crates
  - [nsi_core/](nsi_core/) — domain logic (encoders, spikes, GSE, hypergraph, STII)
  - [py_nsi/](py_nsi/) — Python bindings ([#[pymodule] fn py_nsi](py_nsi/src/lib.rs:1))
- Python pipelines
  - [python/](python/) — demos, metrics, plotting, dashboard
- Configs
  - [configs/](configs/) — config-first runs; demo-specific YAMLs
- Artifacts
  - [outputs/](outputs/) — stamped runs: configs, logs, metrics, plots, hif, circuits, reports

## Architecture pointers (clickable constructs)

- Encoder trait and ensemble
  - [pub trait Encoder](nsi_core/src/ensemble.rs:1)
  - [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1)
  - [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1)
- Spike/time encoding
  - [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1)
  - [#[derive(Clone, Debug)] pub struct Spike](nsi_core/src/encoding.rs:1)
- GSE / Hypergraph
  - [pub struct Gse](nsi_core/src/hypergraph.rs:1)
  - [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1)
  - [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
- STII
  - [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)
- Python module entry
  - [#[pymodule] fn py_nsi](py_nsi/src/lib.rs:1)
  - Methods referenced: [PyEnsemble.encode_all](py_nsi/src/lib.rs:1), [PyEnsemble.intersect](py_nsi/src/lib.rs:1), [PyHypergraphStore.export_hif](py_nsi/src/lib.rs:1)

## Metrics and acceptance

- See [METRICS.md](METRICS.md) for polysemanticity, entropy, STII/ACDC, and acceptance targets.
- “What investors see” per demo is summarized in [DEMO_CORRIDOR.md](DEMO_CORRIDOR.md) (index and takeaways).

## Reproducibility

- Config-first runs: schemas and constraints in [CONFIG_SPECS.md](CONFIG_SPECS.md)
- Bundle/manifest:
  - Use [collect_artifacts](python/repro/bundle.py:1), [write_manifest](python/repro/bundle.py:1), [make_zip](python/repro/bundle.py:1)
  - Stamped outputs under [outputs/](outputs/) include metrics JSON, plots, HIF, minimal circuits, dashboard reports.
- Full runbook in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## Hardware mapping (NSI-CP summary)

- GSE (temporal coincidence), GMF (hypergraph storage/format), PTA (STII/ACDC), FSM (motif mining) — software abstractions align with hardware pipeline; see [shg.md](shg.md) and code references [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1), [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1).

## Demos at a glance (config • script • notebook)

- Demo 0 — Metrics & Acceptance
  - Config/docs: [METRICS.md](METRICS.md)
  - Artifacts root: [outputs/](outputs/)
- Demo 1 — Baseline SAE Polysemanticity
  - Config: [configs/demo1_baseline.yaml](configs/demo1_baseline.yaml)
  - Script: [python/demo1_baseline.py](python/demo1_baseline.py)
  - Notebook: [notebooks/01_baseline_sae_polysemanticity.ipynb](notebooks/01_baseline_sae_polysemanticity.ipynb)
- Demo 2 — Orthogonal Ensemble + Intersections
  - Config: [configs/demo2_ensemble.yaml](configs/demo2_ensemble.yaml)
  - Script: [python/demo2_ensemble.py](python/demo2_ensemble.py)
  - Notebook: [notebooks/02_ensemble_intersection_reduction.ipynb](notebooks/02_ensemble_intersection_reduction.ipynb)
- Demo 3 — Spike Encoding, GSE, Hypergraph (HIF)
  - Config: [configs/demo3_spike_hypergraph.yaml](configs/demo3_spike_hypergraph.yaml)
  - Script: [python/demo3_spike_hypergraph.py](python/demo3_spike_hypergraph.py)
  - Notebook: [notebooks/03_spike_temporal_hypergraph.ipynb](notebooks/03_spike_temporal_hypergraph.ipynb)
- Demo 4 — STII + ACDC Causal Verification
  - Config: [configs/demo4_causal.yaml](configs/demo4_causal.yaml)
  - Script: [python/demo4_causal.py](python/demo4_causal.py)
  - Notebook: [notebooks/04_stii_causal_circuits_fairness.ipynb](notebooks/04_stii_causal_circuits_fairness.ipynb)
- Demo 5 — Investor Dashboard
  - Config: [configs/demo5_dashboard.yaml](configs/demo5_dashboard.yaml)
  - Script(s): [python/demo5_dashboard.py](python/demo5_dashboard.py) • [python/dashboard/app.py](python/dashboard/app.py)
  - Notebook: [notebooks/05_investor_dashboard.ipynb](notebooks/05_investor_dashboard.ipynb)

At-a-glance table

| Demo | Config | Script(s) | Notebook |
| --- | --- | --- | --- |
| 0 — Metrics & Acceptance | [METRICS.md](METRICS.md) | — | — |
| 1 — Baseline SAE | [configs/demo1_baseline.yaml](configs/demo1_baseline.yaml) | [python/demo1_baseline.py](python/demo1_baseline.py) | [notebooks/01_baseline_sae_polysemanticity.ipynb](notebooks/01_baseline_sae_polysemanticity.ipynb) |
| 2 — Ensemble + Intersections | [configs/demo2_ensemble.yaml](configs/demo2_ensemble.yaml) | [python/demo2_ensemble.py](python/demo2_ensemble.py) | [notebooks/02_ensemble_intersection_reduction.ipynb](notebooks/02_ensemble_intersection_reduction.ipynb) |
| 3 — Spike/Hypergraph (HIF) | [configs/demo3_spike_hypergraph.yaml](configs/demo3_spike_hypergraph.yaml) | [python/demo3_spike_hypergraph.py](python/demo3_spike_hypergraph.py) | [notebooks/03_spike_temporal_hypergraph.ipynb](notebooks/03_spike_temporal_hypergraph.ipynb) |
| 4 — STII + ACDC | [configs/demo4_causal.yaml](configs/demo4_causal.yaml) | [python/demo4_causal.py](python/demo4_causal.py) | [notebooks/04_stii_causal_circuits_fairness.ipynb](notebooks/04_stii_causal_circuits_fairness.ipynb) |
| 5 — Dashboard | [configs/demo5_dashboard.yaml](configs/demo5_dashboard.yaml) | [python/demo5_dashboard.py](python/demo5_dashboard.py) • [python/dashboard/app.py](python/dashboard/app.py) | [notebooks/05_investor_dashboard.ipynb](notebooks/05_investor_dashboard.ipynb) |

Note
- py_nsi build is required to regenerate Demos 2–4 ([#[pymodule] fn py_nsi](py_nsi/src/lib.rs:1)). Demo 5 reads artifacts from [outputs/](outputs/).

## Status

- Demos 0–5 completed; artifacts under [outputs/](outputs/) per stamped run:
  - metrics/*.json, plots/*.png, hif/hypergraph.json, circuits/minimal_circuits.json, reports/investor_dashboard.html
- py_nsi note: required for Demos 2–4 regeneration; Dashboard (Demo 5) loads prior artifacts.