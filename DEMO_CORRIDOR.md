# Demo Corridor: From Superposition to Causal, Auditable Circuits

A technically literate investor’s pain today: models remain opaque, safety claims are correlation-only, and “interpretability” stalls at linear additivity. The simple fix: keep concepts small and reproducible, then add time and topology to observe interactions rather than magnitudes. This moves us from correlation to causation via spike-native temporal coincidence and hypergraph circuits, delivering product and IP: causal, auditable circuits verified by STII/ACDC, exported as HIF, and naturally mapped to the NSI-CP hardware pipeline (GSE, GMF, PTA, FSM) described in [shg.md](shg.md).

## How to run (scripts and configs)

- Demo 1 — Baseline
  - Config: [configs/demo1_baseline.yaml](configs/demo1_baseline.yaml)
  - Script: [python/demo1_baseline.py](python/demo1_baseline.py)
- Demo 2 — Orthogonal Ensemble + Intersections
  - Config: [configs/demo2_ensemble.yaml](configs/demo2_ensemble.yaml)
  - Script: [python/demo2_ensemble.py](python/demo2_ensemble.py) (requires [#[pymodule] fn py_nsi](py_nsi/src/lib.rs:1))
- Demo 3 — Spike Encoding, GSE, and Hypergraph (HIF)
  - Config: [configs/demo3_spike_hypergraph.yaml](configs/demo3_spike_hypergraph.yaml)
  - Script: [python/demo3_spike_hypergraph.py](python/demo3_spike_hypergraph.py)
- Demo 4 — STII + ACDC Causal Verification
  - Config: [configs/demo4_causal.yaml](configs/demo4_causal.yaml)
  - Script: [python/demo4_causal.py](python/demo4_causal.py)
- Demo 5 — Investor Dashboard
  - Config: [configs/demo5_dashboard.yaml](configs/demo5_dashboard.yaml)
  - App: streamlit run [python/dashboard/app.py](python/dashboard/app.py)

## Demo Index (0–5) — Investor Takeaways

- Demo 0 — Metrics and Baselines: We formalize polysemanticity and causal verification metrics; lock acceptance criteria to de-risk “goalpost moving.”
- Demo 1 — Baseline SAE: Single-SAE features are measurably polysemantic; monosemantic rate is limited without topology/time.
- Demo 2 — Orthogonal Ensemble: Intersections across encoders reduce polysemanticity while preserving model accuracy.
- Demo 3 — Spike and Hypergraph: Temporal coincidence (GSE) forms causal “islands” and exports a minimal HIF hypergraph.
- Demo 4 — Causal Verification: STII and ACDC cut spurious correlations; necessary circuits remain for fairness and hallucination cases.
- Demo 5 — Investor Dashboard: One page linking metrics to topology and safety posture; highlights the hardware/IP moat.

## Demo 0 — Define and Lock Metrics and Acceptance Criteria

- Goal
  - Lock operational definitions and target thresholds for polysemanticity, causal verification (STII/ACDC), and reproducibility.
- Setup
  - Model/dataset: distilgpt2 (tiny LLM) on a small labeled concept set (e.g., ambiguity: “bank” river/finance; toy fairness: protected attribute labels).
  - Artifacts baseline directory under [outputs/](outputs/).
- Procedure
  - Specify metric formulas in [METRICS.md](METRICS.md), referencing constructs: [pub fn poly_count](nsi_core/src/metrics.rs:1), [pub fn entropy](nsi_core/src/metrics.rs:1), [pub struct StiiResult](nsi_core/src/metrics.rs:1).
  - Freeze acceptance thresholds and deltas in [CONFIG_SPECS.md](CONFIG_SPECS.md) examples.
- Metrics
  - Definition completeness: all formulas and reporting schemas specified.
  - Target stability: bootstrap CI width for key metrics <= 10% of median.
- Artifacts (manual, if recorded)
  - `outputs/baseline/<run_tag>/metrics_spec.json`
  - `outputs/baseline/<run_tag>/locked_acceptance.yaml`
- Acceptance Criteria
  - [ ] All metric definitions present in [METRICS.md](METRICS.md), including monosemantic rate and STII reporting.
  - [ ] Locked thresholds serialized to `outputs/baseline/<run_tag>/locked_acceptance.yaml` if tracked.

## Demo 1 — Baseline SAE Polysemanticity

- Goal
  - Show a single encoder (SAE-like) yields significant polysemanticity on a tiny model, with minimal task accuracy loss constraints defined.
- Setup
  - Model: distilgpt2 or 1–2 layer toy transformer.
  - Data: small labeled concept set (ambiguity + toy fairness).
  - Notebook: [notebooks/01_baseline_sae_polysemanticity.ipynb](notebooks/01_baseline_sae_polysemanticity.ipynb).
- Procedure
  - Extract activations; train/infer a simple SAE-like encoder in Python as baseline.
  - Compute polysemanticity distribution and monosemantic rate.
- Metrics
  - Median polysemanticity ≥ baseline B0; monosemantic % ≤ M0 (expect low).
  - Task accuracy delta vs. raw model ≤ 0.5% absolute.
- Artifacts
  - `outputs/baseline/<run_tag>/poly_hist.png`
  - `outputs/baseline/<run_tag>/metrics.json`
  - `outputs/baseline/<run_tag>/config.yaml`
- Acceptance Criteria
  - [ ] Polysemanticity histogram and JSON exist with seeds and git hash.
  - [ ] Accuracy delta ≤ 0.5% vs. raw baseline.

## Demo 2 — Orthogonal Ensemble + Intersections

- Goal
  - Reduce polysemanticity via ensemble diversity and intersection without harming accuracy.
- Setup
  - K encoders with differing seeds/sparsity configs; Python orchestrates Rust core calls via [py_nsi/src/lib.rs](py_nsi/src/lib.rs).
  - Notebook: [notebooks/02_ensemble_intersection_reduction.ipynb](notebooks/02_ensemble_intersection_reduction.ipynb).
- Procedure
  - Encode with an ensemble via [PyEnsemble.encode_all](py_nsi/src/lib.rs:1); compute intersections with [PyEnsemble.intersect](py_nsi/src/lib.rs:1) (Rust: [pub fn intersect_features](nsi_core/src/ensemble.rs:1)).
  - Compare polysemanticity distributions: single vs. intersection.
- Metrics
  - Median polysemanticity decreases ≥ 20% vs. Demo 1.
  - Monosemantic rate increases ≥ 15% absolute.
  - Task accuracy delta ≤ 0.5% absolute vs. Demo 1.
- Artifacts
  - `outputs/ensemble/<run_tag>/poly_hist_single.png`
  - `outputs/ensemble/<run_tag>/poly_hist_intersection.png`
  - `outputs/ensemble/<run_tag>/poly_hist_dual.png`
  - `outputs/ensemble/<run_tag>/metrics_single.json`
  - `outputs/ensemble/<run_tag>/metrics_intersection.json`
  - `outputs/ensemble/<run_tag>/compare.json`
  - `outputs/ensemble/<run_tag>/config.yaml`
- Acceptance Criteria
  - [ ] Intersection method and threshold recorded in `outputs/ensemble/<run_tag>/config.yaml`.
  - [ ] Polysemanticity and monosemantic improvements meet targets with accuracy delta ≤ 0.5%.

## Demo 3 — Spike Encoding, GSE, and Hypergraph (HIF Export)

- Goal
  - Convert activations to spikes, detect temporal coincidences (GSE), build hypergraph islands, and export HIF.
- Setup
  - Use Rust core functions in [nsi_core/src/encoding.rs](nsi_core/src/encoding.rs) and [nsi_core/src/hypergraph.rs](nsi_core/src/hypergraph.rs) via Python bindings [py_nsi/src/lib.rs](py_nsi/src/lib.rs).
  - Notebook: [notebooks/03_spike_temporal_hypergraph.ipynb](notebooks/03_spike_temporal_hypergraph.ipynb).
- Procedure
  - Map activation → spike time via [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1); package as [pub struct Spike](nsi_core/src/encoding.rs:1).
  - Stream to GSE [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub fn ingest](nsi_core/src/hypergraph.rs:1) to form islands; store in [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1) via [pub fn add_island](nsi_core/src/hypergraph.rs:1). Python call-through: [PyGse.ingest](py_nsi/src/lib.rs:1), [PyHypergraphStore.add_island](py_nsi/src/lib.rs:1).
  - Export HIF with [pub fn export_hif](nsi_core/src/hypergraph.rs:1). Python: [PyHypergraphStore.export_hif](py_nsi/src/lib.rs:1).
- Metrics
  - Cross-ensemble coincidence ratio ≥ 90% of retained islands (enforces diversity).
  - HIF export deterministic under fixed seeds (byte-identical).
- Artifacts
  - `outputs/spike_hypergraph/<run_tag>/hypergraph.hif.json`
  - `outputs/spike_hypergraph/<run_tag>/metrics_hyperedges.json`
  - `outputs/spike_hypergraph/<run_tag>/poly_hist_hyperedges.png`
  - `outputs/spike_hypergraph/<run_tag>/edge_keys.json`
  - `outputs/spike_hypergraph/<run_tag>/config.yaml`
- Acceptance Criteria
  - [ ] HIF JSON valid and loadable by HyperNetX/Cytoscape.
  - [ ] Deterministic export verified across two seeded runs (diff == empty).

## Demo 4 — STII + ACDC Causal Verification (Hallucination and Fairness)

- Goal
  - Verify necessary circuits using STII and ACDC; demonstrate reduction of spurious correlations and identification of causal bias.
- Setup
  - STII over hyperedges computed via [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1).
  - ACDC traversal in Python using exported HIF; PTA semantics mirrored in software.
  - Notebook: [notebooks/04_stii_causal_circuits_fairness.ipynb](notebooks/04_stii_causal_circuits_fairness.ipynb).
- Procedure
  - For candidate hyperedges, compute STII deltas; rank and threshold.
  - Run ACDC pruning to minimal necessary circuits for: (a) hallucination test, (b) toy fairness decision.
- Metrics
  - For fairness: if biased, protected-attribute hyperedges present in minimal circuit with nonzero STII; if not, absent with CI including zero.
  - For hallucination: high-STII hyperpath density between context and answer increases in grounded cases; drops in confabulations.
- Artifacts
  - `outputs/causal/<run_tag>/stii_values.json`
  - `outputs/causal/<run_tag>/acdc_minimal_circuit.json`
  - `outputs/causal/<run_tag>/fairness_report.json`
  - `outputs/causal/<run_tag>/hypergraph_stii.hif.json`
  - `outputs/causal/<run_tag>/config.yaml`
- Acceptance Criteria
  - [ ] Minimal circuits serialized and reproducible (seed-stable).
  - [ ] At least one fairness or hallucination case yields a causal verdict aligned with labels.

## Demo 5 — Investor Dashboard (Metrics, Topology, Safety)

- Goal
  - Present one-page dashboard tying metrics to hypergraph topology and safety posture; showcase reproducibility and artifact lineage.
- Setup
  - Notebook: [notebooks/05_investor_dashboard.ipynb](notebooks/05_investor_dashboard.ipynb).
  - Loads metrics JSON, HIF, minimal circuits; renders plots and topology summaries.
- Procedure
  - Summarize monosemantic rate trend (Demo 1 → 2), HIF stats (Demo 3), and causal verification outcomes (Demo 4).
  - Surface lineage: seeds, git hash, config diffs, artifact paths.
- Metrics
  - Dashboard completeness: all prior artifacts linked and rendered.
  - Latency: dashboard run < 30s on laptop without GPU.
- Artifacts
  - `outputs/investor/<run_tag>/dashboard_metrics.json`
  - `outputs/investor/<run_tag>/dashboard_summary.md`
  - `outputs/investor/<run_tag>/*.png` (figures referenced by the summary)
- Acceptance Criteria
  - [ ] Dashboard file renders locally with working links to all prior artifacts.
  - [ ] “Green” badges for monosemantic improvement, HIF determinism, and at least one causal verification success.

## Hardware Mapping (NSI-CP)

- Demo 2 → Orthogonal Ensemble Intersection
  - Software: [pub struct Ensemble<E: Encoder>](nsi_core/src/ensemble.rs:1), [pub trait Encoder](nsi_core/src/ensemble.rs:1), [pub struct SimpleSaeEncoder](nsi_core/src/ensemble.rs:1).
  - Hardware: Ensemble diversity requirement aligns with sidecar observers; intersection logic prepares inputs for spike-native coincidence.
- Demo 3 → GSE and GMF
  - GSE: [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub fn ingest](nsi_core/src/hypergraph.rs:1) implement the spike-native temporal coincidence window.
  - GMF: [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1), [pub fn add_island](nsi_core/src/hypergraph.rs:1), [pub fn export_hif](nsi_core/src/hypergraph.rs:1) mirror the in-memory hypergraph state and HIF serialization.
- Demo 4 → PTA (STII/ACDC)
  - PTA semantics: [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1) stands in for hardware STII; ACDC pruning corresponds to on-chip traversal and pruning.
- Demo 4 → FSM
  - Motif mining of verified hyperpaths (minimal circuits) is the software analog of FSM’s canonical labeling and CAM counting.
- Moat Summary
  - On-chip, real-time causal hypergraph + STII acceleration yields:
    - Orders-of-magnitude faster verification and motif mining than software-only stacks.
    - Deterministic, reproducible, causal circuits exportable via HIF.
    - IP defensibility via NSI-CP modules (GSE, GMF, PTA, FSM) executing the same abstractions we validate in Demos 2–4.
