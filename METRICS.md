# Metrics: Superposition Elimination and Evaluation Protocols

This document formalizes the operational definitions, pipelines, and reporting used across the demo corridor. All named constructs link to the implementable APIs in Rust and Python bindings (for references see [ARCHITECTURE.md](ARCHITECTURE.md)).

## Polysemanticity: Operational Definitions

- Count threshold definition
  - For a feature f and labeled concepts C_k, define poly(f) = |{k : P(C_k | f_active) > ε}|.
  - ε is set in demo configs (see `configs/demo1_baseline.yaml` and `configs/demo2_ensemble.yaml`, plus [CONFIG_SPECS.md](CONFIG_SPECS.md)); typical ε ∈ [0.05, 0.2] depending on class granularity.
- Monosemantic rate
  - Fraction of features with poly(f) = 1.
- Entropy definition
  - Let p_k = P(C_k | f_active), define H(f) = −∑_k p_k log_b(p_k).
  - Smoothing: add α to counts (Laplace) before normalization; record α and base b (e.g., b = 2) in outputs.
  - Implemented with [pub fn entropy](nsi_core/src/metrics.rs:1).
- Polysemanticity utilities
  - [pub fn poly_count(prob: &[f32], eps: f32) -> usize](nsi_core/src/metrics.rs:1) returns poly(f) for a given probability vector and ε.

## Measurement Pipelines

- Feature granularities
  - Single-encoder features (SAE-like baseline)
    - Extract features f_i from a single encoder; compute P(C_k | f_i active) via probes or labeled tasks.
  - Ensemble intersection features
    - Compute agreement across K encoders and derive an intersection-activated mask via [pub fn intersect_features](nsi_core/src/ensemble.rs:1).
    - Treat intersection features as candidate “pure” features for polysemanticity measurement.
  - Spike/hypergraph nodes
    - Convert activations to spikes with [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1); represent events via [pub struct Spike](nsi_core/src/encoding.rs:1).
    - Temporal islands/hyperedges formed by [pub struct Gse](nsi_core/src/hypergraph.rs:1) and persisted in [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1).
    - Measure concept selectivity at the level of hyperedges or island templates (see below).

### Estimating P(C_k | f_active)

- Direct label co-occurrence
  - For a labeled dataset, f_active is true if feature/spike/hyperedge fired in the window associated with the input; estimate p_k as frequency of label k within f_active events.
- Probe-based estimation
  - Train a simple logistic regression or linear probe per feature/hyperedge to map f_active to labels C_k; convert logits to probabilities via softmax.
- Time/Topology-aware variant
  - For spikes, define f_active if Spike.t ∈ [t_ctx, t_ctx + ΔT] for the current token; for hyperedges, f_active if at least one incidence island overlaps the token’s causal window obtained from [pub fn ingest](nsi_core/src/hypergraph.rs:1).

## Serialization of results

Per run directory under `outputs/<demo>/<run_tag>/`:

- Demo 1 (baseline)
  - `metrics.json` (summary metrics)
  - `poly_hist.png`
  - `probs.npy`, `poly_counts.npy`, `entropy.npy`
- Demo 2 (ensemble)
  - `metrics_single.json`, `metrics_intersection.json`, `compare.json`
  - `poly_hist_single.png`, `poly_hist_intersection.png`, `poly_hist_dual.png`
  - `probs_single.npy`, `poly_counts_single.npy`, `entropy_single.npy`
  - `probs_intersection.npy`, `poly_counts_intersection.npy`, `entropy_intersection.npy`
- Demo 3 (spike + hypergraph)
  - `metrics_hyperedges.json`, `poly_hist_hyperedges.png`
  - `hypergraph.hif.json` exported by [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
  - `edge_keys.json`, `features_hyperedges.npy`
- Demo 4 (causal)
  - `stii_values.json`
  - `acdc_minimal_circuit.json`
  - `fairness_report.json`
  - `hypergraph_stii.hif.json`

All demos also write `config.yaml` in the run directory.

Note
- Entropy base b and smoothing α are serialized alongside metrics; all entropy calculations call [pub fn entropy](nsi_core/src/metrics.rs:1).

## Reporting

- Distributions
  - Histograms for poly(f) and H(f), with median and 25/75% quantiles.
  - Monosemantic rate (poly = 1) reported as percentage with bootstrap CIs (see `configs/demo4_causal.yaml` STII settings).
- Downstream performance coupling
  - Report correlation between monosemantic rate and downstream task accuracy (absolute change vs. baseline) to ensure no regressions.
- Calibration
  - Expected Calibration Error (ECE) for task predictions; report delta ECE when switching from single-encoder to intersection features.
- Reproducibility
  - All plots and JSON metrics are written directly into `outputs/<demo>/<run_tag>/` alongside `config.yaml` (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## Causal Verification Metrics

- STII (Shapley–Taylor Interaction Index)
  - Purpose: quantify interaction (synergy or redundancy) of a subset S of features/hyperedges.
  - Sign: positive indicates synergy (superadditivity), negative indicates interference/redundancy.
  - Software implementation
    - Placeholder aggregation exposed via [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1) with deltas supplied from masked forward passes.
    - Control complexity via `configs/demo4_causal.yaml` `stii` block: `max_order_k`, `subset_sampling.enabled`, `subset_sampling.per_edge_samples`.
  - Reporting
    - Per-edge STII values with bootstrap intervals; rank curves and concentration plots (top-N edges capturing X% of mass).
- ACDC (Automated Circuit Discovery)
  - Minimal circuit verification on exported HIF using iterative pruning:
    - Remove edges whose masking yields negligible output change (e.g., Δlogit or KL divergence below threshold τ).
    - Report final minimal circuit size, retained edges, and associated STII values.
  - Targets
    - For fairness: if biased, protected-attribute hyperedges appear in minimal circuit with STII confidence interval strictly above 0.
    - For hallucination: minimal grounded circuit contains high-STII hyperpaths connecting context to answer; confabulations lack such connectivity.

## Hallucination Risk Connectivity Score

- Definition
  - Let H be the exported hypergraph (HIF) from [pub fn export_hif](nsi_core/src/hypergraph.rs:1).
  - Define high-STII set E⁺ = {e ∈ H : STII(e) ≥ s_min}, where s_min is a threshold from the `stii` block in `configs/demo4_causal.yaml`.
  - For a query–answer pair (q,a), compute the density of high-STII hyperpaths between q-context nodes and a-output nodes within a causal time window.
- Metric
  - HRCS(q,a) = (∑_paths∈E⁺ w(path)) / Z, where w(path) decreases with length and recency; Z normalizes for graph size.
  - Report
    - Higher HRCS in grounded responses; significantly lower in hallucinations. Provide ROC/AUC across a labeled set.

## Acceptance-Oriented Targets (Guidance)

- Demo 1 (Baseline SAE)
  - Expect low monosemantic rate and high median polysemanticity; task accuracy delta ≤ 0.5% absolute.
- Demo 2 (Ensemble Intersection)
  - Median polysemanticity ↓ ≥ 20% vs. Demo 1; monosemantic rate ↑ ≥ 15% absolute; accuracy delta ≤ 0.5%.
- Demo 3 (Spike/Hypergraph)
  - ≥ 90% of retained islands satisfy cross-ensemble coincidence; HIF exports are deterministic under fixed seeds.
- Demo 4 (Causal Verification)
  - At least one fairness or hallucination case yields a causal verdict aligned with labels; minimal circuits are seed-stable.

## Traceability to Implementations

- Probability and entropy
  - [pub fn poly_count](nsi_core/src/metrics.rs:1), [pub fn entropy](nsi_core/src/metrics.rs:1)
- Spike and temporal windows
  - [pub fn activation_to_spike_time](nsi_core/src/encoding.rs:1), [pub struct Spike](nsi_core/src/encoding.rs:1)
- Temporal coincidence and islands
  - [pub struct Gse](nsi_core/src/hypergraph.rs:1), [pub fn ingest](nsi_core/src/hypergraph.rs:1)
- Hypergraph persistence and export
  - [pub struct HypergraphStore](nsi_core/src/hypergraph.rs:1), [pub fn add_island](nsi_core/src/hypergraph.rs:1), [pub fn export_hif](nsi_core/src/hypergraph.rs:1)
- STII computation
  - [impl HypergraphStore { pub fn compute_stii }](nsi_core/src/metrics.rs:1)

All thresholds, seeds, and configuration values must be recorded in the run’s `outputs/<demo>/<run_tag>/config.yaml`, ensuring full reproducibility across environments.
