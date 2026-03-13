# Config Specs: Demo YAML Schemas

This document defines the YAML schemas used by the demo scripts in `python/`. Each demo reads a single YAML file from `configs/` and writes a frozen copy to `outputs/<demo>/<run_tag>/config.yaml`.

## Config Files
- `configs/demo1_baseline.yaml`
- `configs/demo2_ensemble.yaml`
- `configs/demo3_spike_hypergraph.yaml`
- `configs/demo4_causal.yaml`
- `configs/demo5_dashboard.yaml`

## Demo 1: Baseline (`demo1_baseline.yaml`)

Top-level fields:
- `model_name` (str): Hugging Face model id
- `layer_index` (int): layer index to probe
- `dataset` (mapping):
  - `n_per_class` (int)
  - `seed` (int)
  - `concepts` (list[str])
- `sae` (mapping):
  - `hidden_dim` (int)
  - `top_k` (int)
  - `epochs` (int)
  - `lr` (float)
  - `l1_lambda` (float)
  - `seed` (int)
  - `active_threshold` (float)
- `metrics` (mapping):
  - `eps` (float)
  - `hist_bins` (int)
- `outputs` (mapping):
  - `base_dir` (str, default demo output root)
  - `run_tag` (str or null; if null, a timestamp is used)

## Demo 2: Ensemble + Intersections (`demo2_ensemble.yaml`)

Additional fields:
- `sae_single` (mapping): same shape as Demo 1 `sae` for baseline comparison
- `ensemble` (mapping):
  - `feature_dim` (int)
  - `top_k` (int)
  - `seeds` (list[int])
  - `intersect_threshold` (float)
- `outputs.base_dir`: default `outputs/ensemble`

Note: The Python ensemble builder expects `PY_NSI_INPUT_DIM` to be set to the activation dimension before constructing `PyEnsemble` (see `python/ensemble/intersection.py`).

## Demo 3: Spike + Hypergraph (`demo3_spike_hypergraph.yaml`)

Additional fields:
- `ensemble` (mapping): same shape as Demo 2
- `spike` (mapping):
  - `t_start` (float)
  - `delta_t` (float)
  - `min_sigmoid` (float)
  - `gse_window` (float)
- `metrics` adds:
  - `active_threshold_intersection` (float)
  - `active_threshold_hyperedge` (float)
- `outputs.base_dir`: default `outputs/spike_hypergraph`

## Demo 4: Causal (`demo4_causal.yaml`)

Additional fields:
- `dataset` (mapping):
  - `name` (str)
  - `n_samples` (int)
  - `seed` (int)
  - `bias_strength` (float)
  - `noise` (float)
- `ensemble` (mapping): same shape as Demo 2
- `spike` (mapping): same shape as Demo 3
- `stii` (mapping):
  - `max_order_k` (int)
  - `subset_sampling.enabled` (bool)
  - `subset_sampling.per_edge_samples` (int)
  - `stability_checks.bootstrap` (int)
- `acdc` (mapping):
  - `metric` (str)
  - `tolerance_drop` (float)
  - `max_edges` (int)
- `outputs.base_dir`: default `outputs/causal`

## Demo 5: Dashboard (`demo5_dashboard.yaml`)

Fields:
- `runs` (mapping): base directories for each prior demo
  - `baseline_dir`, `ensemble_dir`, `spike_hypergraph_dir`, `causal_dir`
- `selection` (mapping):
  - `mode` (`latest` or `manual`)
  - `manual_paths.*` (per-demo run directory overrides)
- `plots` (mapping): `hist_bins`, `top_k_stii`, `top_k_gender_nodes`
- `toggles` (mapping): show/hide panels

## Output Layout (All Demos)
Each demo writes artifacts under its configured base directory:

```
outputs/<demo>/<run_tag>/
  config.yaml
  ... demo-specific artifacts ...
```

The exact subfolders depend on the demo; see `REPRODUCIBILITY.md` for a full artifact map.
