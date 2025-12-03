from __future__ import annotations

import os
import sys
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Make local modules importable (namespace package "python" at repo root)
sys.path.append(os.path.abspath("."))

from python.utils.config import load_yaml  # noqa: E402
from python.utils.artifacts import create_run_dir, dump_json, dump_yaml  # noqa: E402
from python.datasets.bank_sentences import generate_bank_dataset  # noqa: E402
from python.activations.extract import get_model_and_tokenizer, capture_layer_activations  # noqa: E402
from python.ensemble.intersection import build_pyensemble  # noqa: E402
from python.hypergraph.pipeline import build_hypergraph  # noqa: E402
from python.metrics.polysemanticity import (  # noqa: E402
    concept_probs,
    poly_count,
    entropy,
    summarize_polysemanticity,
)
from python.metrics.downstream import evaluate_logreg  # noqa: E402
from python.plots.hist import plot_histogram  # noqa: E402


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _preflight_py_nsi() -> None:
    """
    Ensure py_nsi is importable before proceeding. If not, raise with instruction.
    """
    try:
        import py_nsi  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "py_nsi is not importable. Build the local Rust wheel first:\n"
            "  cd py_nsi && maturin develop --release\n"
            "Then re-run the demo."
        ) from e


def _metrics_bundle(prob: np.ndarray, eps: float, extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a metrics dict with summary poly stats + extras.
    """
    summary = summarize_polysemanticity(prob, eps=eps)
    out = {
        **summary,
        **extra,
        "eps": eps,
    }
    return out


def main(config_path: str = "configs/demo3_spike_hypergraph.yaml") -> None:
    # Load config
    cfg = load_yaml(config_path)
    model_name: str = cfg["model_name"]
    layer_index: int = int(cfg["layer_index"])
    ds_cfg = cfg["dataset"]
    ens_cfg = cfg["ensemble"]
    spk_cfg = cfg["spike"]
    met_cfg = cfg["metrics"]
    out_cfg = cfg["outputs"]

    # Determinism: prefer dataset seed for global sampling unless otherwise specified
    global_seed = int(ds_cfg.get("seed", 1337))
    _seed_all(global_seed)

    # 1) Data
    n_per_class = int(ds_cfg["n_per_class"])
    texts, labels = generate_bank_dataset(n_per_class=n_per_class, seed=global_seed)
    labels_np = np.asarray(labels, dtype=np.int32)
    num_concepts = int(len(set(labels)))
    concept_names = ds_cfg.get("concepts", [f"concept_{i}" for i in range(num_concepts)])

    # 2) HF model and activations (GPT-2 tiny)
    model, tokenizer = get_model_and_tokenizer(model_name)
    acts = capture_layer_activations(model, tokenizer, texts, layer_index=layer_index)  # [N, D]
    if acts.shape[0] != len(texts):
        raise RuntimeError(f"Activation rows {acts.shape[0]} != #texts {len(texts)}")
    input_dim = int(acts.shape[1])

    # Preflight py_nsi availability
    _preflight_py_nsi()

    # 3) Ensemble (Rust via py_nsi)
    feature_dim = int(ens_cfg["feature_dim"])
    top_k_ens = int(ens_cfg["top_k"])
    seeds_ens: List[int] = [int(s) for s in ens_cfg["seeds"]]

    # Provide input dimension to the PySimpleSaeEncoder wrappers
    os.environ["PY_NSI_INPUT_DIM"] = str(input_dim)
    ensemble = build_pyensemble(feature_dim=feature_dim, top_k=top_k_ens, seeds=seeds_ens)

    # 4) Hypergraph pipeline (spikes + GSE + aggregation)
    t_start = float(spk_cfg["t_start"])
    delta_t = float(spk_cfg["delta_t"])
    min_sigmoid = float(spk_cfg["min_sigmoid"])
    gse_window = float(spk_cfg["gse_window"])

    store, features_bool, edge_keys = build_hypergraph(
        ensemble=ensemble,
        acts=acts,
        labels=labels_np,
        t_start=t_start,
        delta_t=delta_t,
        min_sigmoid=min_sigmoid,
        gse_window=gse_window,
    )
    N, E = int(features_bool.shape[0]), int(features_bool.shape[1])

    # 5) Metrics on hyperedge features
    eps = float(met_cfg["eps"])
    active_threshold_hyperedge = float(met_cfg["active_threshold_hyperedge"])
    bins = int(met_cfg["hist_bins"])

    features_float = features_bool.astype(np.float32)
    if E == 0:
        # Graceful path: no hyperedges formed
        base_dir = out_cfg["base_dir"]
        run_tag = out_cfg.get("run_tag", None)
        run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)

        # Save minimal artifacts
        np.save(os.path.join(run_dir, "features_hyperedges.npy"), features_bool)
        dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))
        with open(os.path.join(run_dir, "edge_keys.json"), "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        # Empty HIF (no edges)
        store.export_hif(os.path.join(run_dir, "hypergraph.hif.json"))

        metrics_h = {
            "representation": "hyperedges_spike_temporal",
            "num_features": 0,
            "num_concepts": int(num_concepts),
            "concepts": list(concept_names),
            "num_samples": int(N),
            "num_edges": 0,
            "edge_size_median": 0.0,
            "edge_size_p90": 0.0,
            "gse_window": gse_window,
            "spike": {"t_start": t_start, "delta_t": delta_t, "min_sigmoid": min_sigmoid},
            "thresholds": {
                "eps": eps,
                "active_threshold_hyperedge": active_threshold_hyperedge,
            },
            "accuracy": 0.0,
            "median_poly": 0.0,
            "p90_poly": 0.0,
            "monosemantic_rate": 0.0,
            "note": "No hyperedges formed; check gse_window/min_sigmoid/top_k.",
        }
        dump_json(metrics_h, os.path.join(run_dir, "metrics_hyperedges.json"))
        print("Demo3 hypergraph: no hyperedges formed; exported empty HIF and minimal artifacts.")
        return

    prob_h = concept_probs(
        features_float,
        labels_np,
        num_concepts=num_concepts,
        active_threshold=active_threshold_hyperedge,
    )  # [E, m]
    poly_h = poly_count(prob_h, eps=eps)  # [E]
    ent_h = entropy(prob_h)  # [E]
    acc_h = evaluate_logreg(features_float, labels_np, seed=global_seed)["accuracy"]

    # Edge size summary
    edge_sizes = np.asarray([len(k) for k in edge_keys], dtype=np.int32)
    edge_size_median = float(np.median(edge_sizes)) if len(edge_sizes) > 0 else 0.0
    edge_size_p90 = float(np.percentile(edge_sizes, 90.0)) if len(edge_sizes) > 0 else 0.0

    metrics_h: Dict[str, Any] = _metrics_bundle(
        prob_h,
        eps=eps,
        extra={
            "representation": "hyperedges_spike_temporal",
            "num_features": int(prob_h.shape[0]),
            "num_concepts": int(prob_h.shape[1]),
            "concepts": list(concept_names),
            "num_samples": int(N),
            "num_edges": int(E),
            "edge_size_median": edge_size_median,
            "edge_size_p90": edge_size_p90,
            "gse_window": gse_window,
            "spike": {"t_start": t_start, "delta_t": delta_t, "min_sigmoid": min_sigmoid},
            "thresholds": {
                "eps": eps,
                "active_threshold_hyperedge": active_threshold_hyperedge,
            },
            "accuracy": float(acc_h),
        },
    )

    # 6) Artifacts
    base_dir = out_cfg["base_dir"]
    run_tag = out_cfg.get("run_tag", None)
    run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)

    # Save arrays and configs
    np.save(os.path.join(run_dir, "features_hyperedges.npy"), features_bool)
    with open(os.path.join(run_dir, "edge_keys.json"), "w", encoding="utf-8") as f:
        json.dump([list(map(int, t)) for t in edge_keys], f, indent=2)
    dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))
    dump_json(metrics_h, os.path.join(run_dir, "metrics_hyperedges.json"))

    # HIF export
    store.export_hif(os.path.join(run_dir, "hypergraph.hif.json"))

    # Plot
    title_h = (
        f"Spike-temporal Hyperedge polysemanticity (eps={eps:.2g}) — "
        f"median={metrics_h['median_poly']:.2f}, mono={metrics_h['monosemantic_rate']:.1%}"
    )
    plot_histogram(poly_h, bins=bins, title=title_h, path=os.path.join(run_dir, "poly_hist_hyperedges.png"))

    # One-line investor summary
    print(
        "Demo3 hypergraph: "
        f"hyperedge poly median={metrics_h['median_poly']:.2f}, "
        f"monosemantic_rate={metrics_h['monosemantic_rate']:.3f}, "
        f"accuracy={metrics_h['accuracy']:.3f}, edges={E}"
    )


if __name__ == "__main__":
    main()