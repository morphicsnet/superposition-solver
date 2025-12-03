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
from python.models.sae import train_sae, encode_topk  # noqa: E402
from python.metrics.polysemanticity import (  # noqa: E402
    concept_probs,
    poly_count,
    entropy,
    summarize_polysemanticity,
)
from python.metrics.downstream import evaluate_logreg  # noqa: E402
from python.plots.hist import plot_histogram  # noqa: E402
from python.plots.compare import plot_dual_hist  # noqa: E402
from python.ensemble.intersection import build_pyensemble, encode_all_and_intersect  # noqa: E402


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only; keep deterministic algorithms default to avoid perf hits.


def _metrics_bundle(
    prob: np.ndarray,
    eps: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
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


def main(config_path: str = "configs/demo2_ensemble.yaml") -> None:
    # Load config
    cfg = load_yaml(config_path)
    model_name: str = cfg["model_name"]
    layer_index: int = int(cfg["layer_index"])
    ds_cfg = cfg["dataset"]
    sae_cfg = cfg["sae_single"]
    ens_cfg = cfg["ensemble"]
    metrics_cfg = cfg["metrics"]
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

    # 3) Baseline single SAE (Python)
    _seed_all(int(sae_cfg.get("seed", 42)))
    hidden_dim = int(sae_cfg["hidden_dim"])
    top_k_single = int(sae_cfg["top_k"])
    epochs = int(sae_cfg["epochs"])
    lr = float(sae_cfg["lr"])
    l1_lambda = float(sae_cfg["l1_lambda"])
    active_threshold_single = float(sae_cfg["active_threshold"])

    sae_model, _ = train_sae(
        acts=acts,
        hidden_dim=hidden_dim,
        top_k=top_k_single,
        l1_lambda=l1_lambda,
        seed=int(sae_cfg.get("seed", 42)),
        epochs=epochs,
        lr=lr,
        device="cpu",
    )

    features_single = encode_topk(sae_model, acts, top_k=top_k_single, device="cpu")  # [N, H_single]
    prob_single = concept_probs(
        features_single,
        labels_np,
        num_concepts=num_concepts,
        active_threshold=active_threshold_single,
    )  # [H_single, m]
    eps = float(metrics_cfg["eps"])
    poly_single = poly_count(prob_single, eps=eps)  # [H_single]
    ent_single = entropy(prob_single)  # [H_single]
    acc_single = evaluate_logreg(features_single, labels_np, seed=global_seed)["accuracy"]

    # 4) Ensemble + intersection (Rust via py_nsi)
    feature_dim = int(ens_cfg["feature_dim"])
    top_k_ens = int(ens_cfg["top_k"])
    seeds_ens: List[int] = [int(s) for s in ens_cfg["seeds"]]
    intersect_threshold = float(ens_cfg["intersect_threshold"])

    # Provide input dimension to the PySimpleSaeEncoder wrappers
    os.environ["PY_NSI_INPUT_DIM"] = str(input_dim)

    # Build ensemble and compute boolean intersection masks
    ensemble = build_pyensemble(feature_dim=feature_dim, top_k=top_k_ens, seeds=seeds_ens)
    masks_bool = encode_all_and_intersect(ensemble, acts=acts, threshold=intersect_threshold)  # [N, H]
    features_intersection = masks_bool.astype(np.float32)  # treat boolean mask as 0/1 features

    # For intersection features, treat boolean masks as active with threshold 0.5
    prob_intersection = concept_probs(
        features_intersection,
        labels_np,
        num_concepts=num_concepts,
        active_threshold=0.5,
    )  # [H, m]
    poly_intersection = poly_count(prob_intersection, eps=eps)
    ent_intersection = entropy(prob_intersection)
    acc_intersection = evaluate_logreg(features_intersection, labels_np, seed=global_seed)["accuracy"]

    # 5) Artifacts
    base_dir = out_cfg["base_dir"]
    run_tag = out_cfg.get("run_tag", None)
    run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)

    # Save arrays and configs
    np.save(os.path.join(run_dir, "probs_single.npy"), prob_single)
    np.save(os.path.join(run_dir, "poly_counts_single.npy"), poly_single)
    np.save(os.path.join(run_dir, "entropy_single.npy"), ent_single)

    np.save(os.path.join(run_dir, "probs_intersection.npy"), prob_intersection)
    np.save(os.path.join(run_dir, "poly_counts_intersection.npy"), poly_intersection)
    np.save(os.path.join(run_dir, "entropy_intersection.npy"), ent_intersection)

    dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    # Metrics JSONs
    metrics_single: Dict[str, Any] = _metrics_bundle(
        prob_single,
        eps=eps,
        extra={
            "representation": "single_sae",
            "active_threshold": active_threshold_single,
            "intersect_threshold": None,
            "num_features": int(prob_single.shape[0]),
            "num_concepts": int(prob_single.shape[1]),
            "concepts": list(concept_names),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "top_k": top_k_single,
            "epochs": epochs,
            "lr": lr,
            "l1_lambda": l1_lambda,
            "accuracy": float(acc_single),
        },
    )
    dump_json(metrics_single, os.path.join(run_dir, "metrics_single.json"))

    metrics_intersection: Dict[str, Any] = _metrics_bundle(
        prob_intersection,
        eps=eps,
        extra={
            "representation": "ensemble_intersection",
            "active_threshold": 0.5,
            "intersect_threshold": intersect_threshold,
            "num_features": int(prob_intersection.shape[0]),
            "num_concepts": int(prob_intersection.shape[1]),
            "concepts": list(concept_names),
            "input_dim": input_dim,
            "feature_dim": feature_dim,
            "top_k": top_k_ens,
            "ensemble_seeds": seeds_ens,
            "accuracy": float(acc_intersection),
        },
    )
    dump_json(metrics_intersection, os.path.join(run_dir, "metrics_intersection.json"))

    compare = {
        "single": {
            "median_poly": metrics_single["median_poly"],
            "p90_poly": metrics_single["p90_poly"],
            "monosemantic_rate": metrics_single["monosemantic_rate"],
            "accuracy": metrics_single["accuracy"],
            "active_threshold": metrics_single["active_threshold"],
            "eps": metrics_single["eps"],
        },
        "intersection": {
            "median_poly": metrics_intersection["median_poly"],
            "p90_poly": metrics_intersection["p90_poly"],
            "monosemantic_rate": metrics_intersection["monosemantic_rate"],
            "accuracy": metrics_intersection["accuracy"],
            "active_threshold": metrics_intersection["active_threshold"],
            "intersect_threshold": metrics_intersection["intersect_threshold"],
            "eps": metrics_intersection["eps"],
        },
    }
    dump_json(compare, os.path.join(run_dir, "compare.json"))

    # Plots
    bins = int(metrics_cfg["hist_bins"])
    title_single = (
        f"Single SAE polysemanticity (eps={eps:.2g}) — "
        f"median={metrics_single['median_poly']:.2f}, mono={metrics_single['monosemantic_rate']:.1%}"
    )
    plot_histogram(poly_single, bins=bins, title=title_single, path=os.path.join(run_dir, "poly_hist_single.png"))

    title_inter = (
        f"Ensemble Intersection polysemanticity (eps={eps:.2g}) — "
        f"median={metrics_intersection['median_poly']:.2f}, mono={metrics_intersection['monosemantic_rate']:.1%}"
    )
    plot_histogram(
        poly_intersection,
        bins=bins,
        title=title_inter,
        path=os.path.join(run_dir, "poly_hist_intersection.png"),
    )

    plot_dual_hist(
        values_a=poly_single,
        values_b=poly_intersection,
        bins=bins,
        labels=("Single SAE", "Ensemble ∩"),
        title=f"Polysemanticity distribution — Single vs. Intersection (eps={eps:.2g})",
        path=os.path.join(run_dir, "poly_hist_dual.png"),
    )

    # One-line investor summary
    print(
        "Demo2 ensemble intersection: "
        f"poly median single={metrics_single['median_poly']:.2f}, intersect={metrics_intersection['median_poly']:.2f}; "
        f"monosemantic_rate single={metrics_single['monosemantic_rate']:.3f}, intersect={metrics_intersection['monosemantic_rate']:.3f}; "
        f"accuracy single={metrics_single['accuracy']:.3f}, intersect={metrics_intersection['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()