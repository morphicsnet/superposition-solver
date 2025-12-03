from __future__ import annotations

import os
import sys
import json
import random
from typing import Any, Dict, List

import numpy as np
import torch

# Make local modules importable (namespace package "python" at repo root)
sys.path.append(os.path.abspath("."))

from python.utils.config import load_yaml  # noqa: E402
from python.datasets.bank_sentences import generate_bank_dataset  # noqa: E402
from python.activations.extract import get_model_and_tokenizer, capture_layer_activations  # noqa: E402
from python.models.sae import SAE, train_sae, encode_topk  # noqa: E402
from python.metrics.polysemanticity import (  # noqa: E402
    concept_probs,
    poly_count,
    entropy,
    summarize_polysemanticity,
)
from python.plots.hist import plot_histogram  # noqa: E402
from python.utils.artifacts import create_run_dir, dump_json, dump_yaml  # noqa: E402


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only; keep deterministic algorithms default to avoid perf hits.


def main(config_path: str = "configs/demo1_baseline.yaml") -> None:
    # Load config
    cfg = load_yaml(config_path)
    model_name: str = cfg["model_name"]
    layer_index: int = int(cfg["layer_index"])
    ds_cfg = cfg["dataset"]
    sae_cfg = cfg["sae"]
    metrics_cfg = cfg["metrics"]
    out_cfg = cfg["outputs"]

    # Determinism: prefer dataset seed for global sampling unless otherwise specified
    _seed_all(int(ds_cfg.get("seed", 1337)))

    # 1) Data
    n_per_class = int(ds_cfg["n_per_class"])
    texts, labels = generate_bank_dataset(n_per_class=n_per_class, seed=int(ds_cfg["seed"]))
    labels_np = np.asarray(labels, dtype=np.int32)
    num_concepts = int(len(set(labels)))
    concept_names = ds_cfg.get("concepts", [f"concept_{i}" for i in range(num_concepts)])

    # 2) HF model and activations (GPT-2 tiny)
    model, tokenizer = get_model_and_tokenizer(model_name)
    acts = capture_layer_activations(model, tokenizer, texts, layer_index=layer_index)  # [N, D]
    if acts.shape[0] != len(texts):
        raise RuntimeError(f"Activation rows {acts.shape[0]} != #texts {len(texts)}")
    input_dim = int(acts.shape[1])

    # 3) SAE training
    _seed_all(int(sae_cfg.get("seed", 42)))
    hidden_dim = int(sae_cfg["hidden_dim"])
    top_k = int(sae_cfg["top_k"])
    epochs = int(sae_cfg["epochs"])
    lr = float(sae_cfg["lr"])
    l1_lambda = float(sae_cfg["l1_lambda"])
    active_threshold = float(sae_cfg["active_threshold"])

    sae_model, train_stats = train_sae(
        acts=acts,
        hidden_dim=hidden_dim,
        top_k=top_k,
        l1_lambda=l1_lambda,
        seed=int(sae_cfg.get("seed", 42)),
        epochs=epochs,
        lr=lr,
        device="cpu",
    )

    # 4) Encode sparse features (top-k), compute metrics
    features = encode_topk(sae_model, acts, top_k=top_k, device="cpu")  # [N, H]
    prob = concept_probs(features, labels_np, num_concepts=num_concepts, active_threshold=active_threshold)  # [H, m]
    eps = float(metrics_cfg["eps"])
    poly = poly_count(prob, eps=eps)  # [H]
    ent = entropy(prob)  # [H]
    summary = summarize_polysemanticity(prob, eps=eps)

    # 5) Artifacts
    base_dir = out_cfg["base_dir"]
    run_tag = out_cfg.get("run_tag", None)
    run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)

    # Save arrays and metrics/configs
    np.save(os.path.join(run_dir, "probs.npy"), prob)
    np.save(os.path.join(run_dir, "poly_counts.npy"), poly)
    np.save(os.path.join(run_dir, "entropy.npy"), ent)
    dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))
    # Augment summary with a few extra run details
    metrics_out: Dict[str, Any] = {
        **summary,
        "eps": eps,
        "active_threshold": active_threshold,
        "num_features": int(prob.shape[0]),
        "num_concepts": int(prob.shape[1]),
        "concepts": list(concept_names),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "epochs": epochs,
        "lr": lr,
        "l1_lambda": l1_lambda,
    }
    dump_json(metrics_out, os.path.join(run_dir, "metrics.json"))

    # Plot histogram of polysemanticity counts with overlays in title
    median_poly = metrics_out["median_poly"]
    mono_rate = metrics_out["monosemantic_rate"]
    bins = int(metrics_cfg["hist_bins"])
    title = f"SAE polysemanticity (eps={eps:.2g}) — median={median_poly:.2f}, monosemantic_rate={mono_rate:.1%}"
    plot_histogram(poly, bins=bins, title=title, path=os.path.join(run_dir, "poly_hist.png"))

    # Log one-line summary per acceptance criteria
    print(f"Baseline SAE polysemanticity: median={median_poly:.2f}, monosemantic_rate={mono_rate:.3f}")


if __name__ == "__main__":
    main()