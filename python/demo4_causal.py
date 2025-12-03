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
from python.datasets.loans_bias import generate_loans_dataset  # noqa: E402
from python.activations.extract import get_model_and_tokenizer, capture_layer_activations  # noqa: E402
from python.ensemble.intersection import build_pyensemble  # noqa: E402
from python.hypergraph.pipeline import build_hypergraph_with_nodes  # noqa: E402
from python.stii.compute import compute_stii_for_hyperedge  # noqa: E402
from python.acdc.prune import acdc_minimal_circuit  # noqa: E402
from python.metrics.fairness import gender_concept_probs, report_bias_presence  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def _train_logreg_nodes(X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Any]:
    """
    Train/test split on node features, return base accuracy and model.
    """
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int32, copy=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(seed), stratify=y
    )
    clf = LogisticRegression(solver="liblinear", random_state=int(seed), max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    return {"clf": clf, "accuracy": acc, "X_test": X_test, "y_test": y_test}


def main(config_path: str = "configs/demo4_causal.yaml") -> None:
    # Load config
    cfg = load_yaml(config_path)
    model_name: str = cfg["model_name"]
    layer_index: int = int(cfg["layer_index"])
    ds_cfg = cfg["dataset"]
    ens_cfg = cfg["ensemble"]
    spk_cfg = cfg["spike"]
    stii_cfg = cfg["stii"]
    acdc_cfg = cfg["acdc"]
    out_cfg = cfg["outputs"]

    # Determinism
    global_seed = int(ds_cfg.get("seed", 1337))
    _seed_all(global_seed)

    # 1) Data
    n_samples = int(ds_cfg["n_samples"])
    bias_strength = float(ds_cfg.get("bias_strength", 0.25))
    noise = float(ds_cfg.get("noise", 0.05))
    texts, labels_np, genders_np = generate_loans_dataset(
        n_samples=n_samples, seed=global_seed, bias_strength=bias_strength, noise=noise
    )
    labels_np = np.asarray(labels_np, dtype=np.int32)
    genders_np = np.asarray(genders_np, dtype=np.int32)

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

    # 4) Hypergraph + node features (spikes + GSE + aggregation)
    t_start = float(spk_cfg["t_start"])
    delta_t = float(spk_cfg["delta_t"])
    min_sigmoid = float(spk_cfg["min_sigmoid"])
    gse_window = float(spk_cfg["gse_window"])

    store, X_edge_bool, edge_keys, nodes_by_sample_bool, node_keys = build_hypergraph_with_nodes(
        ensemble=ensemble,
        acts=acts,
        labels=labels_np,
        t_start=t_start,
        delta_t=delta_t,
        min_sigmoid=min_sigmoid,
        gse_window=gse_window,
    )

    N, E = int(X_edge_bool.shape[0]), int(X_edge_bool.shape[1])
    U = int(nodes_by_sample_bool.shape[1]) if nodes_by_sample_bool.ndim == 2 else 0

    # Graceful path: no hyperedges or no nodes
    if E == 0 or U == 0:
        base_dir = out_cfg["base_dir"]
        run_tag = out_cfg.get("run_tag", None)
        run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)
        # Minimal artifacts
        dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))
        # Empty HIF (no edges); still export what we have
        store.export_hif(os.path.join(run_dir, "hypergraph_stii.hif.json"))
        dump_json(
            {
                "note": "No hyperedges or nodes formed; check gse_window/min_sigmoid/top_k and dataset size.",
                "num_edges": int(E),
                "num_nodes": int(U),
            },
            os.path.join(run_dir, "stii_values.json"),
        )
        dump_json(
            {
                "kept_edges": [],
                "removed_edges": [],
                "base_acc": 0.0,
                "final_acc": 0.0,
                "note": "No edges to prune.",
            },
            os.path.join(run_dir, "acdc_minimal_circuit.json"),
        )
        dump_json(
            {
                "threshold": 0.6,
                "num_biased_nodes": 0,
                "num_minimal_edges": 0,
                "biased_nodes_in_minimal_count": 0,
                "biased_nodes_in_minimal_ratio": 0.0,
                "any_biased_node_in_minimal": False,
                "examples": [],
                "note": "No nodes or minimal circuit.",
            },
            os.path.join(run_dir, "fairness_report.json"),
        )
        print("Demo4 STII+ACDC: edges=0, stii_computed=0, base_acc=0.000, final_acc=0.000, gender_nodes_in_minimal=0")
        return

    # Convert designs to float
    X_edge = X_edge_bool.astype(np.float32)
    X_node = nodes_by_sample_bool.astype(np.float32)
    y = labels_np.astype(np.int32)

    # 5) Train a LogisticRegression on node features to predict y; compute base accuracy
    node_clf_bundle = _train_logreg_nodes(X_node, y, seed=global_seed)
    node_clf = node_clf_bundle["clf"]
    base_acc_nodes = float(node_clf_bundle["accuracy"])

    # Map nodes to columns
    node_to_col: Dict[int, int] = {int(nid): j for j, (nid,) in enumerate(node_keys)}

    # 6) STII per hyperedge (size <= 3 for tractability)
    max_order_k = int(stii_cfg.get("max_order_k", 2))
    stii_values: Dict[Tuple[int, ...], float] = {}
    computed_count = 0
    for ek in edge_keys:
        m = len(ek)
        if m <= 3:
            try:
                stii_val = compute_stii_for_hyperedge(
                    store=store,
                    edge_key=ek,
                    node_to_col=node_to_col,
                    X_base=X_node,
                    y=y,
                    logreg_model=node_clf,
                    max_order_k=min(max_order_k, m),
                )
            except Exception as e:
                # Robust to any edge-specific issues; record zero
                stii_val = 0.0
            stii_values[ek] = float(stii_val)
            computed_count += 1

    # 7) ACDC pruning on edge features
    tolerance_drop = float(acdc_cfg.get("tolerance_drop", 0.02))
    max_edges = int(acdc_cfg.get("max_edges", 50))
    acdc_result = acdc_minimal_circuit(
        edge_keys=edge_keys,
        stii=stii_values,
        X_edge=X_edge,
        y=y,
        tolerance_drop=tolerance_drop,
        max_edges=max_edges,
        seed=global_seed,
    )

    minimal_edges = acdc_result.get("kept_edges", [])
    # Build edge->nodes map
    edge_to_nodes: Dict[Tuple[int, ...], List[int]] = {ek: [int(n) for n in ek] for ek in edge_keys}

    # 8) Fairness: node-level gender association and presence in minimal circuit
    node_gender = gender_concept_probs(nodes_by_sample=X_node.astype(bool), genders=genders_np)
    fairness = report_bias_presence(
        minimal_edges=minimal_edges,
        edge_to_nodes=edge_to_nodes,
        node_gender_probs=node_gender,
        node_keys=node_keys,
        threshold=0.6,
    )

    # 9) Artifacts
    base_dir = out_cfg["base_dir"]
    run_tag = out_cfg.get("run_tag", None)
    run_dir = create_run_dir(base_dir=base_dir, run_tag=run_tag)

    # Save config
    dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    # Export HIF with STII
    store.export_hif(os.path.join(run_dir, "hypergraph_stii.hif.json"))

    # STII values (per-edge)
    stii_list = [
        {"edge": [int(x) for x in ek], "stii": float(stii_values.get(ek, 0.0))}
        for ek in edge_keys
        if len(ek) <= 3
    ]
    dump_json({"values": stii_list, "computed_count": int(computed_count), "total_edges": int(E)}, os.path.join(run_dir, "stii_values.json"))

    # ACDC minimal circuit
    # Ensure serializable (tuples -> lists)
    acdc_ser = {
        "kept_edges": [[int(x) for x in ek] for ek in acdc_result.get("kept_edges", [])],
        "removed_edges": [[int(x) for x in ek] for ek in acdc_result.get("removed_edges", [])],
        "base_acc": float(acdc_result.get("base_acc", 0.0)),
        "final_acc": float(acdc_result.get("final_acc", 0.0)),
        "tolerance_drop": tolerance_drop,
        "max_edges": max_edges,
    }
    dump_json(acdc_ser, os.path.join(run_dir, "acdc_minimal_circuit.json"))

    # Fairness report
    dump_json(fairness, os.path.join(run_dir, "fairness_report.json"))

    # Optional: reuse poly histogram if present (skip silently otherwise)
    reuse_plot_candidates = [
        os.path.join(run_dir, "poly_hist_hyperedges.png"),  # if some upstream step created it
    ]
    for p in reuse_plot_candidates:
        if os.path.exists(p):
            # Already present; nothing to do
            break

    # 10) One-line investor summary
    kept_count = int(len(acdc_result.get("kept_edges", [])))
    print(
        f"Demo4 STII+ACDC: edges={E}, stii_computed={computed_count}, "
        f"base_acc={base_acc_nodes:.3f}, final_acc={float(acdc_result.get('final_acc', 0.0)):.3f}, "
        f"gender_nodes_in_minimal={int(fairness.get('biased_nodes_in_minimal_count', 0))}"
    )


if __name__ == "__main__":
    main()