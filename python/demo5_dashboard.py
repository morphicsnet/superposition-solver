from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# repo-local imports
sys.path.append(os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt

from python.utils.config import load_yaml  # [load_yaml()](python/utils/config.py:7)
from python.utils.artifacts import create_run_dir, dump_json  # [create_run_dir()](python/utils/artifacts.py:11)
from python.dashboard.run_discovery import resolve_selection  # [resolve_selection()](python/dashboard/run_discovery.py:1)
from python.dashboard.hif_utils import load_metrics  # [load_metrics()](python/dashboard/hif_utils.py:1)
from python.dashboard.plots import hist, stii_bar  # [hist()](python/dashboard/plots.py:1)


def _safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        if path and os.path.exists(path):
            return np.load(path)
    except Exception:
        return None
    return None


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def _top_stii_for_bar(stii_dict: Dict[str, Any], k: int) -> List[Tuple[str, float]]:
    values = stii_dict.get("values", []) if isinstance(stii_dict, dict) else []
    if not isinstance(values, list):
        return []
    # Sort deterministically: stii desc then key string
    try:
        values.sort(key=lambda d: (float(d.get("stii", 0.0)), str(d.get("edge", []))), reverse=True)
    except Exception:
        pass
    items: List[Tuple[str, float]] = []
    for d in values[: max(0, int(k))]:
        ek = d.get("edge") or []
        lbl = "{" + ",".join(str(int(x)) for x in ek) + "}"
        items.append((lbl, float(d.get("stii", 0.0))))
    return items


def main(config_path: str = "configs/demo5_dashboard.yaml") -> None:
    # 1) Load cfg and resolve latest/manual selections
    cfg = load_yaml(config_path) or {}
    sel = resolve_selection(cfg)

    # 2) Create output directory for investor report
    out_dir = create_run_dir(base_dir="outputs/investor")
    out_png = os.path.join(out_dir, "plots")
    os.makedirs(out_png, exist_ok=True)

    plots_cfg = cfg.get("plots", {}) if isinstance(cfg, dict) else {}
    bins = int(plots_cfg.get("hist_bins", 30))
    top_k_stii = int(plots_cfg.get("top_k_stii", 20))

    summary: Dict[str, Any] = {"selection": sel, "artifacts": {}}

    # 3) Baseline (Demo 1)
    base_run = sel.get("baseline")
    if base_run:
        m = load_metrics(base_run)
        b = m.get("baseline")
        if b:
            summary["baseline"] = {
                "median_poly": float(b.get("median_poly", 0.0)),
                "p90_poly": float(b.get("p90_poly", 0.0)),
                "monosemantic_rate": float(b.get("monosemantic_rate", 0.0)),
                "num_features": int(b.get("num_features", 0)),
            }
        # Histogram from array if available
        arr = _safe_load_npy(os.path.join(base_run, "poly_counts.npy"))
        if arr is not None and arr.size > 0:
            fig = hist(values=arr, bins=bins, title="Baseline poly(f) counts", xlabel="poly(f)")
            p = os.path.join(out_png, "baseline_poly_hist.png")
            _save_fig(fig, p)
            summary["artifacts"]["baseline_poly_hist"] = p

    # 4) Ensemble (Demo 2)
    ens_run = sel.get("ensemble")
    if ens_run:
        m = load_metrics(ens_run)
        e1, e2, ec = m.get("ensemble_single"), m.get("ensemble_intersection"), m.get("ensemble_compare")
        if e1 or e2:
            summary["ensemble"] = {
                "median_poly_single": float((e1 or {}).get("median_poly", 0.0)),
                "monosemantic_rate_single": float((e1 or {}).get("monosemantic_rate", 0.0)),
                "accuracy_single": float((e1 or {}).get("accuracy", 0.0)),
                "median_poly_intersection": float((e2 or {}).get("median_poly", 0.0)),
                "monosemantic_rate_intersection": float((e2 or {}).get("monosemantic_rate", 0.0)),
                "accuracy_intersection": float((e2 or {}).get("accuracy", 0.0)),
            }
        # Histograms from arrays if available (save as two separate figs)
        arr_single = _safe_load_npy(os.path.join(ens_run, "poly_counts_single.npy"))
        if arr_single is not None and arr_single.size > 0:
            fig1 = hist(values=arr_single, bins=bins, title="Single SAE poly(f)", xlabel="poly(f)")
            p1 = os.path.join(out_png, "ensemble_single_poly_hist.png")
            _save_fig(fig1, p1)
            summary["artifacts"]["ensemble_single_poly_hist"] = p1

        arr_inter = _safe_load_npy(os.path.join(ens_run, "poly_counts_intersection.npy"))
        if arr_inter is not None and arr_inter.size > 0:
            fig2 = hist(values=arr_inter, bins=bins, title="Ensemble ∩ poly(f)", xlabel="poly(f)")
            p2 = os.path.join(out_png, "ensemble_intersection_poly_hist.png")
            _save_fig(fig2, p2)
            summary["artifacts"]["ensemble_intersection_poly_hist"] = p2

    # 5) Hypergraph (Demo 3)
    hyp_run = sel.get("spike_hypergraph")
    if hyp_run:
        m = load_metrics(hyp_run)
        h = m.get("hypergraph")
        if h:
            summary["hypergraph"] = {
                "num_edges": int(h.get("num_edges", 0)),
                "num_concepts": int(h.get("num_concepts", 0)),
                "median_poly": float(h.get("median_poly", 0.0)),
                "monosemantic_rate": float(h.get("monosemantic_rate", 0.0)),
                "accuracy": float(h.get("accuracy", 0.0)),
            }
        # If upstream histogram image exists, copy reference into summary (not copying the file)
        pimg = os.path.join(hyp_run, "poly_hist_hyperedges.png")
        if os.path.exists(pimg):
            summary["artifacts"]["hypergraph_poly_hist_src"] = pimg

    # 6) Causal (Demo 4) — STII + ACDC + fairness
    c_run = sel.get("causal")
    if c_run:
        m = load_metrics(c_run)
        stii = m.get("causal_stii", {})
        acdc = m.get("acdc", {})
        fair = m.get("fairness", {})

        # STII bar (top-K)
        items = _top_stii_for_bar(stii, k=top_k_stii)
        if items:
            fig = stii_bar(items, title=f"Top-{top_k_stii} STII edges")
            p = os.path.join(out_png, "stii_topk_bar.png")
            _save_fig(fig, p)
            summary["artifacts"]["stii_topk_bar"] = p

        summary["causal"] = {
            "stii_computed_count": int(stii.get("computed_count", 0)) if isinstance(stii, dict) else 0,
            "acdc_base_acc": float(acdc.get("base_acc", 0.0)) if isinstance(acdc, dict) else 0.0,
            "acdc_final_acc": float(acdc.get("final_acc", 0.0)) if isinstance(acdc, dict) else 0.0,
            "acdc_kept_edges": int(len(acdc.get("kept_edges", []))) if isinstance(acdc, dict) else 0,
            "fair_any_biased_in_minimal": bool(fair.get("any_biased_node_in_minimal", False))
            if isinstance(fair, dict)
            else False,
            "fair_biased_nodes_in_minimal_count": int(fair.get("biased_nodes_in_minimal_count", 0))
            if isinstance(fair, dict)
            else 0,
        }

    # 7) Write summary JSON and Markdown
    dash_json = os.path.join(out_dir, "dashboard_metrics.json")
    dump_json(summary, dash_json)

    md_lines: List[str] = []
    md_lines.append("# Investor Dashboard Summary")
    md_lines.append("")
    md_lines.append(f"Output directory: `{out_dir}`")
    md_lines.append("")
    md_lines.append("## Selections")
    for k, v in (sel or {}).items():
        md_lines.append(f"- {k}: `{v}`")
    md_lines.append("")
    if "baseline" in summary:
        b = summary["baseline"]
        md_lines.append("## Baseline (Demo 1)")
        md_lines.append(f"- Median poly: {b['median_poly']:.2f}")
        md_lines.append(f"- Monosemantic rate: {b['monosemantic_rate']:.3f}")
    if "ensemble" in summary:
        e = summary["ensemble"]
        md_lines.append("")
        md_lines.append("## Ensemble Intersection (Demo 2)")
        md_lines.append(f"- Single median poly: {e['median_poly_single']:.2f}, mono: {e['monosemantic_rate_single']:.3f}, acc: {e['accuracy_single']:.3f}")
        md_lines.append(f"- ∩ median poly: {e['median_poly_intersection']:.2f}, mono: {e['monosemantic_rate_intersection']:.3f}, acc: {e['accuracy_intersection']:.3f}")
    if "hypergraph" in summary:
        h = summary["hypergraph"]
        md_lines.append("")
        md_lines.append("## Spike–Hypergraph (Demo 3)")
        md_lines.append(f"- #Edges: {h['num_edges']}, accuracy: {h['accuracy']:.3f}")
        md_lines.append(f"- Median poly: {h['median_poly']:.2f}, monosemantic: {h['monosemantic_rate']:.3f}")
    if "causal" in summary:
        c = summary["causal"]
        md_lines.append("")
        md_lines.append("## Causal Circuits (Demo 4: STII + ACDC + Fairness)")
        md_lines.append(f"- ACDC base→final accuracy: {c['acdc_base_acc']:.3f} → {c['acdc_final_acc']:.3f}")
        md_lines.append(f"- Kept edges: {c['acdc_kept_edges']}")
        md_lines.append(
            f"- Fairness: biased nodes present in minimal? {'Yes' if c['fair_any_biased_in_minimal'] else 'No'} "
            f"(count={c['fair_biased_nodes_in_minimal_count']})"
        )

    md_lines.append("")
    md_lines.append("## Artifacts")
    for k, p in (summary.get("artifacts", {}) or {}).items():
        md_lines.append(f"- {k}: `{p}`")

    md_lines.append("")
    md_lines.append("> Acceptance mapping: This report corresponds to the investor story in DEMO_CORRIDOR.md "
                    "(polysemanticity collapse, intersection effects, topology summary, STII/ACDC, fairness).")

    md_path = os.path.join(out_dir, "dashboard_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote investor dashboard summary to: {md_path}")
    print(f"Wrote metrics JSON to: {dash_json}")


if __name__ == "__main__":
    main()