from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Make repository-local "python" package importable
sys.path.append(os.path.abspath("."))

import numpy as np
import streamlit as st

from python.utils.config import load_yaml  # [load_yaml()](python/utils/config.py:7)
from python.dashboard.run_discovery import (  # [resolve_selection()](python/dashboard/run_discovery.py:1)
    list_runs,
    pick_latest,
    resolve_selection,
)
from python.dashboard.hif_utils import (  # [summarize_hif()](python/dashboard/hif_utils.py:1)
    load_hif,
    load_metrics,
    summarize_hif,
)
from python.dashboard.plots import (  # [hypergraph_small_graph()](python/dashboard/plots.py:1)
    hist,
    stii_bar,
    hypergraph_small_graph,
)
from python.repro.bundle import (  # [collect_artifacts()](python/repro/bundle.py:1)
    collect_artifacts,
    write_manifest,
    make_zip,
)

# Optional overlay compare plot (saves to file)
try:
    from python.plots.compare import plot_dual_hist  # [plot_dual_hist()](python/plots/compare.py:1)
except Exception:  # pragma: no cover
    plot_dual_hist = None  # type: ignore


def _safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        if path and os.path.exists(path):
            return np.load(path)
    except Exception:
        return None
    return None


def _cfg_default_path() -> str:
    return "configs/demo5_dashboard.yaml"


def _header():
    st.title("Investor Dashboard — Superposition Elimination Demos")
    st.caption(
        "Zero-code exploration of Demos 1–4: polysemanticity collapse, ensemble intersections, "
        "spike-hypergraph topology, STII/ACDC causal verification, and fairness indication."
    )
    with st.expander("Investor story (acceptance mapping)"):
        st.markdown(
            "- Polysemanticity collapse: show reduced median poly and higher monosemantic rate vs. baseline.\n"
            "- Intersection effects: overlay histograms single vs. ensemble ∩ and compare downstream accuracy.\n"
            "- Spike–hypergraph topology: summarize node/edge counts and small bipartite view.\n"
            "- Causal circuits: list top STII edges, and show ACDC base→final accuracy.\n"
            "- Fairness: indicate presence of gender-associated nodes in minimal circuit.\n\n"
            "Traceability: See DEMO_CORRIDOR.md for corridor acceptance criteria."
        )


def _sidebar_controls(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Optional[str]], Dict[str, Any]]:
    st.sidebar.header("Configuration")
    config_path = st.sidebar.text_input("Config path", value=_cfg_default_path())
    if st.sidebar.button("Reload config"):
        try:
            cfg.update(load_yaml(config_path) or {})
        except Exception:
            st.sidebar.warning("Failed to load config; keeping previous.")

    runs_cfg = cfg.get("runs", {}) if isinstance(cfg, dict) else {}
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg, dict) else {}
    plots_cfg = cfg.get("plots", {}) if isinstance(cfg, dict) else {}
    toggles_cfg = cfg.get("toggles", {}) if isinstance(cfg, dict) else {}

    # Selection mode
    mode = st.sidebar.radio(
        "Selection mode",
        options=["latest", "manual"],
        index=0 if str(sel_cfg.get("mode", "latest")).lower() == "latest" else 1,
        horizontal=True,
    )

    # Base dirs and discovered runs
    baseline_dir = runs_cfg.get("baseline_dir", "outputs/baseline")
    ensemble_dir = runs_cfg.get("ensemble_dir", "outputs/ensemble")
    spike_dir = runs_cfg.get("spike_hypergraph_dir", "outputs/spike_hypergraph")
    causal_dir = runs_cfg.get("causal_dir", "outputs/causal")

    st.sidebar.subheader("Run directories (bases)")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        baseline_dir = st.text_input("Baseline base_dir", value=str(baseline_dir), key="base_baseline")
        spike_dir = st.text_input("Spike/Hypergraph base_dir", value=str(spike_dir), key="base_spike")
    with c2:
        ensemble_dir = st.text_input("Ensemble base_dir", value=str(ensemble_dir), key="base_ensemble")
        causal_dir = st.text_input("Causal base_dir", value=str(causal_dir), key="base_causal")

    discovered = {
        "baseline": list_runs(baseline_dir),
        "ensemble": list_runs(ensemble_dir),
        "spike_hypergraph": list_runs(spike_dir),
        "causal": list_runs(causal_dir),
    }

    # Manual selectors (when mode == manual), otherwise show latest pick
    selected: Dict[str, Optional[str]] = {"baseline": None, "ensemble": None, "spike_hypergraph": None, "causal": None}
    st.sidebar.subheader("Selection")
    for key, label in [
        ("baseline", "Baseline run"),
        ("ensemble", "Ensemble run"),
        ("spike_hypergraph", "Spike/Hypergraph run"),
        ("causal", "Causal run"),
    ]:
        runs = discovered.get(key, [])
        if mode == "manual":
            default_idx = 0
            options = ["(none)"] + runs
            choice = st.sidebar.selectbox(label, options=options, index=0)
            selected[key] = None if choice == "(none)" else choice
        else:
            selected[key] = pick_latest({"baseline": baseline_dir, "ensemble": ensemble_dir,
                                         "spike_hypergraph": spike_dir, "causal": causal_dir}[key])

    # Numeric controls
    st.sidebar.subheader("Plot settings")
    hist_bins = int(st.sidebar.number_input("Histogram bins", min_value=5, max_value=200, value=int(plots_cfg.get("hist_bins", 30))))
    top_k_stii = int(st.sidebar.number_input("Top-K STII edges", min_value=1, max_value=200, value=int(plots_cfg.get("top_k_stii", 20))))
    top_k_gender_nodes = int(
        st.sidebar.number_input("Top-K gender-associated nodes to list", min_value=1, max_value=100, value=int(plots_cfg.get("top_k_gender_nodes", 10)))
    )

    # Toggles
    st.sidebar.subheader("Toggles")
    show_baseline = bool(st.sidebar.checkbox("Show Baseline", value=bool(toggles_cfg.get("show_baseline", True))))
    show_ensemble = bool(st.sidebar.checkbox("Show Ensemble", value=bool(toggles_cfg.get("show_ensemble", True))))
    show_hypergraph = bool(st.sidebar.checkbox("Show Hypergraph", value=bool(toggles_cfg.get("show_hypergraph", True))))
    show_causal = bool(st.sidebar.checkbox("Show Causal", value=bool(toggles_cfg.get("show_causal", True))))

    cfg["runs"] = {
        "baseline_dir": baseline_dir,
        "ensemble_dir": ensemble_dir,
        "spike_hypergraph_dir": spike_dir,
        "causal_dir": causal_dir,
    }
    cfg["selection"] = {"mode": mode, "manual_paths": selected}
    cfg["plots"] = {"hist_bins": hist_bins, "top_k_stii": top_k_stii, "top_k_gender_nodes": top_k_gender_nodes}
    cfg["toggles"] = {
        "show_baseline": show_baseline,
        "show_ensemble": show_ensemble,
        "show_hypergraph": show_hypergraph,
        "show_causal": show_causal,
    }

    # Resolve selection (use our helper to normalize absolute paths, etc.)
    sel = resolve_selection(cfg)

    return cfg, sel, {"discovered": discovered}


def _section_polysemanticity(sel: Dict[str, Optional[str]], cfg: Dict[str, Any]) -> None:
    st.header("Polysemanticity")
    bins = int(cfg.get("plots", {}).get("hist_bins", 30))
    toggles = cfg.get("toggles", {})

    # Baseline (Demo 1)
    if toggles.get("show_baseline", True):
        st.subheader("Baseline SAE")
        run = sel.get("baseline")
        if run:
            m = load_metrics(run)
            b = m.get("baseline")
            if b:
                cols = st.columns(4)
                cols[0].metric("Median poly", f"{b.get('median_poly', 0):.2f}")
                cols[1].metric("P90 poly", f"{b.get('p90_poly', 0):.2f}")
                cols[2].metric("Monosemantic rate", f"{100.0 * float(b.get('monosemantic_rate', 0.0)):.1f}%")
                cols[3].metric("#Features", f"{int(b.get('num_features', 0))}")
            # Histogram: from array or from saved image
            arr = _safe_load_npy(os.path.join(run, "poly_counts.npy"))
            if arr is not None and arr.size > 0:
                fig = hist(values=arr, bins=bins, title="Baseline poly(f) counts", xlabel="poly(f)")
                st.pyplot(fig, clear_figure=True)
            elif os.path.exists(os.path.join(run, "poly_hist.png")):
                st.image(os.path.join(run, "poly_hist.png"), caption="Baseline poly histogram")
            else:
                st.info("No histogram available for baseline.")
        else:
            st.info("No baseline run selected.")

    # Ensemble (Demo 2)
    if toggles.get("show_ensemble", True):
        st.subheader("Ensemble ∩ vs. Single")
        run = sel.get("ensemble")
        if run:
            m = load_metrics(run)
            e1 = m.get("ensemble_single", {})
            e2 = m.get("ensemble_intersection", {})
            if e1 or e2:
                # Dual table
                st.markdown("Comparison")
                rows = []
                if e1:
                    rows.append(
                        {
                            "Representation": "Single SAE",
                            "Median poly": f"{e1.get('median_poly', 0):.2f}",
                            "Monosemantic": f"{100.0 * float(e1.get('monosemantic_rate', 0.0)):.1f}%",
                            "Accuracy": f"{float(e1.get('accuracy', 0.0)):.3f}",
                        }
                    )
                if e2:
                    rows.append(
                        {
                            "Representation": "Ensemble ∩",
                            "Median poly": f"{e2.get('median_poly', 0):.2f}",
                            "Monosemantic": f"{100.0 * float(e2.get('monosemantic_rate', 0.0)):.1f}%",
                            "Accuracy": f"{float(e2.get('accuracy', 0.0)):.3f}",
                        }
                    )
                st.table(rows)

            # Overlay histogram if arrays present
            a_single = _safe_load_npy(os.path.join(run, "poly_counts_single.npy"))
            a_inter = _safe_load_npy(os.path.join(run, "poly_counts_intersection.npy"))
            if a_single is not None and a_inter is not None and a_single.size > 0 and a_inter.size > 0 and plot_dual_hist:
                with tempfile.TemporaryDirectory() as td:
                    out_path = os.path.join(td, "dual.png")
                    try:
                        plot_dual_hist(
                            values_a=a_single,
                            values_b=a_inter,
                            bins=bins,
                            labels=("Single SAE", "Ensemble ∩"),
                            title="Polysemanticity — Single vs. ∩",
                            path=out_path,
                        )
                        st.image(out_path, caption="Single vs. Ensemble ∩ (overlay)")
                    except Exception:
                        # Fallback to separate figs
                        fig1 = hist(values=a_single, bins=bins, title="Single SAE poly(f)", xlabel="poly(f)")
                        fig2 = hist(values=a_inter, bins=bins, title="Ensemble ∩ poly(f)", xlabel="poly(f)")
                        st.pyplot(fig1, clear_figure=True)
                        st.pyplot(fig2, clear_figure=True)
            else:
                # Separate figs (if we have either)
                shown = False
                if a_single is not None and a_single.size > 0:
                    fig1 = hist(values=a_single, bins=bins, title="Single SAE poly(f)", xlabel="poly(f)")
                    st.pyplot(fig1, clear_figure=True)
                    shown = True
                if a_inter is not None and a_inter.size > 0:
                    fig2 = hist(values=a_inter, bins=bins, title="Ensemble ∩ poly(f)", xlabel="poly(f)")
                    st.pyplot(fig2, clear_figure=True)
                    shown = True
                if not shown:
                    # Try pre-made images
                    any_img = False
                    for name in ("poly_hist_dual.png", "poly_hist_single.png", "poly_hist_intersection.png"):
                        p = os.path.join(run, name)
                        if os.path.exists(p):
                            st.image(p, caption=name)
                            any_img = True
                    if not any_img:
                        st.info("No ensemble histograms available.")
        else:
            st.info("No ensemble run selected.")

    # Spike/Hypergraph (Demo 3)
    if toggles.get("show_hypergraph", True):
        st.subheader("Spike–Hypergraph (hyperedge selectivity)")
        run = sel.get("spike_hypergraph")
        if run:
            m = load_metrics(run)
            h = m.get("hypergraph", {})
            if h:
                cols = st.columns(5)
                cols[0].metric("#Edges", f"{int(h.get('num_edges', 0))}")
                cols[1].metric("#Concepts", f"{int(h.get('num_concepts', 0))}")
                cols[2].metric("Median poly", f"{h.get('median_poly', 0):.2f}")
                cols[3].metric("Monosemantic", f"{100.0 * float(h.get('monosemantic_rate', 0.0)):.1f}%")
                cols[4].metric("Accuracy", f"{float(h.get('accuracy', 0.0)):.3f}")
            # Histogram image if exists
            pimg = os.path.join(run, "poly_hist_hyperedges.png")
            if os.path.exists(pimg):
                st.image(pimg, caption="Hyperedge poly histogram")
            else:
                st.info("No hyperedge histogram available.")
        else:
            st.info("No hypergraph run selected.")


def _section_topology(sel: Dict[str, Optional[str]], cfg: Dict[str, Any]) -> None:
    st.header("Hypergraph topology")
    # Prefer causal HIF (has STII weights) else hypergraph HIF
    paths = []
    if sel.get("causal"):
        c = load_metrics(sel["causal"])
        if c.get("hif_path"):
            paths.append(c["hif_path"])
    if sel.get("spike_hypergraph"):
        h = load_metrics(sel["spike_hypergraph"])
        if h.get("hif_path"):
            paths.append(h["hif_path"])

    hif_path = next((p for p in paths if p and os.path.exists(p)), None)
    if not hif_path:
        st.info("No HIF found in selected runs.")
        return

    hif = load_hif(hif_path)
    summary = summarize_hif(hif)
    cols = st.columns(4)
    cols[0].metric("#Nodes", f"{summary.get('num_nodes', 0)}")
    cols[1].metric("#Edges", f"{summary.get('num_edges', 0)}")
    st.write("Edge size distribution:", summary.get("edge_size_hist", {}))
    try:
        fig = hypergraph_small_graph(hif, top_k_edges=25)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.info("Failed to render small topology graph.")


def _section_causal(sel: Dict[str, Optional[str]], cfg: Dict[str, Any]) -> None:
    st.header("Causal circuits (STII + ACDC)")
    run = sel.get("causal")
    if not run:
        st.info("No causal run selected.")
        return

    m = load_metrics(run)

    # STII values
    stii = m.get("causal_stii", {})
    values: List[Dict[str, Any]] = list(stii.get("values", [])) if isinstance(stii.get("values"), list) else []
    top_k = int(cfg.get("plots", {}).get("top_k_stii", 20))
    if values:
        # Sort by stii desc then edge key
        values.sort(key=lambda d: (float(d.get("stii", 0.0)), str(d.get("edge", []))), reverse=True)
        top_vals = values[:top_k]
        bar_items = [("{" + ",".join(map(str, (d.get("edge") or []))) + "}", float(d.get("stii", 0.0))) for d in top_vals]
        fig = stii_bar(bar_items, title=f"Top-{top_k} STII edges")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No STII values available.")

    # ACDC stats
    acdc = m.get("acdc", {})
    if acdc:
        cols = st.columns(4)
        cols[0].metric("Base acc", f"{float(acdc.get('base_acc', 0.0)):.3f}")
        cols[1].metric("Final acc", f"{float(acdc.get('final_acc', 0.0)):.3f}")
        cols[2].metric("#Kept edges", f"{len(acdc.get('kept_edges', []))}")
        cols[3].metric("Max edges cap", f"{int(acdc.get('max_edges', 0)) if 'max_edges' in acdc else '-'}")
    else:
        st.info("No ACDC minimal circuit available.")


def _section_fairness(sel: Dict[str, Optional[str]], cfg: Dict[str, Any]) -> None:
    st.header("Fairness")
    run = sel.get("causal")
    if not run:
        st.info("No causal run selected.")
        return

    m = load_metrics(run)
    fair = m.get("fairness", {})
    if not fair:
        st.info("No fairness report found.")
        return

    cols = st.columns(4)
    cols[0].metric("#Biased nodes", f"{int(fair.get('num_biased_nodes', 0))}")
    cols[1].metric("#Minimal edges", f"{int(fair.get('num_minimal_edges', 0))}")
    cols[2].metric("Biased in minimal", f"{int(fair.get('biased_nodes_in_minimal_count', 0))}")
    cols[3].metric("Any biased in minimal?", "Yes" if bool(fair.get("any_biased_node_in_minimal", False)) else "No")

    # List a few node IDs
    k = int(cfg.get("plots", {}).get("top_k_gender_nodes", 10))
    examples = fair.get("examples", []) or []
    if examples:
        rows = []
        for ex in examples[:k]:
            rows.append(
                {
                    "node_id": ex.get("node_id"),
                    "p_male": round(float(ex.get("p_male", 0.0)), 3),
                    "p_female": round(float(ex.get("p_female", 0.0)), 3),
                    "in_minimal": bool(ex.get("in_minimal", False)),
                }
            )
        st.table(rows)
    else:
        st.info("No example biased nodes reported.")


def _section_downloads(sel: Dict[str, Optional[str]], cfg: Dict[str, Any]) -> None:
    st.header("Downloads")
    # Per-run key artifacts
    def _offer_file(label: str, path: str):
        try:
            if path and os.path.exists(path) and os.path.isfile(path):
                with open(path, "rb") as f:
                    st.download_button(label=label, data=f.read(), file_name=os.path.basename(path))
        except Exception:
            pass

    keys = ["baseline", "ensemble", "spike_hypergraph", "causal"]
    for key in keys:
        run = sel.get(key)
        if not run:
            continue
        with st.expander(f"{key} artifacts"):
            for name in ("config.yaml", "hypergraph.hif.json", "hypergraph_stii.hif.json", "stii_values.json", "acdc_minimal_circuit.json", "fairness_report.json"):
                p = os.path.join(run, name)
                if os.path.exists(p):
                    _offer_file(f"Download {key}/{name}", p)

    # Bundle reproducibility zip
    st.subheader("Reproducibility bundle")
    run_map = {"baseline": sel.get("baseline"), "ensemble": sel.get("ensemble"), "spike_hypergraph": sel.get("spike_hypergraph"), "causal": sel.get("causal")}
    artifacts = collect_artifacts(run_map)

    if not artifacts:
        st.info("No artifacts to bundle from current selection.")
        return

    with tempfile.TemporaryDirectory() as td:
        manifest_path = os.path.join(td, "manifest.json")
        write_manifest(artifacts, out_path=manifest_path, extra={"selection": run_map, "note": "Dashboard-generated bundle"})
        # Include manifest in zip
        artifacts_with_manifest = artifacts + [(manifest_path, "manifest.json")]
        zip_path = os.path.join(td, "bundle.zip")
        make_zip(artifacts_with_manifest, zip_path=zip_path)
        with open(zip_path, "rb") as fz:
            st.download_button(label="Download bundle.zip", data=fz.read(), file_name="investor_bundle.zip")


def main() -> None:
    st.set_page_config(page_title="Investor Dashboard", layout="wide", page_icon="📈")

    # Load config (initial)
    cfg = load_yaml(_cfg_default_path()) or {}

    _header()
    cfg, sel, _ctx = _sidebar_controls(cfg)

    # Sections (robust to missing artifacts)
    _section_polysemanticity(sel, cfg)
    _section_topology(sel, cfg)
    _section_causal(sel, cfg)
    _section_fairness(sel, cfg)
    _section_downloads(sel, cfg)


if __name__ == "__main__":
    # Allow running as `streamlit run python/dashboard/app.py`
    main()