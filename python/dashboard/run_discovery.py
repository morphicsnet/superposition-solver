from __future__ import annotations

import os
from typing import Dict, List, Optional


def list_runs(base_dir: str) -> list[str]:
    """
    List run directories under base_dir (non-recursive), sorted by mtime desc.
    Returns absolute paths.
    """
    if not base_dir or not os.path.isdir(base_dir):
        return []
    entries: List[str] = []
    try:
        for name in os.listdir(base_dir):
            if name.startswith("."):
                continue
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                entries.append(os.path.abspath(p))
    except Exception:
        return []
    # Sort by modification time (desc), stable by name as tiebreaker
    entries.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)
    return entries


def pick_latest(base_dir: str) -> str | None:
    """
    Pick the latest run directory under base_dir or None if none exists.
    """
    runs = list_runs(base_dir)
    return runs[0] if runs else None


def _sel_mode(cfg: dict) -> str:
    try:
        return str(cfg.get("selection", {}).get("mode", "latest")).strip().lower()
    except Exception:
        return "latest"


def resolve_selection(cfg: dict) -> dict[str, str | None]:
    """
    Resolve selection for each demo area based on cfg.
    Expected cfg schema (see [configs/demo5_dashboard.yaml](configs/demo5_dashboard.yaml)):
      runs:
        baseline_dir: ...
        ensemble_dir: ...
        spike_hypergraph_dir: ...
        causal_dir: ...
      selection:
        mode: "latest" | "manual"
        manual_paths:
          baseline: ...
          ensemble: ...
          spike_hypergraph: ...
          causal: ...
    Returns:
      {
        "baseline": <path or None>,
        "ensemble": <path or None>,
        "spike_hypergraph": <path or None>,
        "causal": <path or None>,
      }
    """
    runs_cfg = cfg.get("runs", {}) if isinstance(cfg, dict) else {}
    mode = _sel_mode(cfg)
    manual = (cfg.get("selection", {}) or {}).get("manual_paths", {}) if isinstance(cfg, dict) else {}

    def _latest_or_none(d: Optional[str]) -> Optional[str]:
        return pick_latest(str(d)) if d else None

    out: Dict[str, Optional[str]] = {
        "baseline": None,
        "ensemble": None,
        "spike_hypergraph": None,
        "causal": None,
    }

    if mode == "manual":
        out["baseline"] = manual.get("baseline") or None
        out["ensemble"] = manual.get("ensemble") or None
        out["spike_hypergraph"] = manual.get("spike_hypergraph") or None
        out["causal"] = manual.get("causal") or None
        # Normalize to absolute paths where possible
        for k, v in list(out.items()):
            if v and isinstance(v, str):
                out[k] = os.path.abspath(v)
        return out

    # default: latest
    out["baseline"] = _latest_or_none(runs_cfg.get("baseline_dir"))
    out["ensemble"] = _latest_or_none(runs_cfg.get("ensemble_dir"))
    out["spike_hypergraph"] = _latest_or_none(runs_cfg.get("spike_hypergraph_dir"))
    out["causal"] = _latest_or_none(runs_cfg.get("causal_dir"))
    return out