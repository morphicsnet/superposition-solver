from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import yaml


def create_run_dir(base_dir: str, run_tag: Optional[str] = None) -> str:
    """
    Create an output run directory under base_dir.
    If run_tag is None, use a timestamp tag.
    Returns the created directory path.
    """
    tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def dump_json(obj: Dict[str, Any], path: str) -> None:
    """
    Serialize obj as pretty JSON to path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def dump_yaml(obj: Dict[str, Any], path: str) -> None:
    """
    Serialize obj as YAML to path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)