from __future__ import annotations

from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}