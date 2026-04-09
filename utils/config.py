"""Configuration loader with inheritance support."""

import yaml
import os
from pathlib import Path
from copy import deepcopy


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> dict:
    """Load a YAML config with inheritance support (inherits field)."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Resolve inheritance
    if "inherits" in config:
        parent_path = config_path.parent / config.pop("inherits")
        parent_config = load_config(str(parent_path))
        config = deep_merge(parent_config, config)

    return config


def get_param(config: dict, key_path: str, default=None):
    """Get nested config value using dot-separated path.
    Example: get_param(cfg, "model.sv_moe.num_experts", 4)
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
