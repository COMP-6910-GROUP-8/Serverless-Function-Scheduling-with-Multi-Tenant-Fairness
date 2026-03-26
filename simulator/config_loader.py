"""
/* USAGE:
  from simulator.config_loader import load_config

  # Load defaults only
  config = load_config(config_path=None)

  # Load experiment config merged on top of defaults
  config = load_config("configs/burst_test.yaml")

  # Override seed
  config = load_config("configs/burst_test.yaml", seed_override=123)
*/
"""

import os
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins for non-dict values."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = None, defaults_path: str = None,
                seed_override: int = None) -> dict:
    """
    Load config by merging experiment-specific YAML on top of shared defaults.
    Uses yaml.safe_load exclusively for security.
    """
    # Resolve defaults path relative to this file's directory
    if defaults_path is None:
        defaults_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(defaults_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge experiment config on top if provided
    if config_path is not None:
        with open(config_path, "r") as f:
            experiment_config = yaml.safe_load(f)
        if experiment_config:
            config = deep_merge(config, experiment_config)

    # Override seed if requested
    if seed_override is not None:
        config.setdefault("simulation", {})["random_seed"] = seed_override

    return config
