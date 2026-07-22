"""CLI for ResNetLDDMM experiments (STEP T17).

Entry point for YAML config-based training runs.
Loads config, validates it, and orchestrates registration.
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.resnet_lddmm.config import ExperimentCfg
from src.resnet_lddmm.runner import run


def load_config(config_path: str) -> ExperimentCfg:
    """Load and parse YAML config file.

    Args:
        config_path: path to .yaml config file

    Returns:
        ExperimentCfg instance

    Raises:
        FileNotFoundError: if config file does not exist
        ValueError: if config is invalid (missing required keys, unknown keys)
        yaml.YAMLError: if YAML parsing fails
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dict, got {type(data).__name__}")

    return ExperimentCfg.from_dict(data)


def main():
    """Parse CLI args and run experiment."""
    parser = argparse.ArgumentParser(
        description="ResNetLDDMM: neural ODE registration with learned velocity fields"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, yaml.YAMLError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ctx = run(cfg)
        print(f"Training complete. Output saved to: {cfg.output_dir}")
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
