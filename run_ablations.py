"""Routing analysis and ablation utilities for MOE-RRG.

Provides tools for:
- Expert usage distribution visualization
- Stage/view routing tables
- Stage-stratified performance
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_ablation(config_path: str, base_checkpoint: str,
                 ablation_name: str, gpu: int = 0):
    """Run a single ablation experiment.

    Args:
        config_path: Path to ablation config YAML
        base_checkpoint: Path to base model checkpoint (for reference)
        ablation_name: Name of the ablation
        gpu: GPU device ID
    """
    from utils.config import load_config
    from utils.logger import setup_logger

    config = load_config(config_path)
    logger = setup_logger(f"ablation_{ablation_name}", config["logging"]["log_dir"])
    logger.info(f"Running ablation: {ablation_name}")
    logger.info(f"Config: {config_path}")

    # Import train and evaluate
    # For now, just log the ablation setup
    logger.info("Ablation configuration loaded. Run train.py and evaluate.py with this config.")

    return config


def main():
    parser = argparse.ArgumentParser(description="MOE-RRG Ablations")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["no_svmoe", "no_hpqf", "no_aux",
                                 "stage_only", "no_token_weighting",
                                 "no_curriculum", "cmn_disabled",
                                 "prefix_early", "prefix_late", "prefix_sparse"],
                        help="Which ablation to run")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    ablation_configs = {
        "no_svmoe": "CONFIG/config_ablation_no_svmoe.yaml",
        "no_hpqf": "CONFIG/config_ablation_no_hp_qf.yaml",
        "no_aux": "CONFIG/config_ablation_no_aux.yaml",
        "stage_only": "CONFIG/config_ablation_stage_only.yaml",
        "no_token_weighting": "CONFIG/config_ablation_no_token_weighting.yaml",
        "no_curriculum": "CONFIG/config_ablation_no_curriculum.yaml",
        "cmn_disabled": "CONFIG/config_ablation_cmn_disabled.yaml",
        "prefix_early": "CONFIG/config_ablation_prefix_early.yaml",
        "prefix_late": "CONFIG/config_ablation_prefix_late.yaml",
        "prefix_sparse": "CONFIG/config_ablation_prefix_sparse.yaml",
    }

    config_path = ablation_configs[args.ablation]
    run_ablation(config_path, None, args.ablation, args.gpu)


if __name__ == "__main__":
    main()
