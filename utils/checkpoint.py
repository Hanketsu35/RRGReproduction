"""Checkpoint manager for MOE-RRG training."""

import torch
import os
import json
from pathlib import Path
from typing import Optional


class CheckpointManager:
    """Manages model checkpoints with top-K saving.

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_top_k: Number of best checkpoints to keep
        metric_name: Metric to track for "best" checkpoint
        mode: "min" or "max" — whether lower or higher metric is better
    """

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 save_top_k: int = 3, metric_name: str = "val_loss",
                 mode: str = "min"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.save_top_k = save_top_k
        self.metric_name = metric_name
        self.mode = mode

        # Track saved checkpoints
        self.checkpoints = []  # List of (metric_value, path)

    def save(self, state: dict, metric_value: float,
             epoch: int, is_best: bool = False) -> str:
        """Save a checkpoint.

        Args:
            state: Dictionary with model, optimizer, scheduler, etc.
            metric_value: Current metric value
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        # Save checkpoint
        filename = f"checkpoint_epoch{epoch:03d}_{self.metric_name}{metric_value:.4f}.pt"
        path = self.checkpoint_dir / filename
        torch.save(state, path)

        # Track checkpoint
        self.checkpoints.append((metric_value, str(path)))

        # Sort by metric
        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove excess checkpoints
        while len(self.checkpoints) > self.save_top_k:
            _, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(state, best_path)

        return str(path)

    def load(self, path: str = None, load_best: bool = False) -> Optional[dict]:
        """Load a checkpoint.

        Args:
            path: Specific checkpoint path
            load_best: Whether to load the best checkpoint

        Returns:
            Checkpoint state dict, or None if not found
        """
        if load_best:
            path = self.checkpoint_dir / "best_checkpoint.pt"
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_latest(self) -> Optional[dict]:
        """Load the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return torch.load(latest, map_location="cpu", weights_only=False)
