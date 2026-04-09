"""Cross-platform runtime helpers for training/evaluation scripts."""

from __future__ import annotations

import os
import platform
from typing import Tuple

import torch


def select_torch_device(gpu_index: int = 0) -> torch.device:
    """Pick the best available torch device in a deterministic order."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")

    # `torch.backends.mps` is available only on Apple platforms/builds.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def dataloader_runtime_settings(
    requested_num_workers: int,
    requested_pin_memory: bool,
    device: torch.device,
) -> Tuple[int, bool]:
    """Return DataLoader settings that are safe across macOS/Windows/Linux.

    Rules:
    - CUDA: allow multi-worker loading.
    - macOS (especially MPS): force workers to 0 for better stability.
    - Windows/Linux CPU: allow workers but cap to CPU count and non-negative.
    - pin_memory: enable only for CUDA.
    """
    pin_memory = bool(requested_pin_memory and device.type == "cuda")

    if device.type == "cuda":
        safe_workers = max(0, int(requested_num_workers))
        return safe_workers, pin_memory

    system_name = platform.system().lower()
    if system_name == "darwin":
        return 0, pin_memory

    cpu_count = os.cpu_count() or 1
    safe_workers = max(0, min(int(requested_num_workers), cpu_count))
    return safe_workers, pin_memory