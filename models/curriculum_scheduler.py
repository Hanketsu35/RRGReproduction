"""Curriculum Learning Scheduler for Progressive Report Focus.

Implements epoch-based curriculum for radiology report generation:
- Early epochs: Broad focus on all tokens equally
- Mid epochs: Progressive sharpening on findings/impression sections
- Late epochs: Sharp focus on clinically important tokens

This module provides temperature-annealed position weighting that 
progresses from uniform (epoch 0) to focused (final epoch).

Integration with Phase 3.1 (learnable weights):
    combined_weight_t = learnable_weight_t * curriculum_weight_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CurriculumLearningScheduler(nn.Module):
    """Epoch-based curriculum scheduler for position-aware loss weighting.
    
    Uses temperature annealing to control focus sharpness:
    - High temperature (early): Broad, nearly uniform focus
    - Low temperature (late): Sharp focus on report center (findings/impression)
    
    Args:
        max_seq_len: Maximum sequence length (for position computation)
        max_epochs: Total number of training epochs
        temperature_base: Starting temperature (default: 2.0)
        temperature_min: Final temperature (default: 0.3)
        schedule: "linear", "cosine", or "exponential" annealing
        center: Center position for Gaussian (0.0-1.0, default: 0.5)
        width: Gaussian width as fraction of seq_len (default: 0.25)
    
    Example:
        scheduler = CurriculumLearningScheduler(max_epochs=50)
        
        for epoch in range(50):
            curriculum_weights = scheduler(seq_len=100, epoch=epoch)
            # Use curriculum_weights * learnable_weights in loss computation
    """
    
    def __init__(self, max_seq_len: int = 512, max_epochs: int = 50,
                 temperature_base: float = 2.0, temperature_min: float = 0.3,
                 schedule: str = "linear", center: float = 0.5,
                 width: float = 0.25):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_epochs = max_epochs
        self.temperature_base = temperature_base
        self.temperature_min = temperature_min
        self.schedule = schedule
        self.center = center
        self.width = width
        
        # Validate schedule
        if schedule not in ["linear", "cosine", "exponential"]:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def _get_temperature(self, epoch: int) -> float:
        """Get temperature for this epoch based on schedule.
        
        Args:
            epoch: Current training epoch (0-indexed)
        
        Returns:
            Temperature value for this epoch
        """
        progress = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        
        if self.schedule == "linear":
            # Linear: T_base -> T_min
            temp = self.temperature_base - (self.temperature_base - self.temperature_min) * progress
        
        elif self.schedule == "cosine":
            # Cosine annealing: smooth decay
            temp = self.temperature_min + 0.5 * (self.temperature_base - self.temperature_min) * (1 + torch.cos(torch.tensor(3.14159 * progress)))
            temp = temp.item() if isinstance(temp, torch.Tensor) else temp
        
        elif self.schedule == "exponential":
            # Exponential: aggressive final sharpening
            ratio = self.temperature_min / self.temperature_base
            temp = self.temperature_base * (ratio ** progress)
        
        return temp
    
    def forward(self, seq_len: int, epoch: int, 
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Get position weights for this epoch.
        
        Args:
            seq_len: Actual sequence length (without padding)
            epoch: Current training epoch (0-indexed)
            device: Torch device (CPU or GPU)
        
        Returns:
            Position weights of shape [seq_len], summing to 1.0
            
        Notes:
            - Weights favor the middle of the report (findings/impression)
            - Shape of Gaussian is fixed, controlled by center and width
            - Temperature controls how sharp the focus becomes
        """
        if device is None:
            device = torch.device('cpu')
        
        # Get temperature for this epoch
        temperature = self._get_temperature(epoch)
        
        # Create position logits: Gaussian centered at report middle
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        center_pos = self.center * seq_len
        width_pos = self.width * seq_len
        
        # Gaussian: peaks at center, tails off at edges
        # Var: Gaussian(mean=center, std=width)
        position_logits = -((positions - center_pos) ** 2) / (2 * width_pos ** 2)
        
        # Apply temperature scaling to softmax
        # Higher temperature = softer (more uniform)
        # Lower temperature = sharper (more focused)
        weights = F.softmax(position_logits / temperature, dim=0)
        
        return weights  # [seq_len] summing to 1.0
    
    def get_temperature_schedule(self) -> list:
        """Get temperature values for all epochs (for visualization).
        
        Returns:
            List of temperatures for epochs 0..max_epochs
        """
        temps = []
        for epoch in range(self.max_epochs):
            temp = self._get_temperature(epoch)
            temps.append(temp)
        return temps


def combine_learnable_and_curriculum_weights(
    learnable_weights: torch.Tensor,
    curriculum_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine learnable (Phase 3.1) and curriculum (Phase 3.2) weights.
    
    Args:
        learnable_weights: [T] from TokenPositionWeightingModule
        curriculum_weights: [T] from CurriculumLearningScheduler
    
    Returns:
        Combined weights [T] summing to 1.0
    
    Notes:
        Uses element-wise product: weights_combined = w_learn * w_curric
        Then renormalizes to sum to 1.0
        
        Intuition: 
        - Learnable weights encode DATA importance (what's important in THIS reports)
        - Curriculum weights encode TRAINING importance (what to focus on THIS epoch)
        - Product combines both signals
    """
    # Element-wise product
    combined = learnable_weights * curriculum_weights
    
    # Renormalize to sum to 1.0
    combined = combined / (combined.sum() + 1e-8)
    
    return combined


def apply_curriculum_weighting(
    ce_loss: torch.Tensor,
    seq_lens: torch.Tensor,
    curriculum: CurriculumLearningScheduler,
    epoch: int,
    learnable_weighter: Optional[nn.Module] = None,
    flatten_batch: bool = False,
) -> torch.Tensor:
    """Apply curriculum weighting to cross-entropy loss.
    
    Args:
        ce_loss: CE loss tensor [B, T] or [B*T]
        seq_lens: Sequence lengths [B]
        curriculum: CurriculumLearningScheduler instance
        epoch: Current training epoch
        learnable_weighter: Optional TokenPositionWeightingModule (for combining with 3.1)
        flatten_batch: If True, ce_loss is [B*T]; if False, [B, T]
    
    Returns:
        Weighted CE loss, same shape as input
    
    Example:
        # Standard loss computation
        ce_loss = F.cross_entropy(..., reduction='none')  # [B*T]
        
        # Apply curriculum weighting
        weighted_loss = apply_curriculum_weighting(
            ce_loss, seq_lens, curriculum=scheduler, epoch=epoch
        )
        
        # With learnable weights combined:
        weighted_loss = apply_curriculum_weighting(
            ce_loss, seq_lens, curriculum=scheduler, epoch=epoch,
            learnable_weighter=token_weighter
        )
        
        # Average
        final_loss = weighted_loss.mean()
    """
    if not flatten_batch:
        # Reshape: [B, T] -> [B*T]
        B, T = ce_loss.shape
        ce_loss_flat = ce_loss.reshape(-1)
        device = ce_loss.device
    else:
        B = len(seq_lens)
        T = ce_loss.shape[0] // B
        ce_loss_flat = ce_loss
        device = ce_loss.device
    
    # Get curriculum weights for this epoch
    max_len = seq_lens.max().item()
    curriculum_weights_by_pos = curriculum(seq_len=max_len, epoch=epoch, device=device)
    
    # If learnable weighter provided, combine
    if learnable_weighter is not None:
        learnable_weights_by_pos = learnable_weighter.forward(max_len, device)
        combined_by_pos = combine_learnable_and_curriculum_weights(
            learnable_weights_by_pos,
            curriculum_weights_by_pos,
        )
        weights_by_pos = combined_by_pos
    else:
        weights_by_pos = curriculum_weights_by_pos
    
    # Replicate weights for each batch sample
    batch_weights = []
    for length in seq_lens:
        sample_weights = weights_by_pos[:length]
        batch_weights.append(sample_weights)
    
    # Concatenate all batch weights
    all_weights = torch.cat(batch_weights, dim=0)  # [sum(lengths)]
    
    # Pad if needed to match flattened loss size
    if all_weights.shape[0] < ce_loss_flat.shape[0]:
        padding_size = ce_loss_flat.shape[0] - all_weights.shape[0]
        all_weights = torch.cat([
            all_weights,
            torch.zeros(padding_size, device=device)
        ], dim=0)
    
    # Trim if needed
    all_weights = all_weights[:ce_loss_flat.shape[0]]
    
    # Apply weights
    weighted_loss = ce_loss_flat * all_weights
    
    return weighted_loss


# ─────────────────────────────────────────────────────────────────
# Integration with MOE-RRG Training
# ─────────────────────────────────────────────────────────────────

def integrate_curriculum_into_training_loop(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
    curriculum: CurriculumLearningScheduler,
    epoch: int,
    learnable_weighter: Optional[nn.Module] = None,
    vocab_size: int = 30522,
) -> torch.Tensor:
    """Compute CE loss with curriculum weighting (and optional learnable weights).
    
    Designed for direct integration into training loop.
    
    Args:
        logits: [B, T, V] decoder logits
        targets: [B, T] target token IDs
        seq_lens: [B] actual sequence lengths
        curriculum: CurriculumLearningScheduler instance
        epoch: Current training epoch
        learnable_weighter: Optional TokenPositionWeightingModule
        vocab_size: Vocabulary size for CE computation
    
    Returns:
        Scalar loss with curriculum (and optionally learnable) weighting
    
    Example (in training loop):
        for epoch, batch in enumerate(train_loader):
            logits = model(batch)
            loss = integrate_curriculum_into_training_loop(
                logits, batch["targets"], batch["seq_lens"],
                curriculum=scheduler, epoch=epoch,
                learnable_weighter=token_weighter,
            )
            loss.backward()
    """
    B, T, V = logits.shape
    
    # Flatten for CE computation
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Compute CE loss without reduction
    ce_loss_flat = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction='none',
    )  # [B*T]
    
    # Apply curriculum (and optionally learnable) weighting
    weighted_loss_flat = apply_curriculum_weighting(
        ce_loss_flat,
        seq_lens,
        curriculum=curriculum,
        epoch=epoch,
        learnable_weighter=learnable_weighter,
        flatten_batch=True,
    )  # [B*T]
    
    # Average over batch and time
    final_loss = weighted_loss_flat.mean()
    
    return final_loss
