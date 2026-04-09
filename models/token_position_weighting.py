"""Learnable Token Position Weighting Module for MOE-RRG.

Learns per-position weights during training to prioritize semantically 
important tokens (findings, impression) over structural tokens (BOS, padding).

This module implements the position weighting strategy from Phase 3.1.

Architecture:
- nn.Embedding(max_seq_len, 1): Learns a scalar weight per position
- Softmax normalization to maintain loss scale
- Batch-aware masking to zero out padding positions
- Integration into cross-entropy loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPositionWeightingModule(nn.Module):
    """Learns position-aware weights for token loss contribution.
    
    During training, this module learns to upweight semantically important 
    positions (findings, impression) and downweight structural positions 
    (BOS, padding). Weights are shared across the batch and learned jointly 
    with model parameters.
    
    Args:
        max_seq_len: Maximum sequence length (default: 512)
        hidden_size: Model hidden size (for compatibility, not used)
        initialization: "uniform" or "by_position"
            - "uniform": Start with uniform weights (1.0 everywhere)
            - "by_position": Start with heuristic weights favoring middle positions
    
    Example:
        weighter = TokenPositionWeightingModule(max_seq_len=512)
        weights = weighter(seq_len=100, device=device)  # [100]
        # weights are position-aware, sum to 1.0
        
        # In loss computation:
        ce_loss = F.cross_entropy(..., reduction='none')  # [B*T]
        weighted_loss = ce_loss * weights.repeat(B)
        final_loss = weighted_loss.mean()
    """
    
    def __init__(self, max_seq_len: int = 512, hidden_size: int = 512,
                 initialization: str = "uniform"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.initialization = initialization
        
        # Learnable position weights: [max_seq_len, 1]
        # Each position gets a scalar weight
        self.position_weights = nn.Embedding(max_seq_len, 1)
        
        # Initialize weights
        if initialization == "uniform":
            # Start near zero (softmax will normalize to uniform)
            nn.init.normal_(self.position_weights.weight, mean=0.0, std=0.1)
        elif initialization == "by_position":
            # Heuristic init: favor middle positions (findings/impression section)
            logits = torch.zeros(max_seq_len, 1)
            positions = torch.arange(max_seq_len).float()
            
            # Shape: low weight at start/end, high weight in middle
            # Gaussian-like centered at 25%-75% of sequence
            center = 0.4 * max_seq_len
            width = 0.3 * max_seq_len
            logits[:, 0] = torch.exp(-((positions - center) ** 2) / (2 * width ** 2))
            
            self.position_weights.weight.data = logits
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get position weights for a sequence of length seq_len.
        
        Args:
            seq_len: Actual sequence length (without padding)
            device: Torch device (CPU or GPU)
        
        Returns:
            Weights tensor of shape [seq_len], sums to 1.0
            
        Note: Padding positions should be masked separately in loss computation.
              This function assumes seq_len is the actual content length, not with padding.
        """
        # Get position embeddings for all positions up to seq_len
        positions = torch.arange(seq_len, device=device)
        weights = self.position_weights(positions)  # [seq_len, 1]
        weights = weights.squeeze(-1)  # [seq_len]
        
        # Apply softmax to normalize
        # This maintains loss scale: sum of weights = 1.0
        weights = F.softmax(weights, dim=0)  # [seq_len]
        
        return weights
    
    def get_batch_weights(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Get weights for a batch with variable sequence lengths.
        
        Args:
            seq_lens: [B] actual sequence lengths for each sample
        
        Returns:
            Weights for flattened batch of shape [sum(seq_lens)]
            Can be directly multiplied with flattened CE loss
            
        Example:
            logits shape: [B, T, V]
            targets shape: [B, T]
            seq_lens shape: [B]
            
            # Flatten
            logits_flat = logits.reshape(-1, V)  # [B*T, V]
            targets_flat = targets.reshape(-1)   # [B*T]
            
            # Compute CE loss
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*T]
            
            # Get weights and apply
            batch_weights = self.get_batch_weights(seq_lens)  # [sum(seq_lens)]
            # Pad if needed to match flattened size
            
            weighted_loss = ce_loss[:len(batch_weights)] * batch_weights
            loss = weighted_loss.mean()
        """
        device = seq_lens.device
        max_len = seq_lens.max().item()
        
        # Get weights for max length in batch
        weights_by_pos = self.forward(max_len, device)  # [max_len]
        
        # For each sample, take weights up to its length
        batch_weights = []
        for length in seq_lens:
            batch_weights.append(weights_by_pos[:length])  # [length]
        
        # Concatenate to flat tensor
        batch_weights_flat = torch.cat(batch_weights, dim=0)  # [sum(seq_lens)]
        
        return batch_weights_flat


def apply_position_weighting(
    ce_loss: torch.Tensor,
    seq_lens: torch.Tensor,
    token_weighter: TokenPositionWeightingModule,
    flatten_batch: bool = False,
) -> torch.Tensor:
    """Apply position weighting to cross-entropy loss.
    
    Args:
        ce_loss: CE loss tensor of shape [B, T] or [B*T]
        seq_lens: Sequence lengths [B]
        token_weighter: TokenPositionWeightingModule instance
        flatten_batch: If True, ce_loss is [B*T]; if False, [B, T]
    
    Returns:
        Weighted CE loss, same shape
    
    Example:
        # Standard loss computation
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        ).reshape(B, T)  # [B, T]
        
        # Apply position weighting
        weighted_loss = apply_position_weighting(ce_loss, seq_lens, weighter)
        
        # Average
        final_loss = weighted_loss.mean()
    """
    if not flatten_batch:
        # Reshape: [B, T] -> [B*T]
        B, T = ce_loss.shape
        ce_loss_flat = ce_loss.reshape(-1)
    else:
        B = len(seq_lens)
        T = ce_loss.shape[0] // B
        ce_loss_flat = ce_loss
    
    # Get position weights for each sequence in batch
    device = ce_loss.device
    max_len = seq_lens.max().item()
    weights_by_pos = token_weighter.forward(max_len, device)  # [max_len]
    
    # Replicate weights for each batch sample
    batch_weights = []
    for length in seq_lens:
        # Take weights up to actual sequence length
        sample_weights = weights_by_pos[:length]  # [length]
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
    
    # Trim if needed (shouldn't happen with correct seq_lens)
    all_weights = all_weights[:ce_loss_flat.shape[0]]
    
    # Apply weights and return
    weighted_loss = ce_loss_flat * all_weights
    
    return weighted_loss


# ─────────────────────────────────────────────────────────────────
# Integration with MOE-RRG Loss Module
# ─────────────────────────────────────────────────────────────────

def integrate_token_weighting_into_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
    token_weighter: TokenPositionWeightingModule,
    vocab_size: int,
) -> torch.Tensor:
    """Compute CE loss with position weighting.
    
    Designed for integration into models/losses.py.
    
    Args:
        logits: [B, T, V] decoder logits
        targets: [B, T] target token IDs
        seq_lens: [B] actual sequence lengths (for masking)
        token_weighter: TokenPositionWeightingModule instance
        vocab_size: Vocabulary size
    
    Returns:
        Scalar loss with position weighting applied
    
    Example (in MOERRGLoss.forward):
        # Existing code
        logits = model_output["logits"]        # [B, T, V]
        targets = batch["report_ids"][:, 1:]   # [B, T]
        
        # Apply position weighting
        ce_loss = integrate_token_weighting_into_loss(
            logits, targets, seq_lens, self.token_weighter, vocab_size
        )
        
        # Rest of loss computation
        loss = ce_loss + ...
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
    
    # Apply position weighting
    weighted_loss_flat = apply_position_weighting(
        ce_loss_flat,
        seq_lens,
        token_weighter,
        flatten_batch=True,
    )  # [B*T]
    
    # Average over batch and time
    final_loss = weighted_loss_flat.mean()
    
    return final_loss
