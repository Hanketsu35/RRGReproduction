"""Prior-Report Copy Mechanism - Attention-Based Copying

Implements a copy mechanism that attends to prior report tokens and gates
between copying from prior vs generating from vocabulary.

This replaces the original learned memory approximation with a data-driven
approach that directly leverages the paper's input (prior reports).

Architecture:
1. Encode prior report tokens
2. Attention from decoder hidden states to prior report
3. Gate: P(copy) computed from decoder + prior context
4. Blend: copy_prob * P_copy(tokens) + (1-copy_prob) * P_gen(tokens)

Reference: Pointer-Generator Networks (See et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorReportCopyMemory(nn.Module):
    """Attention-based copying from prior report text.
    
    For each decoding step, computes:
    1. Attention over prior report tokens
    2. Gating probability (copy weight vs generate weight)
    3. Logits that blend copy and generation distributions
    
    Args:
        hidden_size: Decoder hidden dimension
        vocab_size: Vocabulary size for generation
        num_heads: Number of attention heads
        dropout: Dropout probability
        copy_loss_weight: Weight for copy supervising loss (optional)
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        vocab_size: int = 30522,
        num_heads: int = 8,
        dropout: float = 0.1,
        copy_loss_weight: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.copy_loss_weight = copy_loss_weight
        
        # 1. Attention to prior report tokens
        self.prior_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 2. Copy gating MLP: decide copy vs generate
        self.copy_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # [0, 1]
        )
        
        # 3. Attention layer norm
        self.attn_ln = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,        # [B, T, D]
        prior_report_emb: torch.Tensor,     # [B, P, D] (embedded prior tokens)
        prior_report_tokens: torch.Tensor | None = None,  # [B, P] (token IDs, optional)
        return_gate: bool = False,
    ) -> dict:
        """Compute copy context and blending weights.
        
        Args:
            decoder_hidden: Decoder hidden states [B, T, D]
            prior_report_emb: Encoded prior report [B, P, D]
            prior_report_tokens: Prior report token IDs [B, P] (for analysis)
            return_gate: If True, return intermediate gating values
        
        Returns:
            {
                "context": [B, T, D] copy context (attention output),
                "copy_prob": [B, T, 1] gate probability,
                "attn_weights": [B, T, P] attention weights (for visualization),
                "gate_dist": gate probability statistics (if return_gate=True),
            }
        """
        B, T, D = decoder_hidden.size()
        P = prior_report_emb.size(1)
        
        # Step 1: Attention over prior report
        # Input: query=decoder hidden, key/value=prior embeddings
        norm_hidden = self.attn_ln(decoder_hidden)
        copy_context, attn_weights = self.prior_attn(
            query=norm_hidden,           # [B, T, D]
            key=prior_report_emb,        # [B, P, D]
            value=prior_report_emb,      # [B, P, D]
            return_attention_weights=True,
        )
        # copy_context: [B, T, D]
        # attn_weights: [B, T, P]
        
        # Step 2: Compute copy gate (copy vs generate probability)
        # Concatenate decoder hidden + copy context
        combined = torch.cat([norm_hidden, copy_context], dim=-1)  # [B, T, 2D]
        copy_prob = self.copy_gate(combined)  # [B, T, 1]
        
        # Step 3: Prepare output
        result = {
            "context": copy_context,
            "copy_prob": copy_prob,
            "attn_weights": attn_weights,
        }
        
        # Optional: gather statistics for analysis
        if return_gate:
            result["gate_dist"] = {
                "mean": copy_prob.mean().item(),
                "std": copy_prob.std().item(),
                "min": copy_prob.min().item(),
                "max": copy_prob.max().item(),
            }
        
        return result
    
    @staticmethod
    def blend_logits(
        gen_logits: torch.Tensor,      # [B, T, V] generation logits
        copy_prob: torch.Tensor,       # [B, T, 1] gate probability
        attn_weights: torch.Tensor,    # [B, T, P] attention weights
        prior_tokens: torch.Tensor,    # [B, P] token IDs
        vocab_size: int = 30522,
    ) -> torch.Tensor:
        """Blend copy and generation distributions (Pointer-Generator style).
        
        P(w_t) = copy_prob * P_copy(w_t) + (1-copy_prob) * P_gen(w_t)
        
        Where:
        - P_copy(w_t) = attention weights over prior tokens
                        (0 for tokens not in prior, sum of attn weights if token appears multiple times)
        - P_gen(w_t) = softmax(gen_logits)
        
        Args:
            gen_logits: Vocabulary logits [B, T, V]
            copy_prob: Copy gate probability [B, T, 1]
            attn_weights: Attention weights over prior [B, T, P]
            prior_tokens: Token IDs in prior [B, P]
            vocab_size: Size of vocabulary
        
        Returns:
            Blended logits [B, T, V] that can be used for loss/sampling
        """
        B, T, V = gen_logits.size()
        device = gen_logits.device
        
        # Convert to probabilities
        gen_probs = F.softmax(gen_logits, dim=-1)  # [B, T, V]
        
        # Build copy distribution: scatter attention weights to vocabulary indices
        copy_probs = torch.zeros_like(gen_probs)  # [B, T, V]
        
        # For each batch and timestep, scatter prior token attention weights
        for b in range(B):
            for t in range(T):
                for p in range(prior_tokens.size(1)):
                    token_id = prior_tokens[b, p].item()
                    if 0 <= token_id < vocab_size:
                        # Add attention weight to this token's probability
                        copy_probs[b, t, token_id] += attn_weights[b, t, p]
        
        # Normalize copy probabilities (in case token appears multiple times)
        copy_mass = copy_probs.sum(dim=-1, keepdim=True)
        copy_mass = torch.clamp(copy_mass, min=1e-8)
        copy_probs = copy_probs / copy_mass
        
        # Blend: copy_prob controls the mixture weight
        blended_probs = (
            copy_prob * copy_probs +                    # [B, T, 1] * [B, T, V]
            (1 - copy_prob) * gen_probs                 # [B, T, 1] * [B, T, V]
        )
        
        # Convert back to "logits" (log-probs for numerical stability in loss)
        blended_logits = torch.log(blended_probs + 1e-8)
        
        return blended_logits


class PriorPrefixAdapter(nn.Module):
    """Optional: Adapt prior reports into additional decoder prefix tokens.
    
    Simpler alternative to prior report copy mechanism.
    Just learns a projection from prior embedding to more prefix tokens.
    
    Used in Design C (Prefix Tuning variant).
    
    Args:
        hidden_size: Decoder hidden dimension
        num_prefix_tokens: How many prefix tokens to generate from prior
    """
    
    def __init__(self, hidden_size: int = 512, num_prefix_tokens: int = 16):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_prefix = num_prefix_tokens
        
        # Learn projection from prior embedding to prefix
        self.prior_to_prefix = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * num_prefix_tokens),
        )
    
    def forward(self, prior_emb: torch.Tensor) -> torch.Tensor:
        """Project prior embedding to prefix tokens.
        
        Args:
            prior_emb: Pooled prior report embedding [B, D]
        
        Returns:
            Prefix tokens [B, num_prefix, D]
        """
        B, D = prior_emb.size()
        prefix_flat = self.prior_to_prefix(prior_emb)  # [B, D*num_prefix]
        prefix = prefix_flat.reshape(B, self.num_prefix, self.hidden_size)  # [B, num_prefix, D]
        return prefix
