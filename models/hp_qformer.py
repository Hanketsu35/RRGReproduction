"""Hierarchical Prefix Q-Former (HP-QF) module.

Uses P learnable query tokens that interact with multimodal features through
cross-attention and self-attention. The Q-Former output is projected into
PrefixKV pairs that are injected into the decoder's self-attention and
cross-attention key/value states at every decoding step.

[APPROXIMATE] Architecture follows BLIP-2 Q-Former pattern since the paper
does not fully specify layer composition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormerLayer(nn.Module):
    """Single Q-Former layer with self-attention, cross-attention, and FFN.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        ffn_dim: FFN intermediate dimension
        dropout: Dropout probability
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()

        # Self-attention for query-query interactions
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_ln = nn.LayerNorm(hidden_size)

        # Cross-attention: queries attend to multimodal features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_ln = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_size)

    def forward(self, queries: torch.Tensor, encoder_features: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through one Q-Former layer.

        Args:
            queries: [B, P, D] learnable query tokens
            encoder_features: [B, T, D] multimodal features to cross-attend to
            encoder_mask: [B, T] attention mask for encoder features

        Returns:
            [B, P, D] updated query tokens
        """
        # Self-attention among queries
        residual = queries
        queries_norm = self.self_attn_ln(queries)
        queries_attn, _ = self.self_attn(
            queries_norm, queries_norm, queries_norm,
        )
        queries = residual + queries_attn

        # Cross-attention: queries attend to encoder features
        residual = queries
        queries_norm = self.cross_attn_ln(queries)

        # Build key_padding_mask for cross-attention
        key_padding_mask = None
        if encoder_mask is not None:
            key_padding_mask = (encoder_mask == 0)

        queries_cross, _ = self.cross_attn(
            query=queries_norm,
            key=encoder_features,
            value=encoder_features,
            key_padding_mask=key_padding_mask,
        )
        queries = residual + queries_cross

        # FFN
        residual = queries
        queries = residual + self.ffn(self.ffn_ln(queries))

        return queries


class HPQFormer(nn.Module):
    """Hierarchical Prefix Q-Former.

    Produces PrefixKV pairs from learnable queries that have interacted
    with multimodal features through alternating self-attention and
    cross-attention layers.

    Args:
        num_queries: Number of learnable query tokens (P = 32)
        num_layers: Number of Q-Former layers (4)
        hidden_size: Hidden dimension (768)
        num_heads: Number of attention heads (8)
        encoder_hidden_size: Dimension of encoder features (input)
        decoder_hidden_size: Dimension of decoder (output PrefixKV)
        ffn_dim: FFN intermediate dimension
        cross_attention_freq: Cross-attention every N layers
        dropout: Dropout probability
        prefix_injection_depth: "all" or int for specific layers
    """

    def __init__(self, num_queries: int = 32, num_layers: int = 4,
                 hidden_size: int = 768, num_heads: int = 8,
                 encoder_hidden_size: int = 512,
                 decoder_hidden_size: int = 512,
                 ffn_dim: int = 3072,
                 cross_attention_freq: int = 1,
                 dropout: float = 0.1,
                 prefix_injection_depth: str = "all"):
        super().__init__()

        self.num_queries = num_queries
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.prefix_injection_depth = prefix_injection_depth

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_size) * 0.02
        )

        # Input projection from encoder dim to Q-Former dim
        self.input_proj = nn.Linear(encoder_hidden_size, hidden_size)

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_ln = nn.LayerNorm(hidden_size)

        # Project Q-Former output to PrefixKV for decoder injection
        # Separate projections for K and V
        self.prefix_k_proj = nn.Linear(hidden_size, decoder_hidden_size)
        self.prefix_v_proj = nn.Linear(hidden_size, decoder_hidden_size)

    def forward(self, encoder_features: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> dict:
        """Run Q-Former and produce PrefixKV pairs.

        Args:
            encoder_features: [B, T, D_enc] multimodal features
            encoder_mask: [B, T] attention mask

        Returns:
            Dictionary with:
                - prefix_k: [B, P, D_dec] prefix keys for decoder
                - prefix_v: [B, P, D_dec] prefix values for decoder
                - query_outputs: [B, P, D_qf] raw Q-Former outputs (for analysis)
        """
        B = encoder_features.size(0)

        # Project encoder features to Q-Former dimension
        projected_features = self.input_proj(encoder_features)  # [B, T, D_qf]

        # Initialize queries
        queries = self.query_tokens.expand(B, -1, -1)  # [B, P, D_qf]

        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, projected_features, encoder_mask)

        queries = self.final_ln(queries)  # [B, P, D_qf]

        # Project to PrefixKV
        prefix_k = self.prefix_k_proj(queries)  # [B, P, D_dec]
        prefix_v = self.prefix_v_proj(queries)  # [B, P, D_dec]

        return {
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "query_outputs": queries,
        }
