"""Auxiliary text cue processor with sigmoid-gated bias injection.

Encodes indication and prior report with the text encoder, computes a
sigmoid-gated bias from the [CLS] embedding, and injects it into visual
patch tokens via elementwise addition + LayerNorm.

[APPROXIMATE] The exact MLP architecture for the gating mechanism is not
specified. We use: Linear(768->256) -> ReLU -> Linear(256->384) -> Sigmoid.
"""

import torch
import torch.nn as nn


class AuxiliaryGate(nn.Module):
    """Processes auxiliary textual cues and injects them into visual features.

    Args:
        text_hidden_size: Text encoder hidden dim (default: 768)
        visual_hidden_size: Visual encoder hidden dim (default: 384)
        gate_hidden_dim: Hidden dim for the gating MLP (default: 256)
        dropout: Dropout probability
    """

    def __init__(self, text_hidden_size=768, visual_hidden_size=384,
                 gate_hidden_dim=256, dropout=0.1):
        super().__init__()

        # Concatenated auxiliary [CLS] embedding comes from text encoder
        # Indication [CLS] and prior [CLS] are concatenated: dim = 2 * text_hidden
        self.gate_mlp = nn.Sequential(
            nn.Linear(text_hidden_size * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, visual_hidden_size),
        )

        # Sigmoid gate
        self.sigmoid = nn.Sigmoid()

        # LayerNorm for residual-style injection
        self.layer_norm = nn.LayerNorm(visual_hidden_size)

    def forward(self, visual_tokens: torch.Tensor,
                ind_cls: torch.Tensor, pri_cls: torch.Tensor) -> torch.Tensor:
        """Inject auxiliary textual cues into visual patch tokens.

        Args:
            visual_tokens: [B, T, D_vis] visual patch tokens
            ind_cls: [B, D_txt] indication [CLS] embedding
            pri_cls: [B, D_txt] prior report [CLS] embedding

        Returns:
            multimodal_tokens: [B, T, D_vis] with auxiliary bias injected
        """
        # Concatenate indication and prior [CLS] embeddings
        aux_combined = torch.cat([ind_cls, pri_cls], dim=-1)  # [B, 2*D_txt]

        # Compute gated bias
        bias = self.gate_mlp(aux_combined)            # [B, D_vis]
        gate = self.sigmoid(bias)                      # [B, D_vis]

        # Elementwise addition with gating + LayerNorm
        gated_bias = gate * bias                       # [B, D_vis]
        multimodal_tokens = visual_tokens + gated_bias.unsqueeze(1)  # [B, T, D_vis]
        multimodal_tokens = self.layer_norm(multimodal_tokens)

        return multimodal_tokens
