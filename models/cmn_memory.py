"""Copy Memory Network (CMN) approximation.

[APPROXIMATE] The paper references a CMN mechanism but does not provide
sufficient architectural details for exact reproduction. We implement a
minimal compatible approximation: a learned key-value memory that the
decoder can attend to via an additional cross-attention layer.
"""

import torch
import torch.nn as nn


class CopyMemoryNetwork(nn.Module):
    """Minimal CMN approximation with learned memory buffer.

    The memory is a fixed-size key-value store that the decoder can
    attend to, enabling it to "copy" information from a learned vocabulary
    of common patterns.

    Args:
        hidden_size: Decoder hidden dimension
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, hidden_size: int = 512, memory_size: int = 128,
                 memory_dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # Learned memory keys and values
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.02)

        # Projection to match decoder hidden dim
        self.key_proj = nn.Linear(memory_dim, hidden_size)
        self.value_proj = nn.Linear(memory_dim, hidden_size)

        # Cross-attention from decoder to memory
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(hidden_size)

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """Attend decoder hidden states to the learned memory.

        Args:
            decoder_hidden: [B, T, D] decoder hidden states

        Returns:
            [B, T, D] memory-augmented decoder hidden states
        """
        B = decoder_hidden.size(0)

        # Project memory to decoder space
        mem_K = self.key_proj(self.memory_keys)    # [M, D]
        mem_V = self.value_proj(self.memory_values)  # [M, D]

        # Expand for batch
        mem_K = mem_K.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        mem_V = mem_V.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # Cross-attention
        residual = decoder_hidden
        hidden_norm = self.attn_ln(decoder_hidden)
        memory_out, _ = self.memory_attn(
            query=hidden_norm,
            key=mem_K,
            value=mem_V,
        )

        return residual + memory_out

    def memory_kv(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned memory as projected key/value tensors.

        This matches the paper's CMN-style cache more closely than a post-hoc
        residual attention block: the decoder can concatenate these memory
        slots directly into self-attention KV.

        Returns:
            (mem_K, mem_V): each [B, M, D]
        """
        mem_K = self.key_proj(self.memory_keys).unsqueeze(0).expand(batch_size, -1, -1)
        mem_V = self.value_proj(self.memory_values).unsqueeze(0).expand(batch_size, -1, -1)
        return mem_K.to(device), mem_V.to(device)
