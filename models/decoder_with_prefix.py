"""Decoder with PrefixKV injection into self-attention and cross-attention.

The PrefixKV from HP-QFormer is prepended to both self-attention K/V and
cross-attention K/V at every decoding step. The injection depth is configurable
for ablation studies.

[APPROXIMATE] Separate linear projections for K and V injection are used;
the paper does not specify whether shared or separate projections are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderLayerWithPrefix(nn.Module):
    """Single Transformer decoder layer with prefix injection.

    PrefixKV is prepended to:
      - Self-attention key/value (attending to prefix + previous tokens)
      - Cross-attention key/value (attending to prefix + encoder features)

    Args:
        hidden_size: Hidden dimension (512)
        num_heads: Number of attention heads (8)
        ffn_dim: FFN intermediate dimension
        dropout: Dropout probability
        inject_prefix: Whether this layer injects prefix (for depth ablation)
    """

    def __init__(self, hidden_size: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1,
                 inject_prefix: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.inject_prefix = inject_prefix

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_ln = nn.LayerNorm(hidden_size)

        # Cross-attention (to encoder features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_ln = nn.LayerNorm(hidden_size)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_size)

    def _inject_prefix(self, K: torch.Tensor, V: torch.Tensor,
                       prefix_k: torch.Tensor = None,
                       prefix_v: torch.Tensor = None) -> tuple:
        """Prepend prefix KV to existing K, V tensors.

        Args:
            K: [B, T, D] keys
            V: [B, T, D] values
            prefix_k: [B, P, D] prefix keys (or None)
            prefix_v: [B, P, D] prefix values (or None)

        Returns:
            (K_with_prefix, V_with_prefix) or (K, V) if no prefix
        """
        if not self.inject_prefix or prefix_k is None:
            return K, V

        K = torch.cat([prefix_k, K], dim=1)  # [B, P+T, D]
        V = torch.cat([prefix_v, V], dim=1)  # [B, P+T, D]
        return K, V

    def forward(self, x: torch.Tensor, encoder_features: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                cmn_k_self: torch.Tensor = None,
                cmn_v_self: torch.Tensor = None,
                prefix_k_self: torch.Tensor = None,
                prefix_v_self: torch.Tensor = None,
                prefix_k_cross: torch.Tensor = None,
                prefix_v_cross: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through decoder layer.

        Args:
            x: [B, T, D] decoder input (target tokens)
            encoder_features: [B, S, D] encoder output
            tgt_mask: [T, T] causal mask
            tgt_key_padding_mask: [B, T] target padding mask
            memory_key_padding_mask: [B, S] encoder padding mask
            cmn_k_self: [B, M, D] CMN memory keys for self-attention
            cmn_v_self: [B, M, D] CMN memory values for self-attention
            prefix_k_self: [B, P, D] prefix keys for self-attention
            prefix_v_self: [B, P, D] prefix values for self-attention
            prefix_k_cross: [B, P, D] prefix keys for cross-attention
            prefix_v_cross: [B, P, D] prefix values for cross-attention

        Returns:
            [B, T, D] decoder output
        """
        # Self-attention with causal mask and prefix injection
        residual = x
        x_norm = self.self_attn_ln(x)

        # Self-attention: K and V include prefix
        # Q is just the target tokens
        Q = x_norm
        K = x_norm
        V = x_norm

        if cmn_k_self is not None and cmn_v_self is not None:
            K = torch.cat([cmn_k_self, K], dim=1)
            V = torch.cat([cmn_v_self, V], dim=1)

        # Inject prefix into self-attention K/V
        K, V = self._inject_prefix(K, V, prefix_k_self, prefix_v_self)

        # Adjust tgt_mask if CMN memory and/or prefix are injected.
        M = cmn_k_self.size(1) if cmn_k_self is not None else 0
        P = prefix_k_self.size(1) if (self.inject_prefix and prefix_k_self is not None) else 0
        if (M + P) > 0 and tgt_mask is not None:
            T = x.size(1)

            # Q has T rows, K/V have M+P+T columns.
            # Target tokens can attend to all memory/prefix tokens and causal past tokens.
            extra_cols = torch.zeros(T, M + P, device=tgt_mask.device, dtype=tgt_mask.dtype)
            tgt_mask = torch.cat([extra_cols, tgt_mask], dim=1)  # [T, M+P+T]

            # Adjust key_padding_mask for memory/prefix tokens (no padding).
            if tgt_key_padding_mask is not None:
                parts = []
                if M > 0:
                    parts.append(torch.zeros(tgt_key_padding_mask.size(0), M, device=tgt_key_padding_mask.device, dtype=tgt_key_padding_mask.dtype))
                if P > 0:
                    parts.append(torch.zeros(tgt_key_padding_mask.size(0), P, device=tgt_key_padding_mask.device, dtype=tgt_key_padding_mask.dtype))
                parts.append(tgt_key_padding_mask)
                tgt_key_padding_mask = torch.cat(parts, dim=1)

        x_attn, _ = self.self_attn(
            Q, K, V,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = residual + x_attn

        # Cross-attention with prefix injection
        residual = x
        x_norm = self.cross_attn_ln(x)

        # Cross-attention K/V = encoder features + prefix
        enc_K = encoder_features
        enc_V = encoder_features
        enc_K, enc_V = self._inject_prefix(
            enc_K, enc_V, prefix_k_cross, prefix_v_cross
        )

        # Adjust memory_key_padding_mask for prefix
        if self.inject_prefix and prefix_k_cross is not None and memory_key_padding_mask is not None:
            P = prefix_k_cross.size(1)
            prefix_pad = torch.zeros(
                memory_key_padding_mask.size(0), P,
                device=memory_key_padding_mask.device,
                dtype=memory_key_padding_mask.dtype
            )
            memory_key_padding_mask = torch.cat(
                [prefix_pad, memory_key_padding_mask], dim=1
            )

        x_cross, _ = self.cross_attn(
            x_norm, enc_K, enc_V,
            key_padding_mask=memory_key_padding_mask,
        )
        x = residual + x_cross

        # FFN
        residual = x
        x = residual + self.ffn(self.ffn_ln(x))

        return x


class DecoderWithPrefix(nn.Module):
    """Transformer decoder with hierarchical prefix injection.

    Injects PrefixKV from HP-QFormer into self-attention and cross-attention
    key/value states. Injection depth is configurable for ablation studies.

    Args:
        num_layers: Number of decoder layers (3 or 4)
        hidden_size: Hidden dimension (512)
        num_heads: Number of attention heads (8)
        ffn_dim: FFN intermediate dimension
        vocab_size: Vocabulary size
        dropout: Dropout probability
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        prefix_injection_depth: Injection strategy
            - "all": Inject at all layers (0-3, default)
            - "early": Inject at early layers only (0-1)
            - "late": Inject at late layers only (2-3)
            - "sparse": Inject at alternating layers (0, 2)
            - int (str): Integer specifying first N layers (e.g., "2" → layers 0-1)
    """

    def __init__(self, vocab_size: int = 30522, num_layers: int = 3,
                 hidden_size: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1,
                 max_length: int = 256, pad_token_id: int = 0,
                 bos_token_id: int = 101, eos_token_id: int = 102,
                 prefix_injection_depth: str = "all",
                 inject_prefix_into_self: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.inject_prefix_into_self = inject_prefix_into_self

        # Determine which layers get prefix injection
        if prefix_injection_depth == "all":
            inject_at = [True] * num_layers
        elif prefix_injection_depth == "early":
            # Inject at layers 0-1 (first 2 layers)
            inject_at = [i < 2 for i in range(num_layers)]
        elif prefix_injection_depth == "late":
            # Inject at layers 2-3 (last 2 layers)
            inject_at = [i >= 2 for i in range(num_layers)]
        elif prefix_injection_depth == "sparse":
            # Inject at alternating layers 0, 2 (skip 1, 3)
            inject_at = [i % 2 == 0 for i in range(num_layers)]
        else:
            # Backward compatibility: interpret as integer depth
            try:
                depth = int(prefix_injection_depth)
                inject_at = [i < depth for i in range(num_layers)]
            except ValueError:
                raise ValueError(
                    f"Invalid prefix_injection_depth: {prefix_injection_depth}. "
                    f"Must be 'all', 'early', 'late', 'sparse', or an integer."
                )

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayerWithPrefix(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                inject_prefix=inject_at[i],
            )
            for i in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        # Encoder feature projection (if encoder dim != decoder dim)
        self.encoder_proj = None  # Set dynamically if needed

    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask.

        Returns:
            [T, T] boolean mask where True means "do not attend"
        """
        mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        return mask

    def forward(self, target_ids: torch.Tensor,
                encoder_features: torch.Tensor,
                encoder_mask: torch.Tensor = None,
                cmn_k_self: torch.Tensor = None,
                cmn_v_self: torch.Tensor = None,
                prefix_k: torch.Tensor = None,
                prefix_v: torch.Tensor = None,
                encoder_hidden_size: int = None) -> dict:
        """Forward pass: teacher-forced decoding with prefix injection.

        Args:
            target_ids: [B, T] target token IDs (shifted right)
            encoder_features: [B, S, D_enc] encoder output features
            encoder_mask: [B, S] encoder attention mask
            cmn_k_self: [B, M, D] CMN memory keys for self-attention
            cmn_v_self: [B, M, D] CMN memory values for self-attention
            prefix_k: [B, P, D_dec] prefix keys from HP-QFormer
            prefix_v: [B, P, D_dec] prefix values from HP-QFormer
            encoder_hidden_size: Encoder hidden dimension (for projection)

        Returns:
            Dictionary with:
                - logits: [B, T, V] output logits
                - hidden_states: [B, T, D] last layer hidden states
        """
        B, T = target_ids.shape

        # Encoder features are expected to already match decoder hidden size.
        if encoder_features.size(-1) != self.hidden_size:
            raise ValueError(
                f"encoder_features last dim ({encoder_features.size(-1)}) "
                f"must match decoder hidden_size ({self.hidden_size})"
            )

        # Token + positional embeddings
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0)
        x = self.token_embedding(target_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Target padding mask
        tgt_key_padding_mask = (target_ids == self.pad_token_id)

        # Causal mask
        causal_mask = self._generate_causal_mask(T, target_ids.device)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_features=encoder_features,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
                cmn_k_self=cmn_k_self,
                cmn_v_self=cmn_v_self,
                prefix_k_self=prefix_k if self.inject_prefix_into_self else None,
                prefix_v_self=prefix_v if self.inject_prefix_into_self else None,
                prefix_k_cross=prefix_k,
                prefix_v_cross=prefix_v,
            )

        x = self.final_ln(x)

        # Output projection
        logits = self.output_proj(x)  # [B, T, V]

        return {
            "logits": logits,
            "hidden_states": x,
        }

    @torch.no_grad()
    def generate(self, encoder_features: torch.Tensor,
                 encoder_mask: torch.Tensor = None,
                 cmn_k_self: torch.Tensor = None,
                 cmn_v_self: torch.Tensor = None,
                 prefix_k: torch.Tensor = None,
                 prefix_v: torch.Tensor = None,
                 beam_size: int = 5,
                 max_length: int = None,
                 length_penalty: float = 0.6) -> torch.Tensor:
        """Autoregressive generation with beam search.

        Args:
            encoder_features: [B, S, D_enc] encoder features
            encoder_mask: [B, S] encoder mask
            prefix_k: [B, P, D] prefix keys
            prefix_v: [B, P, D] prefix values
            beam_size: Beam search width
            max_length: Max generation length
            length_penalty: Beam search length penalty

        Returns:
            [B, L] generated token IDs
        """
        max_length = max_length or self.max_length
        B = encoder_features.size(0)
        device = encoder_features.device

        # Encoder features are expected to already match decoder hidden size.
        if encoder_features.size(-1) != self.hidden_size:
            raise ValueError(
                f"encoder_features last dim ({encoder_features.size(-1)}) "
                f"must match decoder hidden_size ({self.hidden_size})"
            )

        def _slice_optional(tensor: torch.Tensor, idx: int) -> torch.Tensor:
            if tensor is None:
                return None
            return tensor[idx:idx + 1]

        # Greedy fast-path (beam_size=1)
        if beam_size <= 1:
            generated = torch.full(
                (B, 1), self.bos_token_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_length - 1):
                outputs = self.forward(
                    target_ids=generated,
                    encoder_features=encoder_features,
                    encoder_mask=encoder_mask,
                    cmn_k_self=cmn_k_self,
                    cmn_v_self=cmn_v_self,
                    prefix_k=prefix_k,
                    prefix_v=prefix_v,
                )

                next_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_logits, dim=-1)

                # Keep already-finished sequences terminated.
                eos_fill = torch.full_like(next_token, self.eos_token_id)
                next_token = torch.where(finished, eos_fill, next_token)

                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                finished = finished | (next_token == self.eos_token_id)
                if finished.all():
                    break

            return generated

        # Beam search path
        generated = torch.full(
            (B, max_length), self.pad_token_id, dtype=torch.long, device=device
        )

        for b in range(B):
            enc_b = encoder_features[b:b + 1]
            enc_mask_b = _slice_optional(encoder_mask, b)
            cmn_k_b = _slice_optional(cmn_k_self, b)
            cmn_v_b = _slice_optional(cmn_v_self, b)
            prefix_k_b = _slice_optional(prefix_k, b)
            prefix_v_b = _slice_optional(prefix_v, b)

            beams = [([self.bos_token_id], 0.0, False)]  # (tokens, log_prob, finished)

            for _ in range(max_length - 1):
                all_candidates = []
                all_finished = True

                for tokens, score, is_finished in beams:
                    if is_finished:
                        all_candidates.append((tokens, score, True))
                        continue

                    all_finished = False
                    target_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                    out = self.forward(
                        target_ids=target_ids,
                        encoder_features=enc_b,
                        encoder_mask=enc_mask_b,
                        cmn_k_self=cmn_k_b,
                        cmn_v_self=cmn_v_b,
                        prefix_k=prefix_k_b,
                        prefix_v=prefix_v_b,
                    )

                    log_probs = F.log_softmax(out["logits"][:, -1, :], dim=-1).squeeze(0)
                    topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size)

                    for lp, tok in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                        next_tokens = tokens + [int(tok)]
                        next_score = score + float(lp)
                        next_finished = (int(tok) == self.eos_token_id)
                        all_candidates.append((next_tokens, next_score, next_finished))

                if all_finished:
                    break

                def rank_key(item):
                    seq, seq_score, _ = item
                    norm = ((5.0 + len(seq)) / 6.0) ** max(length_penalty, 0.0)
                    return seq_score / norm

                beams = sorted(all_candidates, key=rank_key, reverse=True)[:beam_size]

            finished_beams = [beam for beam in beams if beam[2]]
            final_beams = finished_beams if finished_beams else beams

            def final_rank(item):
                seq, seq_score, _ = item
                norm = ((5.0 + len(seq)) / 6.0) ** max(length_penalty, 0.0)
                return seq_score / norm

            best_tokens = sorted(final_beams, key=final_rank, reverse=True)[0][0]
            clipped = best_tokens[:max_length]
            generated[b, :len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)

        return generated
