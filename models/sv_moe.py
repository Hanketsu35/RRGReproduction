"""Stage-View Mixture-of-Experts (SV-MoE) module.

Routes using visit stage and image view information.
z = MLP(Emb(StageID) || Emb(ViewID))
Expert scores = Linear(z) + Softmax
Single-expert routing via argmax over K experts.
Experts are FFN modules.

[APPROXIMATE] MLP dimensions and FFN hidden ratio are inferred from standard
practices since the paper does not specify them exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: a standard feed-forward network.

    Args:
        hidden_size: Input/output dimension
        ffn_dim: FFN intermediate dimension
        dropout: Dropout probability
    """

    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward pass.

        Args:
            x: [B, T, D] input tokens

        Returns:
            [B, T, D] transformed tokens
        """
        return self.dropout2(self.fc2(self.dropout1(self.activation(self.fc1(x)))))


class SVMoERouter(nn.Module):
    """Router that computes expert assignment probabilities from stage and view.

    Args:
        num_stages: Number of discrete visit stages
        num_views: Number of imaging views
        embed_dim: Embedding dimension for stage/view IDs
        hidden_dim: MLP hidden dimension
        num_experts: Number of experts to route to
    """

    def __init__(self, num_stages: int, num_views: int, embed_dim: int,
                 hidden_dim: int, num_experts: int):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, embed_dim)
        self.view_embedding = nn.Embedding(num_views, embed_dim)

        # Router MLP: takes concatenation of stage and view embeddings
        self.router_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, stage_ids: torch.Tensor,
                view_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing probabilities.

        Args:
            stage_ids: [B] integer stage IDs
            view_ids: [B] integer view IDs

        Returns:
            expert_probs: [B, K] softmax probabilities over experts
            selected_expert: [B] argmax expert index per sample
        """
        stage_emb = self.stage_embedding(stage_ids)   # [B, embed_dim]
        view_emb = self.view_embedding(view_ids)       # [B, embed_dim]

        # Concatenate and route
        z = torch.cat([stage_emb, view_emb], dim=-1)   # [B, 2*embed_dim]
        logits = self.router_mlp(z)                      # [B, K]
        expert_probs = F.softmax(logits, dim=-1)          # [B, K]
        selected_expert = torch.argmax(expert_probs, dim=-1)  # [B]

        return expert_probs, selected_expert


class SVMoE(nn.Module):
    """Stage-View Mixture-of-Experts layer.

    Routes each sample to a single expert based on stage and view metadata.
    Includes load-balancing loss computation.

    Args:
        hidden_size: Token hidden dimension
        num_experts: Number of FFN experts (K)
        num_stages: Number of visit stage categories
        num_views: Number of imaging view categories
        embed_dim: Embedding dim for stage/view ID embeddings
        router_hidden_dim: Router MLP hidden dimension
        expert_hidden_dim: FFN intermediate dimension per expert
        dropout: Dropout probability
    """

    def __init__(self, hidden_size: int = 512, num_experts: int = 4,
                 num_stages: int = 5, num_views: int = 4,
                 embed_dim: int = 64, router_hidden_dim: int = 128,
                 expert_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size

        # Router
        self.router = SVMoERouter(
            num_stages=num_stages,
            num_views=num_views,
            embed_dim=embed_dim,
            hidden_dim=router_hidden_dim,
            num_experts=num_experts,
        )

        # Expert FFN modules
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, stage_ids: torch.Tensor,
                view_ids: torch.Tensor) -> dict:
        """Route tokens through mixture of experts.

        Args:
            x: [B, T, D] input token sequence
            stage_ids: [B] stage IDs
            view_ids: [B] view IDs

        Returns:
            Dictionary with:
                - output: [B, T, D] expert-processed tokens
                - expert_probs: [B, K] routing probabilities
                - selected_expert: [B] selected expert indices
                - load_balance_loss: scalar load-balancing loss
        """
        B, T, D = x.shape

        # Get routing decisions
        expert_probs, selected_expert = self.router(stage_ids, view_ids)
        # expert_probs: [B, K], selected_expert: [B]

        # Process each sample through its selected expert
        output = torch.zeros_like(x)  # [B, T, D]

        for expert_idx in range(self.num_experts):
            # Find samples routed to this expert
            mask = (selected_expert == expert_idx)  # [B]
            if not mask.any():
                continue

            # Process through expert
            expert_input = x[mask]                   # [B', T, D]
            expert_output = self.experts[expert_idx](expert_input)  # [B', T, D]
            output[mask] = expert_output

        # Compute load-balancing loss (Switch Transformer formulation)
        load_balance_loss = self._compute_load_balance_loss(expert_probs)

        return {
            "output": output,                        # [B, T, D]
            "expert_probs": expert_probs,             # [B, K]
            "selected_expert": selected_expert,       # [B]
            "load_balance_loss": load_balance_loss,   # scalar
        }

    def _compute_load_balance_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """Compute Switch Transformer style load-balancing loss.

        L_bal = K * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i

        Args:
            expert_probs: [B, K] routing probabilities

        Returns:
            Scalar load-balancing loss (minimized when uniform)
        """
        K = self.num_experts

        # f_i: fraction of samples routed to each expert
        selected = torch.argmax(expert_probs, dim=-1)  # [B]
        expert_mask = F.one_hot(selected, K).float()   # [B, K]
        f = expert_mask.mean(dim=0)                      # [K]

        # P_i: mean routing probability for each expert
        P = expert_probs.mean(dim=0)                     # [K]

        # Load balance loss
        loss = K * torch.sum(f * P)
        return loss
