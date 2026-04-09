"""Loss functions for MOE-RRG.

Three loss components:
1. L_CE: Cross-entropy loss for report generation
2. L_IMP: Impression contrastive loss (InfoNCE)
3. L_MOE: MoE routing load-balancing loss

Final: L = L_CE + lambda_imp * L_IMP + lambda_moe * L_MOE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_position_weighting import TokenPositionWeightingModule
from .curriculum_scheduler import CurriculumLearningScheduler


class ReportCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for report generation with label smoothing.

    Args:
        label_smoothing: Label smoothing factor
        ignore_index: Index to ignore in loss (padding)
    """

    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                token_weights: torch.Tensor = None) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: [B, T, V] decoder output logits
            targets: [B, T] target token IDs

        Returns:
            Scalar loss
        """
        B, T, V = logits.shape
        loss_flat = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="none",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
        loss_tok = loss_flat.reshape(B, T)

        valid_mask = (targets != self.ignore_index).float()
        if token_weights is None:
            denom = valid_mask.sum().clamp(min=1e-8)
            return (loss_tok * valid_mask).sum() / denom

        weights = token_weights * valid_mask
        denom = weights.sum().clamp(min=1e-8)
        return (loss_tok * weights).sum() / denom


class ImpressionContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss aligning report embeddings with impression embeddings.

    [APPROXIMATE] The pooling method for the decoder output embedding is not
    specified. We use mean pooling over the decoder's last hidden states.

    Args:
        temperature: InfoNCE temperature (tau = 0.07)
        projection_dim: Dimension for projection head (optional)
    """

    def __init__(self, temperature: float = 0.07, projection_dim: int = 256,
                 decoder_hidden_size: int = 512, text_hidden_size: int = 768):
        super().__init__()
        self.temperature = temperature

        # Projection heads for better alignment
        self.report_proj = nn.Sequential(
            nn.Linear(decoder_hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.impression_proj = nn.Sequential(
            nn.Linear(text_hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, decoder_hidden: torch.Tensor,
                impression_embedding: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            decoder_hidden: [B, T, D_dec] decoder last hidden states
            impression_embedding: [B, D_txt] impression [CLS] embedding
            attention_mask: [B, T] decoder attention mask for mean pooling

        Returns:
            Scalar contrastive loss
        """
        # Mean pool decoder hidden states
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()        # [B, T, 1]
            report_emb = (decoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            report_emb = decoder_hidden.mean(dim=1)           # [B, D_dec]

        # Project to common space
        report_emb = self.report_proj(report_emb)              # [B, proj_dim]
        impression_emb = self.impression_proj(impression_embedding)  # [B, proj_dim]

        # L2 normalize
        report_emb = F.normalize(report_emb, dim=-1)
        impression_emb = F.normalize(impression_emb, dim=-1)

        # InfoNCE loss (in-batch negatives)
        # Similarity matrix: [B, B]
        sim_matrix = torch.matmul(report_emb, impression_emb.T) / self.temperature

        # Labels: diagonal (positive pairs)
        B = sim_matrix.size(0)
        labels = torch.arange(B, device=sim_matrix.device)

        # Symmetric loss
        loss_r2i = F.cross_entropy(sim_matrix, labels)
        loss_i2r = F.cross_entropy(sim_matrix.T, labels)

        return (loss_r2i + loss_i2r) / 2


class MOERRGLoss(nn.Module):
    """Combined loss for MOE-RRG training.

    L = L_CE + lambda_imp * L_IMP + lambda_moe * L_MOE

    Args:
        lambda_imp: Weight for impression contrastive loss
        lambda_moe: Weight for MoE load-balancing loss
        label_smoothing: Label smoothing for CE loss
        temperature: InfoNCE temperature
        ignore_index: Padding token index
    """

    def __init__(self, lambda_imp: float = 0.1, lambda_moe: float = 0.2,
                 label_smoothing: float = 0.1, temperature: float = 0.07,
                 ignore_index: int = 0,
                 token_weight_cfg: dict = None,
                 curriculum_cfg: dict = None,
                 max_epochs: int = 50):
        super().__init__()

        self.lambda_imp = lambda_imp
        self.lambda_moe = lambda_moe

        self.ce_loss = ReportCrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        self.imp_loss = ImpressionContrastiveLoss(temperature=temperature)
        self.ignore_index = ignore_index

        token_weight_cfg = token_weight_cfg or {}
        curriculum_cfg = curriculum_cfg or {}

        self.token_weighter = None
        if token_weight_cfg.get("enabled", False):
            self.token_weighter = TokenPositionWeightingModule(
                max_seq_len=token_weight_cfg.get("max_seq_len", 256),
                initialization=token_weight_cfg.get("initialization", "uniform"),
            )

        self.curriculum = None
        if curriculum_cfg.get("enabled", False):
            self.curriculum = CurriculumLearningScheduler(
                max_seq_len=token_weight_cfg.get("max_seq_len", 256),
                max_epochs=curriculum_cfg.get("max_epochs", max_epochs),
                temperature_base=curriculum_cfg.get("temperature_base", 2.0),
                temperature_min=curriculum_cfg.get("temperature_min", 0.3),
                schedule=curriculum_cfg.get("schedule", "linear"),
                center=curriculum_cfg.get("center", 0.5),
                width=curriculum_cfg.get("width", 0.25),
            )

    def _build_token_weights(self, targets: torch.Tensor,
                             target_mask: torch.Tensor,
                             epoch: int) -> torch.Tensor:
        """Build per-token weights combining learnable and curriculum signals."""
        if self.token_weighter is None and self.curriculum is None:
            return None

        B, T = targets.shape
        device = targets.device

        combined = torch.ones(T, device=device)
        if self.token_weighter is not None:
            combined = combined * self.token_weighter(seq_len=T, device=device)
        if self.curriculum is not None:
            combined = combined * self.curriculum(seq_len=T, epoch=epoch, device=device)

        combined = combined / (combined.sum() + 1e-8)
        weights = combined.unsqueeze(0).expand(B, T).clone()

        if target_mask is not None:
            valid = target_mask.float()
        else:
            valid = (targets != self.ignore_index).float()

        weights = weights * valid
        row_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / row_sum
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                decoder_hidden: torch.Tensor,
                impression_embedding: torch.Tensor,
                load_balance_loss: torch.Tensor,
                target_mask: torch.Tensor = None,
                epoch: int = 0) -> dict:
        """Compute total MOE-RRG loss.

        Args:
            logits: [B, T, V] decoder logits
            targets: [B, T] target token IDs
            decoder_hidden: [B, T, D] decoder hidden states
            impression_embedding: [B, D_txt] impression embeddings
            load_balance_loss: scalar MoE load-balance loss
            target_mask: [B, T] target attention mask

        Returns:
            Dictionary with:
                - total_loss: weighted total loss
                - ce_loss: cross-entropy component
                - imp_loss: impression contrastive component
                - moe_loss: load-balancing component
        """
        # Cross-entropy loss (optionally weighted by position and curriculum)
        token_weights = self._build_token_weights(
            targets=targets,
            target_mask=target_mask,
            epoch=epoch,
        )
        l_ce = self.ce_loss(logits, targets, token_weights=token_weights)

        # Impression contrastive loss
        l_imp = self.imp_loss(decoder_hidden, impression_embedding, target_mask)

        # Total loss
        total_loss = l_ce + self.lambda_imp * l_imp + self.lambda_moe * load_balance_loss

        return {
            "total_loss": total_loss,
            "ce_loss": l_ce,
            "imp_loss": l_imp,
            "moe_loss": load_balance_loss,
        }
