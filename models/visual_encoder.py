"""Visual encoder for chest X-rays.

Uses the public microsoft/rad-dino checkpoint when available and falls back to
the previous DINOv2 approximation for offline or legacy configurations.
"""

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision.transforms import functional as TF

try:
    from rad_dino import RadDino
except Exception:  # pragma: no cover - optional dependency
    RadDino = None


class VisualEncoder(nn.Module):
    """Frozen visual encoder producing patch token features from chest X-rays.

    Args:
        model_name: timm model name (default: dinov2_vits14)
        pretrained: Whether to load pretrained weights
        frozen: Whether to freeze all parameters
        output_tokens: Number of tokens after pooling (default: 49 = 7x7)
    """

    # Map from short names to timm model names
    MODEL_NAME_MAP = {
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
    }

    def __init__(self, model_name="dinov2_vits14", pretrained=True,
                 frozen=True, output_tokens=49):
        super().__init__()

        self.model_name = model_name
        self.frozen = frozen
        self.output_tokens = output_tokens
        self.uses_official_rad_dino = model_name in {"rad-dino", "microsoft/rad-dino", "microsoft/rad-dino-maira-2", "rad-dino-maira-2"}

        if self.uses_official_rad_dino:
            if RadDino is None:
                raise ImportError(
                    "rad_dino is not installed. Install the rad-dino package to use the official checkpoint."
                )
            self.backbone = RadDino()
            self.hidden_size = self.backbone.model.config.hidden_size  # 768
            self.patch_size = self.backbone.model.config.patch_size
            self.num_patches = None
        else:
            # Resolve timm model name
            timm_name = self.MODEL_NAME_MAP.get(model_name, model_name)

            # Load DINOv2 ViT-S/14
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                dynamic_img_size=True,
            )

            # Extract dimensions from the model
            self.hidden_size = self.backbone.embed_dim
            self.patch_size = self.backbone.patch_embed.patch_size[0]
            self.num_patches = self.backbone.patch_embed.num_patches

        # Pooling layer to reduce tokens to output_tokens (7x7 grid)
        grid_size = int(output_tokens ** 0.5)  # 7
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        if frozen:
            self._freeze()

    def _freeze(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _ensure_pil_list(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() != 4:
                raise ValueError("Expected a 4D image tensor batch when passing tensors to VisualEncoder.")
            return [TF.to_pil_image(img.detach().cpu()) for img in images]
        if isinstance(images, Image.Image):
            return [images]
        return list(images)

    def forward(self, images: torch.Tensor) -> dict:
        """Extract patch token features from images.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            Dictionary with:
                - patch_tokens: [B, 49, D] pooled patch tokens
                - raw_tokens: [B, N, D] raw patch tokens (before pooling)
                - cls_token: [B, D] CLS token
        """
        pil_images = self._ensure_pil_list(images)

        if self.uses_official_rad_dino:
            if self.frozen:
                with torch.no_grad():
                    cls_token, patch_tokens = self.backbone.extract_features(pil_images)
            else:
                cls_token, patch_tokens = self.backbone.extract_features(pil_images)

            # patch_tokens: [B, D, H, W]
            B, D, H, W = patch_tokens.shape
            spatial = patch_tokens
        else:
            if self.frozen:
                with torch.no_grad():
                    features = self.backbone.forward_features(images)
            else:
                features = self.backbone.forward_features(images)

            # features shape: [B, N+1, D] where N=num_patches, first token is CLS
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]

            B, N, D = patch_tokens.shape
            h = w = int(N ** 0.5)
            if h * w != N:
                h = w = int(N ** 0.5)
                patch_tokens = patch_tokens[:, :h * w]

            # Reshape to spatial grid for pooling: [B, D, H, W]
            spatial = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)

        # Adaptive pool to 7x7 grid: [B, D, 7, 7]
        orig_device = spatial.device
        if orig_device.type == "mps":
            spatial = spatial.cpu()
        pooled = self.adaptive_pool(spatial)
        if orig_device.type == "mps":
            pooled = pooled.to(orig_device)

        # Reshape back to sequence: [B, 49, D]
        grid_size = int(self.output_tokens ** 0.5)
        patch_tokens_pooled = pooled.reshape(B, D, grid_size, grid_size)
        patch_tokens_pooled = patch_tokens_pooled.permute(0, 2, 3, 1).reshape(B, self.output_tokens, D)

        return {
            "patch_tokens": patch_tokens_pooled,
            "raw_tokens": patch_tokens if not self.uses_official_rad_dino else patch_tokens.flatten(2).transpose(1, 2),
            "cls_token": cls_token,
        }
