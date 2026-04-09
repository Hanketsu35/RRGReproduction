"""Text encoder using the public CXR-BERT checkpoints.

The official public checkpoints are microsoft/BiomedVLP-CXR-BERT-general and
microsoft/BiomedVLP-CXR-BERT-specialized.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Text encoder for report text, indications, and prior reports.

    Supports encoding multiple text fields with learnable type prompts.

    Args:
        model_name: HuggingFace model name for the text encoder
        hidden_size: Hidden dimension (default: 768 for BERT-base)
        frozen: Whether to freeze encoder parameters
        max_length: Maximum token length for inputs
    """

    def __init__(self, model_name="microsoft/BiomedVLP-CXR-BERT-specialized",
                 hidden_size=768, frozen=False, max_length=256):
        super().__init__()

        self.hidden_size = hidden_size
        self.frozen = frozen
        self.max_length = max_length

        # Load pretrained model and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Add special tokens for type prompts
        special_tokens = ["[IND]", "[PRI]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Learnable type prompt embeddings (projected from token embedding)
        self.ind_prompt = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pri_prompt = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        if frozen:
            self._freeze()

    def _freeze(self):
        """Freeze encoder parameters (but keep prompt embeddings trainable)."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               prompt_type: str = None) -> dict:
        """Encode a batch of text inputs, optionally prepending a type prompt.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            prompt_type: One of "ind", "pri", or None

        Returns:
            Dictionary with:
                - cls_embedding: [B, D] [CLS] token embedding
                - last_hidden: [B, L+1, D] full sequence (with prompt if given)
                - attention_mask: [B, L+1] updated mask
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, L, D]
        cls_embedding = last_hidden[:, 0]         # [B, D]

        if prompt_type is not None:
            # Prepend learnable type prompt
            if prompt_type == "ind":
                prompt = self.ind_prompt.expand(last_hidden.size(0), -1, -1)
            elif prompt_type == "pri":
                prompt = self.pri_prompt.expand(last_hidden.size(0), -1, -1)
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

            last_hidden = torch.cat([prompt, last_hidden], dim=1)

            # Extend attention mask for the prompt token
            prompt_mask = torch.ones(
                attention_mask.size(0), 1,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

            # CLS embedding after prompt prepending: still use the original CLS
            # (now at position 1 after the prompt)
            cls_embedding = last_hidden[:, 1]

        return {
            "cls_embedding": cls_embedding,      # [B, D]
            "last_hidden": last_hidden,           # [B, L(+1), D]
            "attention_mask": attention_mask,      # [B, L(+1)]
        }

    def tokenize(self, texts: list[str], max_length: int = None) -> dict:
        """Tokenize a list of texts.

        Args:
            texts: List of strings
            max_length: Override max length

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        max_length = max_length or self.max_length
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
