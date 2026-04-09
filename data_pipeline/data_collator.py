"""Data collator for batching MIMIC-CXR samples.

Handles:
- Text tokenization using TextEncoder
- Padding and batching

Images are passed through as raw PIL objects so the visual encoder can apply
the official model-specific preprocessing (RAD-DINO or fallback path).
"""

import torch
from PIL import Image


class DataCollator:
    """Collates raw dataset samples into batched tensors.

    Args:
        text_encoder: TextEncoder instance for tokenization
        max_image_size: Target image size (square)
        max_text_len: Max report token length
        max_indication_len: Max indication token length
        max_prior_len: Max prior report token length
        max_impression_len: Max impression token length
        is_train: Whether in training mode (affects image transforms)
    """

    def __init__(self, text_encoder, max_image_size: int = 518,
                 max_text_len: int = 256, max_indication_len: int = 64,
                 max_prior_len: int = 128, max_impression_len: int = 128,
                 is_train: bool = True):
        self.text_encoder = text_encoder
        self.max_image_size = max_image_size
        self.max_text_len = max_text_len
        self.max_indication_len = max_indication_len
        self.max_prior_len = max_prior_len
        self.max_impression_len = max_impression_len
        self.is_train = is_train

    def __call__(self, batch: list[dict]) -> dict:
        """Collate a list of samples into a batch.

        Args:
            batch: List of dicts from MIMICCXRDataset.__getitem__

        Returns:
            Batched dictionary with tensor values
        """
        # ─── Images ───
        images = [item["image"].convert("RGB") for item in batch]

        # ─── Text tokenization ───
        reports = [item["report"] for item in batch]
        indications = [item["indication"] if item["indication"] else "none" for item in batch]
        prior_reports = [item["prior_report"] if item["prior_report"] else "none" for item in batch]
        impressions = [item["impression"] if item["impression"] else "none" for item in batch]

        report_tok = self.text_encoder.tokenize(reports, self.max_text_len)
        ind_tok = self.text_encoder.tokenize(indications, self.max_indication_len)
        pri_tok = self.text_encoder.tokenize(prior_reports, self.max_prior_len)
        imp_tok = self.text_encoder.tokenize(impressions, self.max_impression_len)

        # ─── Metadata ───
        stage_ids = torch.tensor([item["stage_id"] for item in batch], dtype=torch.long)
        view_ids = torch.tensor([item["view_id"] for item in batch], dtype=torch.long)

        return {
            "images": images,
            "report_ids": report_tok["input_ids"],
            "report_mask": report_tok["attention_mask"],
            "indication_ids": ind_tok["input_ids"],
            "indication_mask": ind_tok["attention_mask"],
            "prior_ids": pri_tok["input_ids"],
            "prior_mask": pri_tok["attention_mask"],
            "impression_ids": imp_tok["input_ids"],
            "impression_mask": imp_tok["attention_mask"],
            "stage_ids": stage_ids,
            "view_ids": view_ids,
        }
