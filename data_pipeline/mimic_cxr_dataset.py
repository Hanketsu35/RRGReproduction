"""MIMIC-CXR Dataset class for MOE-RRG.

Loads preprocessed metadata and provides (image, text, metadata) samples.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset for MOE-RRG training and evaluation.

    Each sample provides:
    - Chest X-ray image
    - Report text (findings) as target
    - Indication text (auxiliary cue)
    - Prior report text (auxiliary cue)
    - Impression text (for contrastive loss)
    - Stage ID (for SV-MoE routing)
    - View ID (for SV-MoE routing)

    Args:
        metadata_csv: Path to preprocessed metadata CSV
        image_root: Root directory for MIMIC-CXR JPEG files
        split: Which split to load ("train", "validate", "test")
        text_encoder: TextEncoder instance for tokenization
        max_image_size: Maximum image size (will be resized/padded)
        max_text_len: Max report token length
        max_indication_len: Max indication token length
        max_prior_len: Max prior report token length
        max_impression_len: Max impression token length
        transform: Image transforms (applied after resize)
        is_train: Whether in training mode (affects augmentation)
    """

    def __init__(self, metadata_csv: str, image_root: str,
                 split: str = "train",
                 text_encoder=None,
                 max_image_size: int = 518,
                 max_text_len: int = 256,
                 max_indication_len: int = 64,
                 max_prior_len: int = 128,
                 max_impression_len: int = 128,
                 transform=None,
                 is_train: bool = True):
        super().__init__()

        self.image_root = Path(image_root)
        self.split = split
        self.max_image_size = max_image_size
        self.max_text_len = max_text_len
        self.max_indication_len = max_indication_len
        self.max_prior_len = max_prior_len
        self.max_impression_len = max_impression_len
        self.text_encoder = text_encoder
        self.is_train = is_train

        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)

        # Filter by split
        if split:
            split_map = {
                "train": "train",
                "val": "validate",
                "valid": "validate",
                "validate": "validate",
                "validation": "validate",
                "test": "test",
            }
            requested = split_map.get(str(split).lower().strip(), str(split).lower().strip())
            if "split" in self.metadata.columns:
                normalized_split = self.metadata["split"].fillna("train").astype(str).str.lower().str.strip().map(split_map)
                self.metadata["split"] = normalized_split.fillna("train")
            else:
                raise ValueError(
                    f"Metadata CSV at {metadata_csv} has no 'split' column; "
                    "run preprocessing with split CSV or provide pre-split metadata."
                )
            self.metadata = self.metadata[self.metadata["split"] == requested].reset_index(drop=True)

        # Filter out samples without valid reports
        if "findings" in self.metadata.columns:
            self.metadata = self.metadata[
                self.metadata["findings"].notna() &
                (self.metadata["findings"].str.len() > 10)
            ].reset_index(drop=True)

        print(f"MIMICCXRDataset [{split}]: {len(self.metadata)} samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with all fields needed by the model.
            Tokenization is done in the collator for efficiency.
        """
        row = self.metadata.iloc[idx]

        # Load image
        image_path = self.image_root / row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a black image if file not found
            image = Image.new("RGB", (self.max_image_size, self.max_image_size), (0, 0, 0))

        # Get text fields
        report = str(row.get("findings", ""))
        indication = str(row.get("indication", ""))
        prior_report = str(row.get("prior_report", ""))
        impression = str(row.get("impression", ""))

        # Get metadata
        stage_id = int(row.get("stage_id", 0))
        view_id = int(row.get("view_id", 0))

        return {
            "image": image,             # PIL Image (tokenized in collator)
            "report": report,
            "indication": indication,
            "prior_report": prior_report,
            "impression": impression,
            "stage_id": stage_id,
            "view_id": view_id,
            "study_id": int(row.get("study_id", 0)) if pd.notna(row.get("study_id", 0)) else 0,
            "subject_id": int(row.get("subject_id", 0)) if pd.notna(row.get("subject_id", 0)) else 0,
        }
