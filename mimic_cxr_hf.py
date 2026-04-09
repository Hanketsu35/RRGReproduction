"""MIMIC-CXR Dataset from HuggingFace (itsanmolgupta/mimic-cxr-dataset).

IMPORTANT: This dataset path is for experimentation/smoke runs only and is
not paper-comparable. Stage/view IDs are synthetic and prior/indication cues
are unavailable in this source.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


class MIMICCXRHFDataset(Dataset):
    """MIMIC-CXR dataset loaded from HuggingFace.

    Provides: image, findings (report), impression, stage_id, view_id

    WARNING:
        This class is explicitly non-paper-comparable. It generates synthetic
        stage/view metadata and does not provide true prior/indication fields.

    Args:
        split: "train", "val", or "test"
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test
        seed: Random seed for splitting
        max_samples: Limit number of samples (for quick tests)
    """

    def __init__(self, split: str = "train", val_ratio: float = 0.1,
                 test_ratio: float = 0.1, seed: int = 42, max_samples: int = None):
        super().__init__()
        self.seed = seed
        print("[EXPERIMENTAL] Loading MIMIC-CXR from HuggingFace (non-paper path)...")
        full_ds = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")

        # Filter out samples with empty reports
        valid_indices = []
        for i, sample in enumerate(full_ds):
            findings = sample.get("findings", "")
            if findings and len(str(findings).strip()) > 20:
                valid_indices.append(i)
        full_ds = full_ds.select(valid_indices)

        # Create deterministic splits by index
        n = len(full_ds)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test

        if split == "train":
            sel_indices = indices[:n_train]
        elif split == "val":
            sel_indices = indices[n_train:n_train + n_val]
        else:  # test
            sel_indices = indices[n_train + n_val:]

        if max_samples:
            sel_indices = sel_indices[:max_samples]

        self.data = full_ds.select(sel_indices.tolist())
        self.split = split

        print(f"MIMICCXRHFDataset [{split}]: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        image = sample["image"].convert("RGB")
        findings = str(sample.get("findings", "") or "")
        impression = str(sample.get("impression", "") or "")

        # Synthetic metadata for experimentation only (not paper-comparable).
        sample_seed = (self.seed * 1000003 + idx) & 0xFFFFFFFF
        stage_id = int(sample_seed % 5)
        view_id = int((sample_seed // 5) % 4)

        indication = ""
        prior_report = ""

        return {
            "image": image,
            "report": findings,
            "indication": indication,
            "prior_report": prior_report,
            "impression": impression,
            "stage_id": stage_id,
            "view_id": view_id,
            "data_protocol": "experimental_hf",
        }
