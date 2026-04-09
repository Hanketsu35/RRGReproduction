"""MIMIC-CXR dataset preprocessing.

Handles:
- Metadata extraction (stage discretization, view mapping)
- Report section extraction (indication, impression, findings)
- Train/val/test split management
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Canonical view names for embedding lookup
VIEW_NAMES = ["AP", "PA", "LA", "LL"]
NUM_VIEWS = len(VIEW_NAMES)

# Stage discretization
STAGE_RULES = [
    (1, 0),       # Stage 0: first visit
    (2, 1),       # Stage 1: second visit
    (3, 2),       # Stage 2: visits 3-5
    (6, 3),       # Stage 3: visits 6-10
    (11, 4),      # Stage 4: visits >10
]


def discretize_stage(visit_number: int) -> int:
    """Map visit number to stage ID.

    Args:
        visit_number: 1-indexed visit count for this patient

    Returns:
        Stage ID (0-4)
    """
    if visit_number == 1:
        return 0
    elif visit_number == 2:
        return 1
    elif 3 <= visit_number <= 5:
        return 2
    elif 6 <= visit_number <= 10:
        return 3
    else:
        return 4


def map_view(view_label: str) -> int:
    """Map MIMIC-CXR view label to canonical view ID.

    Args:
        view_label: Raw view string from metadata

    Returns:
        View ID (0-3), or -1 to exclude
    """
    view_label = view_label.strip().upper()

    direct_map = {
        "AP": 0,
        "AP SUPINE": 0,
        "PA": 1,
        "LA": 2,
        "LATERAL": 2,
        "LL": 3,
    }

    return direct_map.get(view_label, -1)


class MIMICPreprocessor:
    """Preprocesses MIMIC-CXR metadata for MOE-RRG training.

    Extracts stage, view, and sectioned report text from raw MIMIC-CXR files.

    Args:
        metadata_csv: Path to MIMIC-CXR metadata CSV
        report_csv: Path to MIMIC-CXR sectioned reports CSV
        split_csv: Path to official train/val/test split CSV
        output_dir: Directory to save processed metadata
    """

    def __init__(self, metadata_csv: str, report_csv: str = None,
                 split_csv: str = None, output_dir: str = "processed"):
        self.metadata_csv = metadata_csv
        self.report_csv = report_csv
        self.split_csv = split_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def process(self) -> pd.DataFrame:
        """Run full preprocessing pipeline.

        Returns:
            Processed DataFrame with columns:
                - dicom_id, subject_id, study_id
                - view_id: int (0-3)
                - stage_id: int (0-4)
                - split: train/val/test
                - indication: str
                - impression: str
                - findings: str
                - prior_report: str
                - image_path: str
        """
        print("Loading metadata...")
        meta_df = pd.read_csv(self.metadata_csv)

        # Map views
        print("Mapping views...")
        meta_df["view_id"] = meta_df["ViewPosition"].apply(map_view)

        # Filter out excluded views
        meta_df = meta_df[meta_df["view_id"] >= 0].copy()
        print(f"  After view filtering: {len(meta_df)} samples")

        # Compute visit stages
        print("Computing visit stages...")
        meta_df = self._compute_stages(meta_df)

        # Load reports and extract sections
        if self.report_csv and os.path.exists(self.report_csv):
            print("Loading reports...")
            report_df = pd.read_csv(self.report_csv)
            report_df["study_id"] = report_df["study_id"].apply(self._normalize_id)
            meta_df["study_id"] = meta_df["study_id"].apply(self._normalize_id)
            meta_df = meta_df.merge(
                report_df[["study_id", "indication", "impression", "findings"]],
                on="study_id", how="left"
            )
        else:
            # Create empty columns if no report CSV
            for col in ["indication", "impression", "findings"]:
                meta_df[col] = ""

        # Fill NaN text fields
        for col in ["indication", "impression", "findings"]:
            meta_df[col] = meta_df[col].fillna("").astype(str)

        # Compute prior reports
        print("Computing prior reports...")
        meta_df = self._compute_prior_reports(meta_df)

        # Add train/val/test splits
        if self.split_csv and os.path.exists(self.split_csv):
            print("Loading splits...")
            split_df = pd.read_csv(self.split_csv)
            merge_keys = [k for k in ["subject_id", "study_id", "dicom_id"] if k in split_df.columns and k in meta_df.columns]
            if not merge_keys:
                merge_keys = ["dicom_id"]
            for key in merge_keys:
                split_df[key] = split_df[key].apply(self._normalize_id)
                meta_df[key] = meta_df[key].apply(self._normalize_id)
            meta_df = meta_df.merge(
                split_df[merge_keys + ["split"]],
                on=merge_keys, how="left"
            )
            meta_df["split"] = self._normalize_split(meta_df["split"])
        else:
            # Default: stratified random split
            print("  No split file found, using random stratified split...")
            meta_df = self._random_split(meta_df)

        # Build image paths
        meta_df["image_path"] = meta_df.apply(
            self._build_image_path,
            axis=1,
        )

        # Save processed metadata
        output_path = self.output_dir / "processed_metadata.csv"
        meta_df.to_csv(output_path, index=False)
        print(f"Saved processed metadata to {output_path}")
        print(f"  Total samples: {len(meta_df)}")
        if "split" in meta_df.columns:
            print(f"  Train: {(meta_df['split'] == 'train').sum()}")
            print(f"  Val: {(meta_df['split'] == 'validate').sum()}")
            print(f"  Test: {(meta_df['split'] == 'test').sum()}")

        return meta_df

    def _compute_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute visit stage for each sample based on patient history."""
        # Sort by patient and study timestamp when available
        sort_cols = ["subject_id"]
        if "StudyDate" in df.columns:
            sort_cols.append("StudyDate")
        if "StudyTime" in df.columns:
            sort_cols.append("StudyTime")
        df = df.sort_values(sort_cols).copy()

        # Count visits per unique study (not per image row)
        study_order = (
            df[["subject_id", "study_id"]]
            .drop_duplicates()
            .copy()
        )
        study_order["visit_number"] = study_order.groupby("subject_id").cumcount() + 1
        study_order["stage_id"] = study_order["visit_number"].apply(discretize_stage)

        df = df.merge(
            study_order[["subject_id", "study_id", "visit_number", "stage_id"]],
            on=["subject_id", "study_id"],
            how="left",
        )

        return df

    def _compute_prior_reports(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each study, use prior study impression as the prior report."""
        sort_cols = ["subject_id"]
        if "StudyDate" in df.columns:
            sort_cols.append("StudyDate")
        if "StudyTime" in df.columns:
            sort_cols.append("StudyTime")
        df = df.sort_values(sort_cols).copy()
        df["prior_report"] = ""

        for _, group in df.groupby("subject_id", sort=False):
            study_ids = group["study_id"].drop_duplicates().tolist()
            prior_imp = ""
            for study_id in study_ids:
                mask = group["study_id"] == study_id
                row_indices = group.loc[mask].index
                df.loc[row_indices, "prior_report"] = prior_imp

                current_impressions = group.loc[mask, "impression"].astype(str).str.strip()
                non_empty = current_impressions[current_impressions != ""]
                if len(non_empty) > 0:
                    prior_imp = non_empty.iloc[0]

        return df

    @staticmethod
    def _normalize_split(split_series: pd.Series) -> pd.Series:
        mapping = {
            "train": "train",
            "val": "validate",
            "valid": "validate",
            "validate": "validate",
            "validation": "validate",
            "test": "test",
        }
        normalized = split_series.fillna("train").astype(str).str.lower().str.strip().map(mapping)
        return normalized.fillna("train")

    @staticmethod
    def _normalize_id(value) -> str:
        if pd.isna(value):
            return ""
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value).strip()

    def _build_image_path(self, row: pd.Series) -> str:
        subject_id = self._normalize_id(row.get("subject_id", ""))
        study_id = self._normalize_id(row.get("study_id", ""))
        dicom_id = self._normalize_id(row.get("dicom_id", ""))
        prefix = subject_id[:2] if len(subject_id) >= 2 else subject_id
        return f"files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"

    def _random_split(self, df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.2,
                      seed: int = 42) -> pd.DataFrame:
        """Create stratified random splits by stage and view."""
        rng = np.random.RandomState(seed)
        df = df.copy()

        # Group by patient to avoid data leakage
        patients = df["subject_id"].unique()
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train:n_train + n_val])
        test_patients = set(patients[n_train + n_val:])

        df["split"] = "test"
        df.loc[df["subject_id"].isin(train_patients), "split"] = "train"
        df.loc[df["subject_id"].isin(val_patients), "split"] = "validate"

        return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-CXR metadata for MOE-RRG")
    parser.add_argument("--metadata_csv", required=True, help="Path to mimic-cxr metadata CSV")
    parser.add_argument("--report_csv", default=None, help="Path to sectioned reports CSV")
    parser.add_argument("--split_csv", default=None, help="Path to official split CSV")
    parser.add_argument("--output_dir", default="processed", help="Output directory")
    args = parser.parse_args()

    preprocessor = MIMICPreprocessor(
        metadata_csv=args.metadata_csv,
        report_csv=args.report_csv,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
    )
    preprocessor.process()


if __name__ == "__main__":
    main()
