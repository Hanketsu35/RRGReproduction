#!/usr/bin/env python3
"""
Data Split Validation for MOE-RRG Reproduction

This script validates the MIMIC-CXR preprocessing and data splits against paper specifications.
It checks:
1. Split counts and composition
2. Patient leakage between splits
3. Stage and view distributions
4. Prior report availability and consistency
5. Text field completeness

Generates a comprehensive validation report.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def _normalize_split(series: pd.Series) -> pd.Series:
    """Normalize split labels to canonical forms."""
    mapping = {
        "train": "train",
        "val": "validate",
        "valid": "validate",
        "validate": "validate",
        "validation": "validate",
        "test": "test",
    }
    clean = series.fillna("train").astype(str).str.lower().str.strip()
    normalized = clean.map(mapping)
    unknown = sorted(clean[normalized.isna()].unique().tolist())
    if unknown:
        raise ValueError(f"Unknown split labels found: {unknown}")
    return normalized


def validate_data_splits(metadata_csv: str, output_dir: str = "analysis") -> dict:
    """Comprehensive data split validation.
    
    Args:
        metadata_csv: Path to processed_metadata.csv
        output_dir: Directory to save reports
    
    Returns:
        Validation report dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    
    if "split" not in df.columns:
        raise ValueError("Metadata must contain 'split' column")
    
    # Normalize splits
    df["split"] = _normalize_split(df["split"])
    
    total = len(df)
    print(f"Total samples: {total}")
    
    # ─────────────────────────────────────────
    # 1. SPLIT COUNTS
    # ─────────────────────────────────────────
    split_counts = {s: int((df["split"] == s).sum()) for s in ["train", "validate", "test"]}
    split_ratios = {s: 100.0 * split_counts[s] / total for s in split_counts}
    
    print("\n[1] SPLIT COUNTS:")
    for s in ["train", "validate", "test"]:
        print(f"  {s:10s}: {split_counts[s]:6d} ({split_ratios[s]:5.2f}%)")
    
    # ─────────────────────────────────────────
    # 2. PATIENT LEAKAGE CHECK
    # ─────────────────────────────────────────
    train_patients = set(df.loc[df["split"] == "train", "subject_id"].astype(str).unique())
    val_patients = set(df.loc[df["split"] == "validate", "subject_id"].astype(str).unique())
    test_patients = set(df.loc[df["split"] == "test", "subject_id"].astype(str).unique())
    
    leakage = {
        "train_val": len(train_patients & val_patients),
        "train_test": len(train_patients & test_patients),
        "val_test": len(val_patients & test_patients),
    }
    
    print("\n[2] PATIENT-LEVEL LEAKAGE CHECK:")
    print(f"  Patients in train ∩ val : {leakage['train_val']}")
    print(f"  Patients in train ∩ test: {leakage['train_test']}")
    print(f"  Patients in val ∩ test  : {leakage['val_test']}")
    
    if sum(leakage.values()) == 0:
        print("  ✅ NO LEAKAGE DETECTED")
    else:
        print(f"  ⚠️  LEAKAGE DETECTED: {sum(leakage.values())} patient overlaps")
    
    # ─────────────────────────────────────────
    # 3. STAGE DISTRIBUTION
    # ─────────────────────────────────────────
    stage_dist = {}
    if "stage_id" in df.columns:
        print("\n[3] STAGE DISTRIBUTION (visits):")
        for s in sorted(df["stage_id"].dropna().astype(int).unique().tolist()):
            cnt = int((df["stage_id"] == s).sum())
            ratio = 100.0 * cnt / total
            stage_dist[f"stage_{s}"] = {"count": cnt, "ratio_pct": round(ratio, 2)}
            stage_names = {
                0: "visit 1",
                1: "visit 2",
                2: "visits 3-5",
                3: "visits 6-10",
                4: "visits >10",
            }
            name = stage_names.get(s, f"stage {s}")
            print(f"  stage_{s} ({name:15s}): {cnt:6d} ({ratio:5.2f}%)")
    
    # ─────────────────────────────────────────
    # 4. VIEW DISTRIBUTION
    # ─────────────────────────────────────────
    view_names = {0: "AP", 1: "PA", 2: "LA", 3: "LL"}
    view_dist = {}
    if "view_id" in df.columns:
        print("\n[4] VIEW DISTRIBUTION (imaging position):")
        for v in sorted(df["view_id"].dropna().astype(int).unique().tolist()):
            cnt = int((df["view_id"] == v).sum())
            ratio = 100.0 * cnt / total
            view_name = view_names.get(v, f"view_{v}")
            view_dist[view_name] = {"count": cnt, "ratio_pct": round(ratio, 2)}
            print(f"  {view_name:4s}: {cnt:6d} ({ratio:5.2f}%)")
    
    # ─────────────────────────────────────────
    # 5. REPORT TEXT FIELDS
    # ─────────────────────────────────────────
    print("\n[5] REPORT TEXT FIELD COMPLETENESS:")
    for field in ["findings", "impression", "indication", "prior_report"]:
        if field in df.columns:
            non_empty = int((df[field].fillna("").astype(str).str.strip() != "").sum())
            ratio = 100.0 * non_empty / total
            avg_len = df[field].fillna("").astype(str).str.len().mean()
            print(f"  {field:15s}: {non_empty:6d} ({ratio:5.2f}%), avg length {avg_len:6.1f} chars")
    
    # ─────────────────────────────────────────
    # 6. PRIOR REPORT STATISTICS
    # ─────────────────────────────────────────
    prior_stats = {}
    if "prior_report" in df.columns and "impression" in df.columns:
        print("\n[6] PRIOR REPORT STATISTICS:")
        prior = df["prior_report"].fillna("").astype(str).str.strip()
        impression = df["impression"].fillna("").astype(str).str.strip()
        
        prior_non_empty = int((prior != "").sum())
        prior_ratio = 100.0 * prior_non_empty / total
        print(f"  Prior non-empty: {prior_non_empty:d} ({prior_ratio:.2f}%)")
        
        # Check for same as current impression (potential data leakage within sample)
        same_as_imp = int(((prior != "") & (prior == impression)).sum())
        same_ratio = 100.0 * same_as_imp / max(prior_non_empty, 1)
        print(f"  Prior == current impression: {same_as_imp} ({same_ratio:.2f}% of non-empty priors)")
        
        prior_stats["non_empty"] = prior_non_empty
        prior_stats["non_empty_ratio_pct"] = round(prior_ratio, 2)
        prior_stats["equals_impression"] = same_as_imp
        prior_stats["equals_impression_ratio_pct"] = round(same_ratio, 2)
    
    # ─────────────────────────────────────────
    # 7. STAGE-VIEW CROSS-TABULATION
    # ─────────────────────────────────────────
    if "stage_id" in df.columns and "view_id" in df.columns:
        print("\n[7] STAGE × VIEW CROSS-TABULATION:")
        crosstab = pd.crosstab(df["stage_id"], df["view_id"])
        print(crosstab)
    
    # ─────────────────────────────────────────
    # SAVE REPORTS
    # ─────────────────────────────────────────
    
    # JSON report
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "metadata_csv": metadata_csv,
        "total_samples": total,
        "split_counts": split_counts,
        "split_ratios_pct": {k: round(v, 2) for k, v in split_ratios.items()},
        "patient_leakage": leakage,
        "stage_distribution": stage_dist,
        "view_distribution": view_dist,
        "prior_report_stats": prior_stats,
    }
    
    json_path = output_path / "data_split_validation_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    print(f"\n✅ JSON report saved to: {json_path}")
    
    # Text report (already printed above)
    txt_path = output_path / "data_split_validation_report.txt"
    lines = [
        "=" * 72,
        "MIMIC-CXR DATA SPLIT VALIDATION REPORT",
        "=" * 72,
        f"Generated: {datetime.now().isoformat()}",
        f"Metadata: {metadata_csv}",
        f"Total samples: {total}",
        "",
        "[1] SPLIT COUNTS:",
    ]
    for s in ["train", "validate", "test"]:
        lines.append(f"  {s:10s}: {split_counts[s]:6d} ({split_ratios[s]:5.2f}%)")
    
    lines.extend([
        "",
        "[2] PATIENT-LEVEL LEAKAGE:",
        f"  train ∩ val : {leakage['train_val']}",
        f"  train ∩ test: {leakage['train_test']}",
        f"  val ∩ test  : {leakage['val_test']}",
    ])
    
    if stage_dist:
        lines.append("")
        lines.append("[3] STAGE DISTRIBUTION:")
        for key, value in stage_dist.items():
            lines.append(f"  {key}: {value['count']} ({value['ratio_pct']}%)")
    
    if view_dist:
        lines.append("")
        lines.append("[4] VIEW DISTRIBUTION:")
        for key, value in view_dist.items():
            lines.append(f"  {key}: {value['count']} ({value['ratio_pct']}%)")
    
    if prior_stats:
        lines.append("")
        lines.append("[5] PRIOR REPORT STATS:")
        lines.append(f"  Non-empty: {prior_stats['non_empty']} ({prior_stats['non_empty_ratio_pct']}%)")
        lines.append(f"  Equals current impression: {prior_stats['equals_impression']} ({prior_stats['equals_impression_ratio_pct']}%)")
    
    lines.append("")
    lines.append("=" * 72)
    
    txt_path.write_text("\n".join(lines))
    print(f"✅ Text report saved to: {txt_path}")
    
    return json_report


def main():
    parser = argparse.ArgumentParser(description="Validate MIMIC-CXR data splits")
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to processed_metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis",
        help="Output directory for reports",
    )
    args = parser.parse_args()
    
    validate_data_splits(args.metadata_csv, args.output_dir)


if __name__ == "__main__":
    main()
