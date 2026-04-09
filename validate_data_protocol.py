"""Validate processed MIMIC-CXR protocol alignment for reproduction fidelity.

Checks:
- Split counts and patient leakage
- Stage/view distributions
- Prior report consistency rates
"""

import argparse
from pathlib import Path

import pandas as pd


def _normalize_split(series: pd.Series) -> pd.Series:
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


def _format_ratio(num: int, den: int) -> str:
    if den <= 0:
        return "0 (0.00%)"
    return f"{num} ({(100.0 * num / den):.2f}%)"


def validate_protocol(metadata_csv: str, output_path: str | None = None) -> dict:
    df = pd.read_csv(metadata_csv)

    if "split" not in df.columns:
        raise ValueError("Metadata must contain a 'split' column")

    df["split"] = _normalize_split(df["split"])

    total = len(df)
    split_counts = {s: int((df["split"] == s).sum()) for s in ["train", "validate", "test"]}

    train_patients = set(df.loc[df["split"] == "train", "subject_id"].astype(str).unique())
    val_patients = set(df.loc[df["split"] == "validate", "subject_id"].astype(str).unique())
    test_patients = set(df.loc[df["split"] == "test", "subject_id"].astype(str).unique())

    leakage = {
        "train_val": len(train_patients & val_patients),
        "train_test": len(train_patients & test_patients),
        "val_test": len(val_patients & test_patients),
    }

    stage_dist = {}
    if "stage_id" in df.columns:
        for s in sorted(df["stage_id"].dropna().astype(int).unique().tolist()):
            cnt = int((df["stage_id"] == s).sum())
            stage_dist[f"stage_{s}"] = {
                "count": cnt,
                "ratio_pct": round(100.0 * cnt / total, 4) if total else 0.0,
            }

    view_names = {0: "AP", 1: "PA", 2: "LA", 3: "LL"}
    view_dist = {}
    if "view_id" in df.columns:
        for v in sorted(df["view_id"].dropna().astype(int).unique().tolist()):
            cnt = int((df["view_id"] == v).sum())
            view_dist[view_names.get(v, f"view_{v}")] = {
                "count": cnt,
                "ratio_pct": round(100.0 * cnt / total, 4) if total else 0.0,
            }

    prior_stats = {}
    if "prior_report" in df.columns and "impression" in df.columns:
        prior = df["prior_report"].fillna("").astype(str).str.strip()
        impression = df["impression"].fillna("").astype(str).str.strip()
        prior_non_empty = int((prior != "").sum())
        same_as_impression = int(((prior != "") & (prior == impression)).sum())
        prior_stats = {
            "prior_non_empty": prior_non_empty,
            "prior_non_empty_ratio_pct": round(100.0 * prior_non_empty / total, 4) if total else 0.0,
            "prior_equals_current_impression": same_as_impression,
            "prior_equals_current_impression_ratio_pct": round(
                100.0 * same_as_impression / prior_non_empty, 4
            ) if prior_non_empty else 0.0,
        }

    report = {
        "metadata_csv": metadata_csv,
        "total_samples": total,
        "split_counts": split_counts,
        "patient_leakage_counts": leakage,
        "stage_distribution": stage_dist,
        "view_distribution": view_dist,
        "prior_report_stats": prior_stats,
    }

    lines = []
    lines.append("=" * 72)
    lines.append("MIMIC-CXR Protocol Validation")
    lines.append("=" * 72)
    lines.append(f"Metadata: {metadata_csv}")
    lines.append(f"Total samples: {total}")
    lines.append("")
    lines.append("Split counts:")
    for split_name in ["train", "validate", "test"]:
        lines.append(f"  {split_name:8s}: {_format_ratio(split_counts[split_name], total)}")

    lines.append("")
    lines.append("Patient leakage:")
    lines.append(f"  train-val : {leakage['train_val']}")
    lines.append(f"  train-test: {leakage['train_test']}")
    lines.append(f"  val-test  : {leakage['val_test']}")

    if stage_dist:
        lines.append("")
        lines.append("Stage distribution:")
        for key, value in stage_dist.items():
            lines.append(f"  {key:10s}: {value['count']} ({value['ratio_pct']:.2f}%)")

    if view_dist:
        lines.append("")
        lines.append("View distribution:")
        for key, value in view_dist.items():
            lines.append(f"  {key:10s}: {value['count']} ({value['ratio_pct']:.2f}%)")

    if prior_stats:
        lines.append("")
        lines.append("Prior report consistency:")
        lines.append(
            f"  prior non-empty: {_format_ratio(prior_stats['prior_non_empty'], total)}"
        )
        lines.append(
            "  prior == current impression: "
            f"{prior_stats['prior_equals_current_impression']} "
            f"({prior_stats['prior_equals_current_impression_ratio_pct']:.2f}% of non-empty priors)"
        )

    printable = "\n".join(lines)
    print(printable)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(printable + "\n", encoding="utf-8")
        print(f"\nSaved report to: {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate processed MIMIC-CXR protocol")
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to processed metadata CSV (typically processed_metadata.csv)",
    )
    parser.add_argument(
        "--output",
        default="analysis/data_protocol_validation.txt",
        help="Path to save text report",
    )
    args = parser.parse_args()

    validate_protocol(args.metadata_csv, args.output)


if __name__ == "__main__":
    main()
