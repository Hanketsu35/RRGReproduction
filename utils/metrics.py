"""Evaluation metrics for MOE-RRG.

NLG metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
Clinical metrics: CheXpert labeler F1
"""

import re
import numpy as np
from collections import Counter


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def compute_nlg_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """Compute all NLG metrics.

    Args:
        hypotheses: Generated reports
        references: Ground truth reports

    Returns:
        Dictionary of metric scores
    """
    results = {}

    # BLEU scores
    for n in range(1, 5):
        results[f"bleu_{n}"] = compute_bleu(hypotheses, references, n)

    # ROUGE-L
    results["rouge_l"] = compute_rouge_l(hypotheses, references)

    # METEOR
    results["meteor"] = compute_meteor(hypotheses, references)

    return results


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for report text."""
    text = text.lower().strip()
    # Remove special tokens and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def compute_bleu(hypotheses: list[str], references: list[str],
                 n: int = 4) -> float:
    """Compute corpus-level BLEU-n score.

    Args:
        hypotheses: Generated reports
        references: Ground truth reports
        n: BLEU n-gram order

    Returns:
        BLEU-n score
    """
    from collections import Counter
    import math

    if len(hypotheses) == 0:
        return 0.0

    # Tokenize
    hyp_tokens = [tokenize(h) for h in hypotheses]
    ref_tokens = [tokenize(r) for r in references]

    # Compute brevity penalty
    hyp_lengths = [len(t) for t in hyp_tokens]
    ref_lengths = [len(t) for t in ref_tokens]

    total_hyp_len = sum(hyp_lengths)
    total_ref_len = sum(ref_lengths)

    if total_hyp_len == 0:
        return 0.0

    bp = min(1.0, math.exp(1 - total_ref_len / total_hyp_len)) if total_hyp_len < total_ref_len else 1.0

    # Compute n-gram precisions
    precisions = []
    for order in range(1, n + 1):
        total_clipped = 0
        total_count = 0

        for hyp, ref in zip(hyp_tokens, ref_tokens):
            hyp_ngrams = Counter([tuple(hyp[i:i+order]) for i in range(len(hyp) - order + 1)])
            ref_ngrams = Counter([tuple(ref[i:i+order]) for i in range(len(ref) - order + 1)])

            for ngram, count in hyp_ngrams.items():
                total_clipped += min(count, ref_ngrams.get(ngram, 0))
            total_count += sum(hyp_ngrams.values())

        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped / total_count)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / n
    bleu = bp * math.exp(log_avg)

    return bleu * 100  # Scale to 0-100


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level ROUGE-L F1 score.

    Args:
        hypotheses: Generated reports
        references: Ground truth reports

    Returns:
        ROUGE-L F1 score (0-100)
    """
    def lcs(x, y):
        """Longest common subsequence length."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    total_f1 = 0.0
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = tokenize(hyp)
        ref_tokens = tokenize(ref)

        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            continue

        lcs_len = lcs(hyp_tokens, ref_tokens)
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        total_f1 += f1

    return (total_f1 / len(hypotheses)) * 100 if hypotheses else 0.0


def compute_meteor(hypotheses: list[str], references: list[str]) -> float:
    """Compute simplified METEOR score.

    Args:
        hypotheses: Generated reports
        references: Ground truth reports

    Returns:
        METEOR score (0-100)
    """
    total_score = 0.0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = tokenize(hyp)
        ref_tokens = tokenize(ref)

        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            continue

        # Simple unigram matching
        hyp_counts = Counter(hyp_tokens)
        ref_counts = Counter(ref_tokens)

        matches = 0
        for word, count in hyp_counts.items():
            matches += min(count, ref_counts.get(word, 0))

        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens)

        if precision + recall > 0:
            f_mean = 10 * precision * recall / (9 * precision + recall)
        else:
            f_mean = 0.0

        # Simplified fragmentation penalty
        total_score += f_mean

    return (total_score / len(hypotheses)) * 100 if hypotheses else 0.0


def compute_clinical_metrics(hypotheses: list[str], references: list[str],
                              labeler=None) -> dict:
    """Compute clinical efficacy metrics using CheXpert labeler.

    Args:
        hypotheses: Generated reports
        references: Ground truth reports
        labeler: CheXpert labeler instance (optional)

    Returns:
        Dictionary with precision, recall, F1 per label
    """
    def _to_binary(array_like):
        arr = np.asarray(array_like)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = np.nan_to_num(arr, nan=0.0)
        # Many clinical labelers encode uncertain as -1; map to absent.
        arr = (arr > 0).astype(np.int32)
        return arr

    def _extract_with_chexpert_labeler(texts):
        # Preferred backend: chexpert-labeler package.
        from chexpert_labeler import Labeler  # type: ignore
        active_labeler = labeler if labeler is not None else Labeler()
        raw = active_labeler.label(texts)

        if hasattr(raw, "to_numpy"):
            # Pandas DataFrame path
            df = raw
            if hasattr(df, "columns"):
                label_cols = [c for c in CHEXPERT_LABELS if c in df.columns]
                if label_cols:
                    return _to_binary(df[label_cols].to_numpy()), label_cols, "chexpert_labeler:dataframe"
            return _to_binary(df.to_numpy()), CHEXPERT_LABELS, "chexpert_labeler:dataframe"

        # Numpy/list path
        arr = _to_binary(raw)
        return arr, CHEXPERT_LABELS, "chexpert_labeler:array"

    try:
        hyp_labels, label_names, backend = _extract_with_chexpert_labeler(hypotheses)
        ref_labels, _, _ = _extract_with_chexpert_labeler(references)
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "none",
            "error": str(exc),
            "chexpert_macro_f1": None,
            "chexpert_micro_f1": None,
            "chexpert_precision": None,
            "chexpert_recall": None,
            "per_label_f1": {},
        }

    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": backend,
            "error": f"scikit-learn is required for metric aggregation: {exc}",
            "chexpert_macro_f1": None,
            "chexpert_micro_f1": None,
            "chexpert_precision": None,
            "chexpert_recall": None,
            "per_label_f1": {},
        }

    cols = min(hyp_labels.shape[1], ref_labels.shape[1], len(label_names))
    hyp = hyp_labels[:, :cols]
    ref = ref_labels[:, :cols]
    used_labels = label_names[:cols]

    per_label_f1 = {}
    for i, label in enumerate(used_labels):
        per_label_f1[label] = float(f1_score(ref[:, i], hyp[:, i], zero_division=0))

    return {
        "status": "ok",
        "backend": backend,
        "chexpert_macro_f1": float(f1_score(ref, hyp, average="macro", zero_division=0)),
        "chexpert_micro_f1": float(f1_score(ref, hyp, average="micro", zero_division=0)),
        "chexpert_precision": float(precision_score(ref, hyp, average="micro", zero_division=0)),
        "chexpert_recall": float(recall_score(ref, hyp, average="micro", zero_division=0)),
        "per_label_f1": per_label_f1,
    }
