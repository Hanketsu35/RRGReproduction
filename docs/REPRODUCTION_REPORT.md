# MOE-RRG Reproduction Report

## Summary

This document tracks the reproduction status of "Mixture of Experts for Radiology Report Generation" (MOE-RRG).

## Implemented Modules

### Exact Implementation (faithful to paper)
| Module | Status | Notes |
|--------|--------|-------|
| Stage discretization | Done | 5 stages: 0-4 as specified |
| View categorization | Done | AP, PA, LATERAL, OTHER |
| SV-MoE routing (argmax, K=4) | Done | Single-expert routing by argmax |
| Expert FFN modules | Done | Standard FFN per expert |
| HP-QFormer (4 layers, 768d, 8 heads, 32 queries) | Done | Self-attn + cross-attn per layer |
| PrefixKV injection into self-attn + cross-attn | Done | Prepended to K/V at every step |
| Decoder (3-layer, 512d, 8 heads) | Done | Standard Transformer decoder |
| Cross-entropy loss | Done | With label smoothing |
| Impression contrastive loss (InfoNCE) | Done | In-batch negatives, tau=0.07 |
| MoE load-balancing loss | Done | Switch Transformer formulation |
| Differential learning rates | Done | 3 LR groups as specified |
| Token pooling to 49 tokens | Done | Adaptive avg pool to 7x7 grid |

### Approximate Implementation
| Module | Status | Approximation | Rationale |
|--------|--------|---------------|-----------|
| Visual encoder | Done | microsoft/rad-dino | Public RAD-DINO checkpoint |
| Text encoder | Done | microsoft/BiomedVLP-CXR-BERT-specialized | Public CXR-BERT checkpoint |
| Auxiliary gate MLP | Done | 2-layer MLP (768->256->384) | Dimensions inferred |
| Token pooling method | Done | AdaptiveAvgPool2d(7,7) | Method not specified |
| CMN memory | Done | Prior-report attention copy + optional static cache | Paper lacks exact CMN details |
| Impression pooling | Done | Mean pooling over hidden states | Method not specified |
| Routing MLP | Done | 2-layer (128->emb, 64->embed) | Dimensions inferred |
| PrefixKV projection | Done | Separate K/V linear projections | Shared vs separate not specified |

## Ablation Matrix

| Ablation | Config | Status |
|----------|--------|--------|
| w/o SV-MoE | config_ablation_no_svmoe.yaml | Ready |
| w/o HP-QF | config_ablation_no_hp_qf.yaml | Ready |
| w/o Auxiliary cues | config_ablation_no_aux.yaml | Ready |
| Stage-only routing | config_ablation_stage_only.yaml | Ready |
| w/o Token Weighting | config_ablation_no_token_weighting.yaml | Ready |
| w/o Curriculum | config_ablation_no_curriculum.yaml | Ready |
| CMN Disabled | config_ablation_cmn_disabled.yaml | Ready |
| Prefix Early | config_ablation_prefix_early.yaml | Ready |
| Prefix Late | config_ablation_prefix_late.yaml | Ready |
| Prefix Sparse | config_ablation_prefix_sparse.yaml | Ready |

## Evaluation Protocol Notes

- Official paper-comparable path: `train.py` + `evaluate.py` with processed MIMIC metadata.
- `evaluate.py` now exports decoding settings and comparability metadata in output JSON.
- NLG backend is custom internal implementation; treat as partial comparability unless aligned to official toolkit.
- Clinical metrics are marked `not_comparable` when CheXpert backend is unavailable.

## Experimental Paths (Excluded from Paper Tables)

- `train_hf.py` and `mimic_cxr_hf.py` are experimental/smoke-only paths.
- HF path uses synthetic stage/view metadata and empty prior/indication cues.
- Results from HF path should not be used in paper-side metric comparison.

## Metric Results

_Results will be populated after training and evaluation._

### Overall NLG Metrics (MIMIC-CXR Test)

| Metric | This Reproduction | Paper Reported |
|--------|-------------------|----------------|
| BLEU-1 | — | — |
| BLEU-2 | — | — |
| BLEU-3 | — | — |
| BLEU-4 | — | — |
| METEOR | — | — |
| ROUGE-L | — | — |

### Clinical Efficacy (CheXpert F1)

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Overall | — | — | — |

### Routing Analysis

| Expert | Usage % | Assigned Stages/Views |
|--------|---------|----------------------|
| 0 | — | — |
| 1 | — | — |
| 2 | — | — |
| 3 | — | — |

### Stratified Performance

| Stage | BLEU-4 | ROUGE-L | METEOR |
|-------|--------|---------|--------|
| 0 (first visit) | — | — | — |
| 1 (2nd visit) | — | — | — |
| 2 (visits 3-5) | — | — | — |
| 3 (visits 6-10) | — | — | — |
| 4 (visits >10) | — | — | — |

## Failure Cases

_To be documented after evaluation._

## Compute Requirements

| Resource | Value |
|----------|-------|
| GPU | 1x 24GB VRAM |
| Training time (est.) | 24-48 hours |
| Evaluation time (est.) | 2-4 hours |
| Peak memory | ~18 GB |
| Total parameters | ~85M (approx) |
| Trainable parameters | ~65M (approx) |

## Known Limitations

1. **Public checkpoints**: `microsoft/rad-dino` and `microsoft/BiomedVLP-CXR-BERT-specialized` may still differ from internal paper checkpoints.
2. **CMN exactness**: Prior-attention copy is paper-near but still an inferred implementation (no exact CMN spec published).
3. **NLG comparability**: BLEU/METEOR/ROUGE implementations are internal; direct parity with official toolkit is not guaranteed.
4. **Clinical backend dependency**: CheXpert-based clinical metrics require `chexpert-labeler` + `scikit-learn` to be installed.
5. **Long-run evidence pending**: Final metric tables remain empty until full training/evaluation completes.

## Sanity Check Results

Run `python sanity_checks.py` to verify:
- [x] Shape checks: All module outputs have correct dimensions
- [x] Routing distribution: Experts receive diverse assignments
- [x] Prefix injection: Active in all intended decoder layers
- [x] Loss gradients: All components produce valid gradients
- [x] One-batch overfit: Loss decreases on a single batch
