# MOE-RRG Reproduction

Reproduction of "Mixture of Experts for Radiology Report Generation" (MOE-RRG).

## Architecture Overview

```
Image (CXR) ──> Visual Encoder (RAD-DINO, frozen) ──> Patch Tokens [B, 49, 768]
                                                                       │
Indication + [IND] ──> CXR-BERT ──> ind_cls [B, 768] ──┐              │
Prior Report + [PRI] ──> CXR-BERT ──> pri_cls [B, 768] ──┤  Auxiliary  │
                                                        │    Gate     │
                                                        └──> Bias ──> + ──> LayerNorm ──> Multimodal Tokens
                                                                                         │
                                                                   ┌─────────────────────┤
                                                                   │                     │
                                                            SV-MoE (K=4)           HP-QFormer (P=32)
                                                          (stage/view routing)    (learnable queries)
                                                                   │                     │
                                                                   │              PrefixKV [B, 32, 512]
                                                                   │                     │
                                                                   └──── Encoder ────────┤
                                                                                         │
                                                                               Decoder (3-layer)
                                                                          (prefix injection into
                                                                           self-attn + cross-attn)
                                                                                         │
                                                                                   Report Text

Loss = L_CE + 0.1 * L_IMP + 0.2 * L_MOE
```

## Setup

### 1. Environment

```bash
conda create -n moe-rrg python=3.10
conda activate moe-rrg
pip install -r requirements.txt
```

### 2. Data Preparation

Place MIMIC-CXR data in the expected structure:
```
/data/mimic-cxr/
├── files/                          # JPEG images
│   └── p10/
│       └── p10000032/
│           └── s50314267/
│               └── 02aa804e-bde0afdd-112c0b34-7bc16630-4e381014.jpg
├── mimic-cxr-2.0.0-metadata.csv.gz # Official metadata
├── mimic_cxr_sectioned.csv         # Sectioned reports
└── mimic-cxr-2.0.0-split.csv.gz    # Official splits
```

### 3. Preprocessing

```bash
python -m data_pipeline.preprocessing \
    --metadata_csv /data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv.gz \
    --report_csv /data/mimic-cxr/mimic_cxr_sectioned.csv \
    --split_csv /data/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz \
    --output_dir /data/mimic-cxr/processed
```

Not: `train.py` ve `evaluate.py`, `data.processed_metadata_csv` bulunamazsa bu preprocessing adimini otomatik calistirir.

### 4. Sanity Checks

Before training, run sanity checks to verify all modules:

```bash
python sanity_checks.py
```

This verifies:
- Correct tensor shapes at every module boundary
- Routing distribution is not collapsed
- Prefix injection is active in all intended layers
- Loss gradients flow correctly
- One-batch overfitting works

### 5. Data Protocol Validation

Run this before training/evaluation to verify split leakage, stage/view distribution,
and prior-report consistency in processed metadata:

```bash
python validate_data_protocol.py \
    --metadata_csv /data/mimic-cxr/processed/processed_metadata.csv \
    --output analysis/data_protocol_validation.txt
```

### 6. Training

```bash
# Full model
python train.py --config CONFIG/config_base.yaml --gpu 0

# Ablation: no SV-MoE
python train.py --config CONFIG/config_ablation_no_svmoe.yaml --gpu 0

# Ablation: no HP-QFormer
python train.py --config CONFIG/config_ablation_no_hp_qf.yaml --gpu 0

# Ablation: no auxiliary cues
python train.py --config CONFIG/config_ablation_no_aux.yaml --gpu 0

# Resume from checkpoint
python train.py --config CONFIG/config_base.yaml --resume checkpoints/best_checkpoint.pt
```

### 7. Evaluation

```bash
# Full evaluation with routing analysis
python evaluate.py \
    --config CONFIG/config_base.yaml \
    --checkpoint checkpoints/best_checkpoint.pt \
    --split test \
    --output_dir analysis

# This produces:
#   - NLG metrics (BLEU-1/2/3/4, METEOR, ROUGE-L)
#   - Clinical metrics (CheXpert F1)
#   - Expert usage distribution
#   - Stage-wise and view-wise routing tables
#   - Stage-stratified and view-stratified performance

# Optional clinical backend (recommended for paper-comparable clinical metrics):
#   pip install chexpert-labeler scikit-learn
```

### Experimental HF Path (Non-Paper)

`train_hf.py` and `mimic_cxr_hf.py` are intended for quick smoke/iteration runs.
They are not paper-comparable because stage/view metadata is synthetic and prior/
indication cues are missing in that dataset route.

```bash
python train_hf.py --allow_experimental --config CONFIG/config_mac_m3.yaml --max_samples 256
```

### 8. Ablation Experiments

```bash
# Run all ablations
for ablation in no_svmoe no_hpqf no_aux stage_only no_token_weighting no_curriculum cmn_disabled prefix_early prefix_late prefix_sparse; do
    python train.py --config CONFIG/config_ablation_${ablation}.yaml --gpu 0
    python evaluate.py --config CONFIG/config_ablation_${ablation}.yaml \
        --checkpoint checkpoints/best_checkpoint.pt --split test \
        --output_dir analysis/ablation_${ablation}
done
```

## Key Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| K (experts) | 4 | Paper |
| P (queries) | 32 | Paper |
| lambda_moe | 0.2 | Paper |
| lambda_imp | 0.1 | Paper |
| tau | 0.07 | Paper |
| Epochs | 50 | Paper |
| CXR-BERT lr | 2e-5 | Paper |
| Transformer lr | 2e-4 | Paper |
| IMP branch lr | 5e-5 | Paper |
| Batch size | 16 | Assumed |
| Decoder layers | 3 | Paper |
| Decoder hidden | 512 | Paper |
| Q-Former layers | 4 | Paper |
| Q-Former hidden | 768 | Paper |

## Project Structure

```
├── CONFIG/                       # Experiment configurations
│   ├── config_base.yaml          # Full model
│   ├── config_mimic_cxr.yaml     # Dataset-specific overrides
│   └── config_ablation_*.yaml    # Ablation configs
├── models/
│   ├── visual_encoder.py         # microsoft/rad-dino (official public checkpoint)
│   ├── text_encoder.py           # microsoft/BiomedVLP-CXR-BERT-specialized
│   ├── auxiliary_gate.py         # Sigmoid-gated bias injection
│   ├── sv_moe.py                 # Stage-View Mixture of Experts
│   ├── hp_qformer.py             # Hierarchical Prefix Q-Former
│   ├── decoder_with_prefix.py    # Decoder with prefix injection
│   ├── cmn_memory.py             # Copy Memory Network (approx)
│   ├── losses.py                 # CE + Impression contrastive + MoE balance
│   └── model_factory.py          # Full model assembly
├── data_pipeline/
│   ├── preprocessing.py          # MIMIC-CXR metadata processing
│   ├── mimic_cxr_dataset.py      # Dataset class
│   └── data_collator.py          # Batch collation
├── utils/
│   ├── config.py                 # YAML config with inheritance
│   ├── logger.py                 # Logging setup
│   ├── metrics.py                # NLG and clinical metrics
│   └── checkpoint.py             # Checkpoint management
├── train.py                      # Training pipeline
├── evaluate.py                   # Evaluation pipeline
├── run_ablations.py              # Ablation runner
├── sanity_checks.py              # Implementation verification
├── docs/ASSUMPTIONS.md           # All nontrivial decisions documented
└── requirements.txt
```

## Approximations

See [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) for the full list. Key approximations:

1. **RAD-DINO**: We now use the public `microsoft/rad-dino` checkpoint. It is public, but Microsoft notes it differs from the paper checkpoint because some private data was used in the original release.
2. **CXR-BERT**: We now use the public `microsoft/BiomedVLP-CXR-BERT-specialized` checkpoint. This is the closest open checkpoint for the paper’s text encoder.
3. **CMN mechanism**: Minimal learned memory approximation
4. **Auxiliary gate MLP**: Dimensions inferred from context
5. **Token pooling method**: Adaptive average pooling (not specified in paper)

All approximations are tagged `[APPROXIMATE]` in source code.

## Compute Requirements

- **GPU**: 1x GPU with 24GB VRAM (e.g., RTX 3090, A5000)
- **Training**: ~24-48 hours for 50 epochs (estimated)
- **Evaluation**: ~2-4 hours on test set
