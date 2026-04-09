# Documentation Index

This directory contains implementation documentation for the MOE-RRG reproduction project using extended Phase 2.3-3.3 improvements.

## Core Documentation

- **ASSUMPTIONS.md** — Design decisions and architectural choices (all phases)
- **REPRODUCTION_REPORT.md** — Final project results and findings
- **README.md** — Project overview and setup instructions

## Implementation Phases

### Phase 2.3: CMN Prior-Report Attention
- **PHASE_2.3_CMN_INVESTIGATION.md** — Design alternatives and rationale
- **Implementation**: `models/prior_copy_attention.py` (200 lines)
- **Config**: `CONFIG/config_ablation_cmn_disabled.yaml`

### Phase 2.4: Data Split Validation
- **PHASE_2.4_DATA_VALIDATION_PLAN.md** — Execution guide and 7-dimension validation spec
- **Implementation**: `validate_data_splits_comprehensive.py` (500 lines)

### Phase 3.1: Token Position Weighting
- **PHASE_3.1_TOKEN_POOLING_INVESTIGATION.md** — Design analysis (3 approaches compared)
- **PHASE_3.1_IMPLEMENTATION.md** — Technical guide and integration points
- **Implementation**: `models/token_position_weighting.py` (240 lines)
- **Config**: `CONFIG/config_ablation_no_token_weighting.yaml`

### Phase 3.2: Curriculum Learning
- **PHASE_3.2_CURRICULUM_DESIGN.md** — Design rationale and alternatives
- **PHASE_3.2_IMPLEMENTATION_SUMMARY.md** — Technical details and expected improvements
- **Implementation**: `models/curriculum_scheduler.py` (330 lines)
- **Config**: `CONFIG/config_ablation_no_curriculum.yaml`

### Phase 3.3: Prefix Injection Depth Ablation
- **PHASE_3.3_PREFIX_DESIGN.md** — 4 variants and expected outcomes
- **PHASE_3.3_IMPLEMENTATION_SUMMARY.md** — Ablation study design and execution plan
- **Implementation**: `models/decoder_with_prefix.py` (updated +35 lines)
- **Configs**: 
  - `CONFIG/config_ablation_prefix_early.yaml`
  - `CONFIG/config_ablation_prefix_late.yaml`
  - `CONFIG/config_ablation_prefix_sparse.yaml`

## Quick Start

### Data Validation (Before Training)
```bash
python validate_data_splits_comprehensive.py
```

### Training Commands
```bash
# Baseline (all phases enabled)
python train.py --config CONFIG/config_base.yaml --gpu 0

# Ablations
python train.py --config CONFIG/config_ablation_cmn_disabled.yaml --gpu 0
python train.py --config CONFIG/config_ablation_no_token_weighting.yaml --gpu 0
python train.py --config CONFIG/config_ablation_no_curriculum.yaml --gpu 0
python train.py --config CONFIG/config_ablation_prefix_early.yaml --gpu 0
python train.py --config CONFIG/config_ablation_prefix_late.yaml --gpu 0
python train.py --config CONFIG/config_ablation_prefix_sparse.yaml --gpu 0
```

### Experimental Smoke Path (Non-Paper)
```bash
python train_hf.py --allow_experimental --config CONFIG/config_mac_m3.yaml --max_samples 256
```

## Expected Results

| Phase | Feature | Expected ΔBLEU |
|-------|---------|---|
| 2.3 | CMN Prior Attention | +5 to +10 |
| 3.1 | Token Weighting | +1 to +3 |
| 3.2 | Curriculum Learning | +2 to +5 |
| 3.3 | Prefix Ablation | -0.1 to 0 (diagnostic) |
| **Cumulative** | **All phases** | **+8 to +23** |

## File Organization

```
.
├── README.md
├── docs/
│   ├── ASSUMPTIONS.md
│   ├── REPRODUCTION_REPORT.md
│   ├── DOCUMENTATION_INDEX.md (this file)
│   ├── PHASE_2.3_CMN_INVESTIGATION.md
│   ├── PHASE_2.4_DATA_VALIDATION_PLAN.md
│   ├── PHASE_3.1_TOKEN_POOLING_INVESTIGATION.md
│   ├── PHASE_3.1_IMPLEMENTATION.md
│   ├── PHASE_3.2_CURRICULUM_DESIGN.md
│   ├── PHASE_3.2_IMPLEMENTATION_SUMMARY.md
│   ├── PHASE_3.3_PREFIX_DESIGN.md
│   ├── PHASE_3.3_IMPLEMENTATION_SUMMARY.md
│   └── memory.md
├── models/
│   ├── prior_copy_attention.py
│   ├── token_position_weighting.py
│   ├── curriculum_scheduler.py
│   └── decoder_with_prefix.py (updated)
├── validate_data_splits_comprehensive.py
├── CONFIG/
│   ├── config_base.yaml
│   ├── config_ablation_cmn_disabled.yaml
│   ├── config_ablation_no_token_weighting.yaml
│   ├── config_ablation_no_curriculum.yaml
│   ├── config_ablation_prefix_early.yaml
│   ├── config_ablation_prefix_late.yaml
│   └── config_ablation_prefix_sparse.yaml
└── (other project files)
```

## Notes

- All code is syntax-validated and ready for training
- Configs follow consistent structure for easy ablation studies
- Each phase can be enabled/disabled independently via config
- Base config has all phases enabled by default
- See individual phase docs for technical details
