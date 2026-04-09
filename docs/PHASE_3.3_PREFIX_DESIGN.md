# Phase 3.3: Prefix Depth Ablation - Investigation & Design

**Status**: Investigation & Design
**Objective**: Analyze optimal depth for prefix KV injection into decoder layers
**Rationale**: Paper uses "all layers" but doesn't justify - test alternatives

---

## Problem Statement

The HP-QFormer extracts prefix KV that get injected into decoder's self-attention and cross-attention layers. Current approach:

```
prefix_k, prefix_v → [prepended to all decoder layer K/V]
```

**Questions**:
1. Do we need prefix injection at ALL layers or just early/late layers?
2. What's the optimal injection depth?
3. Is there a sweet spot that balances expressiveness with regularization?

**Motivation**: 
- Early layers: Capture general multimodal fusion (might need prefix)
- Late layers: Refine clinical details (might not need prefix guidance)
- Optimal: Find the right depth for best BLEU vs. parameters

---

## Design Space

### Option A: All Layers (Current / Baseline)
```
Decoder Layer 0: prefix_k, prefix_v injected
Decoder Layer 1: prefix_k, prefix_v injected
Decoder Layer 2: prefix_k, prefix_v injected
Decoder Layer 3: prefix_k, prefix_v injected

Coverage: 100%
```

### Option B: Early Layers Only
```
Decoder Layer 0: prefix_k, prefix_v injected
Decoder Layer 1: prefix_k, prefix_v injected
Decoder Layer 2: NO injection
Decoder Layer 3: NO injection

Coverage: 50% (layers 0-1)
```

### Option C: Late Layers Only
```
Decoder Layer 0: NO injection
Decoder Layer 1: NO injection
Decoder Layer 2: prefix_k, prefix_v injected
Decoder Layer 3: prefix_k, prefix_v injected

Coverage: 50% (layers 2-3)
```

### Option D: Sparse (Every Other)
```
Decoder Layer 0: prefix_k, prefix_v injected
Decoder Layer 1: NO injection
Decoder Layer 2: prefix_k, prefix_v injected
Decoder Layer 3: NO injection

Coverage: 50% (alternating)
```

---

## Analysis: Which Layers Matter?

### Layer Functions in Transformer Decoder

**Layer 0-1 (Early)**: 
- Task: Learn basic token representations + multimodal fusion
- Input: Embedded tokens + encoder features
- Role: Foundation layer, heavily influenced by input
- Hypothesis: **BENEFITS from prefix** (guide early fusion)

**Layer 2-3 (Late)**:
- Task: Refine predictions based on context
- Input: Representations from layer 1
- Role: Polish and specialize
- Hypothesis: **May not need prefix** (already refined by early layers)

**Alternative Hypothesis**:
- Late layers more important (closer to output logits)
- Demonstrate learned representations

---

## Recommendation: Test 3 Main Variants

### Variant 1: All (A) - Baseline
- **Depth**: inject at layers [0, 1, 2, 3]
- **Expected BLEU**: 40.5 (with Phase 3.1/3.2 stacked)
- **Parameters**: ~full

### Variant 2: Early (B) - Simpler
- **Depth**: inject at layers [0, 1] only
- **Expected BLEU**: 40.0-40.3 (slight drop, regularization)
- **Parameters**: ~lower (prefix doesn't propagate through layers 2-3)
- **Hypothesis**: Early multimodal fusion sufficient

### Variant 3: Late (C) - Focused
- **Depth**: inject at layers [2, 3] only
- **Expected BLEU**: 39.5-40.0 (likely worse, missing early guidance)
- **Hypothesis**: Early guidance crucial for setup
- **Value**: Negative result helps confirm importance

---

## Expected Results

```
BLEU Scores (Isolated Ablation):

Configuration        │ BLEU  │ Δ from All │ Status
---------------------|-------|-----------|--------
All layers (0-3)     │ 40.5  │ —         │ Best
Early only (0-1)     │ 40.1  │ -0.4      │ Slight drop
Late only (2-3)      │ 39.7  │ -0.8      │ Larger drop
Random (0,2)         │ 39.9  │ -0.6      │ Moderate drop

Conclusion: All layers likely optimal, but early layers critical
```

---

## Implementation Strategy

### Configuration-Driven Design

Add to `CONFIG/config_base.yaml`:

```yaml
model:
  hp_qformer:
    prefix_injection_depth: "all"    # "all" | "early" | "late" | "sparse"
```

Then in `models/decoder_with_prefix.py`:

```python
class DecoderWithPrefix(nn.Module):
    def __init__(self, ..., prefix_injection_depth="all"):
        ...
        self.inject_at_layers = self._parse_injection_depth(prefix_injection_depth)
    
    def _parse_injection_depth(self, depth_spec):
        if depth_spec == "all":
            return [0, 1, 2, 3]
        elif depth_spec == "early":
            return [0, 1]
        elif depth_spec == "late":
            return [2, 3]
        elif depth_spec == "sparse":
            return [0, 2]
        else:
            return [0, 1, 2, 3]  # default
    
    def forward(self, ..., prefix_k=None, prefix_v=None):
        for layer_idx, layer in enumerate(self.decoder_layers):
            if layer_idx in self.inject_at_layers:
                # Inject prefix
                ...
            else:
                # No prefix injection for this layer
                ...
```

---

## Ablation Configs (3 variants)

### `config_ablation_prefix_early.yaml`
```yaml
hp_qformer:
  prefix_injection_depth: "early"  # Layers 0-1 only
```

### `config_ablation_prefix_late.yaml`
```yaml
hp_qformer:
  prefix_injection_depth: "late"   # Layers 2-3 only
```

### `config_ablation_prefix_sparse.yaml`
```yaml
hp_qformer:
  prefix_injection_depth: "sparse" # Layers 0,2 only
```

Base config keeps: `prefix_injection_depth: "all"`

---

## Expected Implementation Effort

| Task | Effort | Notes |
|------|--------|-------|
| Update DecoderWithPrefix | 1h | Add `_parse_injection_depth()` logic |
| Config sections | 0.5h | Add 3 new ablation configs |
| Documentation | 0.5h | Explain design + expected results |
| **Total** | **2h** | Straightforward changes |

---

## Why This Matters

1. **Ablation completeness**: More thorough understanding of architecture
2. **Parameter efficiency**: Might achieve same BLEU with fewer prefix steps
3. **Insights**: Which layers actually use prefix information?
4. **Paper alignment**: Can justify ALL layers vs alternatives

---

## Expected Outcome

This phase will likely **NOT** improve BLEU (all layers probably optimal) but will:
- ✅ Provide evidence that all layers ARE necessary
- ✅ Quantify the cost of removing prefix at various depths
- ✅ Complete the ablation study for final report
- ✅ Answer "why all layers?" with data

**Most likely result**: All layers > Early > Sparse > Late (confidence: 75%)

---

## Implementation (Next)

Ready to implement if time permits (2 hours estimated)
