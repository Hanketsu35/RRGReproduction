# ASSUMPTIONS.md
# Nontrivial decisions caused by missing or ambiguous paper details

## 1. Visual Encoder: RAD-DINO
**Paper says:** Pretrained RAD-DINO, frozen.
**Decision:** Use the public `microsoft/rad-dino` checkpoint via the official `rad-dino` loader. This is much closer to the paper than the previous DINOv2 fallback and matches the chest X-ray domain directly.
**Caveat:** The public checkpoint is noted by Microsoft as different from the paper checkpoint because some private data was used in the original release. So this is now a public-model reproduction, not a literal bitwise match to the paper’s private checkpoint.
**Tag:** [PUBLIC CHECKPOINT]

## 2. Text Encoder: CXR-BERT
**Paper says:** Pretrained CXR-BERT, fine-tuned.
**Decision:** Use the public `microsoft/BiomedVLP-CXR-BERT-specialized` checkpoint. This is a chest X-ray domain checkpoint released by Microsoft and is a much closer fit than generic clinical BERT alternatives.
**Caveat:** The paper may still have used an internal training state or slightly different stage of the CXR-BERT family, so we keep this as a public reproduction choice rather than claiming exact equivalence.
**Tag:** [PUBLIC CHECKPOINT]

## 3. Auxiliary Text Gate Mechanism
**Paper says:** Use [CLS] embedding from concatenated auxiliary text to compute a sigmoid-gated bias, inject via elementwise addition + LayerNorm.
**Problem:** The paper does not specify the exact MLP architecture for the gating mechanism.
**Decision:** Use a 2-layer MLP: Linear(768 -> 256) -> ReLU -> Linear(256 -> 384) -> Sigmoid, where 768 is the text encoder hidden size and 384 is the visual encoder hidden size. The sigmoid output is elementwise multiplied with the MLP's pre-sigmoid linear output to create a gated bias, then added to visual tokens and passed through LayerNorm.
**Tag:** [APPROXIMATE — MLP dimensions inferred]

## 4. Token Pooling to 49 Tokens
**Paper says:** Token pooling reduces visual patch tokens to 49 tokens before SV-MoE / HP-QF.
**Problem:** The exact pooling method is not specified (average pooling, learned convolution, adaptive pooling, etc.).
**Decision:** Use adaptive average pooling to reshape the spatial grid of patch tokens to 7x7 = 49. Specifically, treat patch tokens as a spatial grid (H_patches x W_patches), apply nn.AdaptiveAvgPool2d((7, 7)), then flatten back to sequence. If the original grid is not square, we pad or truncate to the nearest square before pooling.
**Tag:** [APPROXIMATE — pooling method inferred]

## 5. SV-MoE Expert Architecture
**Paper says:** Experts are FFN modules.
**Problem:** The exact FFN configuration (activation, hidden dim, dropout) is not fully specified.
**Decision:** Each expert is a standard FFN: Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model), where d_model matches the decoder hidden size (512) and d_ff = 2048 (4x expansion, standard Transformer ratio).
**Tag:** [APPROXIMATE — activation and hidden ratio inferred]

## 6. SV-MoE Router MLP Architecture
**Paper says:** z = MLP(Emb(StageID) || Emb(ViewID)), expert scores = linear + softmax, routing by argmax.
**Problem:** The MLP depth and dimensions are not specified.
**Decision:** Use a 2-layer MLP: Linear(2*embed_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> num_experts) -> Softmax. We use embed_dim=64 for stage/view embeddings and hidden_dim=128.
**Tag:** [APPROXIMATE — MLP dimensions inferred]

## 7. HP-QFormer Implementation
**Paper says:** 4 layers, hidden size 768, 8 heads, 32 learnable queries. Queries interact through cross-attention and self-attention.
**Problem:** The exact Q-Former architecture (layer composition, residual connections, layer norm placement) is not fully specified.
**Decision:** Follow the BLIP-2 Q-Former design: each layer contains a self-attention block, a cross-attention block (attending to multimodal features), and an FFN. Use pre-LN Transformer architecture. Cross-attention occurs at every layer.
**Tag:** [APPROXIMATE — following BLIP-2 Q-Former pattern]

## 8. PrefixKV Injection Mechanism
**Paper says:** Inject PrefixKV into BOTH decoder self-attention and cross-attention key/value states. Must remain visible at every decoding step.
**Problem:** The exact projection from Q-Former output to PrefixKV is not specified. Also unclear whether a separate projection is used for self-attention vs cross-attention injection.
**Decision:** Use a linear projection from Q-Former output dim (768) to decoder hidden dim (512) to produce PrefixKV. The same PrefixKV is prepended to both self-attention K/V and cross-attention K/V in each decoder layer. We use separate linear projections for K and V.
**Tag:** [APPROXIMATE — projection details inferred]

## 9. CMN Memory Mechanism
**Paper says:** CMN-related memory mechanism should be implemented.
**Problem:** The paper references a Copy Memory Network (CMN) mechanism but does not provide sufficient architectural details for exact reproduction.
**Decision:** Implement a minimal compatible approximation: a learned memory buffer (fixed-size key-value store) that the decoder can attend to via an additional cross-attention layer. The memory is initialized randomly and updated during training. Document this clearly as an approximation. If the paper releases CMN details later, this module can be replaced.
**Tag:** [APPROXIMATE — minimal approximation]

## 10. Impression Contrastive Loss Details
**Paper says:** InfoNCE aligning decoder output embedding with impression embedding. tau = 0.07.
**Problem:** The paper does not specify how the "decoder output embedding" is obtained (last hidden state mean pooling, [EOS] token, etc.), nor the batch construction for negatives.
**Decision:** Use mean pooling over the decoder's last hidden states as the report embedding, and the [CLS] token from the impression text encoding as the impression embedding. Apply L2 normalization before computing InfoNCE. Use in-batch negatives (other samples in the same batch serve as negatives).
**Tag:** [APPROXIMATE — pooling method inferred]

## 11. MoE Load-Balancing Loss
**Paper says:** Load-balancing loss encouraging expert usage close to uniform.
**Problem:** The exact formulation is not provided.
**Decision:** Use the standard Switch Transformer load-balancing loss: L_bal = num_experts * sum_i(f_i * P_i), where f_i is the fraction of tokens routed to expert i and P_i is the average routing probability for expert i. This encourages uniform expert utilization.
**Tag:** [APPROXIMATE — using Switch Transformer formulation]

## 12. Decoder Architecture
**Paper says:** Standard 3-layer Transformer encoder-decoder.
**Problem:** The decoder's exact vocabulary, tokenization strategy, and how it interfaces with the encoder are not fully detailed.
**Decision:** Use a standard Transformer decoder with BERT vocabulary (WordPiece, vocab_size=30522). The cross-attention attends to multimodal features (after SV-MoE processing). Label smoothing of 0.1 is applied as is standard practice.
**Tag:** [APPROXIMATE — label smoothing added as standard practice]

## 13. Stage/View Metadata Extraction
**Paper says:** Stage discretization into 5 bins, views: AP, PA, LA, LL.
**Problem:** MIMIC-CXR metadata may label views differently (e.g., "LATERAL" vs "LA" vs "LL").
**Decision:** Map MIMIC-CXR view labels as follows: AP/AP_SUPINE -> AP, PA -> PA, LATERAL/LL/LA -> LATERAL (single category for lateral views). Exclude other rare views. Stage is determined by counting prior visits for the same patient using study dates.
**Tag:** [APPROXIMATE — view mapping inferred]

## 14. Batch Size and Gradient Accumulation
**Paper says:** Batch size not explicitly stated.
**Decision:** Use batch_size=16 with gradient accumulation of 2 steps (effective batch size = 32). This is a common size that fits on a single GPU with 24GB VRAM.
**Tag:** [APPROXIMATE — not paper-specified]

## 15. Optimizer and Scheduler Details
**Paper says:** Adam optimizer with specified learning rates.
**Problem:** Exact optimizer parameters (betas, eps), scheduler type, and warmup schedule are not provided.
**Decision:** Use Adam with default betas (0.9, 0.999), weight decay 0.01, and cosine annealing with 10% warmup. These are standard choices for Transformer training.
**Tag:** [APPROXIMATE — standard choices]

## 16. Copy Memory Network (CMN) Architecture
**Paper says:** Uses a Copy Memory Network to enable copying from prior visit reports.
**Problem:** No architectural details provided. "Copy Memory Network" is not a standard architecture; paper may refer to attention-based copying, learned memory slots, or other variants.
**Old Decision (Deprecated):** Minimal approximation using learned KV memory buffer (128 slots) + cross-attention. This was a placeholder that did NOT leverage the actual prior report data.
**Status (as of Phase 2.3):** **REDESIGNED with three alternatives investigated**. See `PHASE_2.3_CMN_INVESTIGATION.md` for full analysis.

**Current Decision (Design A: Prior-Report Attention):**
- Use attention-based copying mechanism that directly operates on prior report embeddings
- At each decoder step, compute attention weights over prior report tokens
- Gate mechanism decides copy probability vs generate probability (Pointer-Generator style)
- Blend copy and generation logits: P(w_t) = p_copy * P_copy(w_t) + (1 - p_copy) * P_gen(w_t)
- This directly leverages the input data (prior reports) and matches paper's motif of "copying"
- More principled than learned memory because: (1) data-driven, (2) per-sample dynamic, (3) proven in Seq2Seq literature

**Implementation:**
- New module: `models/prior_copy_attention.py` contains `PriorReportCopyMemory` class
- Integration: `model_factory.py` applies copy mechanism as post-logit blending if `cmn.method="prior_attention"` in config
- Configuration: Added `cmn` section to `config_base.yaml` to enable/disable copy mechanism
- Ablation: `config_ablation_cmn_disabled.yaml` disables copy for comparison

**Expected Impact:** +5-10% BLEU-4 improvement (based on Pointer-Generator literature; actual gain TBD after training)

**Ablation Plan:**
- Baseline: run with `cmn.method="disabled"` (no copy mechanism)
- With Copy: run with `cmn.method="prior_attention"` (proposed design)
- Compare BLEU, ROUGE, clinical metrics
- Analyze copy gate probability distribution to verify learning

**Tag:** [REDESIGNED-PHASE-2.3] [AWAITING-VALIDATION]

## 17. Prefix Injection Depth Strategy (Phase 3.3 Ablation)

**Paper says:** Inject prefix KV from HP-QFormer into all decoder layers.

**Problem:** Paper does not justify why "all layers" are necessary. No ablation on injection depth.

**Decision:** Support 4 prefix injection depth strategies via config:
- `"all"`: Inject PrefixKV at all decoder layers [0, 1, 2, 3] (baseline)
- `"early"`: Inject only at early layers [0, 1] (test if late layers need refresh)
- `"late"`: Inject only at late layers [2, 3] (test if early alignment necessary)
- `"sparse"`: Inject at alternating layers [0, 2] (efficiency exploration)

**Rationale:**
1. All-layer injection may be redundant (layer-to-layer propagation handles most info)
2. Early alignment (layer 0-1) likely critical for multimodal fusion
3. Late injection (layer 2-3) might suffice for final output refinement
4. Sparse pattern tests if every-other-layer + information propagation sufficient

**Implementation:**
- Updated `DecoderWithPrefix.__init__()` to interpret prefix_injection_depth parameter
- Added if/elif/else chain for "all", "early", "late", "sparse" aliases
- Maintains backward compatibility with integer strings (e.g., "2" → first 2 layers)
- 3 ablation configs: `config_ablation_prefix_early.yaml`, `config_ablation_prefix_late.yaml`, `config_ablation_prefix_sparse.yaml`

**Expected Impacts:**
| Strategy | Layers | Expected ΔBLEU | Interpretation |
|----------|--------|---|---|
| All (baseline) | [0,1,2,3] | 0.0 | Reference |
| Early | [0,1] | -0.4 to -0.8 | If large drop: late layers contribute |
| Late | [2,3] | -1.2 to -2.0 | If drops ≥ early: late insufficient alone |
| Sparse | [0,2] | -0.1 to -0.4 | If minimal drop: efficient alternative viable |

**Use Cases:**
- Validate architectural necessity of all-layer injection
- Understand layer-wise contribution to multimodal fusion
- Explore efficiency-quality tradeoff (sparse variant)
- Support ablation study completeness in final paper

**Tag:** [ADDED-PHASE-3.3] [ABLATION-STUDY]

