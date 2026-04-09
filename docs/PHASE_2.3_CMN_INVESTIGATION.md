# CMN Memory Architecture: Deep Investigation & Redesign

**Date:** April 9, 2026  
**Phase:** 2.3 (High-Impact Fixes)  
**Status:** INVESTIGATION COMPLETE → READY FOR IMPLEMENTATION  
**Effort Est:** 6-8 hours  
**Risk:** MEDIUM (architectural redesign)  
**Expected Impact:** +5-15% BLEU-4 (depends on design choice)

---

## EXECUTIVE SUMMARY

The original MOE-RRG paper references a "Copy Memory Network (CMN)" mechanism but provides **no architectural details**. Our current implementation uses a minimal approximation:

**Current State (APPROXIMATE):**
- Learned key-value memory buffer (fixed 128 slots)
- Cross-attention from decoder to memory
- Static memory → no dynamic interaction with prior reports

**Problem:**
- Unknown if current CMN contributes 5% or 50% of reproduction gap
- "Copy" suggests prior-report reuse, but learned memory doesn't leverage priors
- Paper likely used more sophisticated copying/attention mechanism

**Investigation Outcome:**
Three design alternatives identified with trade-offs:

| Design | Fidelity | Effort | Risk | Expected Gain |
|--------|----------|--------|------|---------------|
| **A: Prior-Report Attention** | HIGH (exploits paper's input) | MEDIUM | LOW | +5-10% |
| **B: Learned Cache + Gating** | MEDIUM (infers copying) | MEDIUM | MEDIUM | +3-8% |
| **C: T5-style Prefix Tuning** | MEDIUM (proven method) | LOW | VERY-LOW | +2-6% |

**RECOMMENDATION:** Start with **Design A (Prior-Report Attention)** ← highest fidelity, leverages actual data.

---

## BACKGROUND: WHAT IS A COPY MEMORY NETWORK?

### Definition
A "Copy Mechanism" allows a sequence model to **copy tokens directly from input** (or a memory bank) rather than generating from scratch. Classic examples:

1. **Pointer-Generator Networks** (See et al., 2017)
   - Learns probability of copying vs generating at each step
   - Attends to source tokens; if copy_prob > 0.5, copy attended token
   
2. **Hierarchical Attention** (with memory)
   - Maintains auxiliary memory (learned embeddings or cached inputs)
   - Decoder can attend to memory as alternative to source sequence

3. **Retrieval-Augmented Generation** (modern variant)
   - Retrieve similar examples from corpus
   - Condition generation on retrieved text

### In MOE-RRG Paper Context
The paper states (abstract/methods): "...leveraging prior visit context via Copy Memory Network..."

**Inference:**
- Clinical reports are **repetitive** (same findings across visits)
- Prior visit reports are **available & provided to decoder**
- CMN should **enable reuse** of prior text fragments
- Example: If prior impression = "mild left pleural effusion", and conditions are similar, model can **copy** relevant phrases

---

## CURRENT IMPLEMENTATION: CRITIQUE

### Code
**File:** `models/cmn_memory.py` (89 lines)

```python
class CopyMemoryNetwork(nn.Module):
    def __init__(self, hidden_size=512, memory_size=128, memory_dim=256, ...):
        # Learned memory keys/values (NOT derived from data)
        self.memory_keys = nn.Parameter(torch.randn(128, 256) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(128, 256) * 0.02)
        
    def forward(self, decoder_hidden):
        # Static cross-attention to learned memory
        memory_out, _ = self.memory_attn(
            query=decoder_hidden,
            key=self.memory_keys,
            value=self.memory_values,
        )
        return residual + memory_out
```

### Critique

| Aspect | Current | Problem | Paper Likely |
|--------|---------|---------|--------------|
| **Memory Source** | Random init params | No connection to actual data  | Prior reports / learned patterns |
| **Dynamics** | Static, learned once | Same memory for all samples | Dynamic, sample-specific |
| **Mechanism** | Cross-attention residual | Additive, not explicitly copying | Gate-weighted copy probability |
| **Copy Probability** | Implicit in attention weights | No explicit yes/no signal | Gating mechanism for copy vs generate |
| **Fidelity** | ~30-40% | Doesn't leverage prior context | ~70-80% if prior-conditioned |

---

## DESIGN ALTERNATIVE A: PRIOR-REPORT ATTENTION ⭐ RECOMMENDED

### Rationale
- **Highest data fidelity:** Directly uses prior reports provided in input
- **Lowest guessing:** Leverages what paper explicitly says (prior context)
- **Proven:** Copy mechanisms work well in seq2seq literature
- **Risk:** LOW (standard attention, no novel components)

### Architecture

```python
class PriorReportCopyMemory(nn.Module):
    """Attention-based copying from prior report text.
    
    For each decoding step, compute:
    1. Attention over prior report tokens
    2. Gating probability (copy vs generate)
    3. If copy: select token from prior
       If generate: use decoder vocabulary
    
    This directly implements a copy mechanism over available prior context.
    """
    
    def __init__(self, hidden_size=512, vocab_size=30522, dropout=0.1):
        super().__init__()
        
        # 1. Copy attention (attend to prior report)
        self.copy_attn = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # 2. Gating (decide copy vs generate)
        self.copy_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # [0, 1] → copy probability
        )
        
        # 3. Copy vocabulary reduction (optional: map prior tokens to vocab)
        self.copy_vocab_size = vocab_size
    
    def forward(self, 
                decoder_hidden: torch.Tensor,      # [B, T, D]
                prior_report_emb: torch.Tensor,    # [B, P, D] (prior report embeddings)
                prior_report_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_hidden: Decoder hidden states at current timestep [B, 1, D]
            prior_report_emb: Encoded prior report [B, P, D]
            prior_report_tokens: Prior report token IDs [B, P]
        
        Returns:
            output: [B, 1, D] (either from copy attention or generation)
            copy_probs: [B, 1] (probability of copying)
            ... copy logits for vocabulary ...
        """
        B, T, D = decoder_hidden.size()
        
        # Step 1: Attend over prior report
        copy_context, attn_weights = self.copy_attn(
            query=decoder_hidden,  # [B, 1, D]
            key=prior_report_emb,  # [B, P, D]
            value=prior_report_emb,  # [B, P, D]
            return_attention_weights=True,
        )  # output: [B, 1, D], attn_weights: [B, 1, P]
        
        # Step 2: Compute copy gate
        combined = torch.cat([decoder_hidden, copy_context], dim=-1)
        copy_prob = self.copy_gate(combined)  # [B, 1, 1]
        
        # Step 3: Copy logits
        # Simple: select tokens with higher attention weight
        copy_logits = attn_weights.squeeze(1)  # [B, P]
        
        # Expand to vocabulary: copy if in prior, else 0
        # (This is a design choice; could also use pointer-generator style)
        
        return {
            "copy_context": copy_context,
            "copy_prob": copy_prob,
            "copy_logits": copy_logits,  # [B, P] → distribute to vocab
            "attn_weights": attn_weights,
        }
```

### Integration into Decoder

```python
class DecoderWithPrefixAndCopy(nn.Module):
    def __init__(self, ...):
        # ... existing decoder layers ...
        self.copy_memory = PriorReportCopyMemory(hidden_size, vocab_size)
    
    def forward(self, 
                input_ids, prefix_k=None, prefix_v=None,
                prior_report_emb=None, prior_report_tokens=None):
        # ... existing decoder logic ...
        hidden = self.decode_layer(hidden)
        
        # Add copy mechanism output
        if prior_report_emb is not None:
            copy_out = self.copy_memory(
                hidden,
                prior_report_emb,
                prior_report_tokens,
            )
            # Blend copy and generation logits
            logits = self._blend_copy_generate_logits(
                logits,
                copy_out,
                prior_report_tokens,
            )
        
        return logits
    
    def _blend_copy_generate_logits(self, gen_logits, copy_out, prior_tokens):
        """Pointer-generator style: blend copy and generation.
        
        P(w_t) = copy_prob * P_copy(w_t) + (1-copy_prob) * P_gen(w_t)
        """
        copy_prob = copy_out["copy_prob"]  # [B, T, 1]
        copy_logits = copy_out["copy_logits"]  # [B, T, P]
        
        # Expand copy logits to vocabulary
        copy_logits_full = torch.zeros_like(gen_logits)  # [B, T, V]
        prior_tokens = prior_tokens.unsqueeze(1).expand_as(copy_logits)  # [B, T, P]
        
        # Scatter copy attention weights to vocab indices
        for b, t in enumerate(range(gen_logits.size(1))):
            for p, token_id in enumerate(prior_tokens[b, t]):
                if token_id < self.vocab_size:
                    copy_logits_full[b, t, token_id] += copy_logits[b, t, p]
        
        # Blend
        blended = copy_prob * copy_logits_full + (1 - copy_prob) * torch.softmax(gen_logits, dim=-1)
        return blended
```

### Pros
- ✅ Directly uses prior reports (data already available)
- ✅ Per-sample dynamic (not static learned memory)
- ✅ Well-motivated: explains why priors are input to model
- ✅ Standard copy mechanism (proven in Seq2Seq literature)
- ✅ Can ablate: compare with/without copy gating

### Cons
- ❌ Requires prior_report_emb at decoding time (inference change)
- ❌ More complex numerics (pointer-generator blending)
- ❌ Unknown if paper actually used this vs simpler variant

---

## DESIGN ALTERNATIVE B: LEARNED CACHE WITH GATING

### Rationale
- Compromise: keeps learned memory (paper may have done this)
- But adds **gating** (explicit copy vs generate decision)
- Middle ground between current state and Alternative A

### Architecture

```python
class LearnedCacheMemory(nn.Module):
    """Learned memory slots with explicit copy gating."""
    
    def __init__(self, hidden_size=512, cache_size=256, dropout=0.1):
        super().__init__()
        
        # Learned memory (more slots than current; maybe 256 instead of 128)
        self.cache_keys = nn.Parameter(torch.randn(cache_size, hidden_size) * 0.02)
        self.cache_values = nn.Parameter(torch.randn(cache_size, hidden_size) * 0.02)
        
        # Attention to memory
        self.memory_attn = MultiheadAttention(hidden_size, 8, dropout, batch_first=True)
        
        # Gate: copy vs generate
        self.copy_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, decoder_hidden, return_gate=False):
        # Attend to memory
        cache_context, attn = self.memory_attn(
            query=decoder_hidden,
            key=self.cache_keys.unsqueeze(0),
            value=self.cache_values.unsqueeze(0),
            return_attention_weights=True,
        )
        
        # Gating
        combined = torch.cat([decoder_hidden, cache_context], dim=-1)
        copy_prob = self.copy_gate(combined)
        
        if return_gate:
            return cache_context, copy_prob, attn
        return cache_context * copy_prob  # Only use cache if gate > 0.5
```

### Pros
- ✅ Simpler than Alternative A (no blending logic)
- ✅ Learned memory can capture common patterns
- ✅ Explicit gating (more interpretable)
- ✅ No inference API change

### Cons
- ❌ Still doesn't use actual prior reports
- ❌ Unclear how paper initialized/trained this memory
- ❌ More arbitrary (why 256 slots? how many?)

---

## DESIGN ALTERNATIVE C: T5-STYLE PREFIX TUNING

### Rationale
- Simplest: just learn more prefix tokens from prior reports
- Similar to what HP-QFormer already does
- Proven: T5 prefix tuning is well-established

### Architecture

```python
class PriorPrefixAdapter(nn.Module):
    """Adapt prior reports into decoder prefix."""
    
    def __init__(self, hidden_size=512, num_prefix_tokens=16):
        super().__init__()
        
        # Learn projection from prior embedding to more prefix tokens
        self.prior_to_prefix = nn.Linear(hidden_size, hidden_size * num_prefix_tokens)
        self.num_prefix = num_prefix_tokens
    
    def forward(self, prior_emb):
        # prior_emb: [B, D]
        prefix = self.prior_to_prefix(prior_emb)  # [B, D*num_prefix]
        prefix = prefix.reshape(prefix.size(0), self.num_prefix, -1)  # [B, num_prefix, D]
        return prefix
```

### Pros
- ✅ Simplest to implement (1 linear layer)
- ✅ Minimal code change (adds to existing prefix)
- ✅ Proven effective in literature
- ✅ Easy to ablate

### Cons
- ❌ Doesn't add "copy" capability (just more context)
- ❌ Least novel fidelity (most approximate to "copy memory")
- ❌ Less direct connection to paper's intent

---

## COMPARISON TABLE

| Criterion | A: Prior-Report Attn | B: Learned Cache | C: Prefix Tuning |
|-----------|----------------------|------------------|------------------|
| **Fidelity to "Copy"** | HIGH (explicit copy) | MEDIUM (implicit) | LOW (just context) |
| **Data Exploitation** | HIGH (uses priors) | MEDIUM (ignores priors) | HIGH (uses priors) |
| **Implementation Effort** | 6-8 hours | 4-5 hours | 2-3 hours |
| **Code Complexity** | HIGH | MEDIUM | VERY-LOW |
| **Expected BLEU Gain** | +5-10% | +3-8% | +2-6% |
| **Inference Cost** | +15-20% (prior embed) | Negligible | Negligible |
| **Ablation Ease** | EASY (on/off gate) | EASY (on/off cache) | EASY (on/off prefix) |
| **Risk of Instability** | MEDIUM (numerics) | LOW | VERY-LOW |
| **Paper Alignment** | HIGH (uses priors) | MEDIUM (guesses) | MEDIUM (guesses) |

---

## RECOMMENDATION & IMPLEMENTATION PLAN

### CHOSEN DESIGN: **A (Prior-Report Attention)** ⭐

**Rationale:**
1. Highest fidelity: Directly leverages paper's input (prior reports)
2. Explicit copy mechanism: Matches paper's term "Copy Memory Network"
3. Data-driven: No random initialization or guessing
4. Ablatable: Can test copy gate on/off
5. Compatible: Can run in parallel with existing modules

### Phased Implementation

**Phase 2.3a: Core Implementation (4-5 hours)**
- [ ] Implement `PriorReportCopyMemory` in new file `models/prior_copy_attention.py`
- [ ] Integrate into `models/decoder_with_prefix.py`
- [ ] Update model_factory to handle prior_report_emb at inference
- [ ] Create config option `cmn.method: "prior_attention"` (vs "learned_cache" or "disabled")

**Phase 2.3b: Integration & Testing (2-3 hours)**
- [ ] Update `evaluate.py` to pass prior report embeddings to model.generate()
- [ ] Update `train.py` to pass priors during forward pass
- [ ] Add smoke tests (prior_attn forward pass doesn't crash)
- [ ] Add ablation config `config_ablation_cmn_disabled.yaml` for comparison

**Phase 2.3c: Evaluation (Post-Run)**
- [ ] Compare BLEU/ROUGE with/without copy gating
- [ ] Analyze copy gate probability distribution (is gate learning meaningful split?)
- [ ] Document in REPRODUCTION_REPORT.md

---

## ASSUMPTIONS & UNKNOWNS

**Assumption 1**: Paper uses some form of copy mechanism  
→ **Validation**: BLEU improves with Prior-Report Attention

**Assumption 2**: Copy mechanism operates on decoder-visible prior reports  
→ **Validation**: Design A exploits input data; if not used, model ignores it

**Assumption 3**: Gating (copy vs generate) is learned dynamically  
→ **Validation**: Copy gate learns non-trivial distribution (not collapsed to 0 or 1)

**Assumption 4**: Prior report tokens are embedded before copy attention  
→ **This is YOUR design choice**: Use existing text_encoder to embed priors

---

## NEXT STEPS

1. **Confirm decision**: Is Design A acceptable? (Or prefer B or C?)
2. **Implement Phase 2.3a**: Create `prior_copy_attention.py`
3. **Integrate Phase 2.3b**: Wire into decoder, train.py, evaluate.py
4. **Run ablation**: BLEU with/without prior copy
5. **Document**: Update ASSUMPTIONS.md with design choice & findings

---

## REFERENCES & BACKGROUND

- Pointer-Generator Networks: https://arxiv.org/abs/1704.04368  
- MOE-RRG Paper: (check for citing copy mechanism papers in appendix)
- BLIP-2 (our HP-QFormer template): https://arxiv.org/abs/2301.12597

---

**Author Notes:** This investigation assumes MOE-RRG paper mentions CMN but gives no details. If you have access to paper appendix or supplementary material, search for:
- "copy" or "memory" keyword
- Cited papers: "pointer-generator", "hierarchical attention", "retrieval"
- Architecture diagram or equations for CMN

If found, adjust recommendation accordingly.
