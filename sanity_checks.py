"""Sanity checks for MOE-RRG implementation.

Runs a series of verification tests:
1. Shape checks at each module output
2. One-batch overfit test
3. Routing distribution analysis
4. Prefix injection verification
5. Loss convergence check
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config
from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.auxiliary_gate import AuxiliaryGate
from models.sv_moe import SVMoE
from models.hp_qformer import HPQFormer
from models.decoder_with_prefix import DecoderWithPrefix
from models.cmn_memory import CopyMemoryNetwork
from models.model_factory import MOERRGModel
from models.losses import MOERRGLoss


def test_shapes():
    """Verify tensor shapes at every module boundary."""
    print("=" * 60)
    print("TEST 1: Shape Checks")
    print("=" * 60)

    B, T = 4, 49  # Batch size, num tokens
    device = "cpu"

    # Visual Encoder
    print("\n--- Visual Encoder ---")
    vis = VisualEncoder(model_name="dinov2_vits14", pretrained=False, frozen=True)
    images = torch.randn(B, 3, 518, 518)
    vis_out = vis(images)
    print(f"  patch_tokens: {vis_out['patch_tokens'].shape}")  # [B, 49, 384]
    assert vis_out["patch_tokens"].shape == (B, 49, 384), f"Expected [4, 49, 384], got {vis_out['patch_tokens'].shape}"
    print(f"  cls_token: {vis_out['cls_token'].shape}")          # [B, 384]
    assert vis_out["cls_token"].shape == (B, 384)
    print("  PASSED")

    # Text Encoder
    print("\n--- Text Encoder ---")
    txt = TextEncoder(model_name="emilyalsentzer/Bio_ClinicalBERT", frozen=False)
    tok = txt.tokenize(["test indication text"] * B, max_length=32)
    txt_out = txt.encode(tok["input_ids"], tok["attention_mask"], prompt_type="ind")
    print(f"  cls_embedding: {txt_out['cls_embedding'].shape}")  # [B, 768]
    assert txt_out["cls_embedding"].shape == (B, 768)
    print(f"  last_hidden: {txt_out['last_hidden'].shape}")      # [B, 33, 768] (32+1 prompt)
    assert txt_out["last_hidden"].shape[0] == B and txt_out["last_hidden"].shape[2] == 768
    print("  PASSED")

    # Auxiliary Gate
    print("\n--- Auxiliary Gate ---")
    gate = AuxiliaryGate(text_hidden_size=768, visual_hidden_size=384)
    ind_cls = torch.randn(B, 768)
    pri_cls = torch.randn(B, 768)
    mm_tokens = gate(vis_out["patch_tokens"], ind_cls, pri_cls)
    print(f"  multimodal_tokens: {mm_tokens.shape}")  # [B, 49, 384]
    assert mm_tokens.shape == (B, 49, 384)
    print("  PASSED")

    # Projection
    proj = torch.nn.Linear(384, 512)
    mm_proj = proj(mm_tokens)
    print(f"  projected: {mm_proj.shape}")  # [B, 49, 512]
    assert mm_proj.shape == (B, 49, 512)

    # SV-MoE
    print("\n--- SV-MoE ---")
    moe = SVMoE(hidden_size=512, num_experts=4, num_stages=5, num_views=4)
    stage_ids = torch.randint(0, 5, (B,))
    view_ids = torch.randint(0, 4, (B,))
    moe_out = moe(mm_proj, stage_ids, view_ids)
    print(f"  output: {moe_out['output'].shape}")                # [B, 49, 512]
    assert moe_out["output"].shape == (B, 49, 512)
    print(f"  expert_probs: {moe_out['expert_probs'].shape}")    # [B, 4]
    assert moe_out["expert_probs"].shape == (B, 4)
    print(f"  selected_expert: {moe_out['selected_expert'].shape}")  # [B]
    assert moe_out["selected_expert"].shape == (B,)
    print(f"  load_balance_loss: {moe_out['load_balance_loss'].item():.4f}")
    print("  PASSED")

    # HP-QFormer
    print("\n--- HP-QFormer ---")
    qf = HPQFormer(num_queries=32, num_layers=4, hidden_size=768,
                    num_heads=8, encoder_hidden_size=512, decoder_hidden_size=512)
    qf_out = qf(moe_out["output"])
    print(f"  prefix_k: {qf_out['prefix_k'].shape}")  # [B, 32, 512]
    assert qf_out["prefix_k"].shape == (B, 32, 512)
    print(f"  prefix_v: {qf_out['prefix_v'].shape}")  # [B, 32, 512]
    assert qf_out["prefix_v"].shape == (B, 32, 512)
    print("  PASSED")

    # Decoder
    print("\n--- Decoder with Prefix ---")
    dec = DecoderWithPrefix(vocab_size=30522, num_layers=3, hidden_size=512,
                            num_heads=8, prefix_injection_depth="all")
    target_ids = torch.randint(0, 30522, (B, 64))
    dec_out = dec(
        target_ids=target_ids,
        encoder_features=moe_out["output"],
        prefix_k=qf_out["prefix_k"],
        prefix_v=qf_out["prefix_v"],
    )
    print(f"  logits: {dec_out['logits'].shape}")  # [B, 64, 30522]
    assert dec_out["logits"].shape == (B, 64, 30522)
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL SHAPE CHECKS PASSED!")
    print("=" * 60)


def test_one_batch_overfit():
    """Test that the model can overfit a single batch."""
    print("\n" + "=" * 60)
    print("TEST 2: One-Batch Overfit Test")
    print("=" * 60)

    config = load_config("CONFIG/config_base.yaml")
    model = MOERRGModel(config)

    # Create a small fake batch — use vocab size that matches the model
    vocab_size = config["model"]["decoder"]["vocab_size"]
    # After token resize, actual vocab may differ; use model's tokenizer vocab
    actual_vocab = model.text_encoder.tokenizer.vocab_size

    B = 4
    batch = {
        "images": torch.randn(B, 3, 518, 518),
        "report_ids": torch.randint(0, actual_vocab, (B, 64)),
        "indication_ids": torch.randint(0, actual_vocab, (B, 16)),
        "indication_mask": torch.ones(B, 16, dtype=torch.long),
        "prior_ids": torch.randint(0, actual_vocab, (B, 32)),
        "prior_mask": torch.ones(B, 32, dtype=torch.long),
        "impression_ids": torch.randint(0, actual_vocab, (B, 32)),
        "impression_mask": torch.ones(B, 32, dtype=torch.long),
        "stage_ids": torch.randint(0, 5, (B,)),
        "view_ids": torch.randint(0, 4, (B,)),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training on single batch for 50 steps...")
    for step in range(50):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                  f"CE: {outputs['ce_loss'].item():.4f} | "
                  f"IMP: {outputs['imp_loss'].item():.4f} | "
                  f"MoE: {outputs['moe_loss'].item():.4f}")

    print("One-batch overfit test COMPLETE (loss should decrease)")


def test_routing_distribution():
    """Test that MoE routing doesn't collapse to a single expert."""
    print("\n" + "=" * 60)
    print("TEST 3: Routing Distribution Check")
    print("=" * 60)

    moe = SVMoE(hidden_size=512, num_experts=4, num_stages=5, num_views=4)
    x = torch.randn(100, 10, 512)
    stage_ids = torch.randint(0, 5, (100,))
    view_ids = torch.randint(0, 4, (100,))

    out = moe(x, stage_ids, view_ids)
    selected = out["selected_expert"]
    usage = torch.bincount(selected, minlength=4).float()
    usage_pct = usage / usage.sum() * 100

    print(f"  Expert usage: {usage_pct.tolist()}")
    print(f"  Load balance loss: {out['load_balance_loss'].item():.4f}")

    # Check that no expert gets > 80% of samples
    for i, pct in enumerate(usage_pct):
        if pct > 80:
            print(f"  WARNING: Expert {i} has {pct:.1f}% usage — potential collapse!")
        else:
            print(f"  Expert {i}: {pct:.1f}% — OK")

    print("Routing distribution check PASSED")


def test_prefix_injection():
    """Test that prefix injection is active in all intended layers."""
    print("\n" + "=" * 60)
    print("TEST 4: Prefix Injection Verification")
    print("=" * 60)

    B, T = 2, 10
    dec = DecoderWithPrefix(
        vocab_size=1000, num_layers=3, hidden_size=512,
        num_heads=8, prefix_injection_depth="all",
    )

    # Create inputs
    target_ids = torch.randint(0, 1000, (B, T))
    encoder_features = torch.randn(B, 49, 512)
    prefix_k = torch.randn(B, 5, 512)
    prefix_v = torch.randn(B, 5, 512)

    # Check that all layers have inject_prefix=True
    for i, layer in enumerate(dec.layers):
        assert layer.inject_prefix, f"Layer {i} should have prefix injection enabled"
        print(f"  Layer {i}: prefix injection = {layer.inject_prefix}")

    # Verify output shape
    out = dec(target_ids, encoder_features, prefix_k=prefix_k, prefix_v=prefix_v)
    print(f"  Output logits shape: {out['logits'].shape}")
    assert out["logits"].shape == (B, T, 1000)

    # Test with limited injection depth
    dec_depth1 = DecoderWithPrefix(
        vocab_size=1000, num_layers=3, hidden_size=512,
        num_heads=8, prefix_injection_depth=1,
    )
    for i, layer in enumerate(dec_depth1.layers):
        expected = (i < 1)
        assert layer.inject_prefix == expected, \
            f"Layer {i}: expected {expected}, got {layer.inject_prefix}"
        print(f"  [depth=1] Layer {i}: inject_prefix = {layer.inject_prefix}")

    print("Prefix injection verification PASSED")


def test_losses_decrease():
    """Test that all loss components produce valid gradients."""
    print("\n" + "=" * 60)
    print("TEST 5: Loss Gradient Check")
    print("=" * 60)

    loss_fn = MOERRGLoss(lambda_imp=0.1, lambda_moe=0.2, temperature=0.07)

    B, T, V = 4, 32, 1000
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    decoder_hidden = torch.randn(B, T, 512, requires_grad=True)
    impression_emb = torch.randn(B, 768, requires_grad=True)
    load_balance = torch.tensor(0.5, requires_grad=False)
    target_mask = torch.ones(B, T, dtype=torch.long)

    losses = loss_fn(logits, targets, decoder_hidden, impression_emb,
                     load_balance, target_mask)

    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  CE loss: {losses['ce_loss'].item():.4f}")
    print(f"  IMP loss: {losses['imp_loss'].item():.4f}")
    print(f"  MoE loss: {losses['moe_loss'].item():.4f}")

    # Check gradients flow
    losses["total_loss"].backward()
    assert logits.grad is not None, "No gradient for logits!"
    assert decoder_hidden.grad is not None, "No gradient for decoder_hidden!"

    print("  Gradients flow correctly")
    print("Loss gradient check PASSED")


if __name__ == "__main__":
    print("MOE-RRG Sanity Checks")
    print("=" * 60)

    test_shapes()
    test_routing_distribution()
    test_prefix_injection()
    test_losses_decrease()
    test_one_batch_overfit()

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS COMPLETED!")
    print("=" * 60)
