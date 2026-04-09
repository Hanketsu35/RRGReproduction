"""MOE-RRG Model Factory — assembles all modules into a single model.

Architecture flow:
    1. Visual Encoder (frozen RAD-DINO) -> patch tokens [B, 49, Dv]
    2. Text Encoder (BiomedVLP-CXR-BERT-specialized) encodes:
       - Indication with [IND] prompt -> ind_cls [B, 768]
       - Prior report with [PRI] prompt -> pri_cls [B, 768]
       - Impression -> impression_cls [B, 768]
    3. Auxiliary Gate: sigmoid-gated bias injection into visual tokens
    -> multimodal_tokens [B, 49, Dv]
    4. Project to decoder dim: multimodal_tokens -> [B, 49, 512]
    5. SV-MoE: route multimodal_tokens through stage-view experts
       -> moe_output [B, 49, 512]
    6. HP-QFormer: Q-Former extracts prefix KV from multimodal features
       -> prefix_k, prefix_v [B, 32, 512]
    7. Decoder with Prefix: generate report with prefix injection
       -> logits [B, T, V]
    8. Loss: L_CE + lambda_imp * L_IMP + lambda_moe * L_MOE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder
from .auxiliary_gate import AuxiliaryGate
from .sv_moe import SVMoE
from .hp_qformer import HPQFormer
from .decoder_with_prefix import DecoderWithPrefix
from .cmn_memory import CopyMemoryNetwork
from .prior_copy_attention import PriorReportCopyMemory
from .losses import MOERRGLoss


class MOERRGModel(nn.Module):
    """Full MOE-RRG model for radiology report generation.

    Args:
        config: Full configuration dictionary
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Extract sub-configs
        vis_cfg = config["model"]["visual_encoder"]
        txt_cfg = config["model"]["text_encoder"]
        aux_cfg = config["model"].get("auxiliary", {})
        moe_cfg = config["model"]["sv_moe"]
        qf_cfg = config["model"]["hp_qformer"]
        dec_cfg = config["model"]["decoder"]

        # ─── 1. Visual Encoder (frozen) ───
        self.visual_encoder = VisualEncoder(
            model_name=vis_cfg["name"],
            pretrained=vis_cfg["pretrained"],
            frozen=vis_cfg["frozen"],
            output_tokens=vis_cfg["output_tokens"],
        )
        vis_hidden = self.visual_encoder.hidden_size

        # ─── 2. Text Encoder (fine-tuned) ───
        self.text_encoder = TextEncoder(
            model_name=txt_cfg["name"],
            hidden_size=txt_cfg["hidden_size"],
            frozen=txt_cfg.get("frozen", False),
            max_length=txt_cfg["max_length"],
        )
        txt_hidden = txt_cfg["hidden_size"]  # 768

        # ─── 3. Auxiliary Gate ───
        self.auxiliary_enabled = aux_cfg.get("enabled", True)
        if self.auxiliary_enabled:
            self.auxiliary_gate = AuxiliaryGate(
                text_hidden_size=txt_hidden,
                visual_hidden_size=vis_hidden,
                gate_hidden_dim=aux_cfg.get("gate_hidden_dim", 256),
            )

        # ─── Projection to decoder hidden size ───
        dec_hidden = dec_cfg["hidden_size"]  # 512
        self.vis_proj = nn.Linear(vis_hidden, dec_hidden)

        # ─── 4. SV-MoE ───
        self.svmoe_enabled = moe_cfg.get("enabled", True)
        if self.svmoe_enabled:
            self.sv_moe = SVMoE(
                hidden_size=dec_hidden,
                num_experts=moe_cfg["num_experts"],
                num_stages=moe_cfg["num_stages"],
                num_views=moe_cfg["num_views"],
                embed_dim=moe_cfg["embed_dim"],
                router_hidden_dim=moe_cfg["router_hidden_dim"],
                expert_hidden_dim=moe_cfg["expert_hidden_dim"],
                dropout=moe_cfg["dropout"],
            )

        # ─── 5. HP-QFormer ───
        self.hpqf_enabled = qf_cfg.get("enabled", True)
        if self.hpqf_enabled:
            self.hp_qformer = HPQFormer(
                num_queries=qf_cfg["num_queries"],
                num_layers=qf_cfg["num_layers"],
                hidden_size=qf_cfg["hidden_size"],
                num_heads=qf_cfg["num_heads"],
                encoder_hidden_size=dec_hidden,
                decoder_hidden_size=dec_hidden,
                prefix_injection_depth=qf_cfg["prefix_injection_depth"],
                dropout=qf_cfg["dropout"],
            )

        # ─── 6. CMN/COPY configuration ───
        cmn_cfg = config["model"].get("cmn", {})
        self.cmn_method = cmn_cfg.get("method", "disabled")
        self.use_static_cmn_cache = cmn_cfg.get(
            "use_static_cache", self.cmn_method == "learned_cache"
        )

        # Optional static CMN cache (legacy/minimal path)
        self.cmn = CopyMemoryNetwork(hidden_size=dec_hidden) if self.use_static_cmn_cache else None

        # Prior Report Copy Mechanism (paper-near path)
        if self.cmn_method == "prior_attention":
            # Projection from text encoder hidden (768) to decoder hidden (512) if needed
            txt_hidden = txt_cfg["hidden_size"]
            if txt_hidden != dec_hidden:
                self.prior_proj = nn.Linear(txt_hidden, dec_hidden)
            else:
                self.prior_proj = None
            
            self.prior_copy = PriorReportCopyMemory(
                hidden_size=dec_hidden,
                vocab_size=dec_cfg["vocab_size"],
                num_heads=cmn_cfg.get("num_heads", 8),
                dropout=cmn_cfg.get("dropout", 0.1),
            )
        else:
            self.prior_proj = None
            self.prior_copy = None

        # ─── 7. Decoder with Prefix ───
        self.decoder = DecoderWithPrefix(
            vocab_size=dec_cfg["vocab_size"],
            num_layers=dec_cfg["num_layers"],
            hidden_size=dec_hidden,
            num_heads=dec_cfg["num_heads"],
            ffn_dim=dec_cfg["ffn_dim"],
            dropout=dec_cfg["dropout"],
            max_length=dec_cfg["max_length"],
            pad_token_id=dec_cfg["pad_token_id"],
            bos_token_id=dec_cfg["bos_token_id"],
            eos_token_id=dec_cfg["eos_token_id"],
            prefix_injection_depth=qf_cfg.get("prefix_injection_depth", "all"),
            inject_prefix_into_self=dec_cfg.get("inject_prefix_into_self", True),
        )

        # ─── 8. Loss ───
        train_cfg = config["training"]
        self.loss_fn = MOERRGLoss(
            lambda_imp=train_cfg["lambda_imp"],
            lambda_moe=train_cfg["lambda_moe"],
            temperature=train_cfg["tau"],
            ignore_index=dec_cfg["pad_token_id"],
            token_weight_cfg=config["model"].get("token_weighting", {}),
            curriculum_cfg=config["model"].get("curriculum_learning", {}),
            max_epochs=train_cfg.get("epochs", 50),
        )

        # Store routing info for analysis
        self.last_routing_info = {}

    def _get_cmn_kv(self, batch_size: int, device: torch.device):
        """Return optional static CMN memory KV tensors."""
        if self.cmn is None:
            return None, None
        return self.cmn.memory_kv(batch_size=batch_size, device=device)

    def _prepare_prior_embeddings(self, pri_out: dict):
        """Prepare prior token embeddings for copy mechanism."""
        if self.prior_copy is None or "last_hidden" not in pri_out:
            return None

        # Remove prepended prompt token to align with prior_input_ids.
        prior_embeddings = pri_out["last_hidden"][:, 1:, :]
        if self.prior_proj is not None:
            prior_embeddings = self.prior_proj(prior_embeddings)
        return prior_embeddings

    def _apply_prior_copy(self, logits: torch.Tensor, hidden_states: torch.Tensor,
                          prior_embeddings: torch.Tensor, prior_input_ids: torch.Tensor) -> torch.Tensor:
        """Apply prior-report copy blending to generation logits."""
        if self.prior_copy is None or prior_embeddings is None or prior_input_ids is None:
            return logits

        copy_out = self.prior_copy(
            decoder_hidden=hidden_states,
            prior_report_emb=prior_embeddings,
            prior_report_tokens=prior_input_ids,
        )
        return PriorReportCopyMemory.blend_logits(
            gen_logits=logits,
            copy_prob=copy_out["copy_prob"],
            attn_weights=copy_out["attn_weights"],
            prior_tokens=prior_input_ids,
            vocab_size=self.decoder.vocab_size,
        )

    def forward(self, batch: dict, epoch: int = 0) -> dict:
        """Full forward pass.

        Expected batch keys:
            - images: [B, 3, H, W] chest X-ray images
            - report_ids: [B, T] target report token IDs
            - indication_ids: [B, L_ind] indication token IDs
            - indication_mask: [B, L_ind] indication attention mask
            - prior_ids: [B, L_pri] prior report token IDs
            - prior_mask: [B, L_pri] prior attention mask
            - impression_ids: [B, L_imp] impression token IDs
            - impression_mask: [B, L_imp] impression attention mask
            - stage_ids: [B] visit stage IDs
            - view_ids: [B] imaging view IDs

        Returns:
            Dictionary with losses, logits, and routing info
        """
        # ─── Visual Features ───
        vis_out = self.visual_encoder(batch["images"])
        patch_tokens = vis_out["patch_tokens"]

        # ─── Text Features ───
        # Encode indication with [IND] prompt
        ind_out = self.text_encoder.encode(
            batch["indication_ids"], batch["indication_mask"],
            prompt_type="ind",
        )

        # Encode prior report with [PRI] prompt
        pri_out = self.text_encoder.encode(
            batch["prior_ids"], batch["prior_mask"],
            prompt_type="pri",
        )

        # Encode impression (no prompt, just [CLS] embedding)
        imp_out = self.text_encoder.encode(
            batch["impression_ids"], batch["impression_mask"],
            prompt_type=None,
        )

        # ─── Auxiliary Gate ───
        if self.auxiliary_enabled:
            multimodal_tokens = self.auxiliary_gate(
                patch_tokens,
                ind_out["cls_embedding"],
                pri_out["cls_embedding"],
            )
        else:
            multimodal_tokens = patch_tokens

        # Project to decoder dimension
        multimodal_tokens = self.vis_proj(multimodal_tokens)  # [B, 49, 512]

        # ─── SV-MoE ───
        load_balance_loss = torch.tensor(0.0, device=patch_tokens.device)
        if self.svmoe_enabled:
            moe_out = self.sv_moe(
                multimodal_tokens,
                batch["stage_ids"],
                batch["view_ids"],
            )
            encoder_features = moe_out["output"]  # [B, 49, 512]
            load_balance_loss = moe_out["load_balance_loss"]

            # Store routing info
            self.last_routing_info = {
                "expert_probs": moe_out["expert_probs"].detach(),
                "selected_expert": moe_out["selected_expert"].detach(),
                "stage_ids": batch["stage_ids"].detach(),
                "view_ids": batch["view_ids"].detach(),
            }
        else:
            encoder_features = multimodal_tokens

        # ─── HP-QFormer ───
        prefix_k, prefix_v = None, None
        if self.hpqf_enabled:
            qf_out = self.hp_qformer(encoder_features)
            prefix_k = qf_out["prefix_k"]  # [B, 32, 512]
            prefix_v = qf_out["prefix_v"]  # [B, 32, 512]

        # ─── Optional CMN Memory ───
        cmn_k_self, cmn_v_self = self._get_cmn_kv(
            batch_size=encoder_features.size(0),
            device=encoder_features.device,
        )

        # ─── Decoder ───
        # Teacher forcing: input is target[:-1], target is shifted
        target_input = batch["report_ids"][:, :-1]
        target_output = batch["report_ids"][:, 1:]

        dec_out = self.decoder(
            target_ids=target_input,
            encoder_features=encoder_features,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            cmn_k_self=cmn_k_self,
            cmn_v_self=cmn_v_self,
        )

        logits = dec_out["logits"]            # [B, T-1, V]
        hidden_states = dec_out["hidden_states"]  # [B, T-1, D]

        # ─── Optional: Apply Prior Copy Mechanism ───
        prior_input_ids = batch.get("prior_ids")
        prior_embeddings = self._prepare_prior_embeddings(pri_out)
        logits = self._apply_prior_copy(
            logits=logits,
            hidden_states=hidden_states,
            prior_embeddings=prior_embeddings,
            prior_input_ids=prior_input_ids,
        )

        # ─── Loss ───
        # Create target mask for contrastive loss pooling
        target_mask = (target_input != self.config["model"]["decoder"]["pad_token_id"])

        losses = self.loss_fn(
            logits=logits,
            targets=target_output,
            decoder_hidden=hidden_states,
            impression_embedding=imp_out["cls_embedding"],
            load_balance_loss=load_balance_loss,
            target_mask=target_mask,
            epoch=epoch,
        )

        return {
            "loss": losses["total_loss"],
            "ce_loss": losses["ce_loss"],
            "imp_loss": losses["imp_loss"],
            "moe_loss": losses["moe_loss"],
            "logits": logits,
            "hidden_states": hidden_states,
            "routing_info": self.last_routing_info,
        }

    @torch.no_grad()
    def generate(self, batch: dict, beam_size: int = 5,
                 max_length: int = None,
                 length_penalty: float = 0.6) -> torch.Tensor:
        """Generate reports autoregressively.

        Returns:
            [B, L] generated token IDs
        """
        max_length = max_length or self.config["model"]["decoder"]["max_length"]

        # ─── Visual Features ───
        vis_out = self.visual_encoder(batch["images"])
        patch_tokens = vis_out["patch_tokens"]

        # ─── Text Features ───
        ind_out = self.text_encoder.encode(
            batch["indication_ids"], batch["indication_mask"],
            prompt_type="ind",
        )
        pri_out = self.text_encoder.encode(
            batch["prior_ids"], batch["prior_mask"],
            prompt_type="pri",
        )
        # ─── Auxiliary Gate ───
        if self.auxiliary_enabled:
            multimodal_tokens = self.auxiliary_gate(
                patch_tokens,
                ind_out["cls_embedding"],
                pri_out["cls_embedding"],
            )
        else:
            multimodal_tokens = patch_tokens

        multimodal_tokens = self.vis_proj(multimodal_tokens)

        # ─── SV-MoE ───
        if self.svmoe_enabled:
            moe_out = self.sv_moe(
                multimodal_tokens,
                batch["stage_ids"],
                batch["view_ids"],
            )
            encoder_features = moe_out["output"]
        else:
            encoder_features = multimodal_tokens

        # ─── HP-QFormer ───
        prefix_k, prefix_v = None, None
        if self.hpqf_enabled:
            qf_out = self.hp_qformer(encoder_features)
            prefix_k = qf_out["prefix_k"]
            prefix_v = qf_out["prefix_v"]

        # ─── Generate ───
        cmn_k_self, cmn_v_self = self._get_cmn_kv(
            batch_size=encoder_features.size(0),
            device=encoder_features.device,
        )

        prior_input_ids = batch.get("prior_ids")
        prior_embeddings = self._prepare_prior_embeddings(pri_out)

        # Fast path: no prior-copy required.
        if self.prior_copy is None or prior_embeddings is None or prior_input_ids is None:
            return self.decoder.generate(
                encoder_features=encoder_features,
                cmn_k_self=cmn_k_self,
                cmn_v_self=cmn_v_self,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                beam_size=beam_size,
                max_length=max_length,
                length_penalty=length_penalty,
            )

        # Copy-aware decoding path (keeps train/eval behavior aligned).
        B = encoder_features.size(0)
        device = encoder_features.device

        def _slice_optional(tensor: torch.Tensor, idx: int):
            if tensor is None:
                return None
            return tensor[idx:idx + 1]

        # Greedy fast path.
        if beam_size <= 1:
            generated = torch.full(
                (B, 1), self.decoder.bos_token_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_length - 1):
                dec_out = self.decoder(
                    target_ids=generated,
                    encoder_features=encoder_features,
                    prefix_k=prefix_k,
                    prefix_v=prefix_v,
                    cmn_k_self=cmn_k_self,
                    cmn_v_self=cmn_v_self,
                )

                step_logits = dec_out["logits"][:, -1:, :]
                step_hidden = dec_out["hidden_states"][:, -1:, :]
                step_logits = self._apply_prior_copy(
                    logits=step_logits,
                    hidden_states=step_hidden,
                    prior_embeddings=prior_embeddings,
                    prior_input_ids=prior_input_ids,
                )

                next_token = torch.argmax(step_logits[:, -1, :], dim=-1)
                eos_fill = torch.full_like(next_token, self.decoder.eos_token_id)
                next_token = torch.where(finished, eos_fill, next_token)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                finished = finished | (next_token == self.decoder.eos_token_id)
                if finished.all():
                    break

            return generated

        # Beam-search path (per-sample for clarity and copy consistency).
        generated = torch.full(
            (B, max_length), self.decoder.pad_token_id, dtype=torch.long, device=device
        )

        for b in range(B):
            enc_b = encoder_features[b:b + 1]
            cmn_k_b = _slice_optional(cmn_k_self, b)
            cmn_v_b = _slice_optional(cmn_v_self, b)
            prefix_k_b = _slice_optional(prefix_k, b)
            prefix_v_b = _slice_optional(prefix_v, b)
            prior_emb_b = prior_embeddings[b:b + 1]
            prior_tok_b = prior_input_ids[b:b + 1]

            beams = [([self.decoder.bos_token_id], 0.0, False)]

            for _ in range(max_length - 1):
                all_candidates = []
                all_finished = True

                for tokens, score, is_finished in beams:
                    if is_finished:
                        all_candidates.append((tokens, score, True))
                        continue

                    all_finished = False
                    target_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                    dec_out = self.decoder(
                        target_ids=target_ids,
                        encoder_features=enc_b,
                        prefix_k=prefix_k_b,
                        prefix_v=prefix_v_b,
                        cmn_k_self=cmn_k_b,
                        cmn_v_self=cmn_v_b,
                    )

                    step_logits = dec_out["logits"][:, -1:, :]
                    step_hidden = dec_out["hidden_states"][:, -1:, :]
                    step_logits = self._apply_prior_copy(
                        logits=step_logits,
                        hidden_states=step_hidden,
                        prior_embeddings=prior_emb_b,
                        prior_input_ids=prior_tok_b,
                    )

                    log_probs = F.log_softmax(step_logits[:, -1, :], dim=-1).squeeze(0)
                    topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size)

                    for lp, tok in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                        next_tokens = tokens + [int(tok)]
                        next_score = score + float(lp)
                        next_finished = (int(tok) == self.decoder.eos_token_id)
                        all_candidates.append((next_tokens, next_score, next_finished))

                if all_finished:
                    break

                def rank_key(item):
                    seq, seq_score, _ = item
                    norm = ((5.0 + len(seq)) / 6.0) ** max(length_penalty, 0.0)
                    return seq_score / norm

                beams = sorted(all_candidates, key=rank_key, reverse=True)[:beam_size]

            finished_beams = [beam for beam in beams if beam[2]]
            final_beams = finished_beams if finished_beams else beams

            def final_rank(item):
                seq, seq_score, _ = item
                norm = ((5.0 + len(seq)) / 6.0) ** max(length_penalty, 0.0)
                return seq_score / norm

            best_tokens = sorted(final_beams, key=final_rank, reverse=True)[0][0]
            clipped = best_tokens[:max_length]
            generated[b, :len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)

        return generated

    def get_param_groups(self) -> list[dict]:
        """Get parameter groups with differential learning rates.

        Returns groups for:
        - Text encoder: lr_text_encoder
        - Transformer (decoder, MoE, Q-Former): lr_transformer
        - Impression branch (projection heads in loss): lr_imp_branch
        - Other: lr_default
        """
        train_cfg = self.config["training"]
        lr_text = train_cfg["optimizer"]["lr_text_encoder"]
        lr_transformer = train_cfg["optimizer"]["lr_transformer"]
        lr_imp = train_cfg["optimizer"]["lr_imp_branch"]
        lr_default = train_cfg["optimizer"]["lr_default"]
        weight_decay = train_cfg["optimizer"]["weight_decay"]

        text_encoder_params = []
        transformer_params = []
        imp_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "text_encoder" in name:
                text_encoder_params.append(param)
            elif "loss_fn.impression" in name or "imp_loss" in name:
                imp_params.append(param)
            elif any(k in name for k in ["decoder", "sv_moe", "hp_qformer",
                                           "vis_proj", "auxiliary_gate", "cmn"]):
                transformer_params.append(param)
            else:
                other_params.append(param)

        groups = []

        if text_encoder_params:
            groups.append({
                "params": text_encoder_params,
                "lr": lr_text,
                "weight_decay": weight_decay,
                "name": "text_encoder",
            })
        if transformer_params:
            groups.append({
                "params": transformer_params,
                "lr": lr_transformer,
                "weight_decay": weight_decay,
                "name": "transformer",
            })
        if imp_params:
            groups.append({
                "params": imp_params,
                "lr": lr_imp,
                "weight_decay": weight_decay,
                "name": "imp_branch",
            })
        if other_params:
            groups.append({
                "params": other_params,
                "lr": lr_default,
                "weight_decay": weight_decay,
                "name": "other",
            })

        return groups
