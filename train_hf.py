"""MOE-RRG Training with HuggingFace MIMIC-CXR dataset.

Uses itsanmolgupta/mimic-cxr-dataset (30K samples, no gated access).
Optimized for Mac Air M3 (MPS, small batch, no fp16).

IMPORTANT: This script is experimental and not paper-comparable.
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager
from utils.platform import select_torch_device, dataloader_runtime_settings
from models.model_factory import MOERRGModel
from mimic_cxr_hf import MIMICCXRHFDataset
from data_pipeline.data_collator import DataCollator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_steps, total_steps, name="cosine"):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if name == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="MOE-RRG Training (HF MIMIC-CXR)")
    parser.add_argument("--config", type=str, default="CONFIG/config_mac_m3.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split (for quick testing)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_experimental",
        action="store_true",
        help="Acknowledge this non-paper-comparable training path",
    )
    args = parser.parse_args()

    if not args.allow_experimental:
        raise SystemExit(
            "Refusing to run train_hf.py without explicit acknowledgement. "
            "Re-run with --allow_experimental for non-paper smoke/experimentation."
        )

    config = load_config(args.config)
    seed = args.seed
    set_seed(seed)

    # Device: cross-platform CUDA/MPS/CPU selection
    device = select_torch_device(0)

    logger = setup_logger("moe_rrg_train", config["logging"]["log_dir"])
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.warning("EXPERIMENTAL MODE: train_hf.py is not paper-comparable.")

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    # ─── Build Model ───
    logger.info("Building model...")
    model = MOERRGModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # ─── Build Datasets ───
    logger.info("Loading MIMIC-CXR from HuggingFace...")
    train_dataset = MIMICCXRHFDataset("train", max_samples=args.max_samples, seed=seed)
    val_dataset = MIMICCXRHFDataset("val", max_samples=args.max_samples, seed=seed)

    train_collator = DataCollator(
        text_encoder=model.text_encoder,
        max_image_size=config["data"].get("max_image_size", 224),
        max_text_len=config["data"].get("max_text_len", 256),
        max_indication_len=config["data"].get("max_indication_len", 64),
        max_prior_len=config["data"].get("max_prior_len", 128),
        max_impression_len=config["data"].get("max_impression_len", 128),
        is_train=True,
    )
    val_collator = DataCollator(
        text_encoder=model.text_encoder,
        max_image_size=config["data"].get("max_image_size", 224),
        max_text_len=config["data"].get("max_text_len", 256),
        max_indication_len=config["data"].get("max_indication_len", 64),
        max_prior_len=config["data"].get("max_prior_len", 128),
        max_impression_len=config["data"].get("max_impression_len", 128),
        is_train=False,
    )

    dl_num_workers, dl_pin = dataloader_runtime_settings(
        requested_num_workers=config["data"].get("num_workers", 0),
        requested_pin_memory=config["data"].get("pin_memory", True),
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=dl_num_workers,
        pin_memory=dl_pin,
        collate_fn=train_collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin,
        collate_fn=val_collator,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ─── Optimizer ───
    param_groups = model.get_param_groups()
    optimizer = torch.optim.Adam(
        param_groups,
        betas=(config["training"]["optimizer"]["adam_beta1"],
               config["training"]["optimizer"]["adam_beta2"]),
    )

    # ─── Scheduler ───
    epochs = config["training"]["epochs"]
    grad_accum = config["training"]["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * config["training"]["scheduler"]["warmup_ratio"])
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps,
                                 config["training"]["scheduler"]["name"])
    logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # ─── Checkpointing ───
    ckpt_manager = CheckpointManager(
        checkpoint_dir=config["logging"]["checkpoint_dir"],
        save_top_k=config["logging"]["save_top_k"],
    )

    # ─── Training Loop ───
    use_amp = False  # MPS'de güvenli değil, CPU'da da gerek yok
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    logger.info("Starting training...")
    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"{'='*50}")

        # ─── Train ───
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_imp = 0.0
        train_moe = 0.0
        n_batches = 0
        accum_steps = 0

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch, epoch=epoch)
            loss = outputs["loss"] / grad_accum
            loss.backward()
            accum_steps += 1

            should_step = (accum_steps == grad_accum) or (step == len(train_loader) - 1)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_steps = 0

            train_loss += outputs["loss"].item()
            train_ce += outputs["ce_loss"].item()
            train_imp += outputs["imp_loss"].item()
            train_moe += outputs["moe_loss"].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{train_loss/n_batches:.4f}",
                "ce": f"{train_ce/n_batches:.4f}",
            })

        avg_train = train_loss / n_batches if n_batches > 0 else 0.0

        # ─── Validate ───
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(batch, epoch=epoch)
                val_loss += outputs["loss"].item()
                val_batches += 1

        avg_val = val_loss / val_batches if val_batches > 0 else 0.0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {lr:.6f}"
        )
        logger.info(
            f"  CE: {train_ce/n_batches:.4f} | IMP: {train_imp/n_batches:.4f} | "
            f"MoE: {train_moe/n_batches:.4f}"
        )

        # ─── Checkpoint ───
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_manager.save(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_val_loss,
                "config": config,
            },
            metric_value=avg_val,
            epoch=epoch,
            is_best=is_best,
        )

        logger.info(f"  Best val: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
