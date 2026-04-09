"""MOE-RRG Training Pipeline.

Supports:
- Differential learning rates
- Mixed precision (FP16)
- Gradient accumulation
- Cosine annealing with warmup
- Checkpointing (top-K)
- WandB / TensorBoard logging
- Seed control
"""

import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager
from utils.platform import select_torch_device, dataloader_runtime_settings
from models.model_factory import MOERRGModel
from data_pipeline.mimic_cxr_dataset import MIMICCXRDataset
from data_pipeline.data_collator import DataCollator
from data_pipeline.preprocessing import MIMICPreprocessor


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     scheduler_name: str = "cosine"):
    """Create learning rate scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if scheduler_name == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_processed_metadata_csv(config: dict, logger) -> str:
    """Resolve processed metadata CSV, generating it if configured and missing."""
    data_cfg = config["data"]
    root_dir = data_cfg["root_dir"]
    default_processed = os.path.join(root_dir, "processed", "processed_metadata.csv")
    processed_csv = data_cfg.get("processed_metadata_csv", default_processed)

    if os.path.exists(processed_csv):
        return processed_csv

    metadata_csv = data_cfg.get("metadata_csv")
    if not metadata_csv or not os.path.exists(metadata_csv):
        raise FileNotFoundError(
            f"Processed metadata not found at {processed_csv}. "
            "Set data.processed_metadata_csv or provide valid data.metadata_csv for auto-preprocessing."
        )

    logger.info("Processed metadata not found. Running MIMIC preprocessing...")
    preprocessor = MIMICPreprocessor(
        metadata_csv=metadata_csv,
        report_csv=data_cfg.get("report_csv"),
        split_csv=data_cfg.get("split_csv"),
        output_dir=os.path.dirname(processed_csv),
    )
    preprocessor.process()

    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Preprocessing completed but output not found at {processed_csv}")

    return processed_csv


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                    grad_accum_steps, max_grad_norm, epoch, logger,
                    log_interval=50, use_amp=False, device_type="cpu"):
    """Train for one epoch.

    Returns:
        Dictionary of average losses
    """
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_imp = 0.0
    total_moe = 0.0
    n_batches = 0

    # Routing statistics accumulation
    all_expert_probs = []
    all_selected_experts = []
    all_stage_ids = []
    all_view_ids = []

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass with mixed precision
        if use_amp:
            with autocast(device_type=device_type):
                outputs = model(batch, epoch=epoch)
                loss = outputs["loss"] / grad_accum_steps

            if device_type == "cuda":
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                # MPS or other: no GradScaler needed
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
        else:
            outputs = model(batch, epoch=epoch)
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Accumulate losses
        total_loss += outputs["loss"].item()
        total_ce += outputs["ce_loss"].item()
        total_imp += outputs["imp_loss"].item()
        total_moe += outputs["moe_loss"].item()
        n_batches += 1

        # Collect routing statistics
        if "routing_info" in outputs and outputs["routing_info"]:
            ri = outputs["routing_info"]
            all_expert_probs.append(ri["expert_probs"].cpu())
            all_selected_experts.append(ri["selected_expert"].cpu())
            all_stage_ids.append(ri["stage_ids"].cpu())
            all_view_ids.append(ri["view_ids"].cpu())

        # Log progress
        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | CE: {total_ce/n_batches:.4f} | "
                f"IMP: {total_imp/n_batches:.4f} | MoE: {total_moe/n_batches:.4f} | "
                f"LR: {lr:.6f}"
            )

    # Compute epoch averages
    results = {
        "train_loss": total_loss / n_batches,
        "train_ce_loss": total_ce / n_batches,
        "train_imp_loss": total_imp / n_batches,
        "train_moe_loss": total_moe / n_batches,
    }

    # Compute routing statistics
    if all_expert_probs:
        expert_probs = torch.cat(all_expert_probs, dim=0)  # [N, K]
        selected_experts = torch.cat(all_selected_experts)  # [N]
        stages = torch.cat(all_stage_ids)                    # [N]
        views = torch.cat(all_view_ids)                      # [N]

        # Expert usage distribution
        usage = torch.bincount(selected_experts, minlength=model.sv_moe.num_experts)
        usage_pct = usage.float() / usage.sum() * 100

        results["expert_usage"] = usage_pct.tolist()
        results["routing_entropy"] = -(
            expert_probs.mean(0) * torch.log(expert_probs.mean(0) + 1e-10)
        ).sum().item()

        logger.info(f"  Expert usage: {[f'{p:.1f}%' for p in usage_pct.tolist()]}")

    return results


@torch.no_grad()
def validate(model, dataloader, epoch, logger, use_amp=False, device_type="cpu"):
    """Validate the model.

    Returns:
        Dictionary of validation losses
    """
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_imp = 0.0
    total_moe = 0.0
    n_batches = 0

    for batch in dataloader:
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if use_amp:
            with autocast(device_type=device_type):
                outputs = model(batch, epoch=epoch)
        else:
            outputs = model(batch, epoch=epoch)

        total_loss += outputs["loss"].item()
        total_ce += outputs["ce_loss"].item()
        total_imp += outputs["imp_loss"].item()
        total_moe += outputs["moe_loss"].item()
        n_batches += 1

    results = {
        "val_loss": total_loss / n_batches,
        "val_ce_loss": total_ce / n_batches,
        "val_imp_loss": total_imp / n_batches,
        "val_moe_loss": total_moe / n_batches,
    }

    logger.info(
        f"  Val | Loss: {results['val_loss']:.4f} | "
        f"CE: {results['val_ce_loss']:.4f} | "
        f"IMP: {results['val_imp_loss']:.4f} | "
        f"MoE: {results['val_moe_loss']:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="MOE-RRG Training")
    parser.add_argument("--config", type=str, default="CONFIG/config_base.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Seed
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)

    # Device — supports CUDA, MPS (Apple Silicon), and CPU
    device = select_torch_device(args.gpu)

    # Logger
    logger = setup_logger("moe_rrg_train", config["logging"]["log_dir"])
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")

    # ─── Build Model ───
    logger.info("Building model...")
    model = MOERRGModel(config).to(device)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ─── Build Datasets ───
    logger.info("Building datasets...")
    data_cfg = config["data"]
    processed_metadata_csv = resolve_processed_metadata_csv(config, logger)

    train_dataset = MIMICCXRDataset(
        metadata_csv=processed_metadata_csv,
        image_root=data_cfg["root_dir"],
        split="train",
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        is_train=True,
    )

    val_dataset = MIMICCXRDataset(
        metadata_csv=processed_metadata_csv,
        image_root=data_cfg["root_dir"],
        split="validate",
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        is_train=False,
    )

    train_collator = DataCollator(
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        max_indication_len=data_cfg["max_indication_len"],
        max_prior_len=data_cfg["max_prior_len"],
        max_impression_len=data_cfg["max_impression_len"],
        is_train=True,
    )

    val_collator = DataCollator(
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        max_indication_len=data_cfg["max_indication_len"],
        max_prior_len=data_cfg["max_prior_len"],
        max_impression_len=data_cfg["max_impression_len"],
        is_train=False,
    )

    # Cross-platform DataLoader settings (safe on macOS/Windows/Linux)
    dl_num_workers, dl_pin_memory = dataloader_runtime_settings(
        requested_num_workers=data_cfg.get("num_workers", 4),
        requested_pin_memory=data_cfg.get("pin_memory", True),
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        collate_fn=train_collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        collate_fn=val_collator,
    )

    # ─── Optimizer with differential LRs ───
    logger.info("Setting up optimizer...")
    param_groups = model.get_param_groups()
    optimizer = torch.optim.Adam(
        param_groups,
        betas=(config["training"]["optimizer"]["adam_beta1"],
               config["training"]["optimizer"]["adam_beta2"]),
    )

    # ─── Scheduler ───
    epochs = config["training"]["epochs"]
    grad_accum = config["training"]["grad_accum_steps"]
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * config["training"]["scheduler"]["warmup_ratio"])
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps, total_steps,
        config["training"]["scheduler"]["name"],
    )

    logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # ─── Mixed Precision ───
    # MPS supports float16 autocast; CUDA uses GradScaler too
    use_amp = config["training"].get("fp16", True) and device.type in ("cuda", "mps")
    device_type = device.type
    scaler = GradScaler(enabled=(device_type == "cuda"))

    # ─── Checkpoint Manager ───
    ckpt_manager = CheckpointManager(
        checkpoint_dir=config["logging"]["checkpoint_dir"],
        save_top_k=config["logging"]["save_top_k"],
        metric_name=config["training"]["early_stopping"]["metric"],
        mode=config["training"]["early_stopping"]["mode"],
    )

    # ─── Resume ───
    start_epoch = 0
    best_metric = float("inf")
    if args.resume:
        ckpt = ckpt_manager.load(args.resume)
        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", float("inf"))
            logger.info(f"Resumed from epoch {start_epoch}")

    # ─── Training Loop ───
    logger.info("Starting training...")
    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    # WandB integration
    if config["logging"].get("use_wandb", False):
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity"),
            config=config,
            name=f"moe_rrg_seed{seed}",
        )

    # TensorBoard
    tb_writer = None
    if config["logging"].get("use_tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(config["logging"]["log_dir"])

    for epoch in range(start_epoch, epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_results = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            grad_accum, config["training"]["max_grad_norm"],
            epoch, logger, config["logging"]["log_interval"],
            use_amp=use_amp, device_type=device_type,
        )

        # Validate
        val_results = validate(model, val_loader, epoch, logger,
                               use_amp=use_amp, device_type=device_type)

        # ─── Logging ───
        if tb_writer:
            for k, v in train_results.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_results.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f"val/{k}", v, epoch)

        if config["logging"].get("use_wandb", False):
            wandb.log({"epoch": epoch, **train_results, **val_results})

        # ─── Checkpointing ───
        val_metric = val_results["val_loss"]
        is_best = val_metric < best_metric
        if is_best:
            best_metric = val_metric
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_manager.save(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "config": config,
                "seed": seed,
            },
            metric_value=val_metric,
            epoch=epoch,
            is_best=is_best,
        )

        logger.info(f"  Best val_loss: {best_metric:.4f} | Patience: {patience_counter}/{patience}")

        # ─── Early Stopping ───
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if tb_writer:
        tb_writer.close()

    logger.info("Training complete!")
    logger.info(f"Best val_loss: {best_metric:.4f}")


if __name__ == "__main__":
    main()
