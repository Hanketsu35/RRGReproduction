"""MOE-RRG Evaluation Pipeline.

Supports:
- NLG metrics (BLEU-1/2/3/4, METEOR, ROUGE-L)
- Clinical metrics (CheXpert F1)
- Routing analysis (expert usage, stage/view tables)
- Stage/view stratified evaluation
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import compute_nlg_metrics, compute_clinical_metrics
from utils.platform import select_torch_device, dataloader_runtime_settings
from models.model_factory import MOERRGModel
from data_pipeline.mimic_cxr_dataset import MIMICCXRDataset
from data_pipeline.data_collator import DataCollator
from data_pipeline.preprocessing import MIMICPreprocessor


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


@torch.no_grad()
def generate_reports(model, dataloader, tokenizer, max_gen_length=256,
                     beam_size=5, length_penalty=0.6, device="cpu"):
    """Generate reports for all samples in the dataloader.

    Returns:
        generated_texts: list of generated report strings
        reference_texts: list of reference report strings
        routing_data: dict of routing statistics
    """
    model.eval()
    generated_texts = []
    reference_texts = []
    routing_data = {
        "expert_probs": [],
        "selected_experts": [],
        "stage_ids": [],
        "view_ids": [],
    }

    for batch in tqdm(dataloader, desc="Generating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Generate
        generated_ids = model.generate(
            batch,
            beam_size=beam_size,
            max_length=max_gen_length,
            length_penalty=length_penalty,
        )

        # Decode
        for i in range(generated_ids.size(0)):
            gen_text = tokenizer.decode(
                generated_ids[i], skip_special_tokens=True
            )
            ref_text = tokenizer.decode(
                batch["report_ids"][i], skip_special_tokens=True
            )
            generated_texts.append(gen_text)
            reference_texts.append(ref_text)

        # Collect routing data
        if model.last_routing_info:
            routing_data["expert_probs"].append(
                model.last_routing_info["expert_probs"].cpu()
            )
            routing_data["selected_experts"].append(
                model.last_routing_info["selected_expert"].cpu()
            )
            routing_data["stage_ids"].append(
                model.last_routing_info["stage_ids"].cpu()
            )
            routing_data["view_ids"].append(
                model.last_routing_info["view_ids"].cpu()
            )

    # Concatenate routing data
    for key in routing_data:
        if routing_data[key]:
            routing_data[key] = torch.cat(routing_data[key], dim=0).numpy()
        else:
            routing_data[key] = np.array([])

    return generated_texts, reference_texts, routing_data


def analyze_routing(routing_data: dict, num_experts: int = 4,
                    logger=None) -> dict:
    """Analyze routing patterns.

    Returns:
        Dictionary with routing analysis results
    """
    results = {}

    if len(routing_data["selected_experts"]) == 0:
        return {"error": "No routing data available"}

    selected = routing_data["selected_experts"]
    stages = routing_data["stage_ids"]
    views = routing_data["view_ids"]
    probs = routing_data["expert_probs"]

    # Overall expert usage
    usage_counts = np.bincount(selected.astype(int), minlength=num_experts)
    usage_pct = usage_counts / usage_counts.sum() * 100
    results["expert_usage_pct"] = usage_pct.tolist()

    if logger:
        logger.info("Expert Usage Distribution:")
        for i, pct in enumerate(usage_pct):
            logger.info(f"  Expert {i}: {pct:.1f}%")

    # Routing entropy
    if len(probs) > 0:
        mean_probs = probs.mean(axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(num_experts)
        results["routing_entropy"] = float(entropy)
        results["max_entropy"] = float(max_entropy)
        results["entropy_ratio"] = float(entropy / max_entropy)

        if logger:
            logger.info(f"Routing entropy: {entropy:.3f} / {max_entropy:.3f} "
                        f"({entropy/max_entropy*100:.1f}%)")

    # Stage-wise routing table
    stage_expert_table = {}
    for stage_id in range(5):
        mask = stages == stage_id
        if mask.sum() > 0:
            stage_selected = selected[mask]
            counts = np.bincount(stage_selected.astype(int), minlength=num_experts)
            pct = counts / counts.sum() * 100
            stage_expert_table[f"stage_{stage_id}"] = pct.tolist()

    results["stage_routing_table"] = stage_expert_table

    if logger:
        logger.info("\nStage-wise Expert Routing:")
        header = f"{'Stage':<10}" + "".join(f"{'Expert '+str(i):<12}" for i in range(num_experts))
        logger.info(header)
        for stage_key, pcts in stage_expert_table.items():
            row = f"{stage_key:<10}" + "".join(f"{p:<12.1f}" for p in pcts)
            logger.info(row)

    # View-wise routing table
    view_names = ["AP", "PA", "LA", "LL"]
    view_expert_table = {}
    for view_id in range(4):
        mask = views == view_id
        if mask.sum() > 0:
            view_selected = selected[mask]
            counts = np.bincount(view_selected.astype(int), minlength=num_experts)
            pct = counts / counts.sum() * 100
            view_expert_table[view_names[view_id]] = pct.tolist()

    results["view_routing_table"] = view_expert_table

    if logger:
        logger.info("\nView-wise Expert Routing:")
        logger.info(header)
        for view_key, pcts in view_expert_table.items():
            row = f"{view_key:<10}" + "".join(f"{p:<12.1f}" for p in pcts)
            logger.info(row)

    return results


def stratified_evaluation(hypotheses: list, references: list,
                          routing_data: dict, logger=None) -> dict:
    """Evaluate metrics stratified by stage and view."""
    results = {}
    stages = routing_data["stage_ids"]
    views = routing_data["view_ids"]

    # Stage-stratified metrics
    for stage_id in range(5):
        mask = stages == stage_id
        if mask.sum() > 0:
            hyps = [h for h, m in zip(hypotheses, mask) if m]
            refs = [r for r, m in zip(references, mask) if m]
            metrics = compute_nlg_metrics(hyps, refs)
            results[f"stage_{stage_id}"] = metrics

            if logger:
                logger.info(f"\nStage {stage_id} ({mask.sum()} samples):")
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v:.2f}")

    # View-stratified metrics
    view_names = ["AP", "PA", "LA", "LL"]
    for view_id in range(4):
        mask = views == view_id
        if mask.sum() > 0:
            hyps = [h for h, m in zip(hypotheses, mask) if m]
            refs = [r for r, m in zip(references, mask) if m]
            metrics = compute_nlg_metrics(hyps, refs)
            results[f"view_{view_names[view_id]}"] = metrics

            if logger:
                logger.info(f"\nView {view_names[view_id]} ({mask.sum()} samples):")
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MOE-RRG Evaluation")
    parser.add_argument("--config", type=str, default="CONFIG/config_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="analysis",
                        help="Directory to save evaluation results")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = select_torch_device(args.gpu)

    # Logger
    logger = setup_logger("moe_rrg_eval", config["logging"]["log_dir"])
    logger.info(f"Evaluating: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(
        f"Decoding config | beam_size={config['evaluation']['beam_size']} "
        f"length_penalty={config['evaluation'].get('length_penalty', 0.6)} "
        f"max_gen_length={config['evaluation']['max_gen_length']}"
    )

    # Load model
    model = MOERRGModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded successfully")

    # Build dataset
    data_cfg = config["data"]
    processed_metadata_csv = resolve_processed_metadata_csv(config, logger)
    dataset = MIMICCXRDataset(
        metadata_csv=processed_metadata_csv,
        image_root=data_cfg["root_dir"],
        split=args.split,
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        is_train=False,
    )

    collator = DataCollator(
        text_encoder=model.text_encoder,
        max_image_size=data_cfg["max_image_size"],
        max_text_len=data_cfg["max_text_len"],
        max_indication_len=data_cfg["max_indication_len"],
        max_prior_len=data_cfg["max_prior_len"],
        max_impression_len=data_cfg["max_impression_len"],
        is_train=False,
    )

    dl_num_workers, dl_pin_memory = dataloader_runtime_settings(
        requested_num_workers=data_cfg.get("num_workers", 4),
        requested_pin_memory=data_cfg.get("pin_memory", True),
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        collate_fn=collator,
    )

    # Generate reports
    logger.info("Generating reports...")
    generated, references, routing_data = generate_reports(
        model, dataloader, model.text_encoder.tokenizer,
        max_gen_length=config["evaluation"]["max_gen_length"],
        beam_size=config["evaluation"]["beam_size"],
        length_penalty=config["evaluation"].get("length_penalty", 0.6),
        device=device,
    )

    # Compute overall NLG metrics
    logger.info("\nOverall NLG Metrics:")
    nlg_metrics = compute_nlg_metrics(generated, references)
    for k, v in nlg_metrics.items():
        logger.info(f"  {k}: {v:.2f}")

    # Compute clinical metrics
    clinical_metrics = compute_clinical_metrics(generated, references)
    logger.info("\nClinical Metrics:")
    for k, v in clinical_metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            logger.info(f"  {k}:")
            for sub_k, sub_v in v.items():
                logger.info(f"    {sub_k}: {sub_v:.4f}" if isinstance(sub_v, (int, float)) else f"    {sub_k}: {sub_v}")
        else:
            logger.info(f"  {k}: {v}")

    # Routing analysis
    logger.info("\nRouting Analysis:")
    routing_analysis = analyze_routing(
        routing_data, config["model"]["sv_moe"]["num_experts"], logger
    )

    # Stratified evaluation
    logger.info("\nStratified Evaluation:")
    stratified = stratified_evaluation(generated, references, routing_data, logger)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    eval_protocol = {
        "nlg_backend": "custom_internal",
        "clinical_backend": clinical_metrics.get("backend", "none") if isinstance(clinical_metrics, dict) else "none",
        "clinical_status": clinical_metrics.get("status", "unknown") if isinstance(clinical_metrics, dict) else "unknown",
        "decoding": {
            "beam_size": config["evaluation"]["beam_size"],
            "length_penalty": config["evaluation"].get("length_penalty", 0.6),
            "max_gen_length": config["evaluation"]["max_gen_length"],
        },
        "paper_comparability": {
            "nlg": "partial (custom implementation, not sacreBLEU/official METEOR)",
            "clinical": "comparable" if isinstance(clinical_metrics, dict) and clinical_metrics.get("status") == "ok" else "not_comparable",
        },
    }

    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "nlg_metrics": nlg_metrics,
        "clinical_metrics": clinical_metrics,
        "evaluation_protocol": eval_protocol,
        "routing_analysis": routing_analysis,
        "stratified_metrics": stratified,
        "num_samples": len(generated),
    }

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    output_path = output_dir / f"eval_results_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
