"""
Test different ensemble strictness levels (strict, moderate, lax)
to find which achieves the highest accuracy on training and validation datasets.

Plots results comparing accuracy across strictness levels.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from run_QC
from run_QC import (
    load_efficientnet,
    load_convnextv2,
    load_dinov2,
    run_efficientnet,
    run_convnextv2,
    run_dinov2,
    discover_images,
    load_val_files_from_splits,
    EFFICIENTNET_DIR,
    CONVNEXTV2_DIR,
    DINOV2_DIR,
    DATA_ROOT,
    DEVICE,
    MIXED_PRECISION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def get_ensemble_prediction(results: dict, strictness: str) -> str:
    """
    Get ensemble prediction based on strictness level.

    Args:
        results: dict with keys "efficientnet", "convnextv2", "dinov2"
        strictness: "strict" (3/3), "moderate" (2/3), or "lax" (1/3)

    Returns:
        "pass" or "fail"
    """
    votes = []
    for key in ["efficientnet", "convnextv2", "dinov2"]:
        if key in results:
            votes.append(results[key]["label"])

    n_pass = sum(1 for v in votes if v == "pass")

    if strictness == "strict":
        return "pass" if n_pass >= 3 else "fail"
    elif strictness == "moderate":
        return "pass" if n_pass >= 2 else "fail"
    elif strictness == "lax":
        return "pass" if n_pass >= 1 else "fail"
    else:
        raise ValueError(f"Unknown strictness: {strictness}")


def run_inference_on_images(
    image_records: List[Dict],
    models: dict,
    configs: dict,
    device: torch.device,
) -> List[Dict]:
    """
    Run all three models on a list of images.

    Returns list of dicts with keys:
        - "path": image path
        - "true_label": "pass" or "fail"
        - "predictions": dict with "efficientnet", "convnextv2", "dinov2" results
    """
    results = []

    for i, record in enumerate(image_records):
        img_path = record["path"]
        true_label = "pass" if record["label"] == 0 else "fail"

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(image_records)}] {Path(img_path).name}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"    Could not load image {img_path}: {e}")
            continue

        predictions = {}

        # Run each model
        if "efficientnet" in models:
            try:
                predictions["efficientnet"] = run_efficientnet(
                    models["efficientnet"], configs["efficientnet"], img, device
                )
            except Exception as e:
                logger.warning(f"    EfficientNet failed: {e}")

        if "convnextv2" in models:
            try:
                predictions["convnextv2"] = run_convnextv2(
                    models["convnextv2"], configs["convnextv2"], img, device
                )
            except Exception as e:
                logger.warning(f"    ConvNeXt V2 failed: {e}")

        if "dinov2" in models:
            try:
                predictions["dinov2"] = run_dinov2(
                    models["dinov2"], configs["dinov2"], img, device
                )
            except Exception as e:
                logger.warning(f"    DINOv2 failed: {e}")

        results.append({
            "path": img_path,
            "true_label": true_label,
            "predictions": predictions,
        })

    return results


def calculate_accuracies(
    inference_results: List[Dict],
    strictness_levels: List[str] = ["strict", "moderate", "lax"],
) -> Dict[str, float]:
    """
    Calculate accuracy for each strictness level.

    Returns dict mapping strictness level to accuracy (0-1).
    """
    accuracies = {}

    for strictness in strictness_levels:
        correct = 0
        total = 0

        for result in inference_results:
            if not result["predictions"]:
                continue

            pred = get_ensemble_prediction(result["predictions"], strictness)
            true_label = result["true_label"]

            if pred == true_label:
                correct += 1
            total += 1

        accuracies[strictness] = correct / total if total > 0 else 0.0

    return accuracies


def main():
    logger.info("=" * 70)
    logger.info("Testing Ensemble Strictness Levels")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")

    # Load models
    logger.info("\nLoading models...")
    models = {}
    configs = {}

    for name, loader, model_dir in [
        ("efficientnet", load_efficientnet, str(EFFICIENTNET_DIR)),
        ("convnextv2", load_convnextv2, str(CONVNEXTV2_DIR)),
        ("dinov2", load_dinov2, str(DINOV2_DIR)),
    ]:
        try:
            m, c = loader(model_dir, DEVICE)
            models[name] = m
            configs[name] = c
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")

    if not models:
        logger.error("No models loaded!")
        sys.exit(1)

    logger.info(f"Models loaded: {list(models.keys())}\n")

    # Load validation images
    logger.info("Loading validation images...")
    val_records = load_val_files_from_splits()
    logger.info(f"  Total val images: {len(val_records)}\n")

    # Load training images (everything not in validation)
    logger.info("Loading training images...")
    pass_dir = os.path.join(DATA_ROOT, "pass")
    fail_dir = os.path.join(DATA_ROOT, "fail")

    all_pass = discover_images(pass_dir, label=0)
    all_fail = discover_images(fail_dir, label=1)
    all_records = all_pass + all_fail

    val_paths = set(r["path"] for r in val_records)
    train_records = [r for r in all_records if r["path"] not in val_paths]

    logger.info(f"  Total train images: {len(train_records)}\n")

    # Run inference on validation set
    logger.info("Running inference on validation set...")
    val_results = run_inference_on_images(val_records, models, configs, DEVICE)
    logger.info(f"  Processed {len(val_results)} validation images\n")

    # Run inference on training set
    logger.info("Running inference on training set...")
    train_results = run_inference_on_images(train_records, models, configs, DEVICE)
    logger.info(f"  Processed {len(train_results)} training images\n")

    # Calculate accuracies
    logger.info("Calculating accuracies...\n")

    val_accuracies = calculate_accuracies(val_results)
    train_accuracies = calculate_accuracies(train_results)

    logger.info("Validation Set Results:")
    for strictness, acc in val_accuracies.items():
        logger.info(f"  {strictness:12s}: {acc:.4f} ({acc*100:.2f}%)")

    logger.info("\nTraining Set Results:")
    for strictness, acc in train_accuracies.items():
        logger.info(f"  {strictness:12s}: {acc:.4f} ({acc*100:.2f}%)")

    # Find best strictness levels
    best_val = max(val_accuracies, key=val_accuracies.get)
    best_train = max(train_accuracies, key=train_accuracies.get)

    logger.info(f"\nBest for validation: {best_val} ({val_accuracies[best_val]*100:.2f}%)")
    logger.info(f"Best for training: {best_train} ({train_accuracies[best_train]*100:.2f}%)")

    # Plot results
    logger.info("\nGenerating plot...")

    strictness_levels = ["strict", "moderate", "lax"]
    val_accs = [val_accuracies[s] for s in strictness_levels]
    train_accs = [train_accuracies[s] for s in strictness_levels]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(strictness_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_accs, width, label="Training", alpha=0.8, color="#3498db")
    bars2 = ax.bar(x + width/2, val_accs, width, label="Validation", alpha=0.8, color="#e74c3c")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Strictness Level", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Ensemble Strictness Levels: Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strictness_levels)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_path = "/home/brlab/Dropbox/LM2_Leaf_Classifier/strictness_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")

    plt.close(fig)

    logger.info("\n" + "=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
