"""
Leaf Pass/Fail Classifier
=========================
Simple binary image classifier using EfficientNet-B3 (via timm).
Images are resized to the model's native 300x300 input — no tiling.

Architecture
------------
  ┌────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
  │ Input Image│───►│ Resize 300px │───►│ EffNet-B3 │───►│ FC Head  │
  │ (any size) │    │              │    │ Backbone  │    │ pass/fail│
  └────────────┘    └──────────────┘    └───────────┘    └──────────┘

Usage
-----
  1. Set global variables below (paths, hyperparams)
  2. Run:  python train_EfficientNet_B3.py
"""

import os
import sys
import csv
import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from torchvision import transforms

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# --- Paths ---
DATA_ROOT       = "/data/Train_Good_Bad_Leaves_Exported_RGB"
PASS_DIR        = os.path.join(DATA_ROOT, "pass")
FAIL_DIR        = os.path.join(DATA_ROOT, "fail")
OUTPUT_DIR      = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_efficientnet_b3_AugColor"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

# --- Resume from previous best ---
# Path to a previous best checkpoint to initialise weights from before training.
# Set to None to train from scratch (ImageNet pretrained only).
RESUME_FROM     = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_efficientnet_b3/checkpoints/best_model.pt"

# --- Model ---
BACKBONE_NAME   = "efficientnet_b3.ra2_in1k"
IMAGE_SIZE      = 300          # EfficientNet-B3 native input size
FREEZE_BACKBONE = False
FREEZE_EPOCHS   = 2

# --- Training ---
NUM_EPOCHS      = 25
BATCH_SIZE      = 64
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"     # "cosine" or "step"
STEP_SIZE       = 10
STEP_GAMMA      = 0.1
EARLY_STOP_PATIENCE = 7
NUM_WORKERS     = 8
PIN_MEMORY      = True
MIXED_PRECISION = True

# --- Validation holdout ---
VAL_PER_GENUS   = 1

# --- Export ---
EXPORT_ONNX        = True
EXPORT_TORCHSCRIPT = True

# --- Resume / Export-only ---
RESUME      = True   # auto-resume training from last checkpoint if one exists
EXPORT_ONLY = False  # set True to skip training and only run export on best_model.pt

# --- Reproducibility ---
SEED = 42

# --- Logging ---
LOG_LEVEL = "INFO"


# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================================
# DATA DISCOVERY & SPLITTING
# ============================================================================

def discover_images(base_dir: str, label: int) -> List[Dict]:
    """Walk pass/ or fail/ directory and return list of image records."""
    records = []
    base = Path(base_dir)
    for family_dir in sorted(base.iterdir()):
        if not family_dir.is_dir():
            continue
        family = family_dir.name
        for genus_dir in sorted(family_dir.iterdir()):
            if not genus_dir.is_dir():
                continue
            genus = genus_dir.name
            for img_path in sorted(genus_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
                    records.append({
                        "path": str(img_path),
                        "label": label,
                        "family": family,
                        "genus": genus,
                        "filename": img_path.name,
                    })
    return records


def split_train_val(
    pass_records: List[Dict],
    fail_records: List[Dict],
    val_per_genus: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    train, val = [], []

    for class_label, records in [("pass", pass_records), ("fail", fail_records)]:
        by_genus = defaultdict(list)
        for r in records:
            by_genus[r["genus"]].append(r)

        for genus, imgs in by_genus.items():
            random.shuffle(imgs)
            n_val = min(val_per_genus, len(imgs))
            val.extend(imgs[:n_val])
            train.extend(imgs[n_val:])
            if n_val > 0:
                logger.debug(
                    f"  Holdout {class_label}/{genus}: {n_val} val, "
                    f"{len(imgs) - n_val} train"
                )

    pass_genera = set(r["genus"] for r in pass_records)
    fail_genera = set(r["genus"] for r in fail_records)
    pass_only = pass_genera - fail_genera
    fail_only = fail_genera - pass_genera
    if pass_only:
        logger.warning(
            f"  {len(pass_only)} genera appear ONLY in pass: "
            f"{sorted(pass_only)[:10]}{'...' if len(pass_only) > 10 else ''}"
        )
    if fail_only:
        logger.warning(
            f"  {len(fail_only)} genera appear ONLY in fail: "
            f"{sorted(fail_only)[:10]}{'...' if len(fail_only) > 10 else ''}"
        )

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ============================================================================
# DATASET
# ============================================================================

def build_transforms(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 270), expand=False),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class LeafDataset(Dataset):
    def __init__(self, records: List[Dict], augment: bool = False):
        self.records = records
        self.transform = build_transforms(augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        tensor = self.transform(img)
        meta = {
            "path": rec["path"],
            "filename": rec.get("filename", ""),
            "family": rec.get("family", ""),
            "genus": rec.get("genus", ""),
        }
        return tensor, rec["label"], meta


def collate_fn(batch):
    images, labels, metas = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(metas)


# ============================================================================
# MODEL
# ============================================================================

class EfficientNetB3LeafClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        num_classes: int = 2,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feat_dim, num_classes),
        )

        logger.info(
            f"EfficientNetB3LeafClassifier: backbone={backbone_name}, "
            f"feat_dim={self.feat_dim}"
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)   # (B, feat_dim)
        return self.head(feats)    # (B, num_classes)


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(
        self,
        model: EfficientNetB3LeafClassifier,
        train_records: List[Dict],
        val_records: List[Dict],
    ):
        self.model = model.to(DEVICE)

        train_ds = LeafDataset(train_records, augment=True)
        self.val_ds = LeafDataset(val_records, augment=False)

        labels = [r["label"] for r in train_records]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            collate_fn=collate_fn, num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY, persistent_workers=True, prefetch_factor=4,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY, persistent_workers=True, prefetch_factor=4,
        )

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        if LR_SCHEDULER == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=NUM_EPOCHS
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.start_epoch = 1
        self.training_complete = False

        # Resume from last checkpoint if available
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
        if RESUME and os.path.exists(last_ckpt_path):
            state = torch.load(last_ckpt_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.best_val_acc = state["best_val_acc"]
            self.patience_counter = state["patience_counter"]
            self.training_complete = state.get("training_complete", False)
            self.start_epoch = state["epoch"] + 1
            logger.info(
                f"Resumed from epoch {state['epoch']} "
                f"(best_val_acc={self.best_val_acc:.4f}, "
                f"{'training complete' if self.training_complete else f'next epoch {self.start_epoch}'})"
            )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch} batch {batch_idx+1}/{len(self.train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        genus_results = defaultdict(lambda: {
            "family": "", "pass_correct": 0, "pass_total": 0,
            "fail_correct": 0, "fail_total": 0,
        })

        for images, labels, metas in self.val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i, meta in enumerate(metas):
                genus = meta["genus"]
                genus_results[genus]["family"] = meta["family"]
                true_label = labels[i].item()
                is_correct = (preds[i] == labels[i]).item()
                if true_label == 0:
                    genus_results[genus]["pass_total"] += 1
                    genus_results[genus]["pass_correct"] += int(is_correct)
                else:
                    genus_results[genus]["fail_total"] += 1
                    genus_results[genus]["fail_correct"] += int(is_correct)

        return total_loss / max(total, 1), correct / max(total, 1), dict(genus_results)

    def train(self) -> str:
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")

        if self.training_complete:
            logger.info("Training already complete — skipping to export.")
            return best_ckpt_path

        logger.info(f"Training: {len(self.train_loader.dataset)} train, "
                    f"{len(self.val_ds)} val")

        for epoch in range(self.start_epoch, NUM_EPOCHS + 1):
            if FREEZE_BACKBONE and epoch == 1:
                self.model.freeze_backbone()
            if FREEZE_BACKBONE and epoch == FREEZE_EPOCHS + 1:
                self.model.unfreeze_backbone()

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, genus_results = self.validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
                f"LR={lr:.2e}"
            )

            failing_genera = []
            for genus, stats in sorted(genus_results.items()):
                g_total = stats["pass_total"] + stats["fail_total"]
                g_correct = stats["pass_correct"] + stats["fail_correct"]
                if g_correct / max(g_total, 1) < 1.0:
                    failing_genera.append(f"{genus}({g_correct/max(g_total,1):.0%})")
            if failing_genera:
                logger.info(f"  Genera with errors: {', '.join(failing_genera)}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "backbone_name": BACKBONE_NAME,
                        "image_size": IMAGE_SIZE,
                    },
                }, best_ckpt_path)
                logger.info(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_val_acc": self.best_val_acc,
                        "patience_counter": self.patience_counter,
                        "training_complete": True,
                    }, last_ckpt_path)
                    break

            # Save resumable checkpoint after every epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_acc": self.best_val_acc,
                "patience_counter": self.patience_counter,
                "training_complete": epoch == NUM_EPOCHS,
            }, last_ckpt_path)

        _, _, final_genus = self.validate()
        genus_report_path = os.path.join(OUTPUT_DIR, "genus_validation_report.csv")
        with open(genus_report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "family", "genus",
                "pass_correct", "pass_total", "pass_accuracy",
                "fail_correct", "fail_total", "fail_accuracy",
                "overall_correct", "overall_total", "overall_accuracy",
            ])
            for genus, stats in sorted(final_genus.items(), key=lambda x: (x[1]["family"], x[0])):
                p_tot, p_cor = stats["pass_total"], stats["pass_correct"]
                f_tot, f_cor = stats["fail_total"], stats["fail_correct"]
                o_tot, o_cor = p_tot + f_tot, p_cor + f_cor
                writer.writerow([
                    stats["family"], genus,
                    p_cor, p_tot, f"{p_cor/max(p_tot,1):.4f}",
                    f_cor, f_tot, f"{f_cor/max(f_tot,1):.4f}",
                    o_cor, o_tot, f"{o_cor/max(o_tot,1):.4f}",
                ])
        logger.info(f"Genus report saved to {genus_report_path}")

        return best_ckpt_path


# ============================================================================
# EXPORT (ONNX + TorchScript)
# ============================================================================

def export_model(model: EfficientNetB3LeafClassifier, output_dir: str):
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    if EXPORT_ONNX:
        onnx_path = os.path.join(output_dir, "leaf_classifier.onnx")
        try:
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            logger.info(f"ONNX model exported to {onnx_path}")
        except ModuleNotFoundError as e:
            logger.warning(f"ONNX export skipped — missing dependency: {e}")
            logger.warning("Fix with:  pip install onnxscript")

    if EXPORT_TORCHSCRIPT:
        ts_path = os.path.join(output_dir, "leaf_classifier_scripted.pt")
        scripted = torch.jit.trace(model, dummy)
        scripted.save(ts_path)
        logger.info(f"TorchScript model exported to {ts_path}")

    config = {
        "backbone_name": BACKBONE_NAME,
        "image_size": IMAGE_SIZE,
        "class_names": ["pass", "fail"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Leaf Pass/Fail Classifier — EfficientNet-B3")
    logger.info("=" * 60)

    logger.info(f"Scanning pass images from: {PASS_DIR}")
    pass_records = discover_images(PASS_DIR, label=0)
    logger.info(f"  Found {len(pass_records)} pass images")

    logger.info(f"Scanning fail images from: {FAIL_DIR}")
    fail_records = discover_images(FAIL_DIR, label=1)
    logger.info(f"  Found {len(fail_records)} fail images")

    if not pass_records and not fail_records:
        logger.error("No images found! Check DATA_ROOT path.")
        sys.exit(1)

    all_genera = set(r["genus"] for r in pass_records + fail_records)
    logger.info(f"  Total genera: {len(all_genera)}")

    train_records, val_records = split_train_val(
        pass_records, fail_records, val_per_genus=VAL_PER_GENUS
    )
    logger.info(f"Split: {len(train_records)} train, {len(val_records)} val")

    val_genera = set(r["genus"] for r in val_records)
    val_families = set(r["family"] for r in val_records)
    all_families = set(r["family"] for r in pass_records + fail_records)

    split_info = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "val_per_genus": VAL_PER_GENUS,
        "train_pass": sum(1 for r in train_records if r["label"] == 0),
        "train_fail": sum(1 for r in train_records if r["label"] == 1),
        "val_pass": sum(1 for r in val_records if r["label"] == 0),
        "val_fail": sum(1 for r in val_records if r["label"] == 1),
        "total_genera": len(all_genera),
        "val_genera": len(val_genera),
        "total_families": len(all_families),
        "val_families": len(val_families),
        "val_files": [
            {
                "path": r["path"], "label": "pass" if r["label"] == 0 else "fail",
                "family": r["family"], "genus": r["genus"],
            }
            for r in sorted(val_records, key=lambda x: (x["family"], x["genus"]))
        ],
    }
    split_path = os.path.join(OUTPUT_DIR, "data_split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(
        f"  Holdout covers {len(val_genera)} genera across {len(val_families)} families"
    )

    model = EfficientNetB3LeafClassifier(backbone_name=BACKBONE_NAME)

    # Load weights from a previous best checkpoint if specified
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        prev_ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(prev_ckpt["model_state_dict"])
        logger.info(
            f"Initialised weights from {RESUME_FROM} "
            f"(val_acc={prev_ckpt.get('val_acc', 'N/A')})"
        )
    elif RESUME_FROM:
        logger.warning(f"RESUME_FROM path not found: {RESUME_FROM} — training from scratch")

    if EXPORT_ONLY:
        best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        if not os.path.exists(best_ckpt):
            logger.error(f"EXPORT_ONLY=True but no checkpoint found at {best_ckpt}")
            sys.exit(1)
    else:
        trainer = Trainer(model, train_records, val_records)
        best_ckpt = trainer.train()

    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded best checkpoint (val_acc={ckpt['val_acc']:.4f})")

    export_model(model, OUTPUT_DIR)

    logger.info("Training complete!")
    logger.info(f"  Best val accuracy: {ckpt['val_acc']:.4f}")
    logger.info(f"  Outputs saved to:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
