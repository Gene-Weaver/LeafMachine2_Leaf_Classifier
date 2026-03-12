"""
Leaf Pass/Fail Classifier
=========================
Tile-based binary image classifier using ConvNeXt V2 (via timm).
Accepts full-resolution images without resizing — tiles the image into
overlapping patches, extracts features per tile, aggregates, and classifies.

Architecture
------------
  ┌────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
  │ Input Image│───►│ Tile Extractor│───►│ ConvNeXt  │───►│ Tile     │
  │ (any size) │    │ (NxN grid)   │    │ Backbone  │    │ Features │
  └────────────┘    └──────────────┘    └───────────┘    └────┬─────┘
                                                              │
                                                         ┌────▼─────┐
                                                         │Aggregator│
                                                         │(mean/max)│
                                                         └────┬─────┘
                                                              │
                                                         ┌────▼─────┐
                                                         │ FC Head  │
                                                         │ pass/fail│
                                                         └──────────┘

Usage
-----
  1. Set global variables below (paths, hyperparams)
  2. Run:  python leaf_classifier.py          (trains + exports model)
  3. Run:  python leaf_inference.py            (inference on new images)

Author: Generated for leaf quality classification
"""

import os
import sys
import csv
import json
import random
import math
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# --- Paths ---
DATA_ROOT       = "/data/Train_Good_Bad_Leaves_Exported_RGB"
PASS_DIR        = os.path.join(DATA_ROOT, "pass")
FAIL_DIR        = os.path.join(DATA_ROOT, "fail")
OUTPUT_DIR      = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_convnextv2_AugColor"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

# --- Resume from previous best ---
RESUME_FROM     = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_convnextv2/checkpoints/best_model.pt"

# --- Model ---
BACKBONE_NAME   = "convnext_tiny.fb_in22k_ft_in1k"  # timm model string
TILE_SIZE       = 224          # tile height & width (pixels)
TILE_OVERLAP    = 0.25         # fractional overlap between tiles (0-0.5)
MAX_TILES       = 64           # cap on tiles per image (memory safety)
AGGREGATE       = "mean"       # tile aggregation: "mean", "max", or "attention"
FREEZE_BACKBONE = False        # freeze backbone for first N epochs (see below)
FREEZE_EPOCHS   = 2            # epochs to keep backbone frozen (if enabled)

# --- Training ---
NUM_EPOCHS      = 25
BATCH_SIZE      = 64           # images per batch (tiles are internal)
TILES_PER_IMAGE_TRAIN = 8     # random tiles per image during training
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"     # "cosine" or "step"
STEP_SIZE       = 10           # for step scheduler
STEP_GAMMA      = 0.1
EARLY_STOP_PATIENCE = 7
NUM_WORKERS     = 8
PIN_MEMORY      = True
MIXED_PRECISION = True         # use torch amp

# --- Validation holdout ---
VAL_PER_GENUS   = 1            # hold out N pass + N fail images per genus

# --- Export ---
EXPORT_ONNX     = True
EXPORT_TORCHSCRIPT = True

# --- Resume / Export-only ---
RESUME          = True   # auto-resume training from last checkpoint if one exists
EXPORT_ONLY     = False  # set True to skip training and only run export on best_model.pt

# --- Reproducibility ---
SEED            = 42

# --- Logging ---
LOG_LEVEL       = "INFO"


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
    """
    Hold out `val_per_genus` images per genus from pass and fail *independently*.
    A genus that exists only in pass still gets a pass holdout (and vice versa).
    Genera with only 1 image: that image goes to val (prioritise coverage).
    Returns (train_records, val_records).
    """
    train, val = [], []

    for class_label, records in [("pass", pass_records), ("fail", fail_records)]:
        by_genus = defaultdict(list)
        for r in records:
            by_genus[r["genus"]].append(r)

        for genus, imgs in by_genus.items():
            random.shuffle(imgs)
            n_val = min(val_per_genus, len(imgs))
            held = imgs[:n_val]
            rest = imgs[n_val:]
            val.extend(held)
            train.extend(rest)
            if n_val > 0:
                logger.debug(
                    f"  Holdout {class_label}/{genus}: {n_val} val image(s) "
                    f"({len(rest)} train)"
                )

    # Warn about genera that only appear in one class
    pass_genera = set(r["genus"] for r in pass_records)
    fail_genera = set(r["genus"] for r in fail_records)
    pass_only = pass_genera - fail_genera
    fail_only = fail_genera - pass_genera
    if pass_only:
        logger.warning(
            f"  {len(pass_only)} genera appear ONLY in pass (no fail examples): "
            f"{sorted(pass_only)[:10]}{'...' if len(pass_only)>10 else ''}"
        )
    if fail_only:
        logger.warning(
            f"  {len(fail_only)} genera appear ONLY in fail (no pass examples): "
            f"{sorted(fail_only)[:10]}{'...' if len(fail_only)>10 else ''}"
        )

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ============================================================================
# TILE EXTRACTION
# ============================================================================

def compute_tile_positions(
    img_w: int, img_h: int,
    tile_size: int, overlap: float,
) -> List[Tuple[int, int]]:
    """
    Compute top-left (x, y) positions for a grid of overlapping tiles.
    If the image is smaller than tile_size in a dimension, a single tile
    at position 0 is used (the backbone's global pool handles the smaller input).
    """
    stride = max(1, int(tile_size * (1.0 - overlap)))
    positions = []

    xs = list(range(0, max(1, img_w - tile_size + 1), stride))
    if len(xs) == 0 or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))
    xs = sorted(set(xs))

    ys = list(range(0, max(1, img_h - tile_size + 1), stride))
    if len(ys) == 0 or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))
    ys = sorted(set(ys))

    for y in ys:
        for x in xs:
            positions.append((x, y))

    return positions


def extract_tiles_from_pil(
    img: Image.Image,
    tile_size: int,
    overlap: float,
    max_tiles: int,
    random_sample: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract tiles from a PIL image.  Returns tensor of shape (N, 3, tile_size, tile_size).
    If random_sample is set, randomly pick that many tiles (for training).
    Images smaller than tile_size are padded to tile_size.
    """
    img_w, img_h = img.size  # PIL gives (width, height)

    # Pad small images
    if img_w < tile_size or img_h < tile_size:
        new_w = max(img_w, tile_size)
        new_h = max(img_h, tile_size)
        padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        padded.paste(img, (0, 0))
        img = padded
        img_w, img_h = img.size

    positions = compute_tile_positions(img_w, img_h, tile_size, overlap)

    if random_sample is not None and random_sample < len(positions):
        positions = random.sample(positions, random_sample)
    elif max_tiles < len(positions):
        # Evenly subsample to keep spatial coverage
        step = max(1, len(positions) // max_tiles)
        positions = positions[::step][:max_tiles]

    img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    tiles = []
    for (x, y) in positions:
        crop = img_np[y:y + tile_size, x:x + tile_size]
        # Handle edge crops that might be smaller (shouldn't happen with our logic, but safety)
        if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        tiles.append(crop)

    tiles_np = np.stack(tiles, axis=0)  # (N, H, W, 3)
    tiles_t = torch.from_numpy(tiles_np).permute(0, 3, 1, 2)  # (N, 3, H, W)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tiles_t = (tiles_t - mean) / std

    return tiles_t


# ============================================================================
# DATASET
# ============================================================================

class LeafTileDataset(Dataset):
    """
    Yields (tiles_tensor, label, metadata_dict) per image.
    During training, randomly samples a fixed number of tiles.
    During eval/inference, extracts the full tile grid.
    """

    def __init__(
        self,
        records: List[Dict],
        tile_size: int = TILE_SIZE,
        overlap: float = TILE_OVERLAP,
        max_tiles: int = MAX_TILES,
        random_tiles: Optional[int] = None,
        augment: bool = False,
    ):
        self.records = records
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tiles = max_tiles
        self.random_tiles = random_tiles
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")

        # Augmentations at image level before tiling
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (0, 90, 180, 270)
            rot = random.choice([0, 90, 180, 270])
            if rot != 0:
                img = img.rotate(rot, expand=True)
            # Color augmentations
            from torchvision import transforms as T
            color_aug = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
            img = color_aug(img)

        tiles = extract_tiles_from_pil(
            img, self.tile_size, self.overlap, self.max_tiles,
            random_sample=self.random_tiles,
        )

        label = rec["label"]
        meta = {
            "path": rec["path"],
            "filename": rec.get("filename", ""),
            "family": rec.get("family", ""),
            "genus": rec.get("genus", ""),
        }

        return tiles, label, meta


def collate_tiles(batch):
    """
    Custom collate: each sample has a variable number of tiles.
    Returns:
      all_tiles:  (total_tiles, 3, H, W)
      labels:     (batch_size,)
      tile_counts: list of int — how many tiles per image
      metas:      list of metadata dicts
    """
    all_tiles = []
    labels = []
    tile_counts = []
    metas = []

    for tiles, label, meta in batch:
        all_tiles.append(tiles)
        labels.append(label)
        tile_counts.append(tiles.shape[0])
        metas.append(meta)

    all_tiles = torch.cat(all_tiles, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return all_tiles, labels, tile_counts, metas


# ============================================================================
# MODEL
# ============================================================================

class TileAggregator(nn.Module):
    """Aggregates per-tile features into a single image-level feature."""

    def __init__(self, feat_dim: int, method: str = "mean"):
        super().__init__()
        self.method = method
        if method == "attention":
            self.attn = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        """
        tile_features: (num_tiles, feat_dim)
        returns:       (feat_dim,)
        """
        if self.method == "mean":
            return tile_features.mean(dim=0)
        elif self.method == "max":
            return tile_features.max(dim=0).values
        elif self.method == "attention":
            # Learned attention weighting
            scores = self.attn(tile_features)          # (N, 1)
            weights = F.softmax(scores, dim=0)         # (N, 1)
            return (weights * tile_features).sum(dim=0) # (feat_dim,)
        else:
            raise ValueError(f"Unknown aggregation: {self.method}")


class ConvNeXtV2LeafClassifier(nn.Module):
    """
    Tile-based binary leaf classifier.

    Forward pass:
      1. All tiles (from potentially multiple images) go through backbone
      2. Tiles are split back per image using tile_counts
      3. Per-image tile features are aggregated
      4. Classification head produces logits
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        aggregate: str = AGGREGATE,
        num_classes: int = 2,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        # Create backbone — remove its classifier head
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features

        # Aggregator
        self.aggregator = TileAggregator(self.feat_dim, method=aggregate)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feat_dim, num_classes),
        )

        self.num_classes = num_classes
        logger.info(
            f"ConvNeXtV2LeafClassifier: backbone={backbone_name}, "
            f"feat_dim={self.feat_dim}, aggregate={aggregate}"
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen.")

    def forward_features(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: (N, 3, H, W) -> (N, feat_dim)"""
        return self.backbone(tiles)

    def forward(
        self,
        all_tiles: torch.Tensor,
        tile_counts: List[int],
    ) -> torch.Tensor:
        """
        all_tiles:   (total_tiles, 3, H, W)
        tile_counts:  list of int
        returns:      (batch_size, num_classes)
        """
        # Extract features for all tiles at once
        all_feats = self.forward_features(all_tiles)  # (total_tiles, feat_dim)

        # Split per image and aggregate
        image_feats = []
        offset = 0
        for count in tile_counts:
            tile_feats = all_feats[offset: offset + count]
            agg = self.aggregator(tile_feats)
            image_feats.append(agg)
            offset += count

        image_feats = torch.stack(image_feats, dim=0)  # (batch, feat_dim)
        logits = self.head(image_feats)                 # (batch, num_classes)
        return logits


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(
        self,
        model: ConvNeXtV2LeafClassifier,
        train_records: List[Dict],
        val_records: List[Dict],
    ):
        self.model = model.to(DEVICE)
        self.train_records = train_records
        self.val_records = val_records

        # Datasets
        self.train_ds = LeafTileDataset(
            train_records,
            random_tiles=TILES_PER_IMAGE_TRAIN,
            augment=True,
        )
        self.val_ds = LeafTileDataset(
            val_records,
            random_tiles=None,  # use full tile grid
            augment=False,
        )

        # Weighted sampling to handle class imbalance
        labels = [r["label"] for r in train_records]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            self.train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            collate_fn=collate_tiles, num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY, persistent_workers=True, prefetch_factor=4,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            collate_fn=collate_tiles, num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY, persistent_workers=True, prefetch_factor=4,
        )

        # Loss — with class weights
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        # Scheduler
        if LR_SCHEDULER == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=NUM_EPOCHS
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA
            )

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda")

        # Tracking
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

        for batch_idx, (all_tiles, labels, tile_counts, _) in enumerate(self.train_loader):
            all_tiles = all_tiles.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"):
                logits = self.model(all_tiles, tile_counts)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch} batch {batch_idx+1}/{len(self.train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        # Track per-genus, per-class results with family info
        genus_results = defaultdict(lambda: {
            "family": "", "pass_correct": 0, "pass_total": 0,
            "fail_correct": 0, "fail_total": 0,
        })

        for all_tiles, labels, tile_counts, metas in self.val_loader:
            all_tiles = all_tiles.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"):
                logits = self.model(all_tiles, tile_counts)
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
                if true_label == 0:  # pass
                    genus_results[genus]["pass_total"] += 1
                    genus_results[genus]["pass_correct"] += int(is_correct)
                else:               # fail
                    genus_results[genus]["fail_total"] += 1
                    genus_results[genus]["fail_correct"] += int(is_correct)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, dict(genus_results)

    def train(self) -> str:
        """Full training loop.  Returns path to best checkpoint."""
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")

        if self.training_complete:
            logger.info("Training already complete — skipping to export.")
            return best_ckpt_path

        logger.info(f"Training: {len(self.train_records)} train, {len(self.val_records)} val")

        for epoch in range(self.start_epoch, NUM_EPOCHS + 1):
            # Optional backbone freezing for initial epochs
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

            # Log genus-level performance
            failing_genera = []
            for genus, stats in sorted(genus_results.items()):
                g_total = stats["pass_total"] + stats["fail_total"]
                g_correct = stats["pass_correct"] + stats["fail_correct"]
                genus_acc = g_correct / max(g_total, 1)
                if genus_acc < 1.0:
                    failing_genera.append(f"{genus}({genus_acc:.0%})")
            if failing_genera:
                logger.info(f"  Genera with errors: {', '.join(failing_genera)}")

            # Checkpoint best model
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
                        "tile_size": TILE_SIZE,
                        "tile_overlap": TILE_OVERLAP,
                        "max_tiles": MAX_TILES,
                        "aggregate": AGGREGATE,
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

        # Save final genus report with family and per-class breakdown
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
                p_tot = stats["pass_total"]
                p_cor = stats["pass_correct"]
                f_tot = stats["fail_total"]
                f_cor = stats["fail_correct"]
                o_tot = p_tot + f_tot
                o_cor = p_cor + f_cor
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

class ConvNeXtV2LeafClassifierExport(nn.Module):
    """
    Wrapper for export: takes a single fixed-size tile batch (N, 3, H, W)
    and returns (N, feat_dim) features.  Aggregation + head are separated
    so the exported model is a simple feature-extractor + classifier
    that can be used tile-by-tile.

    For full inference, the caller tiles the image, runs each tile through
    this model, then averages the logits or features externally.
    """

    def __init__(self, model: ConvNeXtV2LeafClassifier):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, tile_size, tile_size) -> (N, num_classes)"""
        feats = self.backbone(x)   # (N, feat_dim)
        logits = self.head(feats)  # (N, num_classes)
        return logits


def export_model(model: ConvNeXtV2LeafClassifier, output_dir: str):
    """Export model to ONNX and TorchScript."""
    model.eval()
    export_wrapper = ConvNeXtV2LeafClassifierExport(model).to(DEVICE).eval()
    dummy = torch.randn(1, 3, TILE_SIZE, TILE_SIZE, device=DEVICE)

    if EXPORT_ONNX:
        onnx_path = os.path.join(output_dir, "leaf_classifier.onnx")
        try:
            torch.onnx.export(
                export_wrapper, dummy, onnx_path,
                input_names=["tiles"],
                output_names=["logits"],
                dynamic_axes={
                    "tiles": {0: "num_tiles"},
                    "logits": {0: "num_tiles"},
                },
                opset_version=17,
            )
            logger.info(f"ONNX model exported to {onnx_path}")
        except ModuleNotFoundError as e:
            logger.warning(f"ONNX export skipped — missing dependency: {e}")
            logger.warning("Fix with:  pip install onnxscript")

    if EXPORT_TORCHSCRIPT:
        ts_path = os.path.join(output_dir, "leaf_classifier_scripted.pt")
        scripted = torch.jit.trace(export_wrapper, dummy)
        scripted.save(ts_path)
        logger.info(f"TorchScript model exported to {ts_path}")

    # Save config alongside model for inference
    config = {
        "backbone_name": BACKBONE_NAME,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "max_tiles": MAX_TILES,
        "aggregate": AGGREGATE,
        "class_names": ["pass", "fail"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")


# ============================================================================
# MAIN TRAINING ENTRYPOINT
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Leaf Pass/Fail Classifier — Training")
    logger.info("=" * 60)

    # Discover images
    logger.info(f"Scanning pass images from: {PASS_DIR}")
    pass_records = discover_images(PASS_DIR, label=0)  # 0 = pass
    logger.info(f"  Found {len(pass_records)} pass images")

    logger.info(f"Scanning fail images from: {FAIL_DIR}")
    fail_records = discover_images(FAIL_DIR, label=1)  # 1 = fail
    logger.info(f"  Found {len(fail_records)} fail images")

    if len(pass_records) == 0 and len(fail_records) == 0:
        logger.error("No images found! Check DATA_ROOT path.")
        sys.exit(1)

    # Count genera
    all_genera = set(r["genus"] for r in pass_records + fail_records)
    logger.info(f"  Total genera: {len(all_genera)}")

    # Split
    train_records, val_records = split_train_val(
        pass_records, fail_records, val_per_genus=VAL_PER_GENUS
    )
    logger.info(f"Split: {len(train_records)} train, {len(val_records)} val")

    # Save split metadata
    split_path = os.path.join(OUTPUT_DIR, "data_split.json")

    # Summarise holdout coverage
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
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(
        f"  Holdout covers {len(val_genera)} genera across {len(val_families)} families"
    )

    # Build model
    model = ConvNeXtV2LeafClassifier(
        backbone_name=BACKBONE_NAME,
        aggregate=AGGREGATE,
    )

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
        # Train (resumes automatically if last_checkpoint.pt exists)
        trainer = Trainer(model, train_records, val_records)
        best_ckpt = trainer.train()

    # Reload best checkpoint
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded best checkpoint (val_acc={ckpt['val_acc']:.4f})")

    # Export
    export_model(model, OUTPUT_DIR)

    logger.info("Training complete!")
    logger.info(f"  Best val accuracy: {ckpt['val_acc']:.4f}")
    logger.info(f"  Outputs saved to:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()