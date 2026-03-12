"""
Leaf Pass/Fail Classifier — DINOv2
====================================
Binary image classifier using DINOv2 (Meta) as a frozen feature extractor
with a lightweight trainable classification head.

DINOv2 is a Vision Transformer (ViT) that produces high-quality visual
features without any fine-tuning.  We freeze the backbone entirely and
train only a small MLP head, which is fast and data-efficient.

Variable input sizes are supported: image dimensions are padded to be
divisible by the patch size (14 px).  For very large images, a tiling
strategy is used (same as the ConvNeXt variant).

QC Feature Visualization
-------------------------
After training, the script generates attention-overlay images showing
which spatial regions of the leaf the model relies on for its pass/fail
decision.  This uses the CLS token's attention weights from the last
transformer layer — a natural "what does the model look at" map
analogous to GradCAM but native to the ViT architecture.

Usage
-----
  python leaf_classifier_dinov2.py

Directory structure expected (in-place, no copying):
  /data/Train_Good_Bad_Leaves_Exported_RGB/
    pass/{family}/{genus}/{filename}.jpg
    fail/{family}/{genus}/{filename}.jpg
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
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# --- Paths ---
DATA_ROOT       = "/data/Train_Good_Bad_Leaves_Exported_RGB"
PASS_DIR        = os.path.join(DATA_ROOT, "pass")
FAIL_DIR        = os.path.join(DATA_ROOT, "fail")
OUTPUT_DIR      = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_dinov2_AugColor"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
QC_DIR          = os.path.join(OUTPUT_DIR, "qc_attention_maps")

# --- Resume from previous best ---
RESUME_FROM     = "/home/brlab/Dropbox/LM2_Leaf_Classifier/output_dinov2/checkpoints/best_model_dinov2.pt"

# --- DINOv2 Model ---
# Available: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
#            (append _reg for register variants)
DINOV2_MODEL    = "dinov2_vitb14"
PATCH_SIZE      = 14            # ViT-B/14 patch size — do not change unless model changes

# --- Tiling ---
# DINOv2 tile size must be divisible by PATCH_SIZE.
# 518 = 14*37 (high-res DINOv2 eval size, richer attention maps)
# 224 = 14*16 (standard, faster)
TILE_SIZE       = 518
TILE_OVERLAP    = 0.25
MAX_TILES       = 32            # cap tiles per image
TILES_PER_IMAGE_TRAIN = 4      # random tiles during training (DINOv2 is heavier)

# --- Training ---
# Only the head is trained; backbone is always frozen.
NUM_EPOCHS      = 25
BATCH_SIZE      = 16
LR              = 3e-4
WEIGHT_DECAY    = 1e-3
HEAD_HIDDEN     = 512           # MLP hidden dim
HEAD_DROPOUT    = 0.3
LR_SCHEDULER    = "cosine"
EARLY_STOP_PATIENCE = 7
NUM_WORKERS     = 8
PIN_MEMORY      = True
MIXED_PRECISION = True

# --- Validation ---
VAL_PER_GENUS   = 1             # 1 pass + 1 fail held out per genus

# --- QC Visualization ---
QC_IMAGES_PER_CLASS = 5         # save N random QC images per pass/fail

# --- Export ---
EXPORT_ONNX     = True
EXPORT_TORCHSCRIPT = True

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

for d in [OUTPUT_DIR, CHECKPOINT_DIR, QC_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================================
# DATA DISCOVERY & SPLITTING  (identical logic to ConvNeXt version)
# ============================================================================

def discover_images(base_dir: str, label: int) -> List[Dict]:
    """Walk pass/ or fail/ directory: pass/{family}/{genus}/*.jpg"""
    records = []
    base = Path(base_dir)
    if not base.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return records
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
    Hold out val_per_genus images per genus from pass and fail independently.
    Genera with only 1 image: that image goes to val (prioritise coverage).
    """
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
            logger.debug(f"  Holdout {class_label}/{genus}: {n_val} val, {len(imgs)-n_val} train")

    pass_genera = set(r["genus"] for r in pass_records)
    fail_genera = set(r["genus"] for r in fail_records)
    pass_only = pass_genera - fail_genera
    fail_only = fail_genera - pass_genera
    if pass_only:
        logger.warning(f"  {len(pass_only)} genera ONLY in pass: {sorted(pass_only)[:10]}...")
    if fail_only:
        logger.warning(f"  {len(fail_only)} genera ONLY in fail: {sorted(fail_only)[:10]}...")

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ============================================================================
# TILE EXTRACTION  (DINOv2-aware: dimensions padded to be divisible by 14)
# ============================================================================

def pad_to_patch(size: int, patch: int) -> int:
    """Round up to nearest multiple of patch."""
    return math.ceil(size / patch) * patch


def compute_tile_positions(img_w, img_h, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, img_w - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))
    xs = sorted(set(xs))
    ys = list(range(0, max(1, img_h - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))
    ys = sorted(set(ys))
    return [(x, y) for y in ys for x in xs]


def extract_tiles_from_pil(
    img: Image.Image,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    max_tiles: int = MAX_TILES,
    random_sample: Optional[int] = None,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """
    Extract tiles, each padded so dimensions are divisible by patch_size.
    Returns (N, 3, H, W) normalized tensor.
    """
    img_w, img_h = img.size

    # Ensure tile_size is divisible by patch_size
    tile_size = pad_to_patch(tile_size, patch_size)

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
        step = max(1, len(positions) // max_tiles)
        positions = positions[::step][:max_tiles]

    img_np = np.array(img, dtype=np.float32) / 255.0

    tiles = []
    for (x, y) in positions:
        crop = img_np[y:y + tile_size, x:x + tile_size]
        if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
            p = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            p[:crop.shape[0], :crop.shape[1]] = crop
            crop = p
        tiles.append(crop)

    tiles_np = np.stack(tiles).transpose(0, 3, 1, 2)  # (N, 3, H, W)
    tiles_t = torch.from_numpy(tiles_np)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tiles_t = (tiles_t - mean) / std
    return tiles_t


# ============================================================================
# DATASET
# ============================================================================

class LeafTileDataset(Dataset):
    def __init__(self, records, tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                 max_tiles=MAX_TILES, random_tiles=None, augment=False):
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
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            rot = random.choice([0, 90, 180, 270])
            if rot:
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
        meta = {k: rec.get(k, "") for k in ("path", "filename", "family", "genus")}
        return tiles, rec["label"], meta


def collate_tiles(batch):
    all_tiles, labels, tile_counts, metas = [], [], [], []
    for tiles, label, meta in batch:
        all_tiles.append(tiles)
        labels.append(label)
        tile_counts.append(tiles.shape[0])
        metas.append(meta)
    return torch.cat(all_tiles, 0), torch.tensor(labels, dtype=torch.long), tile_counts, metas


# ============================================================================
# ATTENTION HOOK — captures CLS attention from last transformer block
# ============================================================================

class AttentionHook:
    """
    Register on the last attention layer of DINOv2 to capture the
    CLS token's attention weights over spatial patches.
    """

    def __init__(self):
        self.attention_map = None  # (batch, num_heads, seq_len, seq_len)
        self._handle = None

    def hook_fn(self, module, input, output):
        """
        DINOv2's attention forward returns (attn_output, attn_weights) when
        output_attentions=True, but by default it only returns attn_output.
        We intercept the QKV computation instead.
        """
        # This gets called on the full Attention module.  We need to
        # recompute attention weights from the stored qkv.
        # The hook is on the attention module; `input` is the tuple of args.
        pass

    def register(self, model):
        """
        Register a hook that captures attention weights from the last block.
        Works with DINOv2 models loaded via torch.hub.
        """
        last_block = model.blocks[-1]
        # We hook the attn.qkv linear layer's output and the attn module itself
        self._qkv_output = None
        self._num_heads = last_block.attn.num_heads

        def qkv_hook(module, input, output):
            self._qkv_output = output

        def attn_hook(module, input, output):
            if self._qkv_output is not None:
                qkv = self._qkv_output
                B, N, _ = input[0].shape
                # Reshape qkv: (B, N, 3*head_dim*num_heads) -> (3, B, num_heads, N, head_dim)
                qkv = qkv.reshape(B, N, 3, self._num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = q.shape[-1] ** -0.5
                attn_weights = (q @ k.transpose(-2, -1)) * scale
                attn_weights = attn_weights.softmax(dim=-1)
                self.attention_map = attn_weights.detach()  # (B, heads, N, N)

        self._handle_qkv = last_block.attn.qkv.register_forward_hook(qkv_hook)
        self._handle_attn = last_block.attn.register_forward_hook(attn_hook)

    def remove(self):
        if self._handle_qkv:
            self._handle_qkv.remove()
        if self._handle_attn:
            self._handle_attn.remove()

    def get_cls_attention(self) -> Optional[torch.Tensor]:
        """
        Returns CLS token attention over spatial patches.
        Shape: (batch, num_patches) — averaged over heads.
        """
        if self.attention_map is None:
            return None
        # CLS token is index 0; its attention over all tokens (including itself)
        cls_attn = self.attention_map[:, :, 0, 1:]  # (B, heads, num_patches)
        # Average across heads
        cls_attn = cls_attn.mean(dim=1)  # (B, num_patches)
        return cls_attn


# ============================================================================
# MODEL
# ============================================================================

class DINOv2LeafClassifier(nn.Module):
    """
    DINOv2 frozen backbone + trainable MLP head for binary classification.
    Tile aggregation: pool CLS features across tiles via mean/attention.
    """

    def __init__(
        self,
        model_name: str = DINOV2_MODEL,
        hidden_dim: int = HEAD_HIDDEN,
        drop_rate: float = HEAD_DROPOUT,
        num_classes: int = 2,
    ):
        super().__init__()

        # Load DINOv2 backbone
        logger.info(f"Loading DINOv2 backbone: {model_name} ...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.backbone.eval()  # always eval mode
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feat_dim = self.backbone.embed_dim
        self.patch_size = PATCH_SIZE

        # Classification head (trainable)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Attention hook for QC visualization
        self.attn_hook = AttentionHook()
        self.attn_hook.register(self.backbone)

        logger.info(
            f"DINOv2LeafClassifier: backbone={model_name}, "
            f"feat_dim={self.feat_dim}, head_hidden={hidden_dim}"
        )

    def forward_features(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: (N, 3, H, W) -> (N, feat_dim) using CLS token."""
        with torch.no_grad():
            # DINOv2 forward_features returns dict with keys:
            # 'x_norm_clstoken', 'x_norm_patchtokens', etc.
            # For older torch.hub versions, it may just return the CLS token.
            out = self.backbone(tiles)
            if isinstance(out, dict):
                cls_feat = out.get("x_norm_clstoken", None)
                if cls_feat is None:
                    cls_feat = out.get("cls_token", list(out.values())[0])
            elif isinstance(out, torch.Tensor):
                cls_feat = out
            else:
                cls_feat = out
        return cls_feat  # (N, feat_dim)

    def forward(
        self,
        all_tiles: torch.Tensor,
        tile_counts: List[int],
    ) -> torch.Tensor:
        """
        all_tiles:   (total_tiles, 3, H, W)
        tile_counts:  list[int]
        returns:      (batch_size, num_classes)
        """
        cls_feats = self.forward_features(all_tiles)  # (total, feat_dim)

        # Aggregate per image: mean of CLS features
        image_feats = []
        offset = 0
        for count in tile_counts:
            tile_feats = cls_feats[offset:offset + count]
            image_feats.append(tile_feats.mean(dim=0))
            offset += count

        image_feats = torch.stack(image_feats, dim=0)
        logits = self.head(image_feats)
        return logits

    def get_tile_attention(self) -> Optional[torch.Tensor]:
        """Get CLS attention maps from the last forward pass."""
        return self.attn_hook.get_cls_attention()


# ============================================================================
# QC ATTENTION VISUALIZATION
# ============================================================================

def create_attention_overlay(
    img: Image.Image,
    attention: np.ndarray,
    tile_x: int, tile_y: int,
    tile_size: int,
    patch_size: int,
    prediction: str,
    confidence: float,
    genus: str,
    family: str,
) -> Image.Image:
    """
    Create a visualization image:
      Left:  original image with tile boundary marked
      Right: tile with attention heatmap overlay

    attention: 1D array of shape (num_patches,) — CLS attention per patch
    """
    # Extract the tile from the original image
    img_w, img_h = img.size
    tile_crop = img.crop((tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))

    # Reshape attention to 2D spatial grid
    grid_h = tile_size // patch_size
    grid_w = tile_size // patch_size
    n_patches = grid_h * grid_w

    if len(attention) >= n_patches:
        attn_2d = attention[:n_patches].reshape(grid_h, grid_w)
    else:
        # pad if needed
        padded = np.zeros(n_patches)
        padded[:len(attention)] = attention
        attn_2d = padded.reshape(grid_h, grid_w)

    # Normalize to 0-1
    attn_min, attn_max = attn_2d.min(), attn_2d.max()
    if attn_max > attn_min:
        attn_2d = (attn_2d - attn_min) / (attn_max - attn_min)

    # Upsample attention to tile resolution
    attn_upsampled = np.array(
        Image.fromarray((attn_2d * 255).astype(np.uint8)).resize(
            (tile_size, tile_size), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # Create heatmap overlay
    if HAS_MATPLOTLIB:
        colormap = cm.get_cmap("jet")
        heatmap_rgba = colormap(attn_upsampled)[:, :, :3]  # (H, W, 3) float 0-1
    else:
        # Simple red-blue colormap fallback
        heatmap_rgba = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        heatmap_rgba[:, :, 0] = attn_upsampled         # red = high attention
        heatmap_rgba[:, :, 2] = 1.0 - attn_upsampled   # blue = low attention

    heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))

    # Blend tile crop with heatmap
    tile_arr = np.array(tile_crop.resize((tile_size, tile_size))).astype(np.float32)
    blend_arr = tile_arr * 0.5 + np.array(heatmap_img).astype(np.float32) * 0.5
    blend_img = Image.fromarray(blend_arr.astype(np.uint8))

    # Compose the output: [original_with_box | tile | heatmap_blend]
    # Scale original for display (max 512 on longest side)
    display_max = 512
    scale = min(display_max / img_w, display_max / img_h, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    img_display = img.resize((disp_w, disp_h), Image.BILINEAR)

    # Draw tile rectangle on the original
    draw = ImageDraw.Draw(img_display)
    rx0 = int(tile_x * scale)
    ry0 = int(tile_y * scale)
    rx1 = int((tile_x + tile_size) * scale)
    ry1 = int((tile_y + tile_size) * scale)
    for offset in range(3):
        draw.rectangle([rx0 - offset, ry0 - offset, rx1 + offset, ry1 + offset],
                       outline="lime" if prediction == "pass" else "red")

    # Resize tile panels
    panel_size = max(disp_h, 256)
    tile_display = tile_crop.resize((panel_size, panel_size), Image.BILINEAR)
    blend_display = blend_img.resize((panel_size, panel_size), Image.BILINEAR)

    # Assemble
    gap = 10
    total_w = disp_w + gap + panel_size + gap + panel_size
    total_h = max(disp_h, panel_size) + 40  # extra for text
    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    y_off = 30
    canvas.paste(img_display, (0, y_off))
    canvas.paste(tile_display, (disp_w + gap, y_off))
    canvas.paste(blend_display, (disp_w + gap + panel_size + gap, y_off))

    # Add text label
    draw_canvas = ImageDraw.Draw(canvas)
    color = (100, 255, 100) if prediction == "pass" else (255, 100, 100)
    label_text = (
        f"{prediction.upper()} ({confidence:.1%}) | "
        f"{family} / {genus}"
    )
    draw_canvas.text((5, 5), label_text, fill=color)
    draw_canvas.text((disp_w + gap, 5 + y_off + panel_size + 2),
                     "Original tile", fill=(200, 200, 200))
    draw_canvas.text((disp_w + gap + panel_size + gap, 5 + y_off + panel_size + 2),
                     "Attention overlay", fill=(200, 200, 200))

    return canvas


def generate_qc_images(
    model: DINOv2LeafClassifier,
    records: List[Dict],
    output_dir: str,
    n_per_class: int = QC_IMAGES_PER_CLASS,
):
    """
    Generate QC attention-overlay images for a random sample of pass and fail.
    Saves n_per_class images for each class.
    """
    model.eval()
    logger.info(f"Generating QC attention visualizations ({n_per_class} per class)...")

    # Separate by class
    pass_recs = [r for r in records if r["label"] == 0]
    fail_recs = [r for r in records if r["label"] == 1]

    for class_name, class_recs in [("pass", pass_recs), ("fail", fail_recs)]:
        sample = random.sample(class_recs, min(n_per_class, len(class_recs)))

        for i, rec in enumerate(sample):
            try:
                img = Image.open(rec["path"]).convert("RGB")
                img_w, img_h = img.size

                # Extract tiles for this single image
                tile_size = pad_to_patch(TILE_SIZE, PATCH_SIZE)
                tiles = extract_tiles_from_pil(
                    img, tile_size, TILE_OVERLAP, MAX_TILES, random_sample=None
                )
                positions = compute_tile_positions(
                    max(img_w, tile_size), max(img_h, tile_size),
                    tile_size, TILE_OVERLAP
                )
                if MAX_TILES < len(positions):
                    step = max(1, len(positions) // MAX_TILES)
                    positions = positions[::step][:MAX_TILES]

                tiles_dev = tiles.to(DEVICE)

                with torch.no_grad(), torch.amp.autocast(
                    "cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"
                ):
                    logits = model(tiles_dev, [tiles_dev.shape[0]])
                    probs = torch.softmax(logits, dim=1)

                prob_pass = probs[0, 0].item()
                prob_fail = probs[0, 1].item()
                prediction = "pass" if prob_pass > prob_fail else "fail"
                confidence = max(prob_pass, prob_fail)

                # Get attention maps — one per tile
                cls_attn = model.get_tile_attention()  # (num_tiles, num_patches)

                if cls_attn is None:
                    logger.warning(f"  No attention captured for {rec['filename']}")
                    continue

                cls_attn_np = cls_attn.cpu().numpy()

                # Find the tile with the highest maximum attention (most "decisive")
                tile_max_attn = cls_attn_np.max(axis=1)  # (num_tiles,)
                best_tile_idx = int(tile_max_attn.argmax())

                # Ensure we have a valid position
                if best_tile_idx < len(positions):
                    tx, ty = positions[best_tile_idx]
                else:
                    tx, ty = positions[0]

                attn_vec = cls_attn_np[best_tile_idx]

                qc_img = create_attention_overlay(
                    img, attn_vec, tx, ty, tile_size, PATCH_SIZE,
                    prediction, confidence,
                    rec.get("genus", ""), rec.get("family", ""),
                )

                out_name = f"qc_{class_name}_{i+1}_{rec['genus']}_{rec['filename']}"
                out_name = out_name.replace("/", "_")  # safety
                out_path = os.path.join(output_dir, out_name)
                if not out_path.lower().endswith((".jpg", ".png")):
                    out_path += ".png"
                qc_img.save(out_path)
                logger.info(
                    f"  QC [{class_name}] {rec['genus']}/{rec['filename']}: "
                    f"{prediction} ({confidence:.1%}) -> {out_name}"
                )

            except Exception as e:
                logger.error(f"  QC error for {rec['path']}: {e}")
                import traceback; traceback.print_exc()


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, model: DINOv2LeafClassifier, train_records, val_records):
        self.model = model.to(DEVICE)
        self.train_records = train_records
        self.val_records = val_records

        self.train_ds = LeafTileDataset(
            train_records, random_tiles=TILES_PER_IMAGE_TRAIN, augment=True,
        )
        self.val_ds = LeafTileDataset(
            val_records, random_tiles=None, augment=False,
        )

        # Weighted sampling
        labels = [r["label"] for r in train_records]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            self.train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            collate_fn=collate_tiles, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=True, prefetch_factor=4,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            collate_fn=collate_tiles, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=True, prefetch_factor=4,
        )

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Only optimize head parameters (backbone is frozen)
        self.optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        if LR_SCHEDULER == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=NUM_EPOCHS
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=MIXED_PRECISION and DEVICE.type == "cuda"
        )
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        self.model.head.train()
        self.model.backbone.eval()  # always eval
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
            torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=1.0)
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

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
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
                g = meta["genus"]
                genus_results[g]["family"] = meta["family"]
                lbl = labels[i].item()
                ok = int(preds[i] == labels[i])
                if lbl == 0:
                    genus_results[g]["pass_total"] += 1
                    genus_results[g]["pass_correct"] += ok
                else:
                    genus_results[g]["fail_total"] += 1
                    genus_results[g]["fail_correct"] += ok

        return total_loss / max(total, 1), correct / max(total, 1), dict(genus_results)

    def train(self):
        logger.info(f"Training: {len(self.train_records)} train, {len(self.val_records)} val")
        logger.info("Backbone FROZEN — training only the classification head.")
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_dinov2.pt")

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, genus_results = self.validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | LR={lr:.2e}"
            )

            # Log failing genera
            failing = []
            for g, s in sorted(genus_results.items()):
                t = s["pass_total"] + s["fail_total"]
                c = s["pass_correct"] + s["fail_correct"]
                if c < t:
                    failing.append(f"{g}({c}/{t})")
            if failing:
                logger.info(f"  Genera with errors: {', '.join(failing)}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "head_state_dict": self.model.head.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "dinov2_model": DINOV2_MODEL,
                        "tile_size": TILE_SIZE,
                        "patch_size": PATCH_SIZE,
                        "tile_overlap": TILE_OVERLAP,
                        "max_tiles": MAX_TILES,
                        "head_hidden": HEAD_HIDDEN,
                        "head_dropout": HEAD_DROPOUT,
                        "feat_dim": self.model.feat_dim,
                    },
                }, best_ckpt_path)
                logger.info(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Final genus report
        _, _, final_genus = self.validate()
        report_path = os.path.join(OUTPUT_DIR, "genus_validation_report_dinov2.csv")
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "family", "genus",
                "pass_correct", "pass_total", "pass_accuracy",
                "fail_correct", "fail_total", "fail_accuracy",
                "overall_correct", "overall_total", "overall_accuracy",
            ])
            for g, s in sorted(final_genus.items(), key=lambda x: (x[1]["family"], x[0])):
                pt, pc = s["pass_total"], s["pass_correct"]
                ft, fc = s["fail_total"], s["fail_correct"]
                ot, oc = pt + ft, pc + fc
                w.writerow([
                    s["family"], g,
                    pc, pt, f"{pc/max(pt,1):.4f}",
                    fc, ft, f"{fc/max(ft,1):.4f}",
                    oc, ot, f"{oc/max(ot,1):.4f}",
                ])
        logger.info(f"Genus report saved to {report_path}")

        return best_ckpt_path


# ============================================================================
# EXPORT  (ONNX + TorchScript)
# ============================================================================

class DINOv2Export(nn.Module):
    """
    Export wrapper: frozen DINOv2 backbone + head.
    Takes (N, 3, H, W) tiles -> (N, 2) logits.
    Caller aggregates tile logits externally (mean).
    """

    def __init__(self, backbone, head, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", list(feats.values())[0])
        logits = self.head(feats)
        return logits


def export_model(model: DINOv2LeafClassifier, output_dir: str):
    model.eval()
    export_wrapper = DINOv2Export(
        model.backbone, model.head, model.feat_dim
    ).to(DEVICE).eval()
    tile_size = pad_to_patch(TILE_SIZE, PATCH_SIZE)
    dummy = torch.randn(1, 3, tile_size, tile_size, device=DEVICE)

    if EXPORT_ONNX:
        onnx_path = os.path.join(output_dir, "leaf_classifier_dinov2.onnx")
        try:
            torch.onnx.export(
                export_wrapper, dummy, onnx_path,
                input_names=["tiles"], output_names=["logits"],
                dynamic_axes={"tiles": {0: "num_tiles"}, "logits": {0: "num_tiles"}},
                opset_version=17,
            )
            logger.info(f"ONNX exported to {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed (common with ViT ops): {e}")

    if EXPORT_TORCHSCRIPT:
        ts_path = os.path.join(output_dir, "leaf_classifier_dinov2_traced.pt")
        try:
            scripted = torch.jit.trace(export_wrapper, dummy)
            scripted.save(ts_path)
            logger.info(f"TorchScript exported to {ts_path}")
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")

    config = {
        "dinov2_model": DINOV2_MODEL,
        "tile_size": TILE_SIZE,
        "patch_size": PATCH_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "max_tiles": MAX_TILES,
        "class_names": ["pass", "fail"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    config_path = os.path.join(output_dir, "model_config_dinov2.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Leaf Pass/Fail Classifier — DINOv2")
    logger.info("=" * 60)

    # Discover images
    logger.info(f"Scanning pass: {PASS_DIR}")
    pass_records = discover_images(PASS_DIR, label=0)
    logger.info(f"  {len(pass_records)} pass images")

    logger.info(f"Scanning fail: {FAIL_DIR}")
    fail_records = discover_images(FAIL_DIR, label=1)
    logger.info(f"  {len(fail_records)} fail images")

    if not pass_records and not fail_records:
        logger.error("No images found! Check DATA_ROOT.")
        sys.exit(1)

    all_genera = set(r["genus"] for r in pass_records + fail_records)
    all_families = set(r["family"] for r in pass_records + fail_records)
    logger.info(f"  {len(all_genera)} genera across {len(all_families)} families")

    # Split
    train_records, val_records = split_train_val(
        pass_records, fail_records, val_per_genus=VAL_PER_GENUS
    )
    logger.info(f"Split: {len(train_records)} train, {len(val_records)} val")
    logger.info(
        f"  Train: {sum(1 for r in train_records if r['label']==0)} pass, "
        f"{sum(1 for r in train_records if r['label']==1)} fail"
    )
    logger.info(
        f"  Val:   {sum(1 for r in val_records if r['label']==0)} pass, "
        f"{sum(1 for r in val_records if r['label']==1)} fail"
    )

    # Save split
    split_path = os.path.join(OUTPUT_DIR, "data_split_dinov2.json")
    val_genera = set(r["genus"] for r in val_records)
    split_info = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "val_per_genus": VAL_PER_GENUS,
        "total_genera": len(all_genera),
        "val_genera": len(val_genera),
        "total_families": len(all_families),
        "val_files": [
            {"path": r["path"], "label": "pass" if r["label"]==0 else "fail",
             "family": r["family"], "genus": r["genus"]}
            for r in sorted(val_records, key=lambda x: (x["family"], x["genus"]))
        ],
    }
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    # Build model
    model = DINOv2LeafClassifier()

    # Load head weights from a previous best checkpoint if specified
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        prev_ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        model.head.load_state_dict(prev_ckpt["head_state_dict"])
        logger.info(
            f"Initialised head weights from {RESUME_FROM} "
            f"(val_acc={prev_ckpt.get('val_acc', 'N/A')})"
        )
    elif RESUME_FROM:
        logger.warning(f"RESUME_FROM path not found: {RESUME_FROM} — training from scratch")

    # Train
    trainer = Trainer(model, train_records, val_records)
    best_ckpt = trainer.train()

    # Reload best head weights
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.head.load_state_dict(ckpt["head_state_dict"])
    logger.info(f"Loaded best checkpoint (val_acc={ckpt['val_acc']:.4f})")

    # Generate QC attention-overlay images
    all_records = pass_records + fail_records
    generate_qc_images(model, all_records, QC_DIR, n_per_class=QC_IMAGES_PER_CLASS)

    # Export
    export_model(model, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("DINOv2 training complete!")
    logger.info(f"  Best val accuracy: {ckpt['val_acc']:.4f}")
    logger.info(f"  QC images:         {QC_DIR}")
    logger.info(f"  Outputs:           {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()