"""
Ensemble QC Visualization Tool
===============================
Loads all 3 trained models (EfficientNet-B3, ConvNeXt V2, DINOv2) and
generates rich QC images for a random subset of train + val images.

Visualizations per image:
  1. Per-model panels with GradCAM (EfficientNet, ConvNeXt) or native
     attention maps (DINOv2) showing what each model focuses on.
  2. Ensemble summary panel showing all 3 votes, confidence bars,
     and the final consensus determination.
  3. Tile importance heatmaps for the tile-based models.

Usage:
  python run_QC.py

  Configure paths below or override via environment variables.
  Requires: torch, timm, PIL, numpy, matplotlib
"""

import os
import sys
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Model output directories ---
EFFICIENTNET_DIR = SCRIPT_DIR / "output_efficientnet_b3_AugColor"
CONVNEXTV2_DIR   = SCRIPT_DIR / "output_convnextv2_AugColor"
DINOV2_DIR       = SCRIPT_DIR / "output_dinov2_AugColor"

# --- Data root (for discovering train images) ---
DATA_ROOT = "/data/Train_Good_Bad_Leaves_Exported_RGB"

# --- Output ---
QC_OUTPUT_DIR = SCRIPT_DIR / "qc_output"

# --- Sampling ---
N_TRAIN = 10  # random training images to visualise
N_VAL   = 10  # random validation images to visualise
SEED    = 42

# --- Device ---
GPU_ID = 1  # which CUDA device to use (set to 0, 1, etc.)
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
MIXED_PRECISION = True

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ============================================================================
# TILE EXTRACTION (shared)
# ============================================================================

def pad_to_patch(size: int, patch: int) -> int:
    return math.ceil(size / patch) * patch


def compute_tile_positions(img_w, img_h, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, img_w - tile_size + 1), stride))
    if len(xs) == 0 or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))
    xs = sorted(set(xs))
    ys = list(range(0, max(1, img_h - tile_size + 1), stride))
    if len(ys) == 0 or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))
    ys = sorted(set(ys))
    return [(x, y) for y in ys for x in xs]


def extract_tiles_torch(
    img: Image.Image, tile_size: int, overlap: float, max_tiles: int,
    patch_align: int = 1,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Extract tiles and return (tiles_tensor, positions).
    tiles_tensor: (N, 3, tile_size, tile_size) normalised.
    """
    if patch_align > 1:
        tile_size = pad_to_patch(tile_size, patch_align)

    img_w, img_h = img.size
    if img_w < tile_size or img_h < tile_size:
        padded = Image.new("RGB", (max(img_w, tile_size), max(img_h, tile_size)), (0, 0, 0))
        padded.paste(img, (0, 0))
        img = padded
        img_w, img_h = img.size

    positions = compute_tile_positions(img_w, img_h, tile_size, overlap)
    if max_tiles < len(positions):
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

    tiles_np = np.stack(tiles).transpose(0, 3, 1, 2)
    tiles_t = torch.from_numpy(tiles_np)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tiles_t = (tiles_t - mean) / std

    return tiles_t, positions


# ============================================================================
# GRADCAM UTILITY
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works on any CNN by hooking a target conv layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Returns GradCAM heatmap(s) of shape (B, H, W), values in [0, 1].
        class_idx: which class logit to backprop through (0=pass).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[:, class_idx].sum()
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # Normalise per sample
        B = cam.shape[0]
        cam_np = cam.cpu().numpy()
        for i in range(B):
            c = cam_np[i]
            cmin, cmax = c.min(), c.max()
            if cmax > cmin:
                cam_np[i] = (c - cmin) / (cmax - cmin)
            else:
                cam_np[i] = np.zeros_like(c)
        return cam_np  # (B, H, W)

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ============================================================================
# MODEL LOADERS
# ============================================================================

def load_efficientnet(model_dir: str, device: torch.device):
    """Load EfficientNet-B3 from checkpoint. Returns (model, config)."""
    config_path = os.path.join(model_dir, "model_config.json")
    ckpt_path = os.path.join(model_dir, "checkpoints", "best_model.pt")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    from train_EfficientNet_B3 import EfficientNetB3LeafClassifier
    model = EfficientNetB3LeafClassifier(
        backbone_name=config.get("backbone_name", "efficientnet_b3.ra2_in1k"),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_acc = ckpt.get("val_acc", "N/A")
    logger.info(f"  EfficientNet-B3 loaded (val_acc={val_acc})")
    return model, config


def load_convnextv2(model_dir: str, device: torch.device):
    """Load ConvNeXt V2 from checkpoint. Returns (model, config)."""
    config_path = os.path.join(model_dir, "model_config.json")
    ckpt_path = os.path.join(model_dir, "checkpoints", "best_model.pt")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    from train_ConvNeXt_V2 import ConvNeXtV2LeafClassifier
    model = ConvNeXtV2LeafClassifier(
        backbone_name=config.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k"),
        aggregate=config.get("aggregate", "mean"),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_acc = ckpt.get("val_acc", "N/A")
    logger.info(f"  ConvNeXt V2 loaded (val_acc={val_acc})")
    return model, config


def load_dinov2(model_dir: str, device: torch.device):
    """Load DINOv2 from checkpoint. Returns (model, config)."""
    config_path = os.path.join(model_dir, "model_config_dinov2.json")
    ckpt_path = os.path.join(model_dir, "checkpoints", "best_model_dinov2.pt")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    from train_DinoV2 import DINOv2LeafClassifier
    model = DINOv2LeafClassifier(
        model_name=config.get("dinov2_model", "dinov2_vitb14"),
        hidden_dim=config.get("head_hidden", 512),
        drop_rate=config.get("head_dropout", 0.3),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.to(device).eval()
    val_acc = ckpt.get("val_acc", "N/A")
    logger.info(f"  DINOv2 loaded (val_acc={val_acc})")
    return model, config


# ============================================================================
# PER-MODEL INFERENCE + HEATMAPS
# ============================================================================

def run_efficientnet(model, config, img: Image.Image, device):
    """
    Run EfficientNet-B3 with GradCAM.
    Returns dict with label, confidence, prob_pass, prob_fail, gradcam_heatmap.
    """
    image_size = config.get("image_size", 300)
    img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    input_t = torch.from_numpy(arr).unsqueeze(0).to(device)

    # Find the last conv block in the EfficientNet backbone for GradCAM
    # timm EfficientNet: backbone.blocks is a Sequential of InvertedResidual blocks
    target_layer = model.backbone.blocks[-1]

    # Need gradients for GradCAM
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    gradcam = GradCAM(model, target_layer)
    input_t.requires_grad_(True)

    # Forward + GradCAM for the predicted class
    with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and device.type == "cuda"):
        logits = model(input_t)
        probs = torch.softmax(logits, dim=1)

    prob_pass = probs[0, 0].item()
    prob_fail = probs[0, 1].item()
    pred_class = 0 if prob_pass >= 0.5 else 1

    # GradCAM for the predicted class
    cam_map = gradcam(input_t, class_idx=pred_class)  # (1, H, W)
    gradcam.remove()

    # Reset requires_grad
    for p in model.parameters():
        p.requires_grad_(False)

    # Upsample heatmap to image size
    cam_heatmap = cam_map[0]  # (H, W)
    cam_pil = Image.fromarray((cam_heatmap * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((image_size, image_size), Image.BILINEAR)
    cam_heatmap = np.array(cam_pil).astype(np.float32) / 255.0

    label = "pass" if prob_pass >= 0.5 else "fail"
    return {
        "label": label,
        "confidence": max(prob_pass, prob_fail),
        "prob_pass": prob_pass,
        "prob_fail": prob_fail,
        "heatmap": cam_heatmap,
        "display_img": img_resized,
        "method": "GradCAM",
    }


def run_convnextv2(model, config, img: Image.Image, device):
    """
    Run ConvNeXt V2 with GradCAM on the most important tile,
    plus a tile importance map across the full image.
    Returns dict with label, confidence, heatmaps, tile_importance.
    """
    tile_size = config.get("tile_size", 224)
    overlap = config.get("tile_overlap", 0.25)
    max_tiles = config.get("max_tiles", 64)

    tiles_t, positions = extract_tiles_torch(img, tile_size, overlap, max_tiles)
    tiles_dev = tiles_t.to(device)
    n_tiles = tiles_dev.shape[0]

    # Get per-tile features to see tile importance
    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=MIXED_PRECISION and device.type == "cuda"
    ):
        logits = model(tiles_dev, [n_tiles])
        probs = torch.softmax(logits, dim=1)

    prob_pass = probs[0, 0].item()
    prob_fail = probs[0, 1].item()
    label = "pass" if prob_pass >= 0.5 else "fail"

    # Per-tile logits for importance map (run tiles individually through
    # backbone+head to get per-tile classification confidence)
    tile_pass_probs = []
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=MIXED_PRECISION and device.type == "cuda"
    ):
        all_feats = model.forward_features(tiles_dev)  # (N, feat_dim)
        for ti in range(n_tiles):
            feat = all_feats[ti:ti+1]  # (1, feat_dim)
            tile_logit = model.head(feat)
            tile_prob = torch.softmax(tile_logit, dim=1)
            tile_pass_probs.append(tile_prob[0, 0].item())
    tile_pass_probs = np.array(tile_pass_probs)

    # Build tile importance overlay on original image
    img_w, img_h = img.size
    padded_w = max(img_w, tile_size)
    padded_h = max(img_h, tile_size)
    importance_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    for (x, y), tp in zip(positions, tile_pass_probs):
        # For pass prediction, show pass probability; for fail, show fail probability
        val = tp if label == "pass" else (1.0 - tp)
        importance_map[y:y + tile_size, x:x + tile_size] += val
        count_map[y:y + tile_size, x:x + tile_size] += 1.0

    count_map = np.maximum(count_map, 1.0)
    importance_map = importance_map / count_map
    importance_map = importance_map[:img_h, :img_w]

    # GradCAM on the single most confident tile
    best_tile_idx = int(np.argmax(tile_pass_probs) if label == "pass"
                        else np.argmin(tile_pass_probs))
    best_tile = tiles_dev[best_tile_idx:best_tile_idx + 1]

    # Find last conv stage in ConvNeXt backbone for GradCAM
    target_layer = model.backbone.stages[-1]

    for p in model.parameters():
        p.requires_grad_(True)

    # Wrap backbone+head so GradCAM backprops through class logits,
    # not raw backbone features.
    class _SingleTileModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            feat = self.backbone(x)
            return self.head(feat)

    wrapper = _SingleTileModel(model.backbone, model.head)
    gradcam = GradCAM(wrapper, target_layer)
    pred_class = 0 if prob_pass >= 0.5 else 1

    with torch.amp.autocast("cuda", enabled=MIXED_PRECISION and device.type == "cuda"):
        cam_map = gradcam(best_tile, class_idx=pred_class)
    gradcam.remove()

    for p in model.parameters():
        p.requires_grad_(False)

    # Upsample GradCAM to tile resolution
    cam_heatmap = cam_map[0]  # (h_feat, w_feat)
    cam_pil = Image.fromarray((cam_heatmap * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((tile_size, tile_size), Image.BILINEAR)
    cam_tile = np.array(cam_pil).astype(np.float32) / 255.0

    # Map best tile's GradCAM onto the full original image
    best_x, best_y = positions[best_tile_idx]
    full_heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    x_end = min(best_x + tile_size, img_w)
    y_end = min(best_y + tile_size, img_h)
    full_heatmap[best_y:y_end, best_x:x_end] = cam_tile[:y_end - best_y, :x_end - best_x]

    return {
        "label": label,
        "confidence": max(prob_pass, prob_fail),
        "prob_pass": prob_pass,
        "prob_fail": prob_fail,
        "heatmap": full_heatmap,
        "display_img": img,
        "importance_map": importance_map,
        "tile_positions": positions,
        "tile_pass_probs": tile_pass_probs,
        "best_tile_idx": best_tile_idx,
        "method": "GradCAM (best tile)",
    }


def run_dinov2(model, config, img: Image.Image, device):
    """
    Run DINOv2 with patch-level classification maps.

    Instead of CLS attention (which shows general ViT focus, not classification
    relevance), project each patch token through the classification head to see
    what each image region individually predicts as pass/fail.

    Row 1 heatmap: patch-level classification map (all tiles aggregated)
    Row 2 heatmap: tile-level importance (per-tile CLS → head)
    """
    tile_size = config.get("tile_size", 518)
    overlap = config.get("tile_overlap", 0.25)
    max_tiles = config.get("max_tiles", 32)
    patch_size = config.get("patch_size", 14)

    tiles_t, positions = extract_tiles_torch(
        img, tile_size, overlap, max_tiles, patch_align=patch_size
    )
    actual_tile_size = pad_to_patch(tile_size, patch_size)
    tiles_dev = tiles_t.to(device)
    n_tiles = tiles_dev.shape[0]

    grid_h = actual_tile_size // patch_size
    grid_w = actual_tile_size // patch_size
    n_patches = grid_h * grid_w
    img_w, img_h = img.size

    # Single backbone forward pass — get both CLS and patch tokens
    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=MIXED_PRECISION and device.type == "cuda"
    ):
        backbone_out = model.backbone(tiles_dev)

    # Extract CLS and patch tokens
    if isinstance(backbone_out, dict):
        cls_tokens = backbone_out.get("x_norm_clstoken", None)
        patch_tokens = backbone_out.get("x_norm_patchtokens", None)
        if cls_tokens is None:
            cls_tokens = backbone_out.get("cls_token", list(backbone_out.values())[0])
    else:
        cls_tokens = backbone_out
        patch_tokens = None

    # Classify: aggregate CLS tokens → head
    with torch.no_grad():
        agg_cls = cls_tokens.mean(dim=0, keepdim=True)  # (1, feat_dim)
        logits = model.head(agg_cls)
        probs = torch.softmax(logits, dim=1)

    prob_pass = probs[0, 0].item()
    prob_fail = probs[0, 1].item()
    label = "pass" if prob_pass >= 0.5 else "fail"

    # Per-tile importance: each tile's CLS → head
    tile_pass_probs = []
    with torch.no_grad():
        for ti in range(n_tiles):
            tile_logit = model.head(cls_tokens[ti:ti + 1])
            tile_prob = torch.softmax(tile_logit, dim=1)
            tile_pass_probs.append(tile_prob[0, 0].item())
    tile_pass_probs = np.array(tile_pass_probs)

    # Build tile-level importance map (Row 2)
    padded_w = max(img_w, actual_tile_size)
    padded_h = max(img_h, actual_tile_size)
    importance_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    for (x, y), tp in zip(positions, tile_pass_probs):
        val = tp if label == "pass" else (1.0 - tp)
        importance_map[y:y + actual_tile_size, x:x + actual_tile_size] += val
        count_map[y:y + actual_tile_size, x:x + actual_tile_size] += 1.0

    count_map = np.maximum(count_map, 1.0)
    importance_map = importance_map / count_map
    importance_map = importance_map[:img_h, :img_w]

    # Patch-level classification map (Row 1): project each patch token
    # through the head to see per-region pass/fail predictions
    patch_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    patch_count = np.zeros((padded_h, padded_w), dtype=np.float32)

    if patch_tokens is not None:
        with torch.no_grad():
            for ti in range(n_tiles):
                patches = patch_tokens[ti]  # (num_patches, feat_dim)
                if patches.shape[0] >= n_patches:
                    patches = patches[:n_patches]
                else:
                    pad = torch.zeros(
                        n_patches - patches.shape[0], patches.shape[1],
                        device=device, dtype=patches.dtype
                    )
                    patches = torch.cat([patches, pad], dim=0)

                # Each patch token → head → pass/fail probability
                patch_logits = model.head(patches)  # (n_patches, 2)
                patch_probs = torch.softmax(patch_logits, dim=1)

                # Show predicted-class confidence per patch
                if label == "pass":
                    patch_vals = patch_probs[:, 0].cpu().numpy()
                else:
                    patch_vals = patch_probs[:, 1].cpu().numpy()

                spatial = patch_vals.reshape(grid_h, grid_w)

                # Upsample to tile pixel resolution
                smin, smax = spatial.min(), spatial.max()
                if smax > smin:
                    spatial_norm = (spatial - smin) / (smax - smin)
                else:
                    spatial_norm = np.zeros_like(spatial)
                spatial_up = np.array(
                    Image.fromarray(
                        (spatial_norm * 255).astype(np.uint8)
                    ).resize((actual_tile_size, actual_tile_size), Image.BILINEAR)
                ).astype(np.float32) / 255.0

                x, y = positions[ti]
                patch_map[y:y + actual_tile_size, x:x + actual_tile_size] += spatial_up
                patch_count[y:y + actual_tile_size, x:x + actual_tile_size] += 1.0

    patch_count = np.maximum(patch_count, 1.0)
    patch_map = patch_map / patch_count
    patch_map = patch_map[:img_h, :img_w]

    # Normalise
    pmin, pmax = patch_map.min(), patch_map.max()
    if pmax > pmin:
        patch_map = (patch_map - pmin) / (pmax - pmin)

    return {
        "label": label,
        "confidence": max(prob_pass, prob_fail),
        "prob_pass": prob_pass,
        "prob_fail": prob_fail,
        "heatmap": patch_map,
        "display_img": img,
        "full_attention": importance_map,
        "tile_positions": positions,
        "tile_pass_probs": tile_pass_probs,
        "method": "Patch Classification",
    }


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def apply_heatmap_overlay(img_pil: Image.Image, heatmap: np.ndarray,
                          alpha: float = 0.5, cmap_name: str = "jet") -> np.ndarray:
    """Blend a [0,1] heatmap with an RGB image. Returns (H, W, 3) uint8 array."""
    img_np = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    # Resize heatmap to match image
    hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    hm_pil = hm_pil.resize((w, h), Image.BILINEAR)
    hm = np.array(hm_pil).astype(np.float32) / 255.0

    cmap = matplotlib.colormaps[cmap_name]
    hm_color = cmap(hm)[:, :, :3]  # (H, W, 3)
    blended = (1 - alpha) * img_np + alpha * hm_color
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def confidence_bar_color(prob: float, is_pass: bool) -> str:
    """Return a colour string for a confidence bar."""
    if is_pass:
        return "#2ecc71" if prob >= 0.7 else "#f39c12" if prob >= 0.5 else "#e74c3c"
    else:
        return "#e74c3c" if prob >= 0.7 else "#f39c12" if prob >= 0.5 else "#2ecc71"


# ============================================================================
# MAIN COMPOSITE QC IMAGE GENERATOR
# ============================================================================

def create_per_model_panel(ax, result: dict, model_name: str, original_img: Image.Image):
    """Draw a single model's panel: display image with heatmap overlay + info."""
    overlay = apply_heatmap_overlay(result["display_img"], result["heatmap"], alpha=0.45)
    ax.imshow(overlay)

    label = result["label"]
    conf = result["confidence"]
    method = result["method"]
    color = "#2ecc71" if label == "pass" else "#e74c3c"

    ax.set_title(
        f"{model_name}\n{method}",
        fontsize=10, fontweight="bold", pad=8,
    )

    # Prediction badge at bottom
    ax.text(
        0.5, -0.02,
        f"{label.upper()} ({conf:.1%})",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
    )
    ax.axis("off")


def create_full_image_heatmap_panel(ax, original_img: Image.Image,
                                     heatmap: np.ndarray, title: str):
    """Show a full-image heatmap (tile importance or attention) overlaid with a color key."""
    overlay = apply_heatmap_overlay(original_img, heatmap, alpha=0.5)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    # Add colorbar legend at the bottom showing importance scale
    cmap = matplotlib.colormaps["jet"]
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient_colored = cmap(gradient)[0, :, :3]  # drop alpha channel

    # Create a small inset axis for the colorbar
    cax = inset_axes(ax, width="100%", height="8%", loc="lower center",
                     borderpad=-0.5)
    cax.imshow(gradient_colored.reshape(1, -1, 3))
    cax.set_xticks([0, 255])
    cax.set_xticklabels(["Less", "More"], fontsize=7)
    cax.set_yticks([])
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["left"].set_visible(False)
    cax.set_xlabel("Importance", fontsize=7, labelpad=2)

    ax.axis("off")


def create_ensemble_panel(fig, gs_slot, results: dict, true_label: str,
                           genus: str, family: str):
    """
    Draw the ensemble summary panel with:
      - Horizontal confidence bars per model
      - Vote tally
      - Final determination badges for all strictness levels
    """
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    model_names = ["EfficientNet-B3", "ConvNeXt V2", "DINOv2"]
    model_keys = ["efficientnet", "convnextv2", "dinov2"]

    # Title
    ax.text(5, 9.7, "ENSEMBLE SUMMARY", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#2c3e50")

    # Ground truth badge
    gt_color = "#2ecc71" if true_label == "pass" else "#e74c3c"
    ax.text(5, 9.1, f"Ground Truth: {true_label.upper()}", ha="center", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=gt_color, alpha=0.85))

    ax.text(5, 8.4, f"{family} / {genus}", ha="center", va="top",
            fontsize=8, fontstyle="italic", color="#7f8c8d")

    # Per-model confidence bars
    bar_top = 7.6
    bar_height = 0.45
    bar_left = 3.2
    bar_width = 5.5

    n_pass = 0
    for i, (name, key) in enumerate(zip(model_names, model_keys)):
        r = results.get(key)
        if r is None:
            continue

        y = bar_top - i * 1.4
        pp = r["prob_pass"]
        pf = r["prob_fail"]
        label = r["label"]
        is_pass = label == "pass"
        if is_pass:
            n_pass += 1

        # Model name
        ax.text(bar_left - 0.2, y + bar_height / 2, name, ha="right", va="center",
                fontsize=8, fontweight="bold", color="#2c3e50")

        # Background bar (fail = red side)
        ax.barh(y, bar_width, height=bar_height, left=bar_left,
                color="#fadbd8", edgecolor="#e74c3c", linewidth=0.5)

        # Pass portion (green, from left)
        ax.barh(y, bar_width * pp, height=bar_height, left=bar_left,
                color="#2ecc71", alpha=0.85)

        # Percentages
        ax.text(bar_left + bar_width * pp / 2, y + bar_height / 2,
                f"P:{pp:.0%}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.text(bar_left + bar_width * (pp + pf / 2), y + bar_height / 2,
                f"F:{pf:.0%}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

        # Vote icon
        vote_icon = "PASS" if is_pass else "FAIL"
        vote_color = "#2ecc71" if is_pass else "#e74c3c"
        ax.text(bar_left + bar_width + 0.3, y + bar_height / 2,
                vote_icon, ha="left", va="center",
                fontsize=8, fontweight="bold", color=vote_color)

    # Vote tally
    tally_y = bar_top - 3 * 1.4 + 0.3
    ax.text(5, tally_y, f"Pass Votes: {n_pass} / 3", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#2c3e50")

    # Final determinations for each strictness
    strictness_y = tally_y - 0.9
    for j, (sname, threshold) in enumerate([("Strict (3/3)", 3), ("Moderate (2/3)", 2), ("Lax (1/3)", 1)]):
        final = "PASS" if n_pass >= threshold else "FAIL"
        fc = "#2ecc71" if final == "PASS" else "#e74c3c"
        x = 1.5 + j * 3.0
        ax.text(x, strictness_y, sname, ha="center", va="top",
                fontsize=7, color="#7f8c8d")
        ax.text(x, strictness_y - 0.55, final, ha="center", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, alpha=0.9))

    # Agreement indicator
    agree_y = strictness_y - 1.8
    if n_pass == 3 or n_pass == 0:
        agree_text = "UNANIMOUS"
        agree_color = "#2ecc71" if n_pass == 3 else "#e74c3c"
    elif n_pass == 2:
        agree_text = "MAJORITY PASS"
        agree_color = "#f39c12"
    else:
        agree_text = "MAJORITY FAIL"
        agree_color = "#f39c12"

    ax.text(5, agree_y, agree_text, ha="center", va="top",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=agree_color, alpha=0.9))


def generate_qc_image(
    original_img: Image.Image,
    img_path: str,
    true_label: str,
    genus: str,
    family: str,
    split: str,
    models: dict,
    configs: dict,
    device: torch.device,
    output_dir: str,
    idx: int,
):
    """
    Generate a single composite QC image for one leaf.

    Layout (3 rows):
      Row 1: Original image | EfficientNet GradCAM | ConvNeXt GradCAM (best tile on full img) | DINOv2 Attention (best tile on full img)
      Row 2: [bar chart]    | ConvNeXt tile importance (full img) | DINOv2 full attention (full img) | [empty]
      Row 3: Ensemble summary panel (wide)
    """
    results = {}

    # Run each model
    if "efficientnet" in models:
        try:
            results["efficientnet"] = run_efficientnet(
                models["efficientnet"], configs["efficientnet"], original_img, device
            )
        except Exception as e:
            logger.warning(f"    EfficientNet failed: {e}")

    if "convnextv2" in models:
        try:
            results["convnextv2"] = run_convnextv2(
                models["convnextv2"], configs["convnextv2"], original_img, device
            )
        except Exception as e:
            logger.warning(f"    ConvNeXt V2 failed: {e}")

    if "dinov2" in models:
        try:
            results["dinov2"] = run_dinov2(
                models["dinov2"], configs["dinov2"], original_img, device
            )
        except Exception as e:
            logger.warning(f"    DINOv2 failed: {e}")

    if not results:
        logger.warning(f"    No models produced results for {img_path}")
        return

    # Build composite figure
    fig = plt.figure(figsize=(22, 18), facecolor="white")
    fig.suptitle(
        f"{split.upper()} | {family} / {genus} | "
        f"True: {true_label.upper()} | {Path(img_path).name}",
        fontsize=14, fontweight="bold", y=0.98, color="#2c3e50",
    )

    # Layout: 3 rows
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 0.85, 0.9],
                           hspace=0.35, wspace=0.25,
                           left=0.04, right=0.96, top=0.93, bottom=0.03)

    # Row 1: Original + per-model heatmaps
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(np.array(original_img))
    ax_orig.set_title("Original Image", fontsize=10, fontweight="bold", pad=8)
    gt_color = "#2ecc71" if true_label == "pass" else "#e74c3c"
    ax_orig.text(0.5, -0.02, f"TRUE: {true_label.upper()}", transform=ax_orig.transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=gt_color, alpha=0.9))
    ax_orig.axis("off")

    col = 1
    if "efficientnet" in results:
        ax_eff = fig.add_subplot(gs[0, col])
        create_per_model_panel(ax_eff, results["efficientnet"], "EfficientNet-B3", original_img)
        col += 1

    if "convnextv2" in results:
        ax_cnx = fig.add_subplot(gs[0, col])
        create_per_model_panel(ax_cnx, results["convnextv2"], "ConvNeXt V2", original_img)
        col += 1

    if "dinov2" in results:
        ax_dino = fig.add_subplot(gs[0, col])
        create_per_model_panel(ax_dino, results["dinov2"], "DINOv2", original_img)

    # Row 2: Full-image heatmaps + feature analysis
    # Slot 0: Feature magnitude comparison bar chart
    ax_feat = fig.add_subplot(gs[1, 0])
    model_labels = []
    pass_probs = []
    fail_probs = []
    bar_colors_p = []
    bar_colors_f = []
    for key, name in [("efficientnet", "EffNet"), ("convnextv2", "ConvNeXt"), ("dinov2", "DINOv2")]:
        if key in results:
            model_labels.append(name)
            pass_probs.append(results[key]["prob_pass"])
            fail_probs.append(results[key]["prob_fail"])
            bar_colors_p.append("#2ecc71")
            bar_colors_f.append("#e74c3c")

    if model_labels:
        x_pos = np.arange(len(model_labels))
        width = 0.35
        bars_p = ax_feat.bar(x_pos - width/2, pass_probs, width, color=bar_colors_p,
                             label="P(pass)", alpha=0.85, edgecolor="white", linewidth=1.5)
        bars_f = ax_feat.bar(x_pos + width/2, fail_probs, width, color=bar_colors_f,
                             label="P(fail)", alpha=0.85, edgecolor="white", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars_p, pass_probs):
            ax_feat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, val in zip(bars_f, fail_probs):
            ax_feat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_feat.set_xticks(x_pos)
        ax_feat.set_xticklabels(model_labels, fontsize=9, fontweight="bold")
        ax_feat.set_ylim(0, 1.15)
        ax_feat.set_ylabel("Probability", fontsize=9)
        ax_feat.set_title("Model Confidence Comparison", fontsize=10, fontweight="bold", pad=8)
        ax_feat.legend(loc="upper right", fontsize=8)
        ax_feat.axhline(y=0.5, color="#bdc3c7", linestyle="--", linewidth=1, alpha=0.7)
        ax_feat.spines["top"].set_visible(False)
        ax_feat.spines["right"].set_visible(False)

    # Slot 1: ConvNeXt tile importance on full image
    if "convnextv2" in results and "importance_map" in results["convnextv2"]:
        ax_tile = fig.add_subplot(gs[1, 1])
        create_full_image_heatmap_panel(
            ax_tile, original_img,
            results["convnextv2"]["importance_map"],
            "ConvNeXt V2 — Tile Importance",
        )
        # Draw tile grid overlay
        for (x, y) in results["convnextv2"]["tile_positions"]:
            ts = configs["convnextv2"].get("tile_size", 224)
            rect = plt.Rectangle((x, y), ts, ts, linewidth=0.5,
                                  edgecolor="white", facecolor="none", alpha=0.4)
            ax_tile.add_patch(rect)

    # Slot 2: DINOv2 tile importance on full image
    if "dinov2" in results and "full_attention" in results["dinov2"]:
        ax_attn = fig.add_subplot(gs[1, 2])
        create_full_image_heatmap_panel(
            ax_attn, original_img,
            results["dinov2"]["full_attention"],
            "DINOv2 — Tile Importance",
        )
        # Draw tile grid overlay if positions available
        if "tile_positions" in results["dinov2"]:
            ts = pad_to_patch(
                configs["dinov2"].get("tile_size", 518),
                configs["dinov2"].get("patch_size", 14),
            )
            for (x, y) in results["dinov2"]["tile_positions"]:
                rect = plt.Rectangle((x, y), ts, ts, linewidth=0.5,
                                      edgecolor="white", facecolor="none", alpha=0.4)
                ax_attn.add_patch(rect)

    # Slot 3: Agreement matrix / disagreement heatmap
    ax_agree = fig.add_subplot(gs[1, 3])
    ax_agree.axis("off")
    # Mini agreement table
    ax_agree.set_title("Model Agreement Matrix", fontsize=10, fontweight="bold", pad=8)

    table_data = []
    row_labels = []
    for key, name in [("efficientnet", "EffNet-B3"), ("convnextv2", "ConvNeXt V2"), ("dinov2", "DINOv2")]:
        if key in results:
            r = results[key]
            correct = (r["label"] == true_label)
            row_labels.append(name)
            table_data.append([
                r["label"].upper(),
                f"{r['confidence']:.1%}",
                "CORRECT" if correct else "WRONG",
            ])

    if table_data:
        table = ax_agree.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=["Prediction", "Confidence", "vs Truth"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # Colour cells
        for i, row in enumerate(table_data):
            pred_color = "#d5f5e3" if row[0] == "PASS" else "#fadbd8"
            truth_color = "#d5f5e3" if row[2] == "CORRECT" else "#fadbd8"
            table[(i + 1, 0)].set_facecolor(pred_color)
            table[(i + 1, 2)].set_facecolor(truth_color)
            table[(i + 1, 1)].set_facecolor("#fef9e7")

    # Row 3: Ensemble summary (spans all 4 columns)
    create_ensemble_panel(fig, gs[2, :], results, true_label, genus, family)

    # Save
    fname = f"qc_{split}_{idx+1:02d}_{true_label}_{genus}_{Path(img_path).stem}.png"
    fname = fname.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Log summary
    votes = []
    for key in ["efficientnet", "convnextv2", "dinov2"]:
        if key in results:
            votes.append(results[key]["label"])
    n_pass = sum(1 for v in votes if v == "pass")
    logger.info(
        f"    [{split}] {genus}/{Path(img_path).name}: "
        f"true={true_label} | votes={n_pass}/3 | saved {fname}"
    )


# ============================================================================
# IMAGE SAMPLING
# ============================================================================

def discover_images(root_dir: str, label: int) -> List[Dict]:
    """Discover images under root_dir/{family}/{genus}/. Returns list of records."""
    records = []
    root = Path(root_dir)
    if not root.exists():
        return records
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir():
            continue
        for genus_dir in sorted(family_dir.iterdir()):
            if not genus_dir.is_dir():
                continue
            for f in sorted(genus_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({
                        "path": str(f),
                        "filename": f.name,
                        "label": label,
                        "family": family_dir.name,
                        "genus": genus_dir.name,
                    })
    return records


def load_val_files_from_splits() -> List[Dict]:
    """
    Load validation file lists from all data_split.json files.
    Returns the intersection of val files (those in val for ALL models),
    or the union if intersection is too small.
    """
    split_files = [
        (EFFICIENTNET_DIR / "data_split.json", "data_split.json"),
        (CONVNEXTV2_DIR / "data_split.json", "data_split.json"),
        (DINOV2_DIR / "data_split_dinov2.json", "data_split_dinov2.json"),
    ]

    all_val_sets = []
    all_val_records = {}

    for spath, name in split_files:
        if not spath.exists():
            logger.warning(f"  Split file not found: {spath}")
            continue
        with open(spath) as f:
            split = json.load(f)
        val_files = split.get("val_files", [])
        val_paths = set()
        for vf in val_files:
            path = vf["path"]
            val_paths.add(path)
            if path not in all_val_records:
                all_val_records[path] = {
                    "path": path,
                    "filename": Path(path).name,
                    "label": 0 if vf["label"] == "pass" else 1,
                    "family": vf.get("family", ""),
                    "genus": vf.get("genus", ""),
                }
        all_val_sets.append(val_paths)
        logger.info(f"  Loaded {len(val_files)} val files from {name}")

    if not all_val_sets:
        return []

    # Use union of all val sets
    union_paths = set()
    for s in all_val_sets:
        union_paths |= s

    return [all_val_records[p] for p in union_paths if p in all_val_records]


def sample_images(n_train: int, n_val: int, seed: int):
    """
    Sample random train and val images. Uses the same images for all 3 models.
    Returns (train_sample, val_sample) — lists of record dicts.
    """
    rng = random.Random(seed)

    # Load val images from saved splits
    val_records = load_val_files_from_splits()
    logger.info(f"  Total val images available: {len(val_records)}")

    # Discover all images to find training set (everything NOT in val)
    pass_dir = os.path.join(DATA_ROOT, "pass")
    fail_dir = os.path.join(DATA_ROOT, "fail")

    all_pass = discover_images(pass_dir, label=0)
    all_fail = discover_images(fail_dir, label=1)
    all_records = all_pass + all_fail
    logger.info(f"  Total images discovered: {len(all_records)} ({len(all_pass)} pass, {len(all_fail)} fail)")

    val_paths = set(r["path"] for r in val_records)
    train_records = [r for r in all_records if r["path"] not in val_paths]
    logger.info(f"  Training images (non-val): {len(train_records)}")

    # Sample balanced (try equal pass/fail)
    def balanced_sample(records, n, rng_inst):
        pass_recs = [r for r in records if r["label"] == 0]
        fail_recs = [r for r in records if r["label"] == 1]
        n_each = n // 2
        n_pass = min(n_each, len(pass_recs))
        n_fail = min(n - n_pass, len(fail_recs))
        if n_fail < n - n_pass:
            n_pass = min(n - n_fail, len(pass_recs))
        sample = rng_inst.sample(pass_recs, n_pass) + rng_inst.sample(fail_recs, n_fail)
        rng_inst.shuffle(sample)
        return sample

    train_sample = balanced_sample(train_records, min(n_train, len(train_records)), rng)
    val_sample = balanced_sample(val_records, min(n_val, len(val_records)), rng)

    return train_sample, val_sample


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Ensemble QC Visualization Tool")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    os.makedirs(QC_OUTPUT_DIR, exist_ok=True)

    # Load all 3 models
    logger.info("Loading models...")
    models = {}
    configs = {}

    for name, loader, model_dir in [
        ("efficientnet", load_efficientnet, str(EFFICIENTNET_DIR)),
        ("convnextv2",   load_convnextv2,   str(CONVNEXTV2_DIR)),
        ("dinov2",       load_dinov2,        str(DINOV2_DIR)),
    ]:
        try:
            m, c = loader(model_dir, DEVICE)
            models[name] = m
            configs[name] = c
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")

    if not models:
        logger.error("No models loaded! Check model directories.")
        sys.exit(1)

    logger.info(f"Models loaded: {list(models.keys())}")

    # Sample images
    logger.info("Sampling images...")
    train_sample, val_sample = sample_images(N_TRAIN, N_VAL, SEED)
    logger.info(f"  Selected {len(train_sample)} train, {len(val_sample)} val images")

    # Generate QC images
    all_samples = [(s, "train") for s in train_sample] + [(s, "val") for s in val_sample]

    for i, (rec, split) in enumerate(all_samples):
        img_path = rec["path"]
        true_label = "pass" if rec["label"] == 0 else "fail"
        logger.info(f"  [{i+1}/{len(all_samples)}] Processing {split}/{Path(img_path).name}...")

        try:
            img = Image.open(img_path).convert("RGB")
            generate_qc_image(
                original_img=img,
                img_path=img_path,
                true_label=true_label,
                genus=rec.get("genus", "unknown"),
                family=rec.get("family", "unknown"),
                split=split,
                models=models,
                configs=configs,
                device=DEVICE,
                output_dir=str(QC_OUTPUT_DIR),
                idx=i,
            )
        except Exception as e:
            logger.error(f"    Error: {e}")
            import traceback; traceback.print_exc()

    # Summary stats
    logger.info("=" * 60)
    logger.info(f"QC images saved to: {QC_OUTPUT_DIR}")
    logger.info(f"  Total images generated: {len(all_samples)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
