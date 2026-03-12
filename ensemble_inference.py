"""
Ensemble Leaf Pass/Fail Classifier — Inference
================================================
Runs all 3 models (EfficientNet-B3, ConvNeXt V2, DINOv2) on every input
image and produces a consensus pass/fail determination.

Dependencies (lightweight — no full PyTorch required):
    pip install onnxruntime-gpu numpy pillow

If no GPU is available, falls back to CPU automatically.

Usage:
    python ensemble_inference.py --input /path/to/images \
                                 --strictness moderate \
                                 --output results.csv

Strictness modes:
    strict   — all 3 models must agree "pass"
    moderate — at least 2/3 models must agree "pass"
    lax      — at least 1/3 models says "pass"
"""

import os
import sys
import csv
import json
import math
import time
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL AVAILABILITY CHECK
# ============================================================================

def check_and_download_models(model_dirs: dict) -> bool:
    """
    Check if model .data files exist. If missing, prompt user to run download_models.py.
    Returns True if all models are available, False otherwise.
    """
    required_data_files = {
        "efficientnet_b3": "leaf_classifier_efficientnet_b3.onnx.data",
        "convnextv2": "leaf_classifier_convnextv2.onnx.data",
        "dinov2": "leaf_classifier_dinov2.onnx.data",
    }

    missing_models = []
    for model_name, data_file in required_data_files.items():
        data_path = Path(model_dirs[model_name]) / data_file
        if not data_path.exists():
            missing_models.append(model_name)

    if missing_models:
        logger.warning("=" * 60)
        logger.warning("MODEL WEIGHTS NOT FOUND")
        logger.warning("=" * 60)
        logger.warning(f"\nMissing: {', '.join(missing_models)}")
        logger.warning("\nModel weight files (.onnx.data) are required for inference.")
        logger.warning("They are downloaded separately to keep the repository small.\n")
        logger.warning("To download models, run:")
        logger.warning("  python download_models.py\n")
        logger.warning("This will fetch the latest pre-trained weights from GitHub Releases.")
        logger.warning("=" * 60 + "\n")
        return False

    return True

# ============================================================================
# DEFAULTS
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Default model directories (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_DIRS = {
    "efficientnet_b3": SCRIPT_DIR / "models" / "efficientnet_b3",
    "convnextv2":      SCRIPT_DIR / "models" / "convnextv2",
    "dinov2":          SCRIPT_DIR / "models" / "dinov2",
}

# ONNX file names produced by the training scripts
ONNX_FILENAMES = {
    "efficientnet_b3": "leaf_classifier.onnx",
    "convnextv2":      "leaf_classifier.onnx",
    "dinov2":          "leaf_classifier_dinov2.onnx",
}

CONFIG_FILENAMES = {
    "efficientnet_b3": "model_config.json",
    "convnextv2":      "model_config.json",
    "dinov2":          "model_config_dinov2.json",
}

STRICTNESS_THRESHOLDS = {
    "strict":   3,
    "moderate": 2,
    "lax":      1,
}


# ============================================================================
# TILE / PREPROCESSING UTILITIES
# ============================================================================

def pad_to_patch(size: int, patch: int) -> int:
    """Round up to nearest multiple of patch."""
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


def extract_tiles_numpy(
    img: Image.Image,
    tile_size: int,
    overlap: float,
    max_tiles: int,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    patch_align: int = 1,
) -> np.ndarray:
    """
    Extract normalised tiles from a PIL image.
    Returns (N, 3, tile_size, tile_size) float32 array.
    """
    if patch_align > 1:
        tile_size = pad_to_patch(tile_size, patch_align)

    img_w, img_h = img.size
    if img_w < tile_size or img_h < tile_size:
        new_w, new_h = max(img_w, tile_size), max(img_h, tile_size)
        padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
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

    tiles_np = np.stack(tiles).transpose(0, 3, 1, 2)  # (N, 3, H, W)
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    tiles_np = (tiles_np - mean_arr) / std_arr
    return tiles_np


def preprocess_efficientnet(img: Image.Image, config: dict) -> np.ndarray:
    """Resize full image to 300x300, normalise, return (1, 3, 300, 300)."""
    image_size = config.get("image_size", 300)
    mean = config.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = config.get("imagenet_std", [0.229, 0.224, 0.225])

    img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr.transpose(2, 0, 1)  # (3, H, W)

    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean_arr) / std_arr
    return arr[np.newaxis]  # (1, 3, H, W)


def preprocess_convnextv2(img: Image.Image, config: dict) -> np.ndarray:
    """Extract tiles for ConvNeXt V2."""
    return extract_tiles_numpy(
        img,
        tile_size=config.get("tile_size", 224),
        overlap=config.get("tile_overlap", 0.25),
        max_tiles=config.get("max_tiles", 64),
        mean=config.get("imagenet_mean", [0.485, 0.456, 0.406]),
        std=config.get("imagenet_std", [0.229, 0.224, 0.225]),
    )


def preprocess_dinov2(img: Image.Image, config: dict) -> np.ndarray:
    """Extract tiles for DINOv2 (patch-aligned)."""
    return extract_tiles_numpy(
        img,
        tile_size=config.get("tile_size", 518),
        overlap=config.get("tile_overlap", 0.25),
        max_tiles=config.get("max_tiles", 32),
        mean=config.get("imagenet_mean", [0.485, 0.456, 0.406]),
        std=config.get("imagenet_std", [0.229, 0.224, 0.225]),
        patch_align=config.get("patch_size", 14),
    )


# ============================================================================
# ONNX MODEL WRAPPER
# ============================================================================

class ONNXModel:
    """Wraps an ONNX Runtime session for a single model."""

    def __init__(
        self,
        name: str,
        onnx_path: str,
        config: dict,
        preprocess_fn,
        providers: list,
        batch_tiles: int = 32,
    ):
        import onnxruntime as ort

        self.name = name
        self.config = config
        self.preprocess_fn = preprocess_fn
        self.batch_tiles = batch_tiles

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"  Loaded {name} from {onnx_path}")

    def predict(self, img: Image.Image) -> Tuple[str, float]:
        """Run inference on a single PIL image. Returns (label, confidence)."""
        input_np = self.preprocess_fn(img, self.config)

        # Run tiles in batches to control memory
        all_logits = []
        for i in range(0, len(input_np), self.batch_tiles):
            batch = input_np[i:i + self.batch_tiles]
            out = self.session.run(None, {self.input_name: batch})
            all_logits.append(out[0])

        logits = np.concatenate(all_logits, axis=0)  # (N, 2)

        # Aggregate: mean of logits, then softmax
        mean_logits = logits.mean(axis=0)
        exp_l = np.exp(mean_logits - mean_logits.max())
        probs = exp_l / exp_l.sum()

        prob_pass = float(probs[0])
        label = "pass" if prob_pass >= 0.5 else "fail"
        confidence = float(max(probs[0], probs[1]))
        return label, confidence


# ============================================================================
# VRAM PROBE — decide parallel vs sequential GPU execution
# ============================================================================

def probe_gpu_capacity() -> str:
    """
    Run dummy tensors through ONNX Runtime on GPU to determine whether all 3
    models can be resident simultaneously.  Returns "parallel" or "sequential".
    If no GPU is available at all, returns "cpu".
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime is not installed. pip install onnxruntime-gpu")
        sys.exit(1)

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        logger.info("No CUDA provider available — will run on CPU.")
        return "cpu"

    # Try to allocate dummy arrays sized like the worst-case forward pass
    # for all 3 models simultaneously.
    # EfficientNet: 1x3x300x300, ConvNeXt: 64x3x224x224, DINOv2: 32x3x518x518
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        logger.info(f"GPU free memory: {free_mb} MB")

        # Rough estimates of ONNX model memory (weights + activations):
        # EfficientNet-B3:  ~150 MB
        # ConvNeXt-Tiny:    ~250 MB
        # DINOv2-ViT-B/14:  ~500 MB
        # Total parallel:   ~900 MB + working memory
        estimated_parallel_mb = 1200
        if free_mb >= estimated_parallel_mb:
            logger.info(
                f"Sufficient VRAM ({free_mb} MB >= {estimated_parallel_mb} MB) "
                f"— loading all 3 models on GPU."
            )
            return "parallel"
        else:
            logger.info(
                f"Limited VRAM ({free_mb} MB < {estimated_parallel_mb} MB) "
                f"— will load models sequentially to avoid OOM."
            )
            return "sequential"
    except Exception as e:
        logger.warning(f"Could not query GPU memory ({e}). Assuming parallel is safe.")
        return "parallel"


# ============================================================================
# IMAGE DISCOVERY
# ============================================================================

def find_images(input_paths: List[str], recursive: bool = True) -> List[str]:
    """Collect all image files from one or more paths (files, dirs, or globs)."""
    images = []
    for input_path in input_paths:
        p = Path(input_path)
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(p.resolve()))
            continue
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in sorted(p.glob(pattern)):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(str(f.resolve()))
            continue
        # Try as glob pattern
        matches = sorted(glob.glob(input_path, recursive=recursive))
        for m in matches:
            if Path(m).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(Path(m).resolve()))
    return images


# ============================================================================
# ENSEMBLE RUNNER
# ============================================================================

class EnsembleRunner:
    """Loads all 3 ONNX models once and runs ensemble inference."""

    def __init__(
        self,
        model_dirs: dict,
        gpu_mode: str,
    ):
        import onnxruntime as ort

        self.gpu_mode = gpu_mode
        self.models: dict = {}

        if gpu_mode == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        model_specs = {
            "efficientnet_b3": {
                "onnx_file": ONNX_FILENAMES["efficientnet_b3"],
                "config_file": CONFIG_FILENAMES["efficientnet_b3"],
                "preprocess": preprocess_efficientnet,
            },
            "convnextv2": {
                "onnx_file": ONNX_FILENAMES["convnextv2"],
                "config_file": CONFIG_FILENAMES["convnextv2"],
                "preprocess": preprocess_convnextv2,
            },
            "dinov2": {
                "onnx_file": ONNX_FILENAMES["dinov2"],
                "config_file": CONFIG_FILENAMES["dinov2"],
                "preprocess": preprocess_dinov2,
            },
        }

        if gpu_mode == "sequential":
            # In sequential mode, we still load all 3 but only keep one
            # session active at a time by loading/unloading. Store paths
            # and configs for deferred loading.
            logger.info("Sequential GPU mode — models will be loaded one at a time.")
            self._sequential_specs = {}
            for name, spec in model_specs.items():
                model_dir = model_dirs[name]
                onnx_path = os.path.join(model_dir, spec["onnx_file"])
                config_path = os.path.join(model_dir, spec["config_file"])
                if not os.path.exists(onnx_path):
                    logger.warning(f"ONNX file not found: {onnx_path} — skipping {name}")
                    continue
                config = {}
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                self._sequential_specs[name] = {
                    "onnx_path": onnx_path,
                    "config": config,
                    "preprocess": spec["preprocess"],
                    "providers": providers,
                }
        else:
            # Parallel or CPU: load all models upfront
            logger.info("Loading all 3 models into memory...")
            for name, spec in model_specs.items():
                model_dir = model_dirs[name]
                onnx_path = os.path.join(model_dir, spec["onnx_file"])
                config_path = os.path.join(model_dir, spec["config_file"])

                if not os.path.exists(onnx_path):
                    logger.warning(f"ONNX file not found: {onnx_path} — skipping {name}")
                    continue

                config = {}
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)

                self.models[name] = ONNXModel(
                    name=name,
                    onnx_path=onnx_path,
                    config=config,
                    preprocess_fn=spec["preprocess"],
                    providers=providers,
                )

        loaded = list(self.models.keys()) if gpu_mode != "sequential" else list(self._sequential_specs.keys())
        logger.info(f"Models ready: {loaded}")
        if len(loaded) == 0:
            logger.error("No models could be loaded! Check model directories.")
            sys.exit(1)

    def _get_models_iterator(self):
        """Yields (name, model) pairs. In sequential mode, loads/unloads one at a time."""
        if self.gpu_mode == "sequential":
            import onnxruntime as ort
            for name, spec in self._sequential_specs.items():
                model = ONNXModel(
                    name=name,
                    onnx_path=spec["onnx_path"],
                    config=spec["config"],
                    preprocess_fn=spec["preprocess"],
                    providers=spec["providers"],
                )
                yield name, model
                # Let Python GC reclaim the session memory before loading next
                del model
        else:
            for name, model in self.models.items():
                yield name, model

    def predict_single(self, img: Image.Image) -> dict:
        """
        Run all models on one image.
        Returns dict with per-model results and ensemble info.
        """
        results = {}
        for name, model in self._get_models_iterator():
            try:
                label, confidence = model.predict(img)
                results[name] = {"label": label, "confidence": confidence}
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
                results[name] = {"label": "error", "confidence": 0.0}
        return results

    def run(
        self,
        image_paths: List[str],
        strictness: str,
        output_csv: str,
    ):
        """Run ensemble on all images and write CSV."""
        threshold = STRICTNESS_THRESHOLDS[strictness]
        total = len(image_paths)
        logger.info(f"Processing {total} images with strictness={strictness} (need {threshold}/3 pass)")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "fullpath", "strictness_setting",
                "ensemble_agreed_pass", "final_determination",
            ])

            pass_count = 0
            fail_count = 0
            error_count = 0

            for i, img_path in enumerate(image_paths):
                filename = Path(img_path).name
                try:
                    img = Image.open(img_path).convert("RGB")
                    t0 = time.time()
                    model_results = self.predict_single(img)
                    dt = time.time() - t0

                    # Count how many models said "pass"
                    num_pass = sum(
                        1 for r in model_results.values() if r["label"] == "pass"
                    )
                    final = 1 if num_pass >= threshold else 0

                    if final == 1:
                        pass_count += 1
                    else:
                        fail_count += 1

                    writer.writerow([
                        filename, img_path, strictness,
                        num_pass, final,
                    ])

                    if (i + 1) % 50 == 0 or (i + 1) == total:
                        per_model = "  ".join(
                            f"{n}={r['label']}" for n, r in model_results.items()
                        )
                        logger.info(
                            f"  [{i+1}/{total}] {filename}: "
                            f"pass_votes={num_pass} final={final} "
                            f"({per_model}) [{dt:.2f}s]"
                        )

                except Exception as e:
                    logger.error(f"  Error processing {img_path}: {e}")
                    writer.writerow([filename, img_path, strictness, 0, 0])
                    error_count += 1

        logger.info(f"\nResults: {pass_count} pass, {fail_count} fail, {error_count} errors")
        logger.info(f"CSV saved to: {output_csv}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble leaf pass/fail classifier using 3 ONNX models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strictness modes:
  strict   — all 3 models must say pass  (3/3)
  moderate — at least 2 models say pass   (2/3)
  lax      — at least 1 model says pass   (1/3)

Examples:
  python ensemble_inference.py --input /data/leaves --strictness moderate
  python ensemble_inference.py --input /data/batch1 /data/batch2 --strictness strict
  python ensemble_inference.py --input "/data/leaves/**/*.jpg" --strictness lax
        """,
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="One or more folders, files, or glob patterns containing images to classify.",
    )
    parser.add_argument(
        "--strictness", "-s", choices=["strict", "moderate", "lax"],
        default="moderate",
        help="Ensemble agreement level (default: moderate).",
    )
    parser.add_argument(
        "--output", "-o", default="ensemble_results.csv",
        help="Output CSV path (default: ensemble_results.csv).",
    )
    parser.add_argument(
        "--efficientnet-dir", default=None,
        help=f"EfficientNet-B3 output dir (default: {DEFAULT_MODEL_DIRS['efficientnet_b3']})",
    )
    parser.add_argument(
        "--convnextv2-dir", default=None,
        help=f"ConvNeXt V2 output dir (default: {DEFAULT_MODEL_DIRS['convnextv2']})",
    )
    parser.add_argument(
        "--dinov2-dir", default=None,
        help=f"DINOv2 output dir (default: {DEFAULT_MODEL_DIRS['dinov2']})",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do not search input directories recursively.",
    )
    return parser.parse_args()


def main():
    # ========================================================================
    # STATIC TEST PATHS — Set these to run without CLI args
    # ========================================================================
    USE_STATIC_PATHS = False  # Set True to use static paths below instead of CLI

    if USE_STATIC_PATHS:
        # Example configurations — modify as needed for testing
        static_config = {
            # Option 1: Single folder containing many images
            "input": ["/path/to/folder/with/images"],

            # Option 2: List of specific image files
            # "input": [
            #     "/path/to/image1.jpg",
            #     "/path/to/image2.jpg",
            #     "/path/to/image3.jpg",
            # ],

            # Option 3: Multiple folders
            # "input": [
            #     "/path/to/batch1",
            #     "/path/to/batch2",
            #     "/path/to/batch3",
            # ],

            # Option 4: Mix of files and folders
            # "input": [
            #     "/path/to/image1.jpg",
            #     "/path/to/folder_with_images",
            #     "/path/to/image2.jpg",
            # ],

            "strictness": "moderate",
            "output": "ensemble_results.csv",
            "recursive": True,
            "efficientnet_dir": None,  # Use default if None
            "convnextv2_dir": None,
            "dinov2_dir": None,
        }

        # Create a simple namespace-like object to mimic argparse args
        class Args:
            pass
        args = Args()
        args.input = static_config["input"]
        args.strictness = static_config["strictness"]
        args.output = static_config["output"]
        args.no_recursive = not static_config["recursive"]
        args.efficientnet_dir = static_config["efficientnet_dir"]
        args.convnextv2_dir = static_config["convnextv2_dir"]
        args.dinov2_dir = static_config["dinov2_dir"]
    else:
        args = parse_args()

    logger.info("=" * 60)
    logger.info("Ensemble Leaf Classifier — Inference")
    logger.info("=" * 60)

    # Resolve model directories
    model_dirs = {
        "efficientnet_b3": args.efficientnet_dir or str(DEFAULT_MODEL_DIRS["efficientnet_b3"]),
        "convnextv2":      args.convnextv2_dir   or str(DEFAULT_MODEL_DIRS["convnextv2"]),
        "dinov2":          args.dinov2_dir        or str(DEFAULT_MODEL_DIRS["dinov2"]),
    }
    for name, d in model_dirs.items():
        logger.info(f"  {name}: {d}")

    # Check if model weights are available
    if not check_and_download_models(model_dirs):
        logger.error("Cannot proceed without model weights. Please download them first.")
        sys.exit(1)

    # Probe GPU
    gpu_mode = probe_gpu_capacity()

    # Build ensemble runner (loads models once)
    runner = EnsembleRunner(model_dirs=model_dirs, gpu_mode=gpu_mode)

    # Find images
    images = find_images(args.input, recursive=not args.no_recursive)
    logger.info(f"Found {len(images)} images to classify.")

    if len(images) == 0:
        logger.warning("No images found. Check --input paths.")
        return

    # Run
    runner.run(images, strictness=args.strictness, output_csv=args.output)


if __name__ == "__main__":
    main()
