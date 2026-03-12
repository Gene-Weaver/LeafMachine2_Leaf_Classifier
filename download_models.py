#!/usr/bin/env python3
"""
Download pre-trained model weights from GitHub Releases.

This script downloads the .onnx.data files needed for inference.
The .onnx model architectures are included in the repository;
only the weight files (.onnx.data) are downloaded here.

Usage:
    python download_models.py

The models will be saved to the `models/` directory.
"""

import os
import sys
import urllib.request
import json
from pathlib import Path

# Configuration
GITHUB_REPO = "Gene-Weaver/LeafMachine2_Leaf_Classifier"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# Model files to download (from latest release)
MODEL_FILES = {
    "efficientnet_b3": "leaf_classifier_efficientnet_b3.onnx.data",
    "convnextv2": "leaf_classifier_convnextv2.onnx.data",
    "dinov2": "leaf_classifier_dinov2.onnx.data",
}


def fetch_latest_release():
    """Fetch the latest release info from GitHub API."""
    try:
        with urllib.request.urlopen(GITHUB_API_URL) as response:
            releases = json.loads(response.read().decode())
            if releases:
                return releases[0]  # Latest release
            return None
    except Exception as e:
        print(f"❌ Error fetching releases from GitHub API: {e}")
        return None


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    try:
        print(f"  Downloading: {Path(url).name}...")
        urllib.request.urlretrieve(url, dest_path)
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  ✓ {dest_path.name} ({file_size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {url}: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading Pre-trained Model Weights")
    print("=" * 60)

    # Fetch latest release
    print("\nFetching latest release from GitHub...")
    release = fetch_latest_release()

    if not release:
        print("❌ Could not find any releases. Check your internet connection.")
        print(f"   Repository: https://github.com/{GITHUB_REPO}")
        sys.exit(1)

    release_tag = release.get("tag_name", "unknown")
    print(f"✓ Latest release: {release_tag}\n")

    # Extract download URLs for model files
    assets = release.get("assets", [])
    download_urls = {}

    for model_name, filename in MODEL_FILES.items():
        matching = [a for a in assets if a["name"] == filename]
        if matching:
            download_urls[model_name] = matching[0]["browser_download_url"]
        else:
            print(f"⚠ {filename} not found in release assets")

    if not download_urls:
        print("❌ No model files found in the latest release.")
        print("   Please ensure the release contains model weight files.")
        sys.exit(1)

    # Create model directories if they don't exist
    for model_name in MODEL_FILES:
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

    # Download files
    print("Downloading model weight files...\n")
    success_count = 0
    for model_name, url in download_urls.items():
        dest_path = MODELS_DIR / model_name / MODEL_FILES[model_name]
        if download_file(url, dest_path):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(download_urls):
        print(f"✓ All {success_count} model files downloaded successfully!")
        print(f"\nModels are ready at: {MODELS_DIR}")
        print("\nYou can now run:")
        print("  python ensemble_inference.py --input /path/to/images")
    else:
        print(f"⚠ Downloaded {success_count}/{len(download_urls)} files.")
        print("  Some files may have failed. Please try again or check your connection.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
