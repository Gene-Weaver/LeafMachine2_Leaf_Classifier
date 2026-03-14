"""
check_and_install_onnxruntime.py
================================
Detects the installed CUDA version and installs the correct onnxruntime build.

Run this BEFORE ensemble_inference.py:
    python check_and_install_onnxruntime.py
"""

import subprocess
import sys, os
import re
import logging

LOG_FILE = "GPU_system_check.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Console — INFO and above
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)

# File — DEBUG and above (captures everything)
_file = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_fmt)

logger.addHandler(_console)
logger.addHandler(_file)

# ============================================================================
# CUDA VERSION → correct onnxruntime-gpu package
# Matches Nvidia's CUDA toolkit compatibility with onnxruntime release notes:
# https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
# ============================================================================
CUDA_TO_ORT = {
    (12, 6): "onnxruntime-gpu==1.21.0",
    (12, 5): "onnxruntime-gpu==1.21.0",
    (12, 4): "onnxruntime-gpu==1.21.0",
    (12, 3): "onnxruntime-gpu==1.19.2",
    (12, 2): "onnxruntime-gpu==1.19.2",
    (12, 1): "onnxruntime-gpu==1.18.1",
    (12, 0): "onnxruntime-gpu==1.17.3",
    (11, 8): "onnxruntime-gpu==1.17.3",
    (11, 7): "onnxruntime-gpu==1.14.1",
    (11, 6): "onnxruntime-gpu==1.13.1",
}

# Oldest CUDA we'll attempt GPU support for
MIN_CUDA_MAJOR = 11
MIN_CUDA_MINOR = 6


def get_cuda_version() -> tuple[int, int] | None:
    """
    Try multiple methods to detect CUDA version.
    Returns (major, minor) or None if no CUDA found.
    """
    # Method 1: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            logger.info(f"Detected CUDA {major}.{minor} via nvidia-smi")
            return major, minor
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        match = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            logger.info(f"Detected CUDA {major}.{minor} via nvcc")
            return major, minor
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: /usr/local/cuda/version.txt or version.json (Linux)
    import os
    for path in ["/usr/local/cuda/version.txt", "/usr/local/cuda/version.json"]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    content = f.read()
                match = re.search(r"(\d+)\.(\d+)", content)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    logger.info(f"Detected CUDA {major}.{minor} via {path}")
                    return major, minor
            except Exception:
                pass

    return None


def get_installed_ort() -> str | None:
    """Return the currently installed onnxruntime package name and version, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "onnxruntime-gpu"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            match = re.search(r"Version:\s*(\S+)", result.stdout)
            return f"onnxruntime-gpu=={match.group(1)}" if match else "onnxruntime-gpu"
    except Exception:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "onnxruntime"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            match = re.search(r"Version:\s*(\S+)", result.stdout)
            return f"onnxruntime=={match.group(1)}" if match else "onnxruntime"
    except Exception:
        pass

    return None


def install_package(package: str):
    """Install a pip package into the current Python environment."""
    logger.info(f"Running: pip install {package}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=False  # let pip output stream to console
    )
    if result.returncode != 0:
        logger.error(f"pip install failed for {package}")
        sys.exit(1)


def uninstall_package(package_name: str):
    """Uninstall a pip package by name (without version suffix)."""
    logger.info(f"Uninstalling existing {package_name}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", package_name],
        capture_output=False
    )


def pick_ort_package(cuda_major: int, cuda_minor: int) -> str | None:
    """
    Find the best matching onnxruntime-gpu for the detected CUDA version.
    Walks down minor versions to find the closest supported release.
    Returns None if the CUDA version is too old.
    """
    if cuda_major < MIN_CUDA_MAJOR or (
        cuda_major == MIN_CUDA_MAJOR and cuda_minor < MIN_CUDA_MINOR
    ):
        return None

    # Exact match first
    if (cuda_major, cuda_minor) in CUDA_TO_ORT:
        return CUDA_TO_ORT[(cuda_major, cuda_minor)]

    # Walk down minor versions within same major to find nearest compatible
    for minor in range(cuda_minor - 1, -1, -1):
        if (cuda_major, minor) in CUDA_TO_ORT:
            logger.info(
                f"No exact match for CUDA {cuda_major}.{cuda_minor} — "
                f"using package built for CUDA {cuda_major}.{minor}"
            )
            return CUDA_TO_ORT[(cuda_major, minor)]

    return None


def verify_cuda_provider():
    """Confirm that ONNX Runtime can actually see the GPU after install."""
    try:
        import importlib
        import onnxruntime as ort
        importlib.reload(ort)  # reload in case it was just installed
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info("✓ CUDAExecutionProvider is active — GPU inference is ready.")
            return True
        else:
            logger.warning(
                f"onnxruntime is installed but CUDAExecutionProvider is NOT listed.\n"
                f"  Available providers: {providers}\n"
                f"  This usually means the CUDA libraries are not on the system PATH,\n"
                f"  or the cuDNN version doesn't match. Check your CUDA/cuDNN install."
            )
            return False
    except ImportError:
        logger.error("onnxruntime still not importable after install — something went wrong.")
        return False


# ============================================================================
# MAIN
# ============================================================================

def log_system_info():
    """Write detailed system/environment info to the log at startup."""
    import platform
    import os

    logger.debug("=" * 60)
    logger.debug("SYSTEM INFORMATION")
    logger.debug("=" * 60)
    logger.debug(f"Log file:          {os.path.abspath(LOG_FILE)}")
    logger.debug(f"OS:                {platform.platform()}")
    logger.debug(f"Python:            {sys.version}")
    logger.debug(f"Python executable: {sys.executable}")

    # PATH (useful for diagnosing missing nvidia-smi / nvcc)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    logger.debug("PATH entries:")
    for entry in path_entries:
        logger.debug(f"  {entry}")

    # LD_LIBRARY_PATH (Linux — critical for CUDA shared libs)
    ld = os.environ.get("LD_LIBRARY_PATH", "(not set)")
    logger.debug(f"LD_LIBRARY_PATH:   {ld}")

    # Installed pip packages relevant to this check
    logger.debug("Querying relevant pip packages...")
    for pkg in ["onnxruntime", "onnxruntime-gpu", "numpy", "pillow"]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Flatten to one line for readability
                summary = "  |  ".join(
                    line for line in result.stdout.splitlines()
                    if any(line.startswith(k) for k in ("Name:", "Version:", "Location:"))
                )
                logger.debug(f"  {pkg}: {summary}")
            else:
                logger.debug(f"  {pkg}: NOT INSTALLED")
        except Exception as e:
            logger.debug(f"  {pkg}: query failed ({e})")

    # Raw nvidia-smi output
    logger.debug("Full nvidia-smi output:")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            logger.debug(f"  {line}")
        if result.stderr:
            logger.debug(f"  stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        logger.debug("  nvidia-smi not found on PATH")
    except subprocess.TimeoutExpired:
        logger.debug("  nvidia-smi timed out")

    logger.debug("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("CUDA / ONNX Runtime Compatibility Checker")
    logger.info(f"Full log: {os.path.abspath(LOG_FILE)}")
    logger.info("=" * 60)

    log_system_info()

    # Step 1: Detect CUDA
    cuda_version = get_cuda_version()

    if cuda_version is None:
        logger.warning(
            "No CUDA installation detected on this machine.\n"
            "  → Installing CPU-only onnxruntime as fallback.\n"
            "  Note: Inference will use all CPU cores. See ensemble_inference.py\n"
            "  for how to limit thread count with SessionOptions."
        )
        existing = get_installed_ort()
        if existing and "onnxruntime==" in existing and "gpu" not in existing:
            logger.info(f"CPU-only onnxruntime already installed ({existing}). Nothing to do.")
        else:
            if existing:
                uninstall_package("onnxruntime-gpu")
                uninstall_package("onnxruntime")
            install_package("onnxruntime")
        return

    cuda_major, cuda_minor = cuda_version

    # Step 2: Pick the right package
    target_package = pick_ort_package(cuda_major, cuda_minor)

    if target_package is None:
        logger.error(
            f"CUDA {cuda_major}.{cuda_minor} is below the minimum supported version "
            f"(CUDA {MIN_CUDA_MAJOR}.{MIN_CUDA_MINOR}).\n"
            f"  → Falling back to CPU-only onnxruntime.\n"
            f"  To enable GPU support, upgrade your CUDA drivers."
        )
        existing = get_installed_ort()
        if existing:
            uninstall_package("onnxruntime-gpu")
            uninstall_package("onnxruntime")
        install_package("onnxruntime")
        return

    logger.info(f"Target package for CUDA {cuda_major}.{cuda_minor}: {target_package}")

    # Step 3: Check what's currently installed
    existing = get_installed_ort()
    logger.info(f"Currently installed: {existing or 'nothing'}")

    if existing and existing == target_package:
        logger.info("Correct version already installed. Verifying GPU provider...")
        verify_cuda_provider()
        return

    # Step 4: Remove old install and put the right one in
    if existing:
        logger.info("Version mismatch — reinstalling...")
        uninstall_package("onnxruntime-gpu")
        uninstall_package("onnxruntime")

    install_package(target_package)

    # Step 5: Verify
    logger.info("Verifying installation...")
    verify_cuda_provider()
    logger.info("Done. You can now run ensemble_inference.py.")
    logger.info(f"Full log written to: {os.path.abspath(LOG_FILE)}")


if __name__ == "__main__":
    main()