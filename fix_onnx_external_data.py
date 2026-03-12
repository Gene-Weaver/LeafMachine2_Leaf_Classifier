#!/usr/bin/env python3
"""
Fix ONNX external data references to match renamed .onnx.data files.
Updates the internal metadata so .onnx files point to the correct .data files.
"""

import onnx
from pathlib import Path

def fix_external_data_reference(onnx_path, old_data_filename, new_data_filename):
    """
    Load ONNX model, update external data references, and save.

    Args:
        onnx_path: Path to .onnx file
        old_data_filename: Old external data filename (e.g., "leaf_classifier.onnx.data")
        new_data_filename: New external data filename (e.g., "leaf_classifier_efficientnet_b3.onnx.data")
    """
    print(f"\nProcessing: {Path(onnx_path).name}")
    print(f"  Old data reference: {old_data_filename}")
    print(f"  New data reference: {new_data_filename}")

    # Load the model without loading external data
    model = onnx.load(onnx_path, load_external_data=False)

    # Find and update external data references in initializers
    updated_count = 0
    for initializer in model.graph.initializer:
        # Check external_data field (list of key-value pairs)
        if initializer.HasField('raw_data') or len(initializer.external_data) == 0:
            continue

        for ext_data in initializer.external_data:
            if ext_data.key == 'location':
                old_value = ext_data.value
                if old_value == old_data_filename:
                    ext_data.value = new_data_filename
                    updated_count += 1
                    print(f"  ✓ Updated reference: {old_value} → {new_data_filename}")

    if updated_count == 0:
        print(f"  ⚠ No external data references found matching '{old_data_filename}'")
        print(f"    This model might not use external data or references are different.")
        return False

    # Save the updated model
    onnx.save(model, onnx_path)
    print(f"  ✓ Saved updated model")
    return True

def main():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"

    models_to_fix = [
        {
            "dir": models_dir / "efficientnet_b3",
            "onnx_file": "leaf_classifier_efficientnet_b3.onnx",
            "old_data": "leaf_classifier.onnx.data",
            "new_data": "leaf_classifier_efficientnet_b3.onnx.data",
        },
        {
            "dir": models_dir / "convnextv2",
            "onnx_file": "leaf_classifier_convnextv2.onnx",
            "old_data": "leaf_classifier.onnx.data",
            "new_data": "leaf_classifier_convnextv2.onnx.data",
        },
        {
            "dir": models_dir / "dinov2",
            "onnx_file": "leaf_classifier_dinov2.onnx",
            "old_data": "leaf_classifier.onnx.data",
            "new_data": "leaf_classifier_dinov2.onnx.data",
        },
    ]

    print("=" * 60)
    print("Fixing ONNX External Data References")
    print("=" * 60)

    success_count = 0
    for model_info in models_to_fix:
        onnx_path = model_info["dir"] / model_info["onnx_file"]

        if not onnx_path.exists():
            print(f"\n⚠ Not found: {onnx_path}")
            continue

        try:
            if fix_external_data_reference(
                str(onnx_path),
                model_info["old_data"],
                model_info["new_data"]
            ):
                success_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print(f"Fixed {success_count}/{len(models_to_fix)} models")
    print("=" * 60)

    if success_count == len(models_to_fix):
        print("\n✓ All models updated! You can now remove the symlinks:")
        print("  rm models/*/leaf_classifier.onnx.data")

if __name__ == "__main__":
    main()
