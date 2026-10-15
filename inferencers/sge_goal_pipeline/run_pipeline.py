#!/usr/bin/env python3
"""Run the full inference pipeline: IER mask prediction, FER mask prediction, and image editing."""

import argparse
import subprocess
import sys
import os


def run_command(command, cwd):
    """Run a command in the given directory, streaming output to stdout/stderr."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(command)}")
    print(f"In:      {cwd}")
    print(f"{'='*60}\n")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full inference pipeline.")
    parser.add_argument("--data-dir", required=True, help="Absolute path to the test data folder")
    parser.add_argument("--fer-checkpoint-dir", required=True, help="Absolute path to the FER mask prediction checkpoint folder")
    parser.add_argument("--image-editing-checkpoint-dir", required=True, help="Absolute path to the image editing checkpoint folder")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: IER mask prediction
    run_command(
        ["modal", "run", "modal_ier_mask_prediction.py",
         "--data-dir", args.data_dir],
        cwd=os.path.join(base_dir, "ier-mask-prediction-code"),
    )

    # Step 2: FER mask prediction
    run_command(
        ["modal", "run", "modal_fer_mask_inference.py",
         "--data-dir", args.data_dir,
         "--checkpoint-dir", args.fer_checkpoint_dir],
        cwd=os.path.join(base_dir, "fer-mask-prediction-code"),
    )

    # Step 3: Image editing
    run_command(
        ["modal", "run", "modal_flux_kontext_inference.py",
         "--data-dir", args.data_dir,
         "--checkpoint-dir", args.image_editing_checkpoint_dir],
        cwd=os.path.join(base_dir, "image-editing-code"),
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
