#!/usr/bin/env python3
"""Run image editing on a single custom input (no latency reporting)."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from PIL import Image


def ensure_file(path: str, label: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"{label} not found: {abs_path}")
    return abs_path


def write_single_test_case(
    work_dir: str,
    original_image: str,
    initial_mask: str,
    final_mask: str,
    edit_instruction: str,
) -> str:
    case_dir = os.path.join(work_dir, "0")
    os.makedirs(case_dir, exist_ok=True)

    orig_ext = os.path.splitext(original_image)[1] or ".png"
    init_ext = os.path.splitext(initial_mask)[1] or ".png"
    final_ext = os.path.splitext(final_mask)[1] or ".png"

    original_target = os.path.join(case_dir, f"original_image{orig_ext}")
    init_target = os.path.join(case_dir, f"initial_edit_region_mask{init_ext}")
    final_target = os.path.join(case_dir, f"final_edit_region_mask{final_ext}")

    shutil.copy2(original_image, original_target)

    orig_image = Image.open(original_image).convert("RGB")
    orig_size = orig_image.size

    init_image = Image.open(initial_mask).convert("RGB")
    if init_image.size != orig_size:
        init_image = init_image.resize(orig_size, resample=Image.NEAREST)
    init_image.save(init_target)

    final_image = Image.open(final_mask).convert("RGB")
    if final_image.size != orig_size:
        final_image = final_image.resize(orig_size, resample=Image.NEAREST)
    final_image.save(final_target)

    instruction_path = os.path.join(case_dir, "edit_instruction.txt")
    with open(instruction_path, "w", encoding="utf-8") as f:
        f.write(edit_instruction.strip() + "\n")

    return case_dir


def run_modal_inference(data_dir: str, checkpoint_dir: str, base_dir: str) -> None:
    command = [
        "modal",
        "run",
        "modal_flux_kontext_inference.py",
        "--data-dir",
        data_dir,
        "--checkpoint-dir",
        checkpoint_dir,
    ]
    cwd = os.path.join(base_dir, "image-editing-code")

    print("\n" + "=" * 60)
    print("Running image editing inference")
    print("Command:", " ".join(command))
    print("CWD:    ", cwd)
    print("=" * 60 + "\n")

    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Flux Kontext image editing on a single custom input."
    )
    parser.add_argument("--original-image", required=True, help="Path to the original image")
    parser.add_argument("--initial-mask", required=True, help="Path to the initial edit region mask")
    parser.add_argument("--final-mask", required=True, help="Path to the final edit region mask (GT)")
    parser.add_argument(
        "--edit-instruction",
        required=True,
        help="Edit instruction text (wrap in quotes)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the image editing checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path for edited_image.jpg (default: alongside original image)",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary work directory for debugging",
    )
    args = parser.parse_args()

    original_image = ensure_file(args.original_image, "Original image")
    initial_mask = ensure_file(args.initial_mask, "Initial mask")
    final_mask = ensure_file(args.final_mask, "Final mask")
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    work_parent = tempfile.mkdtemp(prefix="single_image_editing_")
    data_dir = os.path.join(work_parent, "test_data")

    try:
        write_single_test_case(
            data_dir,
            original_image,
            initial_mask,
            final_mask,
            args.edit_instruction,
        )

        run_modal_inference(data_dir, checkpoint_dir, base_dir)

        output_case_dir = os.path.join(data_dir, "0")
        edited_path = os.path.join(output_case_dir, "edited_image.jpg")
        if not os.path.isfile(edited_path):
            raise FileNotFoundError(
                f"edited_image.jpg not found after inference: {edited_path}"
            )

        if args.output_path:
            output_path = os.path.abspath(args.output_path)
        else:
            output_path = os.path.join(os.path.dirname(original_image), "edited_image.jpg")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(edited_path, output_path)
        print(f"\nSaved edited image to: {output_path}")
    finally:
        if args.keep_work_dir:
            print(f"Keeping work dir: {work_parent}")
        else:
            shutil.rmtree(work_parent, ignore_errors=True)


if __name__ == "__main__":
    main()
