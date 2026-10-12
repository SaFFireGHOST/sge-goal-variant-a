#!/usr/bin/env python3
"""Upload SGE-Goal Variant A pre-trained checkpoints to Hugging Face Model Hub.

Usage:
    # If already logged in via `hf login`:
    python3 upload_to_hf.py --repo-id <your-username>/sge-goal-calvin-lora

    # Or pass token directly:
    python3 upload_to_hf.py --repo-id <your-username>/sge-goal-calvin-lora --token hf_xxxxxxxxxxxx
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser(description="Upload LoRA checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repo ID in the format 'username/repo-name'")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face User Access Token (with WRITE permission)")
    parser.add_argument("--private", action="store_true",
                        help="Make the Hugging Face repository private (default is public)")
    return parser.parse_args()

def main():
    args = parse_args()
    api = HfApi(token=args.token)

    print(f"==> Ensuring Hugging Face model repository '{args.repo_id}' exists...")
    try:
        create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
        print(f"✓ Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Error creating/accessing repository: {e}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    files_to_upload = [
        ("calvin-fer-mask-prediction/pytorch_lora_weights.safetensors", "calvin-fer-mask-prediction/pytorch_lora_weights.safetensors"),
        ("calvin-fer-mask-prediction/training_metadata.json", "calvin-fer-mask-prediction/training_metadata.json"),
        ("calvin-image-editing/pytorch_lora_weights.safetensors", "calvin-image-editing/pytorch_lora_weights.safetensors"),
        ("calvin-image-editing/training_state.json", "calvin-image-editing/training_state.json"),
        ("README.md", "README.md"),
    ]

    print("\n==> Uploading checkpoint files to Hugging Face...")
    for local_rel, repo_rel in files_to_upload:
        local_path = script_dir / local_rel
        if not local_path.exists():
            print(f"⚠️  Skipping {local_rel} (file not found)")
            continue

        print(f"Uploading {local_rel} ({local_path.stat().st_size / (1024*1024):.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_rel,
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
        )
        print(f"✓ Uploaded {repo_rel}")

    print(f"\n🎉 All checkpoints successfully uploaded to: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
