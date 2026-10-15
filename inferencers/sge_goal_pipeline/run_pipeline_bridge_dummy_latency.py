#!/usr/bin/env python3
"""Run the bridge pipeline on dummy data and report latency stats."""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime


LATENCY_MARKER_RE = re.compile(r"LATENCY_METRICS_JSON:\s*(\{.*\})")


def run_command(command, cwd):
    """Run a command, streaming output and returning wall time plus reported metrics."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(command)}")
    print(f"In:      {cwd}")
    print(f"{'=' * 60}\n")
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    elapsed = time.perf_counter() - start
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    metrics = None
    for line in result.stdout.splitlines() if result.stdout else []:
        match = LATENCY_MARKER_RE.search(line)
        if match:
            metrics = json.loads(match.group(1))
    return elapsed, metrics or {}


def summarize_timings(name, timings):
    """Return summary stats for a list of timings."""
    mean_s = statistics.mean(timings) if timings else 0.0
    std_s = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return {
        "component": name,
        "runs": len(timings),
        "mean_s": mean_s,
        "std_s": std_s,
        "min_s": min(timings) if timings else 0.0,
        "max_s": max(timings) if timings else 0.0,
        "all_s": timings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run bridge dummy pipeline and report mean/std latency per component."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Absolute path to the dummy test data folder",
    )
    parser.add_argument(
        "--fer-checkpoint-dir",
        required=True,
        help="Absolute path to the FER mask prediction checkpoint folder",
    )
    parser.add_argument(
        "--image-editing-checkpoint-dir",
        required=True,
        help="Absolute path to the image editing checkpoint folder",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run the pipeline (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write timing results as JSON",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    fer_checkpoint_dir = os.path.abspath(args.fer_checkpoint_dir)
    image_editing_checkpoint_dir = os.path.abspath(args.image_editing_checkpoint_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    if not os.path.isdir(fer_checkpoint_dir):
        raise FileNotFoundError(f"FER checkpoint directory not found: {fer_checkpoint_dir}")
    if not os.path.isdir(image_editing_checkpoint_dir):
        raise FileNotFoundError(
            f"Image editing checkpoint directory not found: {image_editing_checkpoint_dir}"
        )
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    timings = {
        "ier": [],
        "fer": [],
        "image_editing": [],
        "total": [],
    }
    metric_runs = {
        "ier": [],
        "fer": [],
        "image_editing": [],
    }

    for run_idx in range(1, args.repeats + 1):
        print(f"\n{'#' * 70}")
        print(f"Run {run_idx}/{args.repeats}")
        print(f"{'#' * 70}")

        run_start = time.perf_counter()

        # Step 1: IER mask prediction
        t_ier, ier_metrics = run_command(
            [
                "modal",
                "run",
                "modal_ier_mask_prediction.py",
                "--data-dir",
                data_dir,
            ],
            cwd=os.path.join(base_dir, "ier-mask-prediction-code"),
        )
        timings["ier"].append(ier_metrics.get("inference_s", t_ier))
        metric_runs["ier"].append(ier_metrics)
        print(f"IER inference latency: {timings['ier'][-1]:.2f}s (outer command: {t_ier:.2f}s)")

        # Step 2: FER mask prediction
        t_fer, fer_metrics = run_command(
            [
                "modal",
                "run",
                "modal_fer_mask_inference.py",
                "--data-dir",
                data_dir,
                "--checkpoint-dir",
                fer_checkpoint_dir,
            ],
            cwd=os.path.join(base_dir, "fer-mask-prediction-code"),
        )
        timings["fer"].append(fer_metrics.get("pipeline_inference_s", t_fer))
        metric_runs["fer"].append(fer_metrics)
        print(f"FER pipeline inference latency: {timings['fer'][-1]:.2f}s (outer command: {t_fer:.2f}s)")

        # Step 3: Image editing
        t_edit, edit_metrics = run_command(
            [
                "modal",
                "run",
                "modal_flux_kontext_inference.py",
                "--data-dir",
                data_dir,
                "--checkpoint-dir",
                image_editing_checkpoint_dir,
            ],
            cwd=os.path.join(base_dir, "image-editing-code"),
        )
        timings["image_editing"].append(edit_metrics.get("pipeline_inference_s", t_edit))
        metric_runs["image_editing"].append(edit_metrics)
        print(
            f"Image editing pipeline inference latency: {timings['image_editing'][-1]:.2f}s "
            f"(outer command: {t_edit:.2f}s)"
        )

        total_elapsed = timings["ier"][-1] + timings["fer"][-1] + timings["image_editing"][-1]
        timings["total"].append(total_elapsed)
        print(f"Total measured inference latency: {total_elapsed:.2f}s")

    summaries = [
        summarize_timings("IER", timings["ier"]),
        summarize_timings("FER", timings["fer"]),
        summarize_timings("IMAGE_EDITING", timings["image_editing"]),
        summarize_timings("TOTAL", timings["total"]),
    ]
    fer_step_timings = [
        metrics["mean_step_s"]
        for metrics in metric_runs["fer"]
        if metrics.get("mean_step_s") is not None
    ]
    image_editing_step_timings = [
        metrics["mean_step_s"]
        for metrics in metric_runs["image_editing"]
        if metrics.get("mean_step_s") is not None
    ]
    if fer_step_timings:
        summaries.append(summarize_timings("FER_STEP", fer_step_timings))
    if image_editing_step_timings:
        summaries.append(summarize_timings("IMAGE_EDIT_STEP", image_editing_step_timings))

    print("\n" + "=" * 70)
    print("Latency Summary (seconds)")
    print("=" * 70)
    for summary in summaries:
        print(
            f"{summary['component']:<14} runs={summary['runs']} "
            f"mean={summary['mean_s']:.2f} std={summary['std_s']:.2f} "
            f"min={summary['min_s']:.2f} max={summary['max_s']:.2f}"
        )

    if args.output_json:
        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_dir": data_dir,
            "fer_checkpoint_dir": fer_checkpoint_dir,
            "image_editing_checkpoint_dir": image_editing_checkpoint_dir,
            "repeats": args.repeats,
            "summaries": summaries,
            "metric_runs": metric_runs,
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote timing JSON to {args.output_json}")


if __name__ == "__main__":
    main()
