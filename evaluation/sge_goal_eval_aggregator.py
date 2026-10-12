#!/usr/bin/env python3
"""
sge_goal_eval_aggregator.py — Aggregate evaluation results across test cases.

Given a test data folder containing numbered sub-directories (each with an
evaluation_results.json), this script:
  1. Reads every evaluation_results.json
  2. Collects the UNION of all keys across test cases
  3. Skips list-valued keys (only numeric and Yes/No boolean keys are aggregated)
  4. Treats "Yes" as 1 and "No" as 0
  5. For each qualifying key, computes the mean over only the test cases that
     contain that key (test cases missing a key are excluded from that key's
     average, not from the entire aggregation)
  6. Writes the aggregated report as a JSON file in the test data folder

The output file is named  aggregated_evaluation_results.json  and lives inside
the test data folder.  Re-running overwrites the previous report.

Usage:
    # Single test data folder
    python sge_goal_eval_aggregator.py /path/to/test_data_folder
    python sge_goal_eval_aggregator.py /path/to/test_data_folder --output custom_name.json

    # Batch mode: process ALL test data folders found recursively under a parent
    python sge_goal_eval_aggregator.py /path/to/parent_folder --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set


def _is_numeric_or_bool(value) -> bool:
    """Return True if value is a number or a Yes/No string."""
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str) and value.strip().lower() in ("yes", "no"):
        return True
    return False


def _to_number(value) -> float:
    """Convert a numeric value or Yes/No string to a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() == "yes" else 0.0
    raise ValueError(f"Cannot convert {value!r} to number")


def _is_test_data_folder(directory: str) -> bool:
    """
    Return True if *directory* looks like a test data folder, i.e. it contains
    at least one numbered sub-directory with an evaluation_results.json inside.
    """
    try:
        for name in os.listdir(directory):
            full = os.path.join(directory, name)
            if os.path.isdir(full) and name.isdigit():
                if os.path.isfile(os.path.join(full, "evaluation_results.json")):
                    return True
    except PermissionError:
        pass
    return False


def discover_test_data_folders(parent_dir: str) -> List[str]:
    """
    Recursively walk *parent_dir* and return paths of all directories that
    qualify as test data folders (contain numbered sub-dirs with
    evaluation_results.json).
    """
    found = []
    for root, dirs, _files in os.walk(parent_dir):
        if _is_test_data_folder(root):
            found.append(root)
            # Don't recurse into numbered sub-dirs of a matched folder
            dirs[:] = [d for d in dirs if not d.isdigit()]
    found.sort()
    return found


def aggregate(test_data_dir: str) -> dict:
    """
    Aggregate evaluation_results.json files across all test cases.

    For each qualifying key (numeric or Yes/No valued), computes the mean
    over only the test cases that contain that key.  Test cases missing a
    key are simply excluded from that key's average.

    Returns a dict mapping each qualifying key to a sub-dict:
        { "mean": <float>, "count": <int> }
    where *count* is the number of test cases that contributed.
    """
    # Discover test-case directories (numbered sub-dirs)
    entries = []
    for name in os.listdir(test_data_dir):
        full = os.path.join(test_data_dir, name)
        if os.path.isdir(full) and name.isdigit():
            results_file = os.path.join(full, "evaluation_results.json")
            if os.path.isfile(results_file):
                entries.append((int(name), results_file))
    entries.sort()

    if not entries:
        print(f"  No evaluation_results.json files found in {test_data_dir}")
        return {}

    # Load all results
    all_results = []
    for case_num, path in entries:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                all_results.append(data)
            else:
                print(f"  Warning: empty/invalid results in case {case_num}, skipping")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: corrupt JSON in case {case_num} ({e}), skipping")

    total = len(all_results)
    print(f"  Found {total} test cases with evaluation results.")

    # Collect the UNION of all keys across every test case
    all_keys: Set[str] = set()
    for r in all_results:
        all_keys |= set(r.keys())

    # For each key, collect numeric/bool values from test cases that have it
    skipped_keys = []
    report = {}
    for key in sorted(all_keys):
        # Gather values only from test cases that contain this key
        values = []
        skip = False
        for r in all_results:
            if key not in r:
                continue  # this test case doesn't have this key — exclude it
            v = r[key]
            if not _is_numeric_or_bool(v):
                skip = True
                break
            values.append(_to_number(v))

        if skip or not values:
            skipped_keys.append(key)
            continue

        report[key] = {
            "mean": round(sum(values) / len(values), 6),
            "count": len(values),
        }

    if skipped_keys:
        print(f"  Skipping {len(skipped_keys)} non-numeric keys: {skipped_keys}")

    print(f"  Aggregated {len(report)} evaluation dimensions "
          f"(max {total} samples per dimension).")
    return report


def _run_single(
    test_data_dir: str,
    output_name: str,
    pooled_dir: Optional[str] = None,
    top_level_dir: Optional[str] = None,
) -> Optional[dict]:
    """Aggregate one test data folder and write the report. Returns the report."""
    print(f"\n{'='*60}")
    print(f"Processing: {test_data_dir}")
    print(f"{'='*60}")

    report = aggregate(test_data_dir)
    if not report:
        print("  Nothing to aggregate.")
        return None

    # Write inside the test data folder itself
    out_path = os.path.join(test_data_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to: {out_path}")

    # Also copy to the pooled directory with a unique name.
    # Build the pooled filename from the relative path between
    # top_level_dir and test_data_dir so that different models don't
    # collide (e.g.  "susie__bridge_data_v2_tel_test_data_agg_eval_results.json").
    if pooled_dir is not None:
        if top_level_dir is not None:
            rel = os.path.relpath(test_data_dir, top_level_dir)
            # Replace path separators with "__" to flatten the hierarchy
            unique_prefix = rel.replace(os.sep, "__")
        else:
            unique_prefix = os.path.basename(os.path.normpath(test_data_dir))
        pooled_filename = f"{unique_prefix}_agg_eval_results.json"
        pooled_path = os.path.join(pooled_dir, pooled_filename)
        with open(pooled_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Pooled copy  to: {pooled_path}")

    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across test cases"
    )
    parser.add_argument(
        "test_data_dir",
        help="Path to a test data folder (with numbered sub-dirs), "
             "or a parent folder when using --all.",
    )
    parser.add_argument(
        "--output", "-o",
        default="aggregated_evaluation_results.json",
        help="Output filename (written inside each test_data_dir). "
             "Default: aggregated_evaluation_results.json",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        dest="run_all",
        help="Batch mode: recursively discover ALL test data folders under "
             "the given directory and aggregate each one.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.test_data_dir):
        print(f"ERROR: directory not found: {args.test_data_dir}")
        sys.exit(1)

    # Set up the pooled output directory at the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    pooled_dir = os.path.join(project_root, "aggregated-eval-results")
    os.makedirs(pooled_dir, exist_ok=True)

    if args.run_all:
        # ---- Batch mode ----
        folders = discover_test_data_folders(args.test_data_dir)
        if not folders:
            print(f"No test data folders found under {args.test_data_dir}")
            sys.exit(0)
        print(f"Discovered {len(folders)} test data folder(s) under "
              f"{args.test_data_dir}:\n")
        for fd in folders:
            print(f"  • {os.path.relpath(fd, args.test_data_dir)}")

        success = 0
        for fd in folders:
            report = _run_single(
                fd, args.output,
                pooled_dir=pooled_dir,
                top_level_dir=args.test_data_dir,
            )
            if report:
                success += 1

        print(f"\n{'='*60}")
        print(f"Batch complete: {success}/{len(folders)} folders aggregated.")
        print(f"Pooled results in: {pooled_dir}")
        print(f"{'='*60}")
    else:
        # ---- Single folder mode ----
        report = _run_single(
            args.test_data_dir, args.output,
            pooled_dir=pooled_dir,
        )
        if not report:
            sys.exit(0)


if __name__ == "__main__":
    main()
