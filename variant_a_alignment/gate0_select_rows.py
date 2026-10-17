#!/usr/bin/env python3
"""Gate 0: pick a task-stratified subset of HF test rows.

Reads ``policy/hf_calvin_mapping.json`` (HF row -> CALVIN task) and samples
``--n`` rows round-robin across tasks with a fixed seed, so every CALVIN task
is represented roughly equally.

Usage:
    python variant_a_alignment/gate0_select_rows.py --n 150
    # -> variant_a_alignment/gate0_rows.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COLORS = ("red", "blue", "pink")


def task_contradicts_instruction(m: dict) -> bool:
    """True if the nn_l2-matched CALVIN task (used as the success oracle) names a
    different block colour or the opposite direction from the HF instruction.
    ~95/1087 rows are affected; scoring them measures the wrong task."""
    ins, task = m["hf_instruction"].lower(), m["task"]
    ic = {c for c in COLORS if c in ins}
    tc = {c for c in COLORS if c in task}
    if ic and tc and ic != tc:
        return True
    return ("left" in ins and task.endswith("right")) or ("right" in ins and task.endswith("left"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mapping", type=Path, default=REPO / "policy/hf_calvin_mapping.json")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO / "variant_a_alignment/gate0_rows.json")
    p.add_argument("--keep_mismatched", action="store_true",
                   help="Keep rows whose oracle task contradicts the instruction.")
    args = p.parse_args()

    mapping = json.loads(args.mapping.read_text())
    by_task = defaultdict(list)
    dropped = 0
    for m in mapping["matches"]:
        if not args.keep_mismatched and task_contradicts_instruction(m):
            dropped += 1
            continue
        by_task[m["task"]].append(m["hf_row"])
    print(f"dropped {dropped} rows whose oracle task contradicts the instruction")

    rng = random.Random(args.seed)
    tasks = sorted(by_task)
    for t in tasks:
        rng.shuffle(by_task[t])

    rows = []
    depth = 0
    while len(rows) < args.n and any(depth < len(by_task[t]) for t in tasks):
        for t in tasks:
            if depth < len(by_task[t]) and len(rows) < args.n:
                rows.append(by_task[t][depth])
        depth += 1

    rows.sort()
    task_of = {m["hf_row"]: m["task"] for m in mapping["matches"]}
    args.out.write_text(json.dumps({
        "seed": args.seed,
        "n": len(rows),
        "rows": rows,
        "tasks": {str(r): task_of[r] for r in rows},
    }, indent=1))
    print(f"wrote {len(rows)} rows over {len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
