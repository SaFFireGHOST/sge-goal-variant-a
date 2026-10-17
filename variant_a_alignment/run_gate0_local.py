#!/usr/bin/env python3
"""Run Gate 0 rollouts on a local GPU server (no Modal credits).

Local equivalent of ``modal_calvin_eval.py::run_gate0``: splits the Gate 0 rows
into ``--workers`` shards and runs one ``eval_method.py`` process per shard, all
on the same GPU. GR-MG needs only a few GB of VRAM; PyBullet is CPU-bound, so set
``--workers`` to roughly (CPU cores / 2), up to ~10 on a 48 GB A6000.

Expected local layout (see plan.md §1.8):

    data/task_D_D/validation/{.hydra/, episode_*.npz}   # from export_calvin_subset
    data/ckpts/gr_mg/epoch=47-step=83712.ckpt
    data/calvin_test.parquet
    data/eval_data/gate0/m{j}a{k}/data/<row+1>/...      # generated goals

Usage:
    python variant_a_alignment/run_gate0_local.py --calvin_repo ~/calvin --workers 8
    python variant_a_alignment/gate0_analyze.py --results results/gate0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO / "policy"


def link_workspace(calvin_repo: Path):
    """Same links ``_enter_workspace_with_calvin_conf`` creates inside Modal."""
    for name, target in (("calvin", calvin_repo),
                         ("evaluate", WORKSPACE / "gr_mg_evaluate"),
                         ("policy", WORKSPACE / "gr_mg_policy")):
        link = WORKSPACE / name
        if link.is_file() and not link.is_symlink():   # git symlink checked out as text
            link.unlink()
        if not link.exists():
            os.symlink(target, link)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calvin_repo", type=Path, required=True, help="mees/calvin clone")
    p.add_argument("--data", type=Path, default=REPO / "data")
    p.add_argument("--rows_json", type=Path, default=REPO / "variant_a_alignment/gate0_rows.json")
    p.add_argument("--method", default="gate0", help="sub-folder of <data>/eval_data")
    p.add_argument("--variants", default="m0a0,m0a1,m1a0,m1a1,m2a0,m2a1,gt")
    p.add_argument("--num_perturb", type=int, default=2)
    p.add_argument("--perturb_std", type=float, default=0.02)
    p.add_argument("--ep_len", type=int, default=240)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rows", default="", help="comma-separated rows (overrides --rows_json)")
    p.add_argument("--out", type=Path, default=REPO / "results/gate0")
    args = p.parse_args()

    link_workspace(args.calvin_repo.expanduser().resolve())
    if args.rows:
        rows = [int(r) for r in args.rows.split(",") if r]
    else:
        rows = json.loads(args.rows_json.read_text())["rows"]
    if args.limit:
        rows = rows[: args.limit]
    shards = [s for s in (rows[i:: args.workers] for i in range(args.workers)) if s]

    data = args.data.resolve()
    ckpt_dir = data / "ckpts/gr_mg"
    for need in (data / "task_D_D/validation/.hydra", ckpt_dir, data / "calvin_test.parquet",
                 data / "eval_data" / args.method):
        if not need.exists():
            raise SystemExit(f"missing {need} (see plan.md §1.8)")

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               MESA_GL_VERSION_OVERRIDE="3.3", PYOPENGL_PLATFORM="egl")

    n_roll = len(rows) * len(args.variants.split(",")) * args.num_perturb
    print(f"[gate0-local] {len(rows)} rows -> {n_roll} rollouts on {len(shards)} workers "
          f"(GPU {args.gpu}); logs in {log_dir}")
    procs = []
    for i, shard in enumerate(shards):
        cmd = [
            sys.executable, "evaluate/eval_method.py",
            "--dataset_path", str(data / "task_D_D"),
            "--ckpt_dir", str(ckpt_dir),
            "--epoch", "47",
            "--config_path", "policy/config/train.json",
            "--mapping_json", str(WORKSPACE / "hf_calvin_mapping.json"),
            "--output_dir", str(args.out.resolve()),
            "--ep_len", str(args.ep_len),
            "--method_name", args.method,
            "--rows", ",".join(map(str, shard)),
            "--shard_tag", f"shard{i:02d}",
            "--variants", args.variants,
            "--num_perturb", str(args.num_perturb),
            "--perturb_std", str(args.perturb_std),
            "--method_dir", str(data / "eval_data" / args.method),
            "--hf_parquet_goal", str(data / "calvin_test.parquet"),
        ]
        log = open(log_dir / f"shard{i:02d}.log", "w")
        procs.append((i, subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdout=log,
                                          stderr=subprocess.STDOUT), log))

    t0 = time.time()
    failed = []
    for i, proc, log in procs:
        rc = proc.wait()
        log.close()
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[gate0-local] shard{i:02d} {status}  ({(time.time() - t0) / 60:.1f} min)")
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"shards failed: {failed}; see {log_dir}")
    print(f"[gate0-local] done -> {args.out}")


if __name__ == "__main__":
    main()
