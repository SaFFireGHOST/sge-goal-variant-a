#!/usr/bin/env python3
"""Gate 0 analysis: is there per-row best-of-N headroom for a frozen policy?

Reads the ``per_sample_*.jsonl`` shards written by ``run_gate0`` (one record per
row x variant x perturbation) and reports:

* single-sample SGE-Goal success (mean over the K_m*K_a variants) vs GT goal,
* fraction of rows whose variants disagree (mixed outcomes),
* how often perturbations change the outcome (sanity check for variance),
* cross-fitted best-of-N headroom: choose the variant on all perturbations but
  one, score it on the held-out one, rotate (this is the Gate 0 number),
* the same split by stage: best final mask (Step 2) vs best appearance (Step 3),
* final-mask diversity (pairwise IoU) if the masks are available locally,
* per-task breakdown, with row-bootstrap 95% CIs on the headroom numbers.

Usage:
    modal volume get rm-policy-runs eval_method/gate0 results/
    modal volume get rm-policy-eval-data gate0/masks results/gate0_masks   # optional
    python variant_a_alignment/gate0_analyze.py --results results/gate0 \
        --masks results/gate0_masks
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

VARIANT_RE = re.compile(r"^m(\d+)a(\d+)$")


def load_records(results: Path):
    """Return ``S[row][variant][perturb] -> 0/1`` and ``task_of[row]``."""
    S = defaultdict(lambda: defaultdict(dict))
    task_of = {}
    files = sorted(results.glob("per_sample_*.jsonl"))
    if not files:
        raise SystemExit(f"no per_sample_*.jsonl under {results}")
    for fp in files:
        for line in fp.read_text().splitlines():
            r = json.loads(line)
            if r.get("skipped") or r.get("variant") is None:
                continue
            S[r["hf_row"]][r["variant"]][r.get("perturb", 0)] = int(r["success"])
            task_of[r["hf_row"]] = r.get("task", "?")
    return S, task_of


def complete_rows(S, gen_variants, pks):
    """Rows where every generated variant (and GT, if present) has every perturbation."""
    keep = []
    for row, vs in S.items():
        if all(v in vs and all(p in vs[v] for p in pks) for v in gen_variants):
            keep.append(row)
    return sorted(keep)


def row_stats(vs, gen_variants, mask_groups, pks):
    """Per-row metrics; ``vs[variant][perturb]`` in {0, 1}."""
    A = np.array([[vs[v][p] for p in pks] for v in gen_variants], dtype=float)  # V x P
    out = {
        "single": A.mean(),
        "single_p0": A[:, 0].mean(),
        "oracle_best": A.mean(axis=1).max(),
        "mixed": float(len(set(A.mean(axis=1))) > 1),
        "perturb_flips": float(np.mean([len(set(a)) > 1 for a in A])) if len(pks) > 1 else np.nan,
    }
    if "gt" in vs and all(p in vs["gt"] for p in pks):
        g = np.array([vs["gt"][p] for p in pks], dtype=float)
        out["gt"], out["gt_p0"] = g.mean(), g[0]

    # Cross-fit: pick on P-1 perturbations, score on the held-out one; ties are
    # averaged (expected value of a random tie-break) so the estimate is deterministic.
    if len(pks) > 1:
        cf, cf_mask, cf_app = [], [], []
        for h in range(len(pks)):
            train = [i for i in range(len(pks)) if i != h]
            tr = A[:, train].mean(axis=1)
            cf.append(A[tr == tr.max(), h].mean())

            # Stage 2 only: pick the mask on train perturbations, average its appearances.
            m_tr = np.array([A[idx][:, train].mean() for idx in mask_groups])
            best_m = [g for g, s in zip(mask_groups, m_tr) if s == m_tr.max()]
            cf_mask.append(np.mean([A[idx, h].mean() for idx in best_m]))

            # Stage 3 only: within each mask, pick the appearance; average over masks.
            per_mask = []
            for idx in mask_groups:
                t = A[idx][:, train].mean(axis=1)
                per_mask.append(A[idx][t == t.max(), h].mean())
            cf_app.append(np.mean(per_mask))
        out["crossfit_best"] = float(np.mean(cf))
        out["crossfit_mask"] = float(np.mean(cf_mask))
        out["crossfit_appearance"] = float(np.mean(cf_app))

    # Variance decomposition of per-variant success (over perturbations).
    vm = A.mean(axis=1)
    mask_means = np.array([vm[idx].mean() for idx in mask_groups])
    out["var_between_masks"] = mask_means.var()
    out["var_within_masks"] = np.mean([vm[idx].var() for idx in mask_groups])
    return out


def mask_iou(mask_dir: Path, row: int):
    from PIL import Image

    files = sorted((mask_dir / str(row + 1)).glob("mf_*.jpg"))
    if len(files) < 2:
        return np.nan
    ms = [np.asarray(Image.open(f).convert("L")) > 127 for f in files]
    ious = []
    for a, b in combinations(ms, 2):
        union = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / union if union else 1.0)
    return float(np.mean(ious))


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True, help="dir with per_sample_*.jsonl")
    ap.add_argument("--masks", type=Path, default=None, help="local copy of gate0/masks")
    ap.add_argument("--out", type=Path, default=None, help="report JSON (default: <results>/gate0_report.json)")
    args = ap.parse_args()

    S, task_of = load_records(args.results)
    all_variants = sorted({v for vs in S.values() for v in vs})
    gen_variants = sorted(v for v in all_variants if VARIANT_RE.match(v))
    if not gen_variants:
        raise SystemExit(f"no m{{j}}a{{k}} variants found (got {all_variants})")
    pks = sorted({p for vs in S.values() for v in vs.values() for p in v})
    by_mask = defaultdict(list)
    for i, v in enumerate(gen_variants):
        by_mask[VARIANT_RE.match(v).group(1)].append(i)
    mask_groups = [np.array(ix) for _, ix in sorted(by_mask.items())]

    rows = complete_rows(S, gen_variants, pks)
    print(f"rows complete: {len(rows)}/{len(S)}  variants: {gen_variants} (+gt: {'gt' in all_variants})  "
          f"perturbations: {pks}")
    per_row = {r: row_stats(S[r], gen_variants, mask_groups, pks) for r in rows}
    if args.masks is not None:
        for r in rows:
            per_row[r]["mask_iou"] = mask_iou(args.masks, r)

    def col(key, subset=None):
        return [per_row[r].get(key, np.nan) for r in (subset or rows)]

    single = np.array(col("single"))
    headroom = np.array(col("crossfit_best")) - single
    report = {
        "n_rows": len(rows),
        "variants": gen_variants,
        "perturbations": pks,
        "single_sample": bootstrap_ci(single),
        "single_sample_p0": bootstrap_ci(col("single_p0")),
        "gt": bootstrap_ci(col("gt")),
        "gt_p0": bootstrap_ci(col("gt_p0")),
        "gen_minus_gt": bootstrap_ci(single - np.array(col("gt"))),
        "mixed_fraction": float(np.nanmean(col("mixed"))),
        "perturb_flip_fraction": float(np.nanmean(col("perturb_flips"))) if len(pks) > 1 else None,
        "oracle_best_of_n": bootstrap_ci(col("oracle_best")),
        "crossfit_best_of_n": bootstrap_ci(col("crossfit_best")),
        "headroom_crossfit": bootstrap_ci(headroom),
        "headroom_mask_stage": bootstrap_ci(np.array(col("crossfit_mask")) - single),
        "headroom_appearance_stage": bootstrap_ci(np.array(col("crossfit_appearance")) - single),
        "var_between_masks": float(np.mean(col("var_between_masks"))),
        "var_within_masks": float(np.mean(col("var_within_masks"))),
        "mask_iou_mean": float(np.nanmean(col("mask_iou"))) if args.masks else None,
    }

    tasks = defaultdict(list)
    for r in rows:
        tasks[task_of[r]].append(r)
    report["per_task"] = {
        t: {
            "n": len(rs),
            "single": float(np.mean(col("single", rs))),
            "gt": float(np.nanmean(col("gt", rs))),
            "crossfit_best": float(np.nanmean(col("crossfit_best", rs))),
            "mixed": float(np.mean(col("mixed", rs))),
        }
        for t, rs in sorted(tasks.items())
    }

    def pct(ci):
        m, lo, hi = ci
        return f"{100 * m:6.1f}%  [{100 * lo:5.1f}, {100 * hi:5.1f}]"

    print("\n=== Gate 0 report (mean, 95% bootstrap CI over rows) ===")
    print(f"single-sample SGE-Goal      {pct(report['single_sample'])}")
    print(f"  (unperturbed start only)  {pct(report['single_sample_p0'])}")
    print(f"GT goal, same rows          {pct(report['gt'])}")
    print(f"SGE-Goal minus GT           {pct(report['gen_minus_gt'])}")
    print(f"oracle best-of-{len(gen_variants)} (optimistic) {pct(report['oracle_best_of_n'])}")
    print(f"cross-fit best-of-{len(gen_variants)}         {pct(report['crossfit_best_of_n'])}")
    print(f"HEADROOM (cross-fit - single) {pct(report['headroom_crossfit'])}")
    print(f"  via mask choice only      {pct(report['headroom_mask_stage'])}")
    print(f"  via appearance only       {pct(report['headroom_appearance_stage'])}")
    print(f"mixed-outcome rows          {100 * report['mixed_fraction']:6.1f}%")
    if report["perturb_flip_fraction"] is not None:
        print(f"(row,variant) flipped by perturbation {100 * report['perturb_flip_fraction']:5.1f}%")
    print(f"variance between masks {report['var_between_masks']:.4f} | within masks "
          f"{report['var_within_masks']:.4f}")
    if report["mask_iou_mean"] is not None:
        print(f"mean pairwise IoU of final masks {report['mask_iou_mean']:.3f}")

    print("\nper task: n  single  gt  crossfit  mixed")
    for t, d in report["per_task"].items():
        print(f"  {t:28s} {d['n']:3d}  {100 * d['single']:5.1f}  {100 * d['gt']:5.1f}  "
              f"{100 * d['crossfit_best']:5.1f}  {100 * d['mixed']:5.1f}")

    h, h_lo, _ = report["headroom_crossfit"]
    mixed = report["mixed_fraction"]
    if h >= 0.05 and h_lo > 0 and mixed >= 0.30:
        verdict = "GO: Variant A has signal (plan.md Phase 2A)"
    elif h >= 0.02:
        verdict = "WEAK: Variant A only as an add-on; lead with Variant C"
    else:
        verdict = "NO-GO on GR-MG: pivot to Variant C or GHIL-Glue (plan.md Phase 2C)"
    if report["mask_iou_mean"] is not None and report["mask_iou_mean"] > 0.95:
        verdict += "  [warning: final masks nearly identical -> Step 2 has collapsed]"
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    out = args.out or (args.results / "gate0_report.json")
    out.write_text(json.dumps({"summary": report, "per_row": {str(r): per_row[r] for r in rows}},
                              indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
