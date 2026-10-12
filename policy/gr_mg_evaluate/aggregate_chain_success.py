"""Aggregate per-row success indicators into the 11x5 success-rate table.

Reads:
  * ``--chain_skeletons``: chain skeleton JSON produced by
    ``evaluate/build_chains.py``.
  * ``--results_dir``: root directory containing ``<method>/per_sample.jsonl``
    (one method subdir per experiment). Each ``per_sample.jsonl`` has lines
    ``{"hf_row": int, "success": bool, "skipped": bool, ...}``.

Produces:
  * ``--out_csv``: an ``M x 5`` CSV (rows = methods, cols = chain length 1..5).
  * ``--out_json``: same data plus per-skeleton breakdown.

Definition (matches user's spec):
  * length-1 success rate = mean(success indicator over all 1087 HF rows).
  * length-n (n>=2) success rate = mean over **all** length-n chain instances
    of (∧ of constituent row successes). A chain instance = pick one HF row
    from each of the n clusters. We enumerate ALL such instances across ALL
    length-n sub-chains of every chain skeleton (sliding window).

Mathematical equivalence (efficient):
  For each length-n sub-chain (C₁,...,C_n) with per-cluster success counts
  s_i and totals t_i:
      sum_of_AND       = Π s_i
      num_instances    = Π t_i
  Aggregate over all sub-chains by summing both quantities, then divide.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_per_row(jsonl_path: Path) -> Dict[int, bool]:
    """Return ``{hf_row: success_bool}``. Skipped rows are treated as failures."""
    out: Dict[int, bool] = {}
    with jsonl_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["hf_row"]] = bool(rec.get("success", False)) and not rec.get(
                "skipped", False
            )
    return out


def _gather_per_method_results(method_dir: Path) -> Dict[int, bool]:
    """Merge all ``per_sample_*.jsonl`` shards under ``method_dir``."""
    merged: Dict[int, bool] = {}
    shard_files = sorted(method_dir.glob("per_sample_*.jsonl")) + \
                  sorted(method_dir.glob("per_sample.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No per_sample_*.jsonl files in {method_dir}")
    for sp in shard_files:
        merged.update(_load_per_row(sp))
    return merged


def _length_n_subchains(
    skeletons: List[Dict], n: int
) -> List[List[Dict]]:
    """Enumerate all length-``n`` sliding-window sub-chains across all skeletons."""
    out: List[List[Dict]] = []
    for sk in skeletons:
        clusters = sk["clusters"]
        if len(clusters) < n:
            continue
        for i in range(len(clusters) - n + 1):
            out.append(clusters[i : i + n])
    return out


def _success_rate_length_n(
    n: int,
    skeletons: List[Dict],
    successes: Dict[int, bool],
    total_hf_rows: int,
) -> Tuple[float, int, int]:
    """Returns (success_rate, sum_of_ANDs, num_instances)."""
    if n == 1:
        sum_succ = sum(1 for r in range(total_hf_rows) if successes.get(r, False))
        return (sum_succ / total_hf_rows, sum_succ, total_hf_rows)

    sub_chains = _length_n_subchains(skeletons, n)
    sum_and = 0
    sum_total = 0
    for sc in sub_chains:
        prod_s = 1
        prod_t = 1
        for cl in sc:
            rows = cl["hf_rows"]
            t_i = len(rows)
            s_i = sum(1 for r in rows if successes.get(r, False))
            prod_s *= s_i
            prod_t *= t_i
        sum_and += prod_s
        sum_total += prod_t
    if sum_total == 0:
        return (0.0, 0, 0)
    return (sum_and / sum_total, sum_and, sum_total)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--chain_skeletons", type=Path, required=True)
    p.add_argument("--results_dir", type=Path, required=True,
                   help="Dir containing <method>/per_sample_*.jsonl")
    p.add_argument("--methods", nargs="+", required=True,
                   help="Method dir names in order (rows of the table)")
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--total_hf_rows", type=int, default=1087)
    args = p.parse_args()

    print(f"[aggregate] loading skeletons {args.chain_skeletons}")
    with args.chain_skeletons.open() as fh:
        sk_doc = json.load(fh)
    skeletons = sk_doc["chain_skeletons"]
    print(f"[aggregate]   {len(skeletons)} skeletons")
    for n in range(1, 6):
        n_sub = len(_length_n_subchains(skeletons, n)) if n > 1 else "n/a"
        print(f"[aggregate]   length={n}: {n_sub} sub-chains")

    table: Dict[str, Dict[int, Dict]] = {}
    for method in args.methods:
        m_dir = args.results_dir / method
        try:
            successes = _gather_per_method_results(m_dir)
        except FileNotFoundError as e:
            print(f"[aggregate] WARN {method}: {e}; skipping")
            continue
        n_eval = len(successes)
        n_succ = sum(1 for v in successes.values() if v)
        print(f"[aggregate] {method}: {n_succ}/{n_eval} length-1 successes")

        table[method] = {}
        for n in range(1, 6):
            sr, num, denom = _success_rate_length_n(
                n, skeletons, successes, args.total_hf_rows
            )
            table[method][n] = {
                "success_rate": sr,
                "sum_successes": num,
                "num_instances": denom,
            }
            print(f"  length {n}: SR={sr*100:.2f}%  ({num}/{denom})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n=1 (%)", "n=2 (%)", "n=3 (%)", "n=4 (%)", "n=5 (%)"])
        for method in args.methods:
            if method not in table:
                w.writerow([method, "", "", "", "", ""])
                continue
            row = [method] + [
                f"{table[method][n]['success_rate']*100:.2f}" for n in range(1, 6)
            ]
            w.writerow(row)

    with args.out_json.open("w") as fh:
        json.dump(
            {
                "skeleton_source": str(args.chain_skeletons),
                "results_dir": str(args.results_dir),
                "total_hf_rows": args.total_hf_rows,
                "methods": args.methods,
                "table": table,
            },
            fh,
            indent=2,
        )
    print(f"[aggregate] wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
