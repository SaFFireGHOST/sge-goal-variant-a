"""Compute and persist sequential manipulation chain skeletons.

Reads the HF↔CALVIN mapping JSON (built by ``build_hf_calvin_mapping.py``)
and finds physically-meaningful chains of manipulation operations that
were performed back-to-back during the original CALVIN data collection.

Definitions
-----------
* **Cluster**: A group of CALVIN segments that overlap in frame range
  by >50% — these represent the same underlying physical action that was
  annotated with multiple slightly-different (start, end) windows. All HF
  rows mapping to any segment in the cluster share the same physical
  start/end states.
* **Chain skeleton**: A maximal directed path of clusters
  ``C₁ → C₂ → ... → C_n`` such that each cluster's earliest start frame is
  within ``max_gap`` frames after the previous cluster's latest end frame
  (i.e. they happened consecutively within one CALVIN rollout).

Output schema
-------------
``{
   "max_gap": int,
   "num_clusters": int,
   "chain_skeletons": [
       {
           "chain_id": int,
           "length": int,
           "clusters": [
               {
                   "cluster_idx": int,
                   "task": str,
                   "start_frame": int,
                   "end_frame": int,
                   "hf_rows": [int, ...]   # 0-indexed HF rows that map here
               }, ...
           ]
       }, ...
   ]
}``

Usage
-----
``python -m evaluate.build_chains \
        --mapping_json .runlogs/hf_calvin_mapping.json \
        --out_json .runlogs/chain_skeletons.json \
        --max_gap 200``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_clusters(mapping_path: Path) -> List[Dict]:
    """Group overlapping CALVIN segments into physical action clusters."""
    with mapping_path.open() as fh:
        mapping = json.load(fh)

    seg_db: Dict[int, Dict] = {}
    for entry in mapping["matches"]:
        idx = entry["calvin_seg_idx"]
        if idx not in seg_db:
            seg_db[idx] = {
                "seg_idx": idx,
                "start": entry["start_frame"],
                "end": entry["end_frame"],
                "task": entry["task"],
                "hf_rows": [],
            }
        seg_db[idx]["hf_rows"].append(entry["hf_row"])

    sorted_segs = sorted(seg_db.values(), key=lambda s: s["start"])

    clusters: List[List[Dict]] = []
    for seg in sorted_segs:
        merged = False
        for cluster in clusters:
            c_start = min(x["start"] for x in cluster)
            c_end = max(x["end"] for x in cluster)
            overlap = min(seg["end"], c_end) - max(seg["start"], c_start)
            seg_dur = seg["end"] - seg["start"]
            if overlap > seg_dur * 0.5:
                cluster.append(seg)
                merged = True
                break
        if not merged:
            clusters.append([seg])

    clusters.sort(key=lambda c: min(x["start"] for x in c))
    return clusters


def _summarize_cluster(cluster_idx: int, segs: List[Dict]) -> Dict:
    hf_rows = sorted({r for s in segs for r in s["hf_rows"]})
    return {
        "cluster_idx": cluster_idx,
        "task": segs[0]["task"],   # all segs in a cluster share the task
        "start_frame": min(s["start"] for s in segs),
        "end_frame": max(s["end"] for s in segs),
        "hf_rows": hf_rows,
    }


def _build_chains(clusters: List[List[Dict]], max_gap: int) -> List[List[int]]:
    """Find maximal chains: each cluster's nearest forward successor with gap<=max_gap."""
    n = len(clusters)
    adj: Dict[int, int] = {}
    for i in range(n):
        c_end_i = max(x["end"] for x in clusters[i])
        best_j, best_gap = None, float("inf")
        for j in range(n):
            if i == j:
                continue
            c_start_j = min(x["start"] for x in clusters[j])
            gap = c_start_j - c_end_i
            if 0 < gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j is not None and best_gap <= max_gap:
            adj[i] = best_j

    successors = set(adj.values())
    predecessors = set(adj.keys())
    starts = predecessors - successors

    chains: List[List[int]] = []
    for s in starts:
        chain = [s]
        cur = s
        while cur in adj:
            cur = adj[cur]
            chain.append(cur)
        chains.append(chain)
    return chains


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mapping_json", required=True, type=Path)
    p.add_argument("--out_json", required=True, type=Path)
    p.add_argument(
        "--max_gap", type=int, default=200,
        help="Max frame gap between consecutive clusters (default 200 ≈ 6.6s @ 30fps).",
    )
    args = p.parse_args()

    print(f"[chains] loading {args.mapping_json}")
    clusters = _load_clusters(args.mapping_json)
    print(f"[chains] {len(clusters)} physical-action clusters")

    chain_idxs = _build_chains(clusters, args.max_gap)
    print(f"[chains] {len(chain_idxs)} maximal chain skeletons (max_gap={args.max_gap})")
    for length in (2, 3, 4, 5, 6, 7, 8, 9):
        n = sum(1 for c in chain_idxs if len(c) >= length)
        if n:
            print(f"[chains]   chains of length >= {length}: {n}")

    out = {
        "max_gap": args.max_gap,
        "num_clusters": len(clusters),
        "chain_skeletons": [
            {
                "chain_id": ci,
                "length": len(chain),
                "clusters": [_summarize_cluster(ci_i, clusters[ci_i]) for ci_i in chain],
            }
            for ci, chain in enumerate(chain_idxs)
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[chains] wrote {args.out_json}")


if __name__ == "__main__":
    main()
