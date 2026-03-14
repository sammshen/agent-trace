"""
Converter for Mooncake traces (Kimi/Moonshot AI production traces).

Source: https://github.com/kvcache-ai/Mooncake
Traces: FAST25-release/traces/{conversation,toolagent,synthetic}_trace.jsonl
        FAST25-release/arxiv-trace/mooncake_trace.jsonl

The Mooncake format is nearly identical to trace_runner's format.
Each line is a request with: timestamp, input_length, output_length, hash_ids.
This converter groups sequential requests into trajectories based on shared
hash_id prefixes (prefix cache sharing = same conversation session).

Usage:
    python converters/mooncake_traces.py \
        --input mooncake_toolagent_trace.jsonl \
        --output trajectories/mooncake_toolagent.jsonl
"""

import argparse
import json
from typing import List, Dict, Any


def load_requests(file_path: str) -> List[Dict[str, Any]]:
    requests = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


def group_into_trajectories(
    requests: List[Dict[str, Any]],
    min_prefix_overlap: int,
) -> List[List[Dict[str, Any]]]:
    """Group requests into trajectories based on hash_id prefix sharing.

    Two consecutive requests belong to the same trajectory if they share
    at least min_prefix_overlap hash_ids at the start (indicating KV cache
    prefix reuse = same conversation session).
    """
    if not requests:
        return []

    # sort by timestamp
    requests.sort(key=lambda r: r["timestamp"])

    # build a map: hash_id prefix tuple -> trajectory
    trajectories = []
    # track active trajectories by their hash_id prefix
    active: Dict[tuple, List[Dict[str, Any]]] = {}

    for req in requests:
        hash_ids = req.get("hash_ids", [])
        matched_key = None

        if len(hash_ids) >= min_prefix_overlap:
            # check if this request's prefix matches any active trajectory
            prefix = tuple(hash_ids[:min_prefix_overlap])
            if prefix in active:
                matched_key = prefix

        if matched_key is not None:
            active[matched_key].append(req)
        else:
            # start a new trajectory
            if hash_ids and len(hash_ids) >= min_prefix_overlap:
                key = tuple(hash_ids[:min_prefix_overlap])
            else:
                key = (id(req),)  # unique key for isolated requests
            active[key] = [req]

    trajectories = list(active.values())
    # sort each trajectory by timestamp
    for t in trajectories:
        t.sort(key=lambda r: r["timestamp"])

    return trajectories


def convert_trajectory(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a group of Mooncake requests into trace_runner format."""
    output = []
    for i, req in enumerate(requests):
        if i == 0:
            pre_gap_ms = 0
        else:
            pre_gap_ms = req["timestamp"] - requests[i - 1]["timestamp"]

        output.append({
            "pre_gap_ms": max(0, pre_gap_ms),
            "total_input_length": req["input_length"],
            "output_length": req["output_length"],
            "hash_ids": req["hash_ids"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert Mooncake traces to trace_runner format")
    parser.add_argument("--input", type=str, required=True, help="Input .jsonl trace file")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file path")
    parser.add_argument("--min-prefix-overlap", type=int, default=1, help="Minimum shared hash_id prefix length to group requests into a trajectory (default: 1)")
    parser.add_argument("--no-group", action="store_true", help="Don't group into trajectories — treat each request as a single-request trajectory")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories to write")
    parser.add_argument("--min-requests", type=int, default=1, help="Skip trajectories with fewer requests than this")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    requests = load_requests(args.input)
    print(f"Loaded {len(requests)} requests")

    if args.no_group:
        trajectories = [[r] for r in requests]
    else:
        trajectories = group_into_trajectories(requests, args.min_prefix_overlap)

    print(f"Grouped into {len(trajectories)} trajectories")

    num_written = 0
    with open(args.output, "w") as f:
        for traj_requests in trajectories:
            if len(traj_requests) < args.min_requests:
                continue
            converted = convert_trajectory(traj_requests)
            f.write(json.dumps(converted) + "\n")
            num_written += 1
            if args.max_trajectories and num_written >= args.max_trajectories:
                break

    print(f"Done: {num_written} trajectories written to {args.output}")


if __name__ == "__main__":
    main()
