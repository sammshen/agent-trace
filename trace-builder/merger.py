"""
Merge intermediate trajectories and arrival schedules into a final trace file.

Reads:
  - trajectories .jsonl (one JSON array of requests per line)
  - arrivals .csv (comma-separated floats, seconds from start)

Produces a trace file where each line is one trajectory with arrival_time_ms,
session_id, and requests array in the final trace format.

Usage:
    python trace-builder/merger.py \
        --trajectories trace-builder/trajectories/swe_agent.jsonl \
        --arrivals trace-builder/arrivals/arrivals.csv \
        --output traces/swe_agent_trace.jsonl
"""

import argparse
import json
from typing import List, Dict, Any


def load_trajectories(file_path: str) -> List[List[Dict[str, Any]]]:
    trajectories = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_arrivals(file_path: str) -> List[float]:
    arrivals = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for val in line.split(","):
                val = val.strip()
                if val:
                    arrivals.append(float(val))
    return arrivals


def ensure_request_fields(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add default request_id and comes_after if missing."""
    for i, req in enumerate(requests):
        if "request_id" not in req:
            req["request_id"] = str(i)
        if "comes_after" not in req:
            req["comes_after"] = [] if i == 0 else [str(i - 1)]
    return requests


def merge(trajectories: List[List[Dict[str, Any]]], arrivals: List[float]) -> List[Dict[str, Any]]:
    num_to_merge = min(len(trajectories), len(arrivals))
    traces = []
    for i in range(num_to_merge):
        requests = ensure_request_fields(trajectories[i])
        arrival_ms = int(arrivals[i] * 1000)
        traces.append({
            "arrival_time_ms": arrival_ms,
            "session_id": str(i),
            "requests": requests,
        })
    return traces


def main():
    parser = argparse.ArgumentParser(description="Merge trajectories and arrivals into a trace file")
    parser.add_argument("--trajectories", type=str, required=True, help="Input trajectories .jsonl")
    parser.add_argument("--arrivals", type=str, required=True, help="Input arrivals .csv")
    parser.add_argument("--output", type=str, required=True, help="Output trace .jsonl")
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectories)
    arrivals = load_arrivals(args.arrivals)

    print(f"Loaded {len(trajectories)} trajectories, {len(arrivals)} arrival times")

    num_to_merge = min(len(trajectories), len(arrivals))
    if len(trajectories) != len(arrivals):
        print(f"Warning: count mismatch — merging first {num_to_merge}")

    traces = merge(trajectories, arrivals)

    with open(args.output, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Wrote {len(traces)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
