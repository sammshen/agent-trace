"""
Converter for BurstGPT traces (10M+ requests from Azure OpenAI GPT services).

Source: https://github.com/HPMLL/BurstGPT
Files:  BurstGPT_1.csv, BurstGPT_2.csv, BurstGPT_3.csv

CSV fields: Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type

Each row is an independent request. This converter groups by time proximity
and optionally filters by model and log type.

Usage:
    python converters/burstgpt.py \
        --input BurstGPT_1.csv \
        --output trajectories/burstgpt.jsonl
"""

import argparse
import csv
import json
import hashlib
from typing import List, Dict, Any


def hash_from_lengths(input_length: int, output_length: int, index: int) -> List[int]:
    """Synthesize hash_ids from token counts."""
    raw = f"burstgpt:{input_length}:{output_length}:{index}".encode()
    num_hashes = max(1, input_length // 256)
    hashes = []
    for i in range(num_hashes):
        chunk = hashlib.sha256(raw + i.to_bytes(4, "big")).digest()
        hashes.append(int.from_bytes(chunk[:4], "big"))
    return hashes


def load_requests(file_path: str, model_filter: str = None, log_type_filter: str = None) -> List[Dict[str, Any]]:
    requests = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if model_filter and row["Model"].strip() != model_filter:
                continue
            if log_type_filter and log_type_filter not in row["Log Type"].strip():
                continue

            requests.append({
                "timestamp_s": float(row["Timestamp"]),
                "input_length": int(row["Request tokens"]),
                "output_length": int(row["Response tokens"]),
                "model": row["Model"].strip(),
            })

    requests.sort(key=lambda r: r["timestamp_s"])
    return requests


def group_into_trajectories(
    requests: List[Dict[str, Any]],
    session_gap_s: float,
) -> List[List[Dict[str, Any]]]:
    if not requests:
        return []

    trajectories = []
    current = [requests[0]]

    for req in requests[1:]:
        gap = req["timestamp_s"] - current[-1]["timestamp_s"]
        if gap > session_gap_s:
            trajectories.append(current)
            current = [req]
        else:
            current.append(req)

    trajectories.append(current)
    return trajectories


def convert_trajectory(requests: List[Dict[str, Any]], traj_index: int) -> List[Dict[str, Any]]:
    output = []
    for i, req in enumerate(requests):
        if i == 0:
            pre_gap_ms = 0
        else:
            pre_gap_ms = int((req["timestamp_s"] - requests[i - 1]["timestamp_s"]) * 1000)

        output.append({
            "pre_gap_ms": max(0, pre_gap_ms),
            "total_input_length": req["input_length"],
            "output_length": req["output_length"],
            "hash_ids": hash_from_lengths(req["input_length"], req["output_length"], traj_index * 1000 + i),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert BurstGPT traces to trace_runner format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file (BurstGPT_*.csv)")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file path")
    parser.add_argument("--model", type=str, default=None, help="Filter by model (e.g. 'ChatGPT', 'GPT-4')")
    parser.add_argument("--log-type", type=str, default=None, help="Filter by log type (e.g. 'Conversation', 'API')")
    parser.add_argument("--session-gap-s", type=float, default=30.0, help="Max gap in seconds between requests in same session (default: 30)")
    parser.add_argument("--no-group", action="store_true", help="Don't group — each request becomes a single-request trajectory")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories to write")
    parser.add_argument("--min-requests", type=int, default=1, help="Skip trajectories with fewer requests than this")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    requests = load_requests(args.input, args.model, args.log_type)
    print(f"Loaded {len(requests)} requests")

    if args.no_group:
        trajectories = [[r] for r in requests]
    else:
        trajectories = group_into_trajectories(requests, args.session_gap_s)

    print(f"Grouped into {len(trajectories)} trajectories")

    num_written = 0
    with open(args.output, "w") as f:
        for i, traj_requests in enumerate(trajectories):
            if len(traj_requests) < args.min_requests:
                continue
            converted = convert_trajectory(traj_requests, i)
            f.write(json.dumps(converted) + "\n")
            num_written += 1
            if args.max_trajectories and num_written >= args.max_trajectories:
                break

    print(f"Done: {num_written} trajectories written to {args.output}")


if __name__ == "__main__":
    main()
