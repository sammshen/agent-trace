"""
Converter for Azure LLM Inference Traces (2023 & 2024).

Source: https://github.com/Azure/AzurePublicDataset
Files:  AzureLLMInferenceTrace_code.csv, AzureLLMInferenceTrace_conv.csv

Each row is an independent request with a timestamp and token counts.
This converter groups consecutive requests into trajectories by time proximity
(requests within a session window are assumed to be part of the same conversation).

Since Azure traces have no prompt content (privacy), hash_ids are synthesized
from token counts to allow deterministic prompt reconstruction.

Usage:
    python converters/azure_llm_inference.py \
        --input AzureLLMInferenceTrace_conv.csv \
        --output trajectories/azure_conv.jsonl
"""

import argparse
import csv
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any


def hash_from_lengths(input_length: int, output_length: int, index: int) -> List[int]:
    """Synthesize hash_ids from token counts since Azure has no prompt content."""
    raw = f"{input_length}:{output_length}:{index}".encode()
    h = hashlib.sha256(raw).digest()
    # produce enough hashes to roughly cover input at 256 tokens/hash
    num_hashes = max(1, input_length // 256)
    hashes = []
    for i in range(num_hashes):
        chunk = hashlib.sha256(raw + i.to_bytes(4, "big")).digest()
        hashes.append(int.from_bytes(chunk[:4], "big"))
    return hashes


def load_requests(file_path: str) -> List[Dict[str, Any]]:
    requests = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["TIMESTAMP"].strip()
            ts = datetime.strptime(ts_str[:26], "%Y-%m-%d %H:%M:%S.%f")
            requests.append({
                "timestamp_ms": int(ts.timestamp() * 1000),
                "input_length": int(row["ContextTokens"]),
                "output_length": int(row["GeneratedTokens"]),
            })
    requests.sort(key=lambda r: r["timestamp_ms"])
    return requests


def group_into_trajectories(
    requests: List[Dict[str, Any]],
    session_gap_ms: int,
) -> List[List[Dict[str, Any]]]:
    """Group requests into trajectories by time proximity.

    Requests separated by more than session_gap_ms start a new trajectory.
    Within a session, increasing input_length suggests multi-turn conversation.
    """
    if not requests:
        return []

    trajectories = []
    current = [requests[0]]

    for req in requests[1:]:
        gap = req["timestamp_ms"] - current[-1]["timestamp_ms"]
        if gap > session_gap_ms:
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
            pre_gap_ms = req["timestamp_ms"] - requests[i - 1]["timestamp_ms"]

        output.append({
            "pre_gap_ms": max(0, pre_gap_ms),
            "total_input_length": req["input_length"],
            "output_length": req["output_length"],
            "hash_ids": hash_from_lengths(req["input_length"], req["output_length"], traj_index * 1000 + i),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert Azure LLM Inference traces to trace_runner format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file (AzureLLMInferenceTrace_*.csv)")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file path")
    parser.add_argument("--session-gap-ms", type=int, default=30000, help="Max gap between requests in same session (default: 30000ms)")
    parser.add_argument("--no-group", action="store_true", help="Don't group — each request becomes a single-request trajectory")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories to write")
    parser.add_argument("--min-requests", type=int, default=1, help="Skip trajectories with fewer requests than this")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    requests = load_requests(args.input)
    print(f"Loaded {len(requests)} requests")

    if args.no_group:
        trajectories = [[r] for r in requests]
    else:
        trajectories = group_into_trajectories(requests, args.session_gap_ms)

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
