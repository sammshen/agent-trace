"""
Converter for nebius/SWE-agent-trajectories HuggingFace dataset.

Converts SWE-agent trajectory data into the intermediate trajectory format:
  - Each trajectory becomes one JSONL line (a list of requests)
  - Each request has: pre_gap_ms, request_id, comes_after,
    full_request_with_conversation_history, output_length

Usage:
    python trace-builder/converters/swe_agent_trajectories.py \
        --output trace-builder/trajectories/swe_agent.jsonl
"""

import argparse
import json
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer

# map dataset model_name values to HuggingFace tokenizer IDs
MODEL_NAME_TO_TOKENIZER = {
    "swe-agent-llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "swe-agent-llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "swe-agent-qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "swe-agent-qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# pre_gap_ms estimation: base latency + per-character factor for tool output
BASE_TOOL_LATENCY_MS = 50       # minimum latency for any tool call
MS_PER_OUTPUT_CHAR = 0.5        # longer outputs imply longer-running commands
MAX_PRE_GAP_MS = 30_000         # cap at 30s


def estimate_pre_gap_ms(tool_output_text: str) -> int:
    """Estimate tool execution time from the tool output between ai turns."""
    n = len(tool_output_text)
    return min(MAX_PRE_GAP_MS, BASE_TOOL_LATENCY_MS + int(n * MS_PER_OUTPUT_CHAR))


def resolve_tokenizer(dataset_model_name: str, override: str = None) -> str:
    """Resolve a HuggingFace tokenizer ID from the dataset's model_name field."""
    if override:
        return override

    if dataset_model_name in MODEL_NAME_TO_TOKENIZER:
        return MODEL_NAME_TO_TOKENIZER[dataset_model_name]

    for key, value in MODEL_NAME_TO_TOKENIZER.items():
        if key in dataset_model_name or dataset_model_name in key:
            return value

    return None


def convert_trajectory(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
) -> List[Dict[str, Any]]:
    """Convert a single SWE-agent trajectory into intermediate request format.

    Each 'ai' message is one LLM request. The input is the full conversation
    history up to (but not including) the ai message. The output is the ai
    message itself. pre_gap_ms is estimated from the tool output between turns.
    """
    requests = []
    conversation_so_far = []
    tool_output_since_last_ai = ""

    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("text", "") or ""

        if role == "system":
            conversation_so_far.append(text)
            continue

        if role == "ai":
            input_text = "\n".join(conversation_so_far)
            output_tokens = tokenizer.encode(text, add_special_tokens=False)
            output_length = len(output_tokens)

            if output_length == 0:
                conversation_so_far.append(text)
                tool_output_since_last_ai = ""
                continue

            request_index = len(requests)
            is_first = request_index == 0

            requests.append({
                "pre_gap_ms": 0 if is_first else estimate_pre_gap_ms(tool_output_since_last_ai),
                "request_id": str(request_index),
                "comes_after": [] if is_first else [str(request_index - 1)],
                "full_request_with_conversation_history": input_text,
                "output_length": output_length,
            })

            conversation_so_far.append(text)
            tool_output_since_last_ai = ""

        elif role == "user":
            conversation_so_far.append(text)
            tool_output_since_last_ai += text

    return requests


def main():
    parser = argparse.ArgumentParser(description="Convert nebius/SWE-agent-trajectories to intermediate trajectory format")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file path")
    parser.add_argument("--model-name", type=str, default=None, help="Override HuggingFace tokenizer (auto-detected from dataset if omitted)")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories to convert")
    parser.add_argument("--min-requests", type=int, default=2, help="Skip trajectories with fewer requests than this")
    parser.add_argument("--resolved-only", action="store_true", help="Only include resolved trajectories (target=true)")
    args = parser.parse_args()

    print("Loading dataset: nebius/SWE-agent-trajectories (streaming)")
    dataset = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

    dataset_iter = iter(dataset)
    first_row = next(dataset_iter)
    dataset_model_name = first_row.get("model_name", "")

    tokenizer_id = resolve_tokenizer(dataset_model_name, args.model_name)
    if not tokenizer_id:
        raise ValueError(
            f"Cannot resolve tokenizer for dataset model_name={dataset_model_name!r}. "
            f"Known mappings: {list(MODEL_NAME_TO_TOKENIZER.keys())}. "
            f"Pass --model-name <hf_model_id> to override."
        )

    print(f"Dataset model_name: {dataset_model_name!r} -> tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    num_written = 0
    num_skipped = 0

    def process_row(row, i):
        nonlocal num_written, num_skipped

        if args.max_trajectories is not None and num_written >= args.max_trajectories:
            return False

        if args.resolved_only and not row.get("target", False):
            num_skipped += 1
            return True

        trajectory_messages = row.get("trajectory", [])
        if not trajectory_messages:
            num_skipped += 1
            return True

        requests = convert_trajectory(trajectory_messages, tokenizer)

        if len(requests) < args.min_requests:
            num_skipped += 1
            return True

        f.write(json.dumps(requests) + "\n")
        num_written += 1

        if num_written % 100 == 0:
            print(f"  Converted {num_written} trajectories (skipped {num_skipped}, scanned {i + 1})")

        return True

    with open(args.output, "w") as f:
        if not process_row(first_row, 0):
            pass
        else:
            for i, row in enumerate(dataset_iter, start=1):
                if not process_row(row, i):
                    break

    print(f"\nDone: {num_written} trajectories written to {args.output} (skipped {num_skipped})")


if __name__ == "__main__":
    main()
