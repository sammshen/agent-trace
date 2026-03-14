import json
import os
import time
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class RequestStats:
    trajectory_id: int
    request_index: int
    input_length: int
    output_length: int
    ttft: float  # seconds, -1 if no tokens generated
    total_time: float  # seconds
    is_compaction: bool


@dataclass
class TrajectoryStats:
    trajectory_id: int
    request_stats: List[RequestStats] = field(default_factory=list)


def generate_token_ids(trajectory: List[Dict[str, Any]], tokenizer: AutoTokenizer):
    def assert_fields_present(request):
        assert isinstance(request, dict)
        assert "pre_gap_ms" in request
        assert "total_input_length" in request
        assert "output_length" in request
        assert "hash_ids" in request

    for request in trajectory:
        assert_fields_present(request)

    def convert_hash_ids_to_token_ids(hash_ids: List[int], target_length: int, tokenizer):
        import random
        import numpy as np

        if not hash_ids:
            seed = 0
            base_offset = 0
        else:
            seed = sum(hash_ids) % (2**31)
            base_offset = hash_ids[0]

        random.seed(seed)
        np.random.seed(seed)

        vocab_size = tokenizer.vocab_size

        buffer_factor = 1.2
        initial_num_tokens = int(target_length * buffer_factor)

        token_ids = []
        for i in range(initial_num_tokens):
            if hash_ids:
                token_id = (base_offset + i + sum(hash_ids[i % len(hash_ids):i % len(hash_ids) + 3])) % vocab_size
            else:
                token_id = (i * 7 + 13) % vocab_size
            token_ids.append(token_id)

        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        final_tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(final_tokens) > target_length:
            final_tokens = final_tokens[:target_length]
        elif len(final_tokens) < target_length:
            needed = target_length - len(final_tokens)
            padding = [(base_offset + len(final_tokens) + i) % vocab_size for i in range(needed)]
            final_tokens.extend(padding)

        return tokenizer.decode(final_tokens, skip_special_tokens=True)

    for request in trajectory:
        request["prompt"] = convert_hash_ids_to_token_ids(request["hash_ids"], request["total_input_length"], tokenizer)

    return trajectory


def extract_content_completions(chunk):
    if chunk.choices and chunk.choices[0].text is not None:
        return chunk.choices[0].text
    return ""


def extract_reasoning_content(chunk):
    delta = chunk.choices[0].delta
    potential_reasoning_keys = [
        "reasoning_content",
        "reasoning",
    ]
    for key in potential_reasoning_keys:
        if hasattr(delta, key) and getattr(delta, key):
            return getattr(delta, key)
    return None


def extract_normal_content(chunk):
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content:
        return delta.content
    return None


def extract_content(chunk, completions_mode=False):
    if completions_mode:
        return extract_content_completions(chunk)

    if normal_content := extract_normal_content(chunk):
        return normal_content
    elif reasoning_content := extract_reasoning_content(chunk):
        return reasoning_content
    else:
        return ""


def has_content(chunk, completions_mode=False):
    if completions_mode:
        return bool(chunk.choices) and (chunk.choices[0].text is not None)

    return (
        chunk.choices
        and chunk.choices[0].delta
        and (
            extract_normal_content(chunk) is not None
            or extract_reasoning_content(chunk) is not None
        )
    )


async def send_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    output_length: int,
    trajectory_id: int,
    request_index: int,
    input_length: int,
    is_compaction: bool,
    completions_mode: bool = False,
    eos_token_id: int = -1,
) -> RequestStats:
    start_time = time.time()
    first_token_time = None
    responses = []

    if completions_mode:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            max_tokens=output_length,
            temperature=0.0,
            stream_options={"include_usage": True},
            logit_bias={str(eos_token_id): -100}
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=output_length,
            temperature=0.0,
            stream_options={"include_usage": True},
            logit_bias={str(eos_token_id): -100}
        )

    async for chunk in response:
        if not chunk.choices:
            continue

        if has_content(chunk, completions_mode):
            content = extract_content(chunk, completions_mode)
            if first_token_time is None and content != "":
                first_token_time = time.time()
            responses.append(content)

    end_time = time.time()

    ttft = (first_token_time - start_time) if first_token_time is not None else -1
    return RequestStats(
        trajectory_id=trajectory_id,
        request_index=request_index,
        input_length=input_length,
        output_length=output_length,
        ttft=ttft,
        total_time=end_time - start_time,
        is_compaction=is_compaction,
    )


async def run_trajectory(
    trajectory: List[Dict[str, Any]],
    trajectory_id: int,
    client: AsyncOpenAI,
    model: str,
    completions_mode: bool,
    eos_token_id: int,
) -> TrajectoryStats:
    stats = TrajectoryStats(trajectory_id=trajectory_id)
    prev_input_length = 0

    for i, request in enumerate(trajectory):
        if request["pre_gap_ms"] > 0:
            await asyncio.sleep(request["pre_gap_ms"] / 1000)

        is_compaction = request["total_input_length"] < prev_input_length
        prev_input_length = request["total_input_length"]

        req_stats = await send_request(
            client=client,
            model=model,
            prompt=request["prompt"],
            output_length=request["output_length"],
            trajectory_id=trajectory_id,
            request_index=i,
            input_length=request["total_input_length"],
            is_compaction=is_compaction,
            completions_mode=completions_mode,
            eos_token_id=eos_token_id,
        )
        stats.request_stats.append(req_stats)

    return stats


def load_trajectories(file_path: str) -> List[List[Dict[str, Any]]]:
    assert file_path.endswith(".jsonl"), f"Trajectory file must be .jsonl, got: {file_path}"
    trajectories = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_arrivals(file_path: str) -> List[float]:
    assert file_path.endswith(".csv"), f"Arrivals file must be .csv, got: {file_path}"
    arrivals = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support both comma-separated values on one line and one value per line
            for val in line.split(","):
                val = val.strip()
                if val:
                    arrivals.append(float(val))
    return arrivals


def print_summary(all_stats: List[TrajectoryStats], wall_time: float):
    all_requests: List[RequestStats] = []
    for ts in all_stats:
        all_requests.extend(ts.request_stats)

    num_trajectories = len(all_stats)
    num_requests = len(all_requests)
    total_input_tokens = sum(r.input_length for r in all_requests)
    total_output_tokens = sum(r.output_length for r in all_requests)
    total_tokens = total_input_tokens + total_output_tokens
    num_compactions = sum(1 for r in all_requests if r.is_compaction)

    ttfts = [r.ttft for r in all_requests if r.ttft >= 0]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")

    avg_requests_per_trajectory = num_requests / num_trajectories if num_trajectories > 0 else 0
    avg_input_tokens_per_request = total_input_tokens / num_requests if num_requests > 0 else 0
    avg_output_tokens_per_request = total_output_tokens / num_requests if num_requests > 0 else 0

    throughput = total_tokens / wall_time if wall_time > 0 else 0
    output_throughput = total_output_tokens / wall_time if wall_time > 0 else 0

    print("\n" + "=" * 60)
    print("TRACE REPLAY SUMMARY")
    print("=" * 60)
    print(f"  Number of trajectories:           {num_trajectories}")
    print(f"  Total requests:                   {num_requests}")
    print(f"  Avg requests per trajectory:      {avg_requests_per_trajectory:.1f}")
    print(f"  Total input tokens:               {total_input_tokens}")
    print(f"  Total output tokens:              {total_output_tokens}")
    print(f"  Avg input tokens per request:     {avg_input_tokens_per_request:.0f}")
    print(f"  Avg output tokens per request:    {avg_output_tokens_per_request:.0f}")
    print(f"  Number of compactions:            {num_compactions}")
    print(f"  Wall time:                        {wall_time:.2f}s")
    print(f"  Total throughput (tok/s):         {throughput:.1f}")
    print(f"  Output throughput (tok/s):        {output_throughput:.1f}")
    print(f"  Mean TTFT:                        {mean_ttft:.4f}s")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Trace replay workload runner for LLM inference servers")
    # input files
    parser.add_argument("--trajectory-file", type=str, required=True)
    parser.add_argument("--arrivals-file", type=str, required=True)

    # model related
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name (for tokenization)")
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name as served by the API (defaults to --model-name)")
    parser.add_argument("--eos-token-id", type=int, default=-1, help="EOS token ID")

    # url related
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the API")
    parser.add_argument("--host", type=str, default=None, help="Host for the API")
    parser.add_argument("--port", type=int, default=None, help="Port for the API")

    # api related
    parser.add_argument("--completions", action="store_true", help="Use completions API instead of chat completions API")
    args = parser.parse_args()

    completions_mode = args.completions
    served_model_name = args.served_model_name or args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.eos_token_id != -1:
        eos_token_id = args.eos_token_id
        if tokenizer.eos_token_id is not None:
            assert tokenizer.eos_token_id == args.eos_token_id, "Passed in the wrong EOS token ID"
    else:
        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token ID is not set for the tokenizer, please find it and manually pass it in to ensure the output length is respected")
        eos_token_id = tokenizer.eos_token_id

    print(f"Using EOS token ID: {eos_token_id}")

    trajectories = load_trajectories(args.trajectory_file)
    arrivals = load_arrivals(args.arrivals_file)

    # match trajectories and arrivals — one arrival time per trajectory
    num_to_run = min(len(trajectories), len(arrivals))
    trajectories = trajectories[:num_to_run]
    arrivals = arrivals[:num_to_run]

    print(f"Loaded {num_to_run} trajectories")

    # pre-generate prompts for all trajectories
    for trajectory in trajectories:
        generate_token_ids(trajectory, tokenizer)

    def get_base_url(args):
        if args.base_url is not None:
            return args.base_url
        host = args.host or "localhost"
        port = args.port or 8000
        return f"http://{host}:{port}/v1"

    base_url = get_base_url(args)
    print(f"Using base URL: {base_url}")
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=None,
    )

    # launch trajectories concurrently according to arrival times
    async def launch_trajectory(trajectory, trajectory_id, delay):
        await asyncio.sleep(delay)
        return await run_trajectory(
            trajectory=trajectory,
            trajectory_id=trajectory_id,
            client=client,
            model=served_model_name,
            completions_mode=completions_mode,
            eos_token_id=eos_token_id,
        )

    tasks = []
    for i in range(num_to_run):
        tasks.append(asyncio.create_task(launch_trajectory(trajectories[i], i, arrivals[i])))

    wall_start = time.time()
    all_stats = await asyncio.gather(*tasks)
    wall_time = time.time() - wall_start

    print_summary(list(all_stats), wall_time)


if __name__ == "__main__":
    asyncio.run(main())
