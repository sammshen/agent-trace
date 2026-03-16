"""
Trace replay workload runner for LLM inference servers.

Reads a single trace file (jsonl) and replays it against an OpenAI-compatible
endpoint. Each trajectory launches at its arrival_time_ms. Within a trajectory,
requests respect the comes_after DAG: a request starts only after all its
dependencies have completed (+ pre_gap_ms delay).

Usage:
    python traces/trace_runner.py \
        --trace-file traces/swe_agent_trace.jsonl \
        --model-name meta-llama/Llama-3.2-1B-Instruct \
        --port 8234
"""

import json
import os
import time
import argparse
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class RequestStats:
    trajectory_id: int
    request_id: str
    input_length: int  # tokens
    output_length: int
    ttft: float  # seconds, -1 if no tokens generated
    total_time: float  # seconds


@dataclass
class TrajectoryStats:
    trajectory_id: int
    request_stats: List[RequestStats] = field(default_factory=list)


def extract_content_completions(chunk):
    if chunk.choices and chunk.choices[0].text is not None:
        return chunk.choices[0].text
    return ""


def extract_reasoning_content(chunk):
    delta = chunk.choices[0].delta
    for key in ("reasoning_content", "reasoning"):
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
    request_id: str,
    completions_mode: bool = False,
    tokenizer=None,
) -> RequestStats:
    start_time = time.time()
    first_token_time = None

    input_length = len(tokenizer.encode(prompt, add_special_tokens=True))

    if completions_mode:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            max_tokens=output_length,
            temperature=0.0,
            stream_options={"include_usage": True},
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=output_length,
            temperature=0.0,
            stream_options={"include_usage": True},
        )

    async for chunk in response:
        if not chunk.choices:
            continue
        if has_content(chunk, completions_mode):
            content = extract_content(chunk, completions_mode)
            if first_token_time is None and content != "":
                first_token_time = time.time()

    end_time = time.time()
    ttft = (first_token_time - start_time) if first_token_time is not None else -1

    return RequestStats(
        trajectory_id=trajectory_id,
        request_id=request_id,
        input_length=input_length,
        output_length=output_length,
        ttft=ttft,
        total_time=end_time - start_time,
    )


async def run_trajectory(
    trajectory: Dict[str, Any],
    trajectory_id: int,
    client: AsyncOpenAI,
    model: str,
    completions_mode: bool,
    tokenizer=None,
) -> TrajectoryStats:
    stats = TrajectoryStats(trajectory_id=trajectory_id)
    requests = trajectory["requests"]

    # build completion events for DAG scheduling
    completion_events: Dict[str, asyncio.Event] = {}
    for req in requests:
        completion_events[req["request_id"]] = asyncio.Event()

    async def run_request(req: Dict[str, Any]) -> RequestStats:
        # wait for all dependencies
        for dep_id in req.get("comes_after", []):
            if dep_id in completion_events:
                await completion_events[dep_id].wait()

        # wait pre_gap_ms after dependencies
        if req.get("pre_gap_ms", 0) > 0:
            await asyncio.sleep(req["pre_gap_ms"] / 1000)

        req_stats = await send_request(
            client=client,
            model=model,
            prompt=req["full_request_with_conversation_history"],
            output_length=req["output_length"],
            trajectory_id=trajectory_id,
            request_id=req["request_id"],
            completions_mode=completions_mode,
            tokenizer=tokenizer,
        )

        completion_events[req["request_id"]].set()
        return req_stats

    # launch all requests concurrently — DAG ordering handled by events
    tasks = [asyncio.create_task(run_request(req)) for req in requests]
    results = await asyncio.gather(*tasks)

    stats.request_stats = list(results)
    return stats


def load_trace(file_path: str) -> List[Dict[str, Any]]:
    trajectories = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def print_summary(all_stats: List[TrajectoryStats], wall_time: float):
    all_requests: List[RequestStats] = []
    for ts in all_stats:
        all_requests.extend(ts.request_stats)

    num_trajectories = len(all_stats)
    num_requests = len(all_requests)
    total_input_tokens = sum(r.input_length for r in all_requests)
    total_output_tokens = sum(r.output_length for r in all_requests)
    total_tokens = total_input_tokens + total_output_tokens

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
    print(f"  Wall time:                        {wall_time:.2f}s")
    print(f"  Total throughput (tok/s):         {throughput:.1f}")
    print(f"  Output throughput (tok/s):        {output_throughput:.1f}")
    print(f"  Mean TTFT:                        {mean_ttft:.4f}s")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Trace replay workload runner for LLM inference servers")
    parser.add_argument("--trace-file", type=str, required=True, help="Trace .jsonl file")

    # model related
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name (for tokenization)")
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name as served by the API (defaults to --model-name)")

    # url related
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the API")
    parser.add_argument("--host", type=str, default=None, help="Host for the API")
    parser.add_argument("--port", type=int, default=None, help="Port for the API")

    # api related
    parser.add_argument("--completions", action="store_true", help="Use completions API instead of chat completions API")
    args = parser.parse_args()

    completions_mode = args.completions
    served_model_name = args.served_model_name or args.model_name

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not served_model_name:
        served_model_name = args.model_name

    trajectories = load_trace(args.trace_file)
    print(f"Loaded {len(trajectories)} trajectories from {args.trace_file}")

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

    async def launch_trajectory(trajectory, trajectory_id):
        delay_s = trajectory["arrival_time_ms"] / 1000
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return await run_trajectory(
            trajectory=trajectory,
            trajectory_id=trajectory_id,
            client=client,
            model=served_model_name,
            completions_mode=completions_mode,
            tokenizer=tokenizer,
        )

    tasks = []
    for i, trajectory in enumerate(trajectories):
        tasks.append(asyncio.create_task(launch_trajectory(trajectory, i)))

    wall_start = time.time()
    all_stats = await asyncio.gather(*tasks)
    wall_time = time.time() - wall_start

    print_summary(list(all_stats), wall_time)


if __name__ == "__main__":
    asyncio.run(main())
