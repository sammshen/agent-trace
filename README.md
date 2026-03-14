# agent-trace

replay recorded LLM agent traces against an OpenAI-compatible inference server. designed for benchmarking serving systems with realistic multi-turn agentic workloads.

## what it does

`trace_runner.py` takes a set of trajectories and an arrivals schedule, then fires them concurrently at a serving endpoint. each trajectory is one agent session (e.g. a coding agent solving a task) — a sequence of requests where each depends on the previous response. the gap between requests (`pre_gap_ms`) simulates tool execution time.

`total_input_length` is the full conversation history at each turn. it can shrink between turns to simulate context compaction.

`hash_ids` allow deterministic prompt reconstruction from the original content without shipping the actual text.

## usage

```
pip install vllm openai transformers
```

```
python3 trace_runner.py \
    --trajectory-file trajectories/example.jsonl \
    --arrivals-file arrivals/arrivals.csv \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --port 8234
```

trajectories are jsonl (one json array per line). arrivals are csv (comma-separated floats, seconds from start). see `converters/` for tools to generate these from public datasets.

## converters

converters turn public datasets into the trajectory format. see `converters/README.md` for details.

available: mooncake (real timestamps + KV cache hashes), azure LLM inference traces, burstgpt (10M+ requests), SWE-agent trajectories (80k coding agent sessions).
