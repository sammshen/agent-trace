# Converters

Converters transform public agent trajectory datasets into the intermediate trajectory format expected by `merger.py`.

## Output format

Each converter produces a `.jsonl` file where every line is one trajectory — a JSON array of requests:

```json
[
    {
        "pre_gap_ms": 0,
        "request_id": "0",
        "comes_after": [],
        "full_request_with_conversation_history": "...",
        "output_length": 596
    }
]
```

| Field | Description |
|-------|-------------|
| `pre_gap_ms` | Delay before sending this request (ms). Simulates tool execution time between turns. Always 0 for the first request. |
| `request_id` | Unique request ID within the trajectory (string, sequential). |
| `comes_after` | List of request_ids that must complete before this request starts. Sequential by default. |
| `full_request_with_conversation_history` | Full prompt text including all conversation history up to this turn. |
| `output_length` | Number of output tokens for this request. |

Converters must produce `full_request_with_conversation_history` with actual text content. Datasets that only have token counts (no text) cannot be used.

## Available converters

| Converter | Dataset | Timing source | Description |
|-----------|---------|---------------|-------------|
| `swe_agent_trajectories.py` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | Estimated from tool output | 80k SWE-agent coding trajectories (~31 steps avg). |

## swe_agent_trajectories.py

```bash
pip install datasets transformers
```

```bash
python trace-builder/converters/swe_agent_trajectories.py \
    --output trace-builder/trajectories/swe_agent.jsonl \
    --max-trajectories 100
```

Tokenizer is auto-detected from the dataset's `model_name` field. Known mappings:

| Dataset `model_name` | Tokenizer |
|----------------------|-----------|
| `swe-agent-llama-70b` | `meta-llama/Llama-3.1-70B-Instruct` |
| `swe-agent-llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `swe-agent-qwen-72b` | `Qwen/Qwen2.5-72B-Instruct` |
| `swe-agent-qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` |

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--output` | yes | — | Output `.jsonl` file path |
| `--model-name` | no | auto-detected | Override HuggingFace tokenizer ID |
| `--max-trajectories` | no | all | Limit number of trajectories to convert |
| `--min-requests` | no | 2 | Skip trajectories with fewer requests |
| `--resolved-only` | no | off | Only include resolved trajectories |

`pre_gap_ms` is estimated per-request from the tool output between agent turns (base 50ms + 0.5ms/char, capped at 30s).

## Adding a new converter

1. Create `converters/<dataset_name>.py`
2. Accept `--output` (output `.jsonl` path) and optionally `--input` or stream from HuggingFace
3. For each trajectory in the source dataset:
   - Produce `full_request_with_conversation_history` — the full prompt text at each turn
   - Count output tokens for `output_length`
   - Set `pre_gap_ms` to 0 for the first request; derive from timestamps or estimate for subsequent requests
   - Assign sequential `request_id` and `comes_after`
4. Write one JSON array per trajectory, one trajectory per line

Optional but recommended flags: `--max-trajectories`, `--min-requests`, any dataset-specific filters.
