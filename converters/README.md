# Converters

Converters transform public agent trajectory datasets into the format expected by `trace_runner.py`.

## Output format

Each converter produces a `.jsonl` file where every line is one trajectory — a JSON array of requests:

```json
[
    {
        "pre_gap_ms": 0,
        "total_input_length": 10753,
        "output_length": 596,
        "hash_ids": [2847103, 991244, 3821057, ...]
    },
    {
        "pre_gap_ms": 300,
        "total_input_length": 16924,
        "output_length": 34,
        "hash_ids": [1123847, 482910, 3399012, ...]
    }
]
```

| Field | Description |
|-------|-------------|
| `pre_gap_ms` | Delay before sending this request (ms). Simulates tool execution time between turns. Always 0 for the first request. |
| `total_input_length` | Total input token count for this request, including full conversation history. A decrease from the previous request indicates a compaction. |
| `output_length` | Number of output tokens for this request. |
| `hash_ids` | One int per `tokens_per_hash`-sized chunk of the input tokens. Used for deterministic prompt reconstruction at runtime. Length ≈ `total_input_length / tokens_per_hash`. |

## Available converters

| Converter | Dataset | Timing source | Description |
|-----------|---------|---------------|-------------|
| `mooncake_traces.py` | [Mooncake](https://github.com/kvcache-ai/Mooncake) | Real timestamps | 23k Kimi production traces with KV-cache block hashes. FAST 2025 Best Paper. |
| `azure_llm_inference.py` | [Azure LLM Inference](https://github.com/Azure/AzurePublicDataset) | Real timestamps | Azure OpenAI traces (code + conversation workloads). 2023 & 2024 releases. |
| `burstgpt.py` | [BurstGPT](https://github.com/HPMLL/BurstGPT) | Real timestamps | 10M+ Azure OpenAI GPT requests over 213 days. KDD 2025. |
| `swe_agent_trajectories.py` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | Estimated from tool output | 80k SWE-agent coding trajectories (~31 steps avg). |

## Adding a new converter

1. Create `converters/<dataset_name>.py`
2. Accept at minimum `--input` (or stream from HuggingFace) and `--output` (output `.jsonl` path)
3. For each trajectory in the source dataset:
   - Identify the LLM turns (requests) and their surrounding context
   - Get or compute `total_input_length` and `output_length` in tokens
   - Set `pre_gap_ms` to 0 for the first request; derive from timestamps, estimate from tool output, or use a reasonable default for subsequent requests
   - Produce `hash_ids` — from source data if available (e.g. Mooncake block hashes), otherwise hash the input content in chunks
4. Write one JSON array per trajectory, one trajectory per line

Optional but recommended flags: `--max-trajectories`, `--min-requests`, any dataset-specific filters.

## Usage

### `mooncake_traces.py`

```bash
# download trace files from https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release
python converters/mooncake_traces.py --input toolagent_trace.jsonl --output trajectories/mooncake.jsonl
```

Groups requests into trajectories by shared `hash_id` prefixes (KV cache prefix sharing = same session). Use `--no-group` to treat each request independently.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | yes | — | Input `.jsonl` trace file |
| `--output` | yes | — | Output `.jsonl` file path |
| `--min-prefix-overlap` | no | 1 | Min shared hash_id prefix to group into a trajectory |
| `--no-group` | no | off | Each request becomes its own trajectory |
| `--max-trajectories` | no | all | Limit number of trajectories to write |
| `--min-requests` | no | 1 | Skip trajectories with fewer requests |

### `azure_llm_inference.py`

```bash
# download from https://github.com/Azure/AzurePublicDataset
python converters/azure_llm_inference.py --input AzureLLMInferenceTrace_conv.csv --output trajectories/azure_conv.jsonl
```

Groups requests into trajectories by time proximity. No prompt content in source (privacy), so `hash_ids` are synthesized from token counts.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | yes | — | Input CSV file |
| `--output` | yes | — | Output `.jsonl` file path |
| `--session-gap-ms` | no | 30000 | Max gap between requests in same session (ms) |
| `--no-group` | no | off | Each request becomes its own trajectory |
| `--max-trajectories` | no | all | Limit number of trajectories to write |
| `--min-requests` | no | 1 | Skip trajectories with fewer requests |

### `burstgpt.py`

```bash
# download from https://github.com/HPMLL/BurstGPT
python converters/burstgpt.py --input BurstGPT_1.csv --output trajectories/burstgpt.jsonl
```

Groups by time proximity. Can filter by model and log type. No prompt content, so `hash_ids` are synthesized.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | yes | — | Input CSV file |
| `--output` | yes | — | Output `.jsonl` file path |
| `--model` | no | all | Filter by model (`ChatGPT`, `GPT-4`) |
| `--log-type` | no | all | Filter by log type (`Conversation`, `API`) |
| `--session-gap-s` | no | 30 | Max gap between requests in same session (seconds) |
| `--no-group` | no | off | Each request becomes its own trajectory |
| `--max-trajectories` | no | all | Limit number of trajectories to write |
| `--min-requests` | no | 1 | Skip trajectories with fewer requests |

### `swe_agent_trajectories.py`

```bash
pip install datasets transformers
```

```bash
python converters/swe_agent_trajectories.py --output trajectories/swe_agent.jsonl --max-trajectories 100
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
| `--tokens-per-hash` | no | 256 | Number of tokens per `hash_ids` entry |
| `--max-trajectories` | no | all | Limit number of trajectories to convert |
| `--min-requests` | no | 2 | Skip trajectories with fewer requests |
| `--resolved-only` | no | off | Only include resolved trajectories |

`pre_gap_ms` is estimated per-request from the tool output between agent turns (base 50ms + 0.5ms/char, capped at 30s). Short outputs like edit confirmations → small gaps; long outputs like test suite results → larger gaps.
