Agents have 3 critical behaviors: 

1. Tool Calling
2. Subagents (Concurrent LLM calls and parallel conversation history branch-offs)
3. Context Compaction

The trace format (see `example_trace.jsonl`) handles these with: 
1. Every request has a `pre_gap_ms` field which is the amount of time that needs to be waited **after** all preceding requests have received a response to mimic synchronous tool calls in the agent harness. If the agent does some tool calls in parallel or in a non-blocking fashion, these should not be accounted for in the `pre_gap_ms`. 
2. The `comes_after` field allows requests to not be purely serialized (e.g. request 2 and request 3 can both come after request 1 instead of request 3 having to come after request 2). 
3. The `full_request_with_conversation_history` field allows editing conversation history on a per-request basis easily and also helps manage subagent contexts. 

`traces/trace_runner.py` reads a trace file and replays it against a serving endpoint. each trajectory is one agent session — a sequence of requests where each depends on previous responses via the `comes_after` DAG. the gap between requests (`pre_gap_ms`) simulates tool execution time.

each request carries `full_request_with_conversation_history` — the complete prompt text sent to the model, including all prior conversation context.

## usage

```
uv pip install openai transformers
```

```
python traces/trace_runner.py \
    --trace-file traces/example_trace.jsonl \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --port 8234
```

`--model-name` is used both for tokenization (accurate token counts in the summary) and as the model name in API requests. use `--served-model-name` to override the API model name if it differs.

see `traces/README.md` for the trace format specification.

## building traces

use `trace-builder/` to convert public datasets into trace files. the pipeline is:

```
converter → trajectories → merger.py + arrivals → trace file
```

see `trace-builder/README.md` for details. currently supports: SWE-agent trajectories (80k coding agent sessions with full conversation text).
