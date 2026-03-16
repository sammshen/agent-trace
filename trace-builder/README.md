# trace-builder

tools for building trace files from public datasets.

## pipeline

```
converter → trajectories/*.jsonl → merger.py + arrivals/*.csv → traces/*.jsonl
```

1. **converters** transform raw datasets into intermediate trajectory format (one jsonl line per trajectory)
2. **merger.py** combines trajectories with an arrival schedule to produce a final trace file

## intermediate trajectory format

each jsonl line is one trajectory — an array of requests:

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
| `pre_gap_ms` | Delay before sending this request (ms). Simulates tool execution time. 0 for first request. |
| `request_id` | Unique request ID within the trajectory (string). |
| `comes_after` | List of request_ids that must complete before this request starts. |
| `full_request_with_conversation_history` | Full prompt text including conversation history. |
| `output_length` | Number of output tokens. |

## merger.py

merges trajectories with arrival times into a final trace file.

```bash
python trace-builder/merger.py \
    --trajectories trace-builder/trajectories/swe_agent.jsonl \
    --arrivals trace-builder/arrivals/arrivals.csv \
    --output traces/swe_agent_trace.jsonl
```

arrivals csv is comma-separated floats (seconds from start). trajectory i gets arrival time arrivals[i].

## converters

see `converters/README.md` for available converters and usage.
