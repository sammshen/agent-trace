# Schema Design

The schema of this trace is designed to be run for inference benchmarking. This means that we are sending requests directly to an LLM instead of to an agentic application. The target metrics to be optimized are inference metrics like TTFT and throughput (tok/s).

This means that the tool calls (and everything in the agent harness) are not meant to be executed. Furthermore, the actual responses from the LLM are not processed and the conversation history accumulates according to the trace deterministically. 

Notes: 
- Tool calling in harnesses can be done concurrently with reasoning (some tool calls will be cost 0) and some tool calls can be done concurrently with each other. We specify the ability to make tool calls "sync" or "async"
- We define a trajectory as a conversation history
    - There are no timestamps inside of a task, the next request (to the LLM) will be sent once a response to the previous request arrives and all necessary tool calls have arrived
    - We will append the necessary tool call context before every request to the conversation
    - The `schema_example/trajectories.jsonl` is an example of a single trajectory trace of a dummy deep research / web task
    - We also support conversation history **compaction** to specify a new reduced conversation history. The way this should be specified
    - We assume reasoning / CoT is not added to the conversation history
- The `schema_example/arrivals.csv` specifies the frequency of tasks (not requests) coming into the system
    - An implicit assumption is that task frequency can be made orthogonal to the actual tasks themselves


