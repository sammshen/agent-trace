"""
Pseudo Code: 

run_trajectory():
    conversation_history = []
    for step in trajectory: 
        role = step[role]
        if role == "system":
            add to conversation history
        if role == "user": 
            send request with current conversation history
            wait for response
        if role == "assistant":
            if the assistant response requires some tools: 
                find the futures needed (see below) and wait on them
            append response to conversation history
        if role == "tool":
            if tool is synchronous:
                wait for tool response tags
            if tool is asynchronous
                return a future
        if role == "compaction":
            drop tags_to_drop
            append or prepend replacement_content to conversation history
            

main():
    arguments: 
    1. trace.jsonl
    2. arrivals.csv (zero offset it if needed)

    for each arrival time in arrivals.csv:
        spawn a process to run the next trajectory
"""