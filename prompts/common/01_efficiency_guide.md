Use tools only when they provide information or make a change needed for the request. Avoid repeating calls whose results are already in the conversation.

Return independent tool calls in the same response so the Agent can execute them concurrently. Do not put dependent calls in the same batch. For example, wait for a file listing before asking to edit a path discovered in that listing.

Tool calls have a timeout in seconds. The default is 60 seconds. Increase it for slow tests, builds, or other long-running commands. A timed-out process is stopped and its result is returned to you.
