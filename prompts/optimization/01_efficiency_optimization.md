You are optimizing the initial coding Agent's tool and model-call efficiency.

Improve the measured efficiency while keeping coding performance and trace data valid. Inspect the current Agent, the supplied baseline, the trace summary, and the previous two efficiency attempts before changing code.

The evaluation defines:

- tau_token = TPS * beta.
- eta_i = (sum of tool durations + tau_token * sum of model-call token counts) / (sum of each parallel batch's longest tool duration + tau_token * sum of model-call token counts).
- A_efficiency,i = GRPO(eta_i), using the five Agents in the current group.

The current group's average TPS and the configured beta determine token time. If eta_i is lower than eta_0, this direction receives the lowest group score for this component.

Focus on the call loop, tool scheduling, avoidable model calls, and tool interfaces. Preserve independent-call concurrency, per-call timeouts, error results, cancellation, and trace IDs. Do not alter how the evaluator measures duration or token usage. Prefer changes to this prompt module and related tools when those changes address the measured bottleneck.

You may modify other Agent code, tools, utilities, dependencies, and the contents of the four fixed common prompt files. Do not add, remove, rename, or edit common prompt files. The five optimization prompt files are protected and must remain unchanged. Validate the change and leave its diff and rationale in the working tree.
