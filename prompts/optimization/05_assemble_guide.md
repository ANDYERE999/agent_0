Integrate the accepted harness changes into the writable output workspace supplied in the request context. The base Agent and accepted candidate workspaces are read-only. Do not copy, create, or manage workspaces; the external scheduler prepares them.

Use the candidate diffs, traces, reward values, and accepted-direction list supplied in the user context. Preserve unrelated base changes. Integrate only the directions provided by the scheduler. Resolve overlapping edits by comparing their intended behavior and validation evidence. Do not change the protected optimization prompts.

The five Agent scores use the following dimensions: efficiency, exploration, code generation, test generation, and ground-truth pass rate. For candidate i and dimension d, normalize each group-relative reward as A_tilde_d,i = tanh(A_d,i), then calculate Score_i = sum_d(w_i,d * A_tilde_d,i) / sum_d(w_i,d). Each candidate uses its own weights:

| Agent | Efficiency | Exploration | Code generation | Test generation | Ground-truth pass rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| a_0 | 1 | 1 | 1 | 1 | 4 |
| a_1 | 3 | 1 | 1 | 1 | 4 |
| a_2 | 1 | 3 | 1 | 1 | 4 |
| a_3 | 1 | 1 | 3 | 1 | 4 |
| a_4 | 1 | 1 | 1 | 3 | 4 |

The scheduler calculates scores and selects the candidates to integrate. If every direction scores below a_0, the scheduler supplies the highest-scoring direction. Do not recalculate acceptance or replace the scheduler's selection.

Run the checks available in the workspace. Leave the integrated code in the output workspace and report the integrated directions, changed files, validation results, and unresolved conflicts.
