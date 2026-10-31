You are optimizing how the initial coding Agent explores a repository.

Improve the required-file coverage while reducing unnecessary repository reads. Inspect the current Agent, the supplied baseline and trace, the ground-truth changed-file set, and the previous two exploration attempts.

For candidate i:

- F_gt is the set of ground-truth changed files. F_i is the set of repository files read during the exploration phase.
- A_must,i = |F_gt intersect F_i| / |F_gt|.
- B_i = Token(F_i), B_gt = Token(F_gt), and H = max(2 * B_gt, 64000).
- A_redundant,i = 0 when B_i <= 2 * H. When B_i > 2 * H, A_redundant,i = A_must,i * exp((B_i - 2 * B_gt) / (Token(F) - B_gt)).
- A_explore,i = GRPO(A_must,i - A_redundant,i).

If the candidate's SWE-bench pass rate is lower than a_0, this direction receives the lowest group score for this component. Use the supplied evaluator definitions and file-token counts. Do not edit trace records, tests, or evaluation data to increase the score.

Improve targeted code discovery, path selection, and file-reading tools. Prefer changes to this prompt module and related tools. You may modify other Agent code, tools, utilities, dependencies, and the contents of the four fixed common prompt files. Do not add, remove, rename, or edit common prompt files. The five optimization prompt files are protected and must remain unchanged. Validate the change and leave its diff and rationale in the working tree.
