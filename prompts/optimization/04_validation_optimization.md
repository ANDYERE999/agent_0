You are optimizing test generation and validation in the initial coding Agent.

Generate tests that pass on the candidate's own implementation and fail on implementations with the relevant defect. Inspect the current Agent, the supplied baseline and trace, the five candidate versions, and the previous two validation attempts.

For candidate i, T_i is its generated test set and P_i is the subset that passes on its own code and is valid under the evaluator. For each t in P_i, test the other four candidate code versions c_j. The base score is:

s_i = (1 / |T_i|) * sum over t in P_i and j != i of 1[t(c_j) = fail] - GRPO(n_i)

n_i is the validation-modification round count. The round term is a subtraction. Use this final definition even if an older body-text formula shows a plus sign. A_test_gen,i = GRPO(s_i). The candidate's own implementation must pass its own tests. If s_i is below s_0, this direction receives the lowest group score for the test-generation component.

Preserve all ground-truth tests and test results. Do not make a test appear valid by changing the code under test or the evaluator. Prefer improvements to test prompts, test discovery, and related tools. You may modify other Agent code, tools, utilities, dependencies, and the contents of the four fixed common prompt files. Do not add, remove, rename, or edit common prompt files. The five optimization prompt files are protected and must remain unchanged. Validate the change and leave its diff and rationale in the working tree.
