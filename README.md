# Initial coding agent

This repository contains the initial Agent used by the RSI experiment. It has one entry point, code_agent.py, and supports the common, o_1, o_2, o_3, o_4, and assemble modes.

## Install and configure

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
```

Set the API key for the configured provider in the process environment. The key is not stored in config.yaml or passed to Bash tools.

Copy config_example.yaml to config.yaml, then set a model under the provider you will use. config.yaml is ignored by Git. For an OpenAI Responses model, set providers.openai_responses.model. For an OpenAI-compatible endpoint, also set providers.openai_compatible.base_url. For Anthropic Messages, set providers.anthropic_messages.model.

The configured request timeout is 300 seconds. Transient API failures receive up to three retries with exponential backoff. Streaming and native function calling are always enabled.

## Request context

The external framework creates the isolated workspace and writes a read-only JSON context file before starting the Agent. The Agent does not create or manage isolation.

The context must provide request_id, session_id, workspace_root, working_directory, platform, shell, read_roots, write_roots, and session_dir. platform is windows or linux; shell is powershell or bash. session_dir must already exist. shell_environment is optional and contains only environment variables that the external framework permits the command tool to receive. prompt_context is optional JSON passed with the user request.

```json
{
  "request_id": "task-001",
  "session_id": "task-001",
  "workspace_root": "E:/work/task-001",
  "working_directory": "E:/work/task-001",
  "platform": "windows",
  "shell": "powershell",
  "read_roots": ["E:/work/task-001"],
  "write_roots": ["E:/work/task-001"],
  "session_dir": "E:/work/sessions",
  "shell_environment": {},
  "prompt_context": {}
}
```

The editor resolves relative paths against workspace_root, resolves symbolic links, and enforces the context's read and write roots. Bash starts in working_directory. The external framework must enforce operating-system-level isolation.

## Run

Headless mode accepts a prompt directly or from a UTF-8 file:

```powershell
python code_agent.py --mode common --context request.json --prompt "Fix the reported issue."
python code_agent.py --mode common --context request.json --prompt-file task.md
```

Start the terminal interface with:

```powershell
python code_agent.py --mode common --context request.json --tui
```

Enter /exit to leave the TUI. Enter /resume or /resume SESSION_ID to resume a saved session. Headless mode resumes with --resume or --resume SESSION_ID. The Agent reads the original mode, provider, and model from the session. It rejects a different request ID or workspace.

The external framework can stop a run by sending SIGINT or SIGTERM. The Agent stops scheduling new tools, terminates active tool process trees, and saves the session and trace.

## Sessions and traces

The Agent appends messages, think records, action records, retries, and task status to session_dir/SESSION_ID.jsonl. Each record is flushed as it is written. Token fields that the provider does not return are null.

If a process stops after a tool started but before its result was saved, resume marks that call as interrupted and asks the model to inspect the workspace rather than repeating a potentially side-effecting command.

The Agent records raw events. The external scheduler builds trace graphs, assigns stages, computes validation rounds and rewards, restores protected optimization prompts, and evaluates code changes.
