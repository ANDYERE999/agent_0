import json
import os
from pathlib import Path

from utilities.context import RequestContext


def make_context(tmp_path: Path, session_id: str = "test-session") -> RequestContext:
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "sessions"
    workspace.mkdir(parents=True)
    sessions.mkdir()
    if os.name == "nt":
        platform, shell = "windows", "powershell"
    else:
        platform, shell = "linux", "bash"
    context_file = tmp_path / "request.json"
    context_file.write_text(
        json.dumps(
            {
                "request_id": "test-request",
                "session_id": session_id,
                "workspace_root": str(workspace),
                "working_directory": str(workspace),
                "platform": platform,
                "shell": shell,
                "read_roots": [str(workspace)],
                "write_roots": [str(workspace)],
                "session_dir": str(sessions),
                "shell_environment": {},
                "prompt_context": {},
            }
        ),
        encoding="utf-8",
    )
    return RequestContext.from_file(context_file)
