from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path

from tools.base import BaseTool, ToolResult


class EditTool(BaseTool):
    name = "editor"
    guide = (
        "View files or directories, create files, or replace an entire existing file. "
        "Use view before editing. Paths are relative to workspace_root unless absolute. "
        "Only access paths allowed by the request context. Edits require the full new file content. "
        "This tool has no partial or line-range edits. Set timeout above 60 seconds only when needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "edit"],
                "description": "Operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path, relative to workspace_root or absolute.",
            },
            "file_text": {
                "type": "string",
                "description": "Complete file content for create or edit.",
            },
        },
        "required": ["command", "path"],
    }

    def execute(self, context, arguments, timeout_seconds, cancellation) -> ToolResult:
        command = arguments["command"]
        path = context.resolve_path(arguments["path"], access="read" if command == "view" else "write")
        if command == "view":
            return self._view(path, timeout_seconds)
        if command == "create":
            return self._write(path, arguments.get("file_text"), create=True, timeout=timeout_seconds)
        if command == "edit":
            return self._write(path, arguments.get("file_text"), create=False, timeout=timeout_seconds)
        return ToolResult.failure(
            f"Unsupported editor command: {command}",
            timeout_seconds=timeout_seconds,
        )

    def _view(self, path: Path, timeout: int) -> ToolResult:
        if not path.exists():
            return ToolResult.failure(f"Path does not exist: {path}", timeout_seconds=timeout)
        if path.is_dir():
            entries = []
            for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
                if child.name.startswith(".") or child.is_symlink():
                    continue
                entries.append(str(child.relative_to(path)))
                if child.is_dir():
                    for nested in sorted(child.iterdir(), key=lambda item: item.name.lower()):
                        if not nested.name.startswith(".") and not nested.is_symlink():
                            entries.append(str(nested.relative_to(path)))
            text = "\n".join(entries)
            return ToolResult(
                status="completed",
                stdout=text,
                timeout_seconds=timeout,
                artifacts={"files_read": [str(path)]},
            )
        if not path.is_file():
            return ToolResult.failure(f"Path is not a regular file: {path}", timeout_seconds=timeout)
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            return ToolResult.failure(
                f"File is not valid UTF-8 text: {error}",
                timeout_seconds=timeout,
            )
        numbered = "\n".join(
            f"{line_number:>6}\t{line}"
            for line_number, line in enumerate(contents.splitlines(), start=1)
        )
        return ToolResult(
            status="completed",
            stdout=numbered,
            timeout_seconds=timeout,
            artifacts={"files_read": [str(path)]},
        )

    def _write(self, path: Path, file_text, create: bool, timeout: int) -> ToolResult:
        if not isinstance(file_text, str):
            return ToolResult.failure(
                "file_text must be a string containing the complete file content.",
                timeout_seconds=timeout,
            )
        if not path.parent.is_dir():
            return ToolResult.failure(
                f"Parent directory does not exist: {path.parent}",
                timeout_seconds=timeout,
            )
        if create and path.exists():
            return ToolResult.failure(f"File already exists: {path}", timeout_seconds=timeout)
        if not create and not path.is_file():
            return ToolResult.failure(f"File does not exist: {path}", timeout_seconds=timeout)

        old_text = path.read_text(encoding="utf-8") if path.exists() else ""
        diff = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                file_text.splitlines(keepends=True),
                fromfile=str(path) if path.exists() else "/dev/null",
                tofile=str(path),
            )
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".agent-tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as temporary_file:
                temporary_file.write(file_text)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if create:
                os.link(temporary_name, path)
            else:
                os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

        return ToolResult(
            status="completed",
            stdout=f"{'Created' if create else 'Updated'} {path}\n{diff}",
            timeout_seconds=timeout,
            artifacts={
                "files_written": [str(path)],
                "diff": diff,
            },
        )
