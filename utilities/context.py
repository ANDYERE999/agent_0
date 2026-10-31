from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ContextError(ValueError):
    pass


def _resolved(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ContextError(f"Cannot resolve path {path!r}: {error}") from error


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    session_id: str
    workspace_root: Path
    working_directory: Path
    platform: str
    shell: str
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    session_dir: Path
    shell_environment: dict[str, str] = field(default_factory=dict)
    prompt_context: dict[str, Any] = field(default_factory=dict)
    additional_paths: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> RequestContext:
        context_path = Path(path).expanduser().resolve(strict=True)
        if not context_path.is_file():
            raise ContextError(f"Request context is not a file: {context_path}")
        try:
            payload = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContextError(f"Cannot read request context {context_path}: {error}") from error
        if not isinstance(payload, dict):
            raise ContextError("Request context must be a JSON object.")

        required = [
            "request_id",
            "session_id",
            "workspace_root",
            "working_directory",
            "platform",
            "shell",
            "read_roots",
            "write_roots",
            "session_dir",
        ]
        missing = [name for name in required if name not in payload]
        if missing:
            raise ContextError(f"Request context is missing fields: {', '.join(missing)}")

        workspace_root = _resolved(payload["workspace_root"])
        working_directory = _resolved(payload["working_directory"])
        read_roots = cls._roots(payload["read_roots"], "read_roots")
        write_roots = cls._roots(payload["write_roots"], "write_roots")
        session_dir = _resolved(payload["session_dir"])
        if not workspace_root.is_dir():
            raise ContextError(f"workspace_root does not exist: {workspace_root}")
        if not working_directory.is_dir():
            raise ContextError(f"working_directory does not exist: {working_directory}")
        if not session_dir.is_dir():
            raise ContextError(f"session_dir does not exist: {session_dir}")
        if not any(_is_within(working_directory, root) for root in read_roots + write_roots):
            raise ContextError("working_directory is outside the allowed read and write roots.")
        if payload["platform"].lower() not in {"windows", "linux"}:
            raise ContextError("platform must be windows or linux.")
        if not isinstance(payload["shell"], str):
            raise ContextError("shell must be a string.")
        shell_name = payload["shell"].lower()
        if payload["platform"].lower() == "windows" and shell_name not in {
            "powershell",
            "powershell 5.1",
            "powershell.exe",
        }:
            raise ContextError("Windows requests must use PowerShell 5.1.")
        if payload["platform"].lower() == "linux" and shell_name not in {
            "bash",
            "/bin/bash",
        }:
            raise ContextError("Linux requests must use Bash.")
        shell_environment = payload.get("shell_environment", {})
        if not isinstance(shell_environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in shell_environment.items()
        ):
            raise ContextError("shell_environment must map strings to strings.")
        prompt_context = payload.get("prompt_context", {})
        if not isinstance(prompt_context, dict):
            raise ContextError("prompt_context must be an object.")
        additional_paths = {
            key: value
            for key, value in payload.items()
            if key not in set(required) | {"shell_environment", "prompt_context"}
        }
        return cls(
            request_id=str(payload["request_id"]),
            session_id=str(payload["session_id"]),
            workspace_root=workspace_root,
            working_directory=working_directory,
            platform=payload["platform"].lower(),
            shell=payload["shell"],
            read_roots=read_roots,
            write_roots=write_roots,
            session_dir=session_dir,
            shell_environment=dict(shell_environment),
            prompt_context=prompt_context,
            additional_paths=additional_paths,
        )

    @staticmethod
    def _roots(value, field_name: str) -> tuple[Path, ...]:
        if not isinstance(value, list) or not value:
            raise ContextError(f"{field_name} must be a non-empty array of paths.")
        roots = tuple(_resolved(path) for path in value)
        if any(not root.exists() for root in roots):
            missing = [str(root) for root in roots if not root.exists()]
            raise ContextError(f"{field_name} contains paths that do not exist: {missing}")
        return roots

    def resolve_path(self, path: str, access: str = "read") -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ContextError("Tool paths must be non-empty strings.")
        supplied = Path(path).expanduser()
        candidate = supplied if supplied.is_absolute() else self.workspace_root / supplied
        resolved = _resolved(candidate)
        roots = self.write_roots if access == "write" else self.read_roots + self.write_roots
        if not any(_is_within(resolved, root) for root in roots):
            raise ContextError(f"Access denied for {access} path: {resolved}")
        return resolved

    def session_path(self, session_id: str | None = None) -> Path:
        selected_id = session_id or self.session_id
        if not selected_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in selected_id):
            raise ContextError("session_id may contain only letters, digits, dots, underscores, and hyphens.")
        return self.session_dir / f"{selected_id}.jsonl"

    def prompt_fields(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "workspace_root": str(self.workspace_root),
            "working_directory": str(self.working_directory),
            "platform": self.platform,
            "shell": self.shell,
            "read_roots": [str(root) for root in self.read_roots],
            "write_roots": [str(root) for root in self.write_roots],
            **self.additional_paths,
        }
