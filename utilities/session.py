from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from tools.base import utc_now


class SessionError(RuntimeError):
    pass


class SessionStore:
    schema_version = 1

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.exists()

    def create(self, metadata: dict[str, Any]):
        if self.path.exists():
            raise SessionError(f"Session already exists: {self.path}")
        entry = {
            "type": "meta",
            "schema_version": self.schema_version,
            "created_at": utc_now(),
            **metadata,
        }
        self._append(entry)

    def read(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        metadata = {}
        messages = []
        events = []
        if not self.path.exists():
            raise SessionError(f"Session does not exist: {self.path}")
        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        for line_number, raw_line in enumerate(raw_lines, start=1):
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as error:
                if line_number == len(raw_lines):
                    break
                raise SessionError(f"Invalid session JSON at line {line_number}: {error}") from error
            if not isinstance(entry, dict):
                raise SessionError(f"Session line {line_number} must contain a JSON object.")
            if entry.get("type") == "meta":
                metadata = entry
            elif entry.get("type") == "message":
                messages.append(entry["message"])
            else:
                events.append(entry)
        if metadata.get("schema_version") != self.schema_version:
            raise SessionError("Unsupported session schema version.")
        return metadata, messages, events

    def append_message(self, message: dict[str, Any]):
        self._append({"type": "message", "message": message, "timestamp": utc_now()})

    def append_event(self, event: dict[str, Any]):
        self._append(event)

    def _append(self, entry: dict[str, Any]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
            stream.flush()

    def recover_incomplete_tool_calls(
        self,
        messages: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        finished_actions = {}
        started_actions = set()
        for event in events:
            if event.get("type") != "action":
                continue
            call_id = event.get("tool_call_id")
            if not call_id:
                continue
            if event.get("state") == "started":
                started_actions.add(call_id)
            elif event.get("state") == "finished":
                finished_actions[call_id] = event.get("result", {})

        completed_ids = {
            message.get("tool_call_id")
            for message in messages
            if message.get("role") == "tool"
        }
        recovered = list(messages)
        for message in list(messages):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not call_id or call_id in completed_ids:
                    continue
                if call_id in finished_actions:
                    result = finished_actions[call_id]
                else:
                    reason = (
                        "The process stopped before this tool returned a saved result. "
                        "Its side effects may have occurred. Inspect the workspace before repeating it."
                        if call_id in started_actions
                        else "The process stopped before this tool was started."
                    )
                    result = {
                        "status": "error",
                        "stdout": "",
                        "stderr": "",
                        "exit_code": None,
                        "error": reason,
                        "timeout_seconds": call.get("timeout_seconds", 60),
                        "timed_out": False,
                        "started_at": None,
                        "ended_at": utc_now(),
                        "duration_seconds": None,
                        "artifacts": {},
                    }
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call.get("name", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                }
                recovered.append(tool_message)
                self.append_message(tool_message)
        return recovered
