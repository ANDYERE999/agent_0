from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from jsonschema import Draft202012Validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error: str | None = None
    timeout_seconds: int = 60
    timed_out: bool = False
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "artifacts": self.artifacts,
        }

    def to_model_content(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def failure(
        cls,
        message: str,
        timeout_seconds: int = 60,
        status: str = "error",
    ) -> ToolResult:
        return cls(status=status, error=message, timeout_seconds=timeout_seconds)


class BaseTool(ABC):
    name: str
    guide: str
    input_schema: dict[str, Any]

    def schema_for_model(self) -> dict[str, Any]:
        schema = copy.deepcopy(self.input_schema)
        if not isinstance(schema, dict):
            raise ValueError(f"Tool {self.name!r} input_schema must be an object")
        if schema.get("type") != "object":
            raise ValueError(f"Tool {self.name!r} input_schema must have type 'object'")
        properties = schema.setdefault("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Tool {self.name!r} input_schema properties must be an object")
        if "timeout" in properties:
            raise ValueError(f"Tool {self.name!r} may not define the reserved timeout parameter")
        properties["timeout"] = {
            "type": "integer",
            "minimum": 1,
            "default": 60,
            "description": "Maximum execution time in seconds. Defaults to 60. Increase it for long-running commands.",
        }
        required = schema.setdefault("required", [])
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            raise ValueError(f"Tool {self.name!r} input_schema required must be an array of names")
        schema["additionalProperties"] = False
        Draft202012Validator.check_schema(schema)
        return schema

    def invoke(self, context, arguments, timeout_seconds, cancellation) -> ToolResult:
        started_at = utc_now()
        started_clock = time.monotonic()
        if cancellation.cancelled:
            result = ToolResult.failure(
                "The request was cancelled before the tool started.",
                timeout_seconds=timeout_seconds,
                status="cancelled",
            )
        else:
            try:
                result = self.execute(
                    context,
                    arguments,
                    timeout_seconds,
                    cancellation,
                )
            except Exception as error:
                result = ToolResult.failure(
                    f"{type(error).__name__}: {error}",
                    timeout_seconds=timeout_seconds,
                )
        return replace(
            result,
            timeout_seconds=timeout_seconds,
            started_at=started_at,
            ended_at=utc_now(),
            duration_seconds=round(time.monotonic() - started_clock, 6),
        )

    @abstractmethod
    def execute(self, context, arguments, timeout_seconds, cancellation) -> ToolResult:
        raise NotImplementedError
