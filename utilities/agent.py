from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tools.base import ToolResult
from tools import load_all_tools
from utilities.model_client import (
    ModelClient,
    ModelRequestError,
    RequestCancelled,
)
from utilities.prompting import build_system_prompt, build_user_message
from utilities.session import SessionError, SessionStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRunResult:
    status: str
    text: str = ""
    session_id: str = ""


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        agent_root,
        mode,
        context,
        settings,
        cancellation,
        resume_session_id: str | None = None,
        model_client=None,
        on_text=None,
        on_status=None,
    ):
        self.agent_root = agent_root
        self.mode = mode
        self.context = context
        self.settings = settings
        self.cancellation = cancellation
        self.on_text = on_text
        self.on_status = on_status
        self._status_lock = threading.Lock()
        self.tools = load_all_tools()
        self.model_client = model_client or ModelClient(settings)
        self.system_prompt = build_system_prompt(
            agent_root,
            mode,
            self.tools,
            context,
        )
        selected_session_id = resume_session_id or context.session_id
        self.session = SessionStore(context.session_path(selected_session_id))
        self.messages: list[dict[str, Any]] = []

        if resume_session_id is not None:
            self._resume(resume_session_id)
        else:
            self._create_session()

    def _session_metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.context.session_id,
            "request_id": self.context.request_id,
            "workspace_root": str(self.context.workspace_root),
            "working_directory": str(self.context.working_directory),
            "mode": self.mode,
            "provider": self.settings.provider,
            "model": self.settings.model,
            "system_prompt": self.system_prompt,
        }

    def _create_session(self):
        if self.session.exists():
            raise SessionError(
                f"Session already exists. Resume it with --resume {self.context.session_id}."
            )
        self.session.create(self._session_metadata())

    def _resume(self, requested_session_id: str):
        metadata, messages, events = self.session.read()
        checks = {
            "session_id": requested_session_id,
            "request_id": self.context.request_id,
            "workspace_root": str(self.context.workspace_root),
            "working_directory": str(self.context.working_directory),
            "mode": self.mode,
            "provider": self.settings.provider,
            "model": self.settings.model,
        }
        mismatches = [
            key
            for key, expected in checks.items()
            if metadata.get(key) != expected
        ]
        if mismatches:
            raise SessionError(
                "Cannot resume because these session fields do not match the current request: "
                + ", ".join(mismatches)
            )
        self.system_prompt = metadata["system_prompt"]
        self.messages = self.session.recover_incomplete_tool_calls(messages, events)

    def run(self, prompt: str | None = None) -> AgentRunResult:
        if prompt is not None:
            if not prompt.strip():
                raise AgentError("The user prompt cannot be empty.")
            message = {
                "role": "user",
                "content": build_user_message(prompt, self.context),
            }
            self.messages.append(message)
            self.session.append_message(message)
        elif not self.messages:
            raise AgentError("A prompt is required for a new session.")

        if prompt is None and self._last_message_is_final():
            last_message = self.messages[-1]
            return AgentRunResult(
                status="completed",
                text=last_message.get("content", ""),
                session_id=self.session.path.stem,
            )

        while not self.cancellation.cancelled:
            try:
                response = self.model_client.complete(
                    self.system_prompt,
                    self.messages,
                    self.tools,
                    self.cancellation,
                    on_text=self.on_text,
                )
            except RequestCancelled:
                return self._finish("cancelled")
            except ModelRequestError as error:
                self.session.append_event(
                    {
                        "type": "error",
                        "event_id": str(uuid.uuid4()),
                        "request_id": self.context.request_id,
                        "session_id": self.session.path.stem,
                        "timestamp": _utc_now(),
                        "error": str(error),
                        "attempts": error.attempts,
                    }
                )
                self._notify_status(str(error))
                return self._finish("error")

            assistant_message = {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in response.tool_calls
                ],
            }
            think_event = {
                "type": "think",
                "event_id": str(uuid.uuid4()),
                "request_id": self.context.request_id,
                "session_id": self.session.path.stem,
                "model_call_id": response.model_call_id,
                "provider": response.provider,
                "model": response.model,
                "response_id": response.response_id,
                "text": response.text,
                "reasoning_text": response.reasoning_text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "started_at": response.started_at,
                "ended_at": response.ended_at,
                "duration_seconds": response.duration_seconds,
                "stop_reason": response.stop_reason,
                "attempts": response.attempts,
                "tool_call_ids": [call.id for call in response.tool_calls],
            }
            self.session.append_event(think_event)
            self.messages.append(assistant_message)
            self.session.append_message(assistant_message)

            if not response.tool_calls:
                return self._finish("completed", response.text)
            if self.cancellation.cancelled:
                return self._finish("cancelled")

            tool_messages = self._run_tool_batch(response)
            for tool_message in tool_messages:
                self.messages.append(tool_message)
                self.session.append_message(tool_message)
            if self.cancellation.cancelled:
                return self._finish("cancelled")

        return self._finish("cancelled")

    def _run_tool_batch(self, response) -> list[dict[str, Any]]:
        batch_id = str(uuid.uuid4())
        calls = response.tool_calls
        parameters_by_call = {}
        started_by_call = {}
        for call in calls:
            timeout_seconds, arguments, validation_error = self._prepare_call(call)
            parameters_by_call[call.id] = (timeout_seconds, arguments, validation_error)
            started_at = _utc_now()
            started_by_call[call.id] = started_at
            self.session.append_event(
                {
                    "type": "action",
                    "state": "started",
                    "event_id": str(uuid.uuid4()),
                    "request_id": self.context.request_id,
                    "session_id": self.session.path.stem,
                    "model_call_id": response.model_call_id,
                    "batch_id": batch_id,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "parameters": arguments,
                    "timeout_seconds": timeout_seconds,
                    "started_at": started_at,
                }
            )
            self._notify_status(
                f"Started {call.name} ({timeout_seconds}s timeout)",
                tool_call_id=call.id,
            )

        results = {}
        max_workers = len(calls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_call,
                    call,
                    parameters_by_call[call.id],
                ): call
                for call in calls
            }
            for future in as_completed(futures):
                call = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    timeout_seconds = parameters_by_call[call.id][0]
                    result = ToolResult.failure(
                        f"{type(error).__name__}: {error}",
                        timeout_seconds=timeout_seconds,
                    )
                results[call.id] = result
                result_data = result.to_dict()
                self.session.append_event(
                    {
                        "type": "action",
                        "state": "finished",
                        "event_id": str(uuid.uuid4()),
                        "request_id": self.context.request_id,
                        "session_id": self.session.path.stem,
                        "model_call_id": response.model_call_id,
                        "batch_id": batch_id,
                        "tool_call_id": call.id,
                        "tool_name": call.name,
                        "parameters": parameters_by_call[call.id][1],
                        "timeout_seconds": result.timeout_seconds,
                        "started_at": result.started_at or started_by_call[call.id],
                        "ended_at": result.ended_at or _utc_now(),
                        "duration_seconds": result.duration_seconds,
                        "result": result_data,
                    }
                )
                self._notify_status(
                    f"Finished {call.name}: {result.status}",
                    tool_call_id=call.id,
                    result=result_data,
                )

        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": results[call.id].to_model_content(),
            }
            for call in calls
        ]

    def _prepare_call(self, call):
        if not isinstance(call.arguments, dict):
            return (
                60,
                call.arguments,
                "Tool arguments must be a JSON object.",
            )
        arguments = dict(call.arguments)
        timeout_seconds = arguments.pop("timeout", 60)
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            return (
                60,
                arguments,
                "timeout must be a positive integer number of seconds.",
            )
        tool = self.tools.get(call.name)
        if tool is None:
            return timeout_seconds, arguments, f"Tool {call.name!r} is not registered."
        schema = tool.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        unknown = [name for name in arguments if name not in properties]
        if missing:
            return timeout_seconds, arguments, f"Missing required arguments: {', '.join(missing)}."
        if unknown:
            return timeout_seconds, arguments, f"Unsupported arguments: {', '.join(unknown)}."
        for name, value in arguments.items():
            expected_type = properties.get(name, {}).get("type")
            if not self._matches_type(value, expected_type):
                return (
                    timeout_seconds,
                    arguments,
                    f"Argument {name!r} must have type {expected_type}.",
                )
        return timeout_seconds, arguments, None

    @staticmethod
    def _matches_type(value, expected_type) -> bool:
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        return True

    def _execute_call(self, call, prepared):
        timeout_seconds, arguments, validation_error = prepared
        if validation_error:
            return ToolResult.failure(
                validation_error,
                timeout_seconds=timeout_seconds,
            )
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult.failure(
                f"Tool {call.name!r} is not registered.",
                timeout_seconds=timeout_seconds,
            )
        return tool.invoke(
            self.context,
            arguments,
            timeout_seconds,
            self.cancellation,
        )

    def _last_message_is_final(self) -> bool:
        if not self.messages:
            return False
        last = self.messages[-1]
        return last.get("role") == "assistant" and not last.get("tool_calls")

    def _finish(self, status: str, text: str = "") -> AgentRunResult:
        self.session.append_event(
            {
                "type": "task",
                "event_id": str(uuid.uuid4()),
                "request_id": self.context.request_id,
                "session_id": self.session.path.stem,
                "status": status,
                "ended_at": _utc_now(),
            }
        )
        return AgentRunResult(
            status=status,
            text=text,
            session_id=self.session.path.stem,
        )

    def _notify_status(self, message: str, **details):
        if self.on_status:
            with self._status_lock:
                self.on_status(message, details)
