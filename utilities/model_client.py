from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
import openai


class RequestCancelled(RuntimeError):
    pass


class ModelProtocolError(RuntimeError):
    pass


class IncompleteModelStream(ModelProtocolError):
    pass


class ModelRequestError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


@dataclass
class ModelToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class ModelResponse:
    provider: str
    model: str
    model_call_id: str
    text: str
    reasoning_text: str | None
    tool_calls: list[ModelToolCall]
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    response_id: str | None = None
    stop_reason: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None


def _value(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    if not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _usage_value(value: Any, name: str) -> int | None:
    result = _value(value, name)
    return result if isinstance(result, int) and not isinstance(result, bool) else None


class ModelClient:
    def __init__(self, settings, client=None):
        self.settings = settings
        self._injected_client = client
        self.client = client or self._create_client()

    def _create_client(self):
        api_key = self.settings.api_key
        if self.settings.provider in {"openai_responses", "openai_compatible"}:
            options = {
                "api_key": api_key or "EMPTY",
                "timeout": self.settings.request_timeout_seconds,
                "max_retries": 0,
            }
            if self.settings.base_url:
                options["base_url"] = self.settings.base_url
            return openai.OpenAI(**options)
        if self.settings.provider == "anthropic_messages":
            return anthropic.Anthropic(
                api_key=api_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=0,
            )
        raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def schemas(self, tools: dict[str, Any]) -> list[dict[str, Any]]:
        if self.settings.provider == "anthropic_messages":
            return [
                {
                    "name": tool.name,
                    "description": tool.guide,
                    "input_schema": tool.schema_for_model(),
                }
                for tool in tools.values()
            ]
        if self.settings.provider == "openai_responses":
            return [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.guide,
                    "parameters": tool.schema_for_model(),
                    "strict": False,
                }
                for tool in tools.values()
            ]
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.guide,
                    "parameters": tool.schema_for_model(),
                    "strict": False,
                },
            }
            for tool in tools.values()
        ]

    def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: dict[str, Any],
        cancellation,
        on_text=None,
    ) -> ModelResponse:
        api_tools = self.schemas(tools)
        attempts = []
        started_at = _utc_now()
        started_clock = time.monotonic()

        for attempt_number in range(self.settings.max_retries + 1):
            if cancellation.cancelled:
                raise RequestCancelled("The request was cancelled.")
            attempt_started_at = _utc_now()
            try:
                if self.settings.provider == "openai_responses":
                    response = self._stream_responses(
                        system_prompt, messages, api_tools, cancellation, on_text
                    )
                elif self.settings.provider == "openai_compatible":
                    response = self._stream_chat_completions(
                        system_prompt, messages, api_tools, cancellation, on_text
                    )
                else:
                    response = self._stream_anthropic(
                        system_prompt, messages, api_tools, cancellation, on_text
                    )
                response.attempts = attempts + [
                    {
                        "attempt": attempt_number + 1,
                        "started_at": attempt_started_at,
                        "ended_at": _utc_now(),
                        "status": "completed",
                    }
                ]
                response.started_at = started_at
                response.ended_at = _utc_now()
                response.duration_seconds = round(time.monotonic() - started_clock, 6)
                return response
            except RequestCancelled:
                raise
            except Exception as error:
                if cancellation.cancelled:
                    raise RequestCancelled("The request was cancelled.") from error
                attempt = {
                    "attempt": attempt_number + 1,
                    "started_at": attempt_started_at,
                    "ended_at": _utc_now(),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                attempts.append(attempt)
                if not self._is_retryable(error) or attempt_number >= self.settings.max_retries:
                    raise ModelRequestError(
                        f"Model request failed after {len(attempts)} attempt(s): "
                        f"{type(error).__name__}: {error}",
                        attempts=attempts,
                    ) from error
                delay = min(
                    self.settings.retry_backoff_base_seconds * (2**attempt_number),
                    self.settings.retry_backoff_max_seconds,
                )
                attempt["retry_delay_seconds"] = delay
                if cancellation.wait(delay):
                    raise RequestCancelled("The request was cancelled during API retry backoff.")

        raise ModelRequestError("Model request ended without a response.", attempts=attempts)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, IncompleteModelStream):
            return True
        transient_types = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            TimeoutError,
            ConnectionError,
        )
        if isinstance(error, transient_types):
            return True
        status_code = getattr(error, "status_code", None)
        return status_code == 429 or isinstance(status_code, int) and 500 <= status_code <= 599

    def _stream_chat_completions(
        self,
        system_prompt,
        messages,
        tools,
        cancellation,
        on_text,
    ) -> ModelResponse:
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(_to_chat_messages(messages))
        options = {
            "model": self.settings.model,
            "messages": api_messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": True,
            "timeout": self.settings.request_timeout_seconds,
        }
        if self.settings.stream_include_usage:
            options["stream_options"] = {"include_usage": True}
        stream = self.client.chat.completions.create(**options)
        unregister = cancellation.register(getattr(stream, "close", lambda: None))
        text_parts = []
        reasoning_parts = []
        calls_by_index = {}
        usage = None
        response_id = None
        finish_reason = None
        try:
            for chunk in stream:
                if cancellation.cancelled:
                    raise RequestCancelled("The request was cancelled.")
                response_id = _value(chunk, "id", response_id)
                usage = _value(chunk, "usage", usage)
                choices = _value(chunk, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = _value(choice, "finish_reason", finish_reason)
                delta = _value(choice, "delta", {})
                text_delta = _value(delta, "content")
                if isinstance(text_delta, str):
                    text_parts.append(text_delta)
                    if on_text:
                        on_text(text_delta)
                reasoning_delta = _value(delta, "reasoning_content")
                if reasoning_delta is None:
                    reasoning_delta = _value(delta, "reasoning")
                if isinstance(reasoning_delta, str):
                    reasoning_parts.append(reasoning_delta)
                for tool_delta in _value(delta, "tool_calls", []) or []:
                    index = _value(tool_delta, "index", 0)
                    state = calls_by_index.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    state["id"] += _value(tool_delta, "id", "") or ""
                    function = _value(tool_delta, "function", {})
                    state["name"] += _value(function, "name", "") or ""
                    state["arguments"] += _value(function, "arguments", "") or ""
        finally:
            unregister()
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        if finish_reason is None:
            raise IncompleteModelStream(
                "The Chat Completions stream ended without a finish reason."
            )
        if finish_reason not in {"stop", "tool_calls", "function_call"}:
            raise ModelProtocolError(
                f"Chat Completions returned unexpected finish reason: {finish_reason}."
            )
        tool_calls = [
            ModelToolCall(
                id=state["id"] or str(uuid.uuid4()),
                name=state["name"],
                arguments=_parse_arguments(state["arguments"]),
            )
            for _, state in sorted(calls_by_index.items())
        ]
        if finish_reason == "length":
            raise ModelProtocolError("The model response reached its output token limit.")
        usage = _value(usage, "usage", usage)
        return ModelResponse(
            provider=self.settings.provider,
            model=self.settings.model,
            model_call_id=str(uuid.uuid4()),
            text="".join(text_parts),
            reasoning_text="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            reasoning_tokens=_usage_value(
                _value(usage, "completion_tokens_details"),
                "reasoning_tokens",
            ),
            response_id=response_id,
            stop_reason=finish_reason,
        )

    def _stream_responses(
        self,
        system_prompt,
        messages,
        tools,
        cancellation,
        on_text,
    ) -> ModelResponse:
        manager = self.client.responses.stream(
            model=self.settings.model,
            instructions=system_prompt,
            input=_to_responses_input(messages),
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            max_output_tokens=self.settings.max_output_tokens,
            timeout=self.settings.request_timeout_seconds,
        )
        text_parts = []
        reasoning_parts = []
        with manager as stream:
            unregister = cancellation.register(getattr(stream, "close", lambda: None))
            try:
                for event in stream:
                    if cancellation.cancelled:
                        raise RequestCancelled("The request was cancelled.")
                    event_type = _value(event, "type", "")
                    if event_type == "response.output_text.delta":
                        delta = _value(event, "delta", "")
                        if isinstance(delta, str):
                            text_parts.append(delta)
                            if on_text:
                                on_text(delta)
                    elif event_type in {
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = _value(event, "delta", "")
                        if isinstance(delta, str):
                            reasoning_parts.append(delta)
                response = stream.get_final_response()
            finally:
                unregister()

        status = _value(response, "status")
        output = _value(response, "output", []) or []
        tool_calls = []
        for item in output:
            if _value(item, "type") != "function_call":
                continue
            tool_calls.append(
                ModelToolCall(
                    id=_value(item, "call_id") or str(uuid.uuid4()),
                    name=_value(item, "name", ""),
                    arguments=_parse_arguments(_value(item, "arguments", "")),
                )
            )
        if status is None:
            raise IncompleteModelStream("The Responses API stream ended without a final status.")
        response_text = _value(response, "output_text")
        if not isinstance(response_text, str):
            response_text = "".join(text_parts)
        if status == "incomplete":
            raise ModelProtocolError("The Responses API ended before completing the response.")
        if status != "completed":
            raise ModelProtocolError(f"The Responses API returned unexpected status: {status}.")
        reasoning = "".join(reasoning_parts) or _responses_reasoning(output)
        usage = _value(response, "usage")
        output_details = _value(usage, "output_tokens_details")
        return ModelResponse(
            provider=self.settings.provider,
            model=self.settings.model,
            model_call_id=str(uuid.uuid4()),
            text=response_text,
            reasoning_text=reasoning or None,
            tool_calls=tool_calls,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            reasoning_tokens=_usage_value(output_details, "reasoning_tokens"),
            response_id=_value(response, "id"),
            stop_reason=status,
        )

    def _stream_anthropic(
        self,
        system_prompt,
        messages,
        tools,
        cancellation,
        on_text,
    ) -> ModelResponse:
        manager = self.client.messages.stream(
            model=self.settings.model,
            max_tokens=self.settings.max_output_tokens,
            system=system_prompt,
            messages=_to_anthropic_messages(messages),
            tools=tools,
            tool_choice={"type": "auto"},
            timeout=self.settings.request_timeout_seconds,
        )
        text_parts = []
        reasoning_parts = []
        with manager as stream:
            unregister = cancellation.register(getattr(stream, "close", lambda: None))
            try:
                for event in stream:
                    if cancellation.cancelled:
                        raise RequestCancelled("The request was cancelled.")
                    event_type = _value(event, "type", "")
                    delta = _value(event, "delta", {})
                    if event_type == "content_block_delta":
                        delta_type = _value(delta, "type")
                        if delta_type == "text_delta":
                            text = _value(delta, "text", "")
                            if isinstance(text, str):
                                text_parts.append(text)
                                if on_text:
                                    on_text(text)
                        elif delta_type in {"thinking_delta", "redacted_thinking_delta"}:
                            reasoning = _value(delta, "thinking")
                            if isinstance(reasoning, str):
                                reasoning_parts.append(reasoning)
                message = stream.get_final_message()
            finally:
                unregister()

        text_parts = []
        reasoning_parts = []
        tool_calls = []
        for block in _value(message, "content", []) or []:
            block_type = _value(block, "type")
            if block_type == "text":
                text = _value(block, "text", "")
                text_parts.append(text)
            elif block_type == "thinking":
                reasoning = _value(block, "thinking")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
            elif block_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        id=_value(block, "id") or str(uuid.uuid4()),
                        name=_value(block, "name", ""),
                        arguments=_parse_arguments(_value(block, "input", {})),
                    )
                )
        text = "".join(text_parts)
        usage = _value(message, "usage")
        stop_reason = _value(message, "stop_reason")
        if stop_reason == "max_tokens":
            raise ModelProtocolError("Anthropic reached its output token limit.")
        if stop_reason is None:
            raise IncompleteModelStream("The Anthropic stream ended without a stop reason.")
        if stop_reason not in {"end_turn", "stop_sequence", "tool_use", "pause_turn", "refusal"}:
            raise ModelProtocolError(f"Anthropic returned unexpected stop reason: {stop_reason}.")
        return ModelResponse(
            provider=self.settings.provider,
            model=self.settings.model,
            model_call_id=str(uuid.uuid4()),
            text=text,
            reasoning_text="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            reasoning_tokens=_usage_value(usage, "reasoning_tokens"),
            response_id=_value(message, "id"),
            stop_reason=stop_reason,
        )


def _to_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        role = message.get("role")
        if role in {"user", "assistant"}:
            output = {"role": role, "content": message.get("content", "")}
            calls = message.get("tool_calls", [])
            if calls:
                output["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call.get("arguments", {}),
                                ensure_ascii=False,
                            )
                            if isinstance(call.get("arguments"), dict)
                            else str(call.get("arguments", "")),
                        },
                    }
                    for call in calls
                ]
            converted.append(output)
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message.get("content", ""),
                }
            )
        else:
            raise ModelProtocolError(f"Unsupported conversation role: {role}")
    return converted


def _to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        elif role == "assistant":
            if content:
                converted.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
            for call in message.get("tool_calls", []):
                arguments = call.get("arguments", {})
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                        if isinstance(arguments, dict)
                        else str(arguments),
                    }
                )
        elif role == "tool":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content", ""),
                }
            )
        else:
            raise ModelProtocolError(f"Unsupported conversation role: {role}")
    return converted


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    pending_results = []

    def flush_results():
        if pending_results:
            converted.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "tool":
            try:
                result = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                result = {}
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": content,
                    "is_error": result.get("status") in {"error", "timeout", "cancelled"}
                    if isinstance(result, dict)
                    else False,
                }
            )
            continue

        flush_results()
        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                }
            )
        elif role == "assistant":
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls", []):
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": arguments,
                    }
                )
            if blocks:
                converted.append({"role": "assistant", "content": blocks})
        else:
            raise ModelProtocolError(f"Unsupported conversation role: {role}")
    flush_results()
    return converted


def _responses_reasoning(output) -> str:
    parts = []
    for item in output or []:
        if _value(item, "type") != "reasoning":
            continue
        for summary in _value(item, "summary", []) or []:
            text = _value(summary, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
