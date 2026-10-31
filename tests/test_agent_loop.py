import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tools.base import BaseTool, ToolResult
from utilities.agent import Agent
from utilities.cancellation import CancellationToken
from utilities.model_client import ModelResponse, ModelToolCall

from tests.helpers import make_context


AGENT_ROOT = Path(__file__).resolve().parents[1]


class ProbeTool(BaseTool):
    name = "probe"
    guide = "A test tool that waits for a peer call."
    input_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "delay": {"type": "number"},
        },
        "required": ["label", "delay"],
    }

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def execute(self, context, arguments, timeout_seconds, cancellation):
        self.barrier.wait(timeout=2)
        time.sleep(arguments["delay"])
        return ToolResult(
            status="completed",
            stdout=arguments["label"],
            timeout_seconds=timeout_seconds,
        )


class FailingTool(BaseTool):
    name = "failing"
    guide = "A test tool that returns an error."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, context, arguments, timeout_seconds, cancellation):
        return ToolResult.failure("deliberate test failure", timeout_seconds=timeout_seconds)


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system_prompt, messages, tools, cancellation, on_text=None):
        self.calls.append([dict(message) for message in messages])
        response_value = self.responses.pop(0)
        if on_text and response_value.text:
            on_text(response_value.text)
        return response_value


def response(text="", calls=None):
    return ModelResponse(
        provider="openai_responses",
        model="fake-model",
        model_call_id=f"model-{time.time_ns()}",
        text=text,
        reasoning_text=None,
        tool_calls=calls or [],
        input_tokens=3,
        output_tokens=2,
        reasoning_tokens=None,
        response_id=None,
        stop_reason="completed",
        attempts=[],
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


def test_agent_runs_same_response_tools_in_parallel_and_returns_results_in_request_order(tmp_path):
    context = make_context(tmp_path)
    model = FakeModelClient(
        [
            response(
                calls=[
                    ModelToolCall(
                        "call-first",
                        "probe",
                        {"label": "first", "delay": 0.15, "timeout": 60},
                    ),
                    ModelToolCall(
                        "call-second",
                        "probe",
                        {"label": "second", "delay": 0.01, "timeout": 60},
                    ),
                ]
            ),
            response("done"),
        ]
    )
    settings = SimpleNamespace(provider="openai_responses", model="fake-model")
    agent = Agent(
        AGENT_ROOT,
        "common",
        context,
        settings,
        CancellationToken(),
        model_client=model,
    )
    agent.tools["probe"] = ProbeTool()

    started = time.monotonic()
    result = agent.run("Inspect the workspace.")
    elapsed = time.monotonic() - started

    assert result.status == "completed"
    assert result.text == "done"
    assert elapsed < 0.35
    tool_messages = [
        message for message in model.calls[1] if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-first",
        "call-second",
    ]
    assert json.loads(tool_messages[0]["content"])["stdout"] == "first"


def test_agent_returns_tool_errors_to_model_and_continues(tmp_path):
    context = make_context(tmp_path)
    model = FakeModelClient(
        [
            response(calls=[ModelToolCall("call-fail", "failing", {"value": "x"})]),
            response("Recovered."),
        ]
    )
    settings = SimpleNamespace(provider="openai_responses", model="fake-model")
    agent = Agent(
        AGENT_ROOT,
        "common",
        context,
        settings,
        CancellationToken(),
        model_client=model,
    )
    agent.tools["failing"] = FailingTool()

    result = agent.run("Handle a tool failure.")

    assert result.status == "completed"
    assert result.text == "Recovered."
    tool_message = next(
        message
        for message in model.calls[1]
        if message.get("role") == "tool"
    )
    assert json.loads(tool_message["content"])["error"] == "deliberate test failure"


def test_agent_resume_recovers_unfinished_calls_without_reexecution(tmp_path):
    context = make_context(tmp_path)
    initial_model = FakeModelClient([response("done")])
    settings = SimpleNamespace(provider="openai_responses", model="fake-model")
    agent = Agent(
        AGENT_ROOT,
        "common",
        context,
        settings,
        CancellationToken(),
        model_client=initial_model,
    )
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "unfinished-call",
                "name": "bash",
                "arguments": {"command": "write a file"},
            }
        ],
    }
    agent.session.append_message(assistant_message)
    agent.session.append_event(
        {
            "type": "action",
            "state": "started",
            "tool_call_id": "unfinished-call",
            "timeout_seconds": 60,
        }
    )

    resumed_model = FakeModelClient([response("Inspected.")])
    resumed = Agent(
        AGENT_ROOT,
        "common",
        context,
        settings,
        CancellationToken(),
        resume_session_id=context.session_id,
        model_client=resumed_model,
    )

    assert len(resumed.messages) == 2
    recovered = json.loads(resumed.messages[-1]["content"])
    assert recovered["status"] == "error"
    assert "side effects may have occurred" in recovered["error"]
    assert resumed.run().text == "Inspected."
