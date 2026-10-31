import json
from types import SimpleNamespace

from utilities.cancellation import CancellationToken
from utilities.model_client import (
    IncompleteModelStream,
    ModelClient,
    _to_anthropic_messages,
    _to_responses_input,
)


def test_responses_message_conversion_preserves_tool_results():
    converted = _to_responses_input(
        [
            {"role": "user", "content": "Fix the issue."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "bash",
                        "arguments": {"command": "pytest"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "bash",
                "content": "{\"status\":\"completed\"}",
            },
        ]
    )

    assert converted[0]["role"] == "user"
    assert converted[1]["type"] == "function_call"
    assert json.loads(converted[1]["arguments"]) == {"command": "pytest"}
    assert converted[2]["type"] == "function_call_output"
    assert converted[2]["call_id"] == "call-1"


def test_anthropic_message_conversion_groups_parallel_tool_results():
    converted = _to_anthropic_messages(
        [
            {"role": "user", "content": "Run tests."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "bash", "arguments": {"command": "pytest"}},
                    {"id": "b", "name": "editor", "arguments": {"command": "view"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": "{\"status\":\"completed\"}",
            },
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": "{\"status\":\"timeout\"}",
            },
        ]
    )

    assert converted[-1]["role"] == "user"
    assert len(converted[-1]["content"]) == 2
    assert converted[-1]["content"][1]["is_error"]


class FakeStream(list):
    def close(self):
        self.closed = True


def test_openai_compatible_stream_assembles_tool_arguments_and_usage():
    text_delta = SimpleNamespace(
        content="Checking.",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="call-1",
                function=SimpleNamespace(name="bash", arguments='{"command":"pytest'),
            )
        ],
    )
    argument_delta = SimpleNamespace(
        content=None,
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='","timeout":90}'),
            )
        ],
    )
    stream = FakeStream(
        [
            SimpleNamespace(
                id="response-1",
                choices=[SimpleNamespace(delta=text_delta, finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                id="response-1",
                choices=[SimpleNamespace(delta=argument_delta, finish_reason="tool_calls")],
                usage=None,
            ),
            SimpleNamespace(
                id="response-1",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
                ),
            ),
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: stream)
        )
    )
    settings = SimpleNamespace(
        provider="openai_compatible",
        model="fake-model",
        stream_include_usage=False,
        request_timeout_seconds=10,
    )
    model_client = ModelClient(settings, client=client)
    displayed = []

    result = model_client._stream_chat_completions(
        "system",
        [{"role": "user", "content": "Run tests."}],
        [],
        CancellationToken(),
        displayed.append,
    )

    assert result.text == "Checking."
    assert displayed == ["Checking."]
    assert result.tool_calls[0].arguments == {"command": "pytest", "timeout": 90}
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert stream.closed


def test_openai_compatible_incomplete_stream_rejects_partial_tool_call():
    stream = FakeStream(
        [
            SimpleNamespace(
                id="response-1",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="bash",
                                        arguments='{"command":"rm -rf',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream))
    )
    settings = SimpleNamespace(
        provider="openai_compatible",
        model="fake-model",
        stream_include_usage=False,
        request_timeout_seconds=10,
    )

    try:
        ModelClient(settings, client=client)._stream_chat_completions(
            "system",
            [{"role": "user", "content": "Run a command."}],
            [],
            CancellationToken(),
            None,
        )
    except IncompleteModelStream:
        pass
    else:
        raise AssertionError("An incomplete stream must not return tool calls.")

    assert stream.closed
