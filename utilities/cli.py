from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from utilities.agent import Agent, AgentError
from utilities.cancellation import CancellationToken
from utilities.config import ConfigError, load_settings
from utilities.context import ContextError, RequestContext
from utilities.prompting import MODES
from utilities.session import SessionError, SessionStore
from tools import ToolLoadError


def build_parser(agent_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the initial coding agent.")
    parser.add_argument("--mode", choices=sorted(MODES), help="Prompt mode.")
    parser.add_argument(
        "--context",
        required=True,
        help="Path to the read-only request-context JSON supplied by the external framework.",
    )
    parser.add_argument("--prompt", help="Task prompt.")
    parser.add_argument("--prompt-file", help="UTF-8 file containing the task prompt.")
    parser.add_argument(
        "--config",
        default=str(agent_root / "config.yaml"),
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "openai_responses", "anthropic_messages"],
        help="Override the configured provider.",
    )
    parser.add_argument("--model", help="Override the configured model.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        help="Resume a session. Omitting the value uses session_id from the request context.",
    )
    parser.add_argument("--tui", action="store_true", help="Start the interactive terminal UI.")
    return parser


class Console:
    def __init__(self, stream_text: bool = True):
        self.stream_text = stream_text
        self._lock = threading.Lock()

    def text(self, chunk: str):
        if not self.stream_text or not chunk:
            return
        with self._lock:
            print(chunk, end="", flush=True)

    def status(self, message: str, details: dict):
        with self._lock:
            print(f"\n[{message}]", file=sys.stderr, flush=True)


def _read_prompt(args, is_tty: bool) -> str | None:
    if args.prompt is not None and args.prompt_file is not None:
        raise AgentError("Use either --prompt or --prompt-file, not both.")
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if not is_tty:
        return sys.stdin.read()
    return None


def _create_agent(
    agent_root,
    mode,
    context,
    settings,
    cancellation,
    resume_session_id,
    console,
):
    if mode is None:
        raise AgentError("--mode is required for a new session.")
    return Agent(
        agent_root=agent_root,
        mode=mode,
        context=context,
        settings=settings,
        cancellation=cancellation,
        resume_session_id=resume_session_id,
        on_text=console.text,
        on_status=console.status,
    )


def _run_tui(agent_factory, initial_prompt, resume_requested, context):
    agent = None
    if resume_requested is not None:
        selected_id = resume_requested or context.session_id
        agent = agent_factory(selected_id)
        result = agent.run(initial_prompt)
        if result.text and not initial_prompt:
            print(result.text)
        if result.status != "completed":
            return 0 if result.status == "cancelled" else 1
    elif initial_prompt:
        agent = agent_factory(None)
        result = agent.run(initial_prompt)
        print()
        if result.status != "completed":
            return 0 if result.status == "cancelled" else 1

    while True:
        try:
            prompt = input("\nAgent> ").strip()
        except EOFError:
            return 0
        if not prompt:
            continue
        if prompt == "/exit":
            return 0
        if prompt == "/resume" or prompt.startswith("/resume "):
            session_id = prompt.removeprefix("/resume").strip() or context.session_id
            try:
                agent = agent_factory(session_id)
                result = agent.run()
            except (AgentError, SessionError) as error:
                print(f"Resume failed: {error}", file=sys.stderr)
                continue
            print()
            if result.status != "completed":
                return 0 if result.status == "cancelled" else 1
            continue
        try:
            if agent is None:
                agent = agent_factory(None)
            result = agent.run(prompt)
        except (AgentError, SessionError) as error:
            print(f"Agent error: {error}", file=sys.stderr)
            return 1
        print()
        if result.status != "completed":
            return 0 if result.status == "cancelled" else 1


def main(argv: list[str] | None = None) -> int:
    agent_root = Path(__file__).resolve().parent.parent
    parser = build_parser(agent_root)
    args = parser.parse_args(argv)
    cancellation = CancellationToken()

    def cancel_handler(signum, frame):
        cancellation.cancel()

    previous_handlers = {}
    for signum in ("SIGINT", "SIGTERM"):
        signal_number = getattr(__import__("signal"), signum, None)
        if signal_number is not None:
            previous_handlers[signal_number] = __import__("signal").signal(
                signal_number,
                cancel_handler,
            )

    try:
        context = RequestContext.from_file(args.context)
        console = Console()
        prompt = _read_prompt(args, is_tty=sys.stdin.isatty())
        if args.resume is not None and prompt == "":
            prompt = None
        if not args.tui and args.resume is None and prompt is None:
            raise AgentError("Provide --prompt, --prompt-file, or pipe a prompt to stdin.")
        if args.mode is None and args.resume is None:
            raise AgentError("--mode is required for a new session.")

        def agent_factory(resume_session_id):
            mode = args.mode
            provider = args.provider
            model = args.model
            if resume_session_id is not None:
                metadata, _, _ = SessionStore(
                    context.session_path(resume_session_id)
                ).read()
                mode = mode or metadata.get("mode")
                provider = provider or metadata.get("provider")
                model = model or metadata.get("model")
            if mode is None:
                raise AgentError("--mode is required for a new session.")
            settings = load_settings(
                args.config,
                provider_override=provider,
                model_override=model,
            )
            return _create_agent(
                agent_root,
                mode,
                context,
                settings,
                cancellation,
                resume_session_id,
                console,
            )

        if args.tui:
            return _run_tui(agent_factory, prompt, args.resume, context)

        resume_session_id = None
        if args.resume is not None:
            resume_session_id = args.resume or context.session_id
        agent = agent_factory(resume_session_id)
        result = agent.run(prompt)
        if result.status == "error":
            return 1
        return 0
    except (AgentError, ConfigError, ContextError, SessionError, ToolLoadError, OSError) as error:
        print(f"Agent error: {error}", file=sys.stderr)
        return 1
    finally:
        signal_module = __import__("signal")
        for signal_number, handler in previous_handlers.items():
            signal_module.signal(signal_number, handler)
