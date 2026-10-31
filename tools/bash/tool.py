from __future__ import annotations

import os
import signal
import subprocess
import time

from tools.base import BaseTool, ToolResult


class BashTool(BaseTool):
    name = "bash"
    guide = (
        "Run one command in a new shell process. The shell is Bash on Linux and PowerShell 5.1 "
        "on Windows. Each call starts in working_directory; cd and environment changes do not "
        "carry into other calls. Only use files allowed by the request context. The timeout "
        "defaults to 60 seconds. Increase timeout for slow tests or builds."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "A command to run in the platform shell.",
            }
        },
        "required": ["command"],
    }

    def execute(self, context, arguments, timeout_seconds, cancellation) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.failure(
                "command must be a non-empty string.",
                timeout_seconds=timeout_seconds,
            )
        if cancellation.cancelled:
            return ToolResult.failure(
                "The request was cancelled before the command started.",
                timeout_seconds=timeout_seconds,
                status="cancelled",
            )

        shell = context.shell.lower()
        if shell in {"powershell", "powershell 5.1", "powershell.exe"}:
            executable = [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
        elif shell in {"bash", "/bin/bash"}:
            executable = ["/bin/bash", "-c", command]
        else:
            return ToolResult.failure(
                f"Unsupported shell in request context: {context.shell}",
                timeout_seconds=timeout_seconds,
            )

        environment = self._environment(context)
        creation_flags = 0
        process_options = {}
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            process_options["start_new_session"] = True

        process = subprocess.Popen(
            executable,
            cwd=context.working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
            **process_options,
        )

        def terminate_process_tree():
            self._terminate_tree(process)

        unregister = cancellation.register(terminate_process_tree)
        started = time.monotonic()
        try:
            while True:
                if cancellation.cancelled:
                    terminate_process_tree()
                    stdout, stderr = process.communicate()
                    return ToolResult(
                        status="cancelled",
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=process.returncode,
                        error="The request was cancelled.",
                        timeout_seconds=timeout_seconds,
                    )
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    terminate_process_tree()
                    stdout, stderr = process.communicate()
                    return ToolResult(
                        status="timeout",
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=process.returncode,
                        error=f"Command exceeded its {timeout_seconds}-second timeout.",
                        timeout_seconds=timeout_seconds,
                        timed_out=True,
                    )
                try:
                    stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                    return ToolResult(
                        status="completed",
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=process.returncode,
                        timeout_seconds=timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    continue
        finally:
            unregister()

    @staticmethod
    def _environment(context) -> dict[str, str]:
        allowed_names = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "TMPDIR",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
        }
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.upper() in allowed_names and not BashTool._is_secret_name(name)
        }
        environment.update(
            {
                name: value
                for name, value in context.shell_environment.items()
                if not BashTool._is_secret_name(name)
            }
        )
        return environment

    @staticmethod
    def _is_secret_name(name: str) -> bool:
        upper_name = name.upper()
        return any(
            marker in upper_name
            for marker in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
        )

    @staticmethod
    def _terminate_tree(process: subprocess.Popen):
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if process.poll() is None:
                process.kill()
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
