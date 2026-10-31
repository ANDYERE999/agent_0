import os

from tools import load_all_tools
from tools.base import BaseTool
from tools.bash.tool import BashTool
from tools.edit.tool import EditTool
from utilities.cancellation import CancellationToken
from utilities.context import ContextError

from tests.helpers import make_context


def test_context_resolves_workspace_paths_and_rejects_traversal(tmp_path):
    context = make_context(tmp_path)
    file_path = context.workspace_root / "module.py"
    file_path.write_text("value = 1\n", encoding="utf-8")

    assert context.resolve_path("module.py", access="read") == file_path.resolve()
    try:
        context.resolve_path("../outside.txt", access="write")
    except ContextError as error:
        assert "Access denied" in str(error)
    else:
        raise AssertionError("Expected an out-of-workspace path to be rejected.")


def test_context_rejects_symlinks_that_escape_allowed_roots(tmp_path):
    context = make_context(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    link = context.workspace_root / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        return

    try:
        context.resolve_path("outside-link/secret.txt", access="read")
    except ContextError:
        pass
    else:
        raise AssertionError("Expected a symlink path outside the workspace to be rejected.")


def test_editor_views_creates_and_replaces_complete_files(tmp_path):
    context = make_context(tmp_path)
    tool = EditTool()
    cancellation = CancellationToken()
    target = context.workspace_root / "new.txt"

    created = tool.invoke(
        context,
        {"command": "create", "path": "new.txt", "file_text": "first\n"},
        60,
        cancellation,
    )
    assert created.status == "completed"
    assert target.read_text(encoding="utf-8") == "first\n"

    edited = tool.invoke(
        context,
        {"command": "edit", "path": str(target), "file_text": "second\n"},
        60,
        cancellation,
    )
    assert edited.status == "completed"
    assert "first" in edited.artifacts["diff"]
    assert target.read_text(encoding="utf-8") == "second\n"

    viewed = tool.invoke(
        context,
        {"command": "view", "path": "new.txt"},
        60,
        cancellation,
    )
    assert viewed.status == "completed"
    assert "second" in viewed.stdout
    assert viewed.artifacts["files_read"] == [str(target)]


def test_editor_directory_view_does_not_follow_symlink_directories(tmp_path):
    context = make_context(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    link = context.workspace_root / "external"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        return

    result = EditTool().invoke(
        context,
        {"command": "view", "path": "."},
        60,
        CancellationToken(),
    )

    assert result.status == "completed"
    assert "external" not in result.stdout
    assert "secret.txt" not in result.stdout


def test_editor_create_does_not_replace_a_file_created_concurrently(tmp_path):
    context = make_context(tmp_path)
    target = context.workspace_root / "created.txt"
    target.write_text("winner", encoding="utf-8")

    result = EditTool().invoke(
        context,
        {"command": "create", "path": "created.txt", "file_text": "loser"},
        60,
        CancellationToken(),
    )

    assert result.status == "error"
    assert target.read_text(encoding="utf-8") == "winner"


def test_tool_discovery_adds_timeout_to_native_schema():
    tools = load_all_tools()
    assert set(tools) == {"bash", "editor"}
    assert issubclass(type(tools["bash"]), BaseTool)
    assert tools["editor"].schema_for_model()["properties"]["timeout"]["default"] == 60


def test_tool_schema_validation_rejects_invalid_json_schema():
    class InvalidSchemaTool(BaseTool):
        name = "invalid"
        guide = "Invalid schema test tool."
        input_schema = {
            "type": "object",
            "properties": {"value": {"type": "not-a-json-schema-type"}},
        }

        def execute(self, context, arguments, timeout_seconds, cancellation):
            raise AssertionError("This test tool must not execute.")

    try:
        InvalidSchemaTool().schema_for_model()
    except Exception as error:
        assert "not-a-json-schema-type" in str(error)
    else:
        raise AssertionError("Invalid JSON Schema must be rejected.")


def test_bash_uses_an_isolated_shell_and_filters_credentials(tmp_path):
    context = make_context(tmp_path)
    context.shell_environment.update(
        {"FAKE_API_KEY": "not-for-tools", "ALLOWED_VALUE": "visible"}
    )
    tool = BashTool()
    if os.name == "nt":
        command = 'Write-Output "$env:FAKE_API_KEY|$env:ALLOWED_VALUE"'
    else:
        command = 'printf "%s|%s" "$FAKE_API_KEY" "$ALLOWED_VALUE"'

    result = tool.invoke(
        context,
        {"command": command},
        10,
        CancellationToken(),
    )

    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.stdout.strip() == "|visible"


def test_bash_timeout_terminates_the_command(tmp_path):
    context = make_context(tmp_path)
    command = "Start-Sleep -Seconds 3" if os.name == "nt" else "sleep 3"

    result = BashTool().invoke(
        context,
        {"command": command},
        1,
        CancellationToken(),
    )

    assert result.status == "timeout"
    assert result.timed_out
    assert result.timeout_seconds == 1
