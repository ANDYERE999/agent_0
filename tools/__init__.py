from tools.base import BaseTool, ToolResult


class ToolLoadError(RuntimeError):
    pass


def load_all_tools():
    from importlib import import_module
    from pathlib import Path
    import re

    tools_root = Path(__file__).parent
    loaded = {}
    for tool_directory in sorted(path for path in tools_root.iterdir() if path.is_dir()):
        module_path = tool_directory / "tool.py"
        if not module_path.is_file():
            continue
        try:
            module = import_module(f"tools.{tool_directory.name}.tool")
        except Exception as error:
            raise ToolLoadError(f"Cannot import tool module {module_path}: {error}") from error
        candidates = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseTool)
            and value is not BaseTool
            and value.__module__ == module.__name__
        ]
        if len(candidates) != 1:
            raise ToolLoadError(
                f"{module_path} must define exactly one BaseTool subclass; found {len(candidates)}"
            )
        try:
            tool = candidates[0]()
            if not isinstance(tool.name, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", tool.name
            ):
                raise ValueError("tool name must contain 1 to 64 letters, digits, underscores, or hyphens")
            if not isinstance(tool.guide, str) or not tool.guide.strip():
                raise ValueError("tool guide must be a non-empty string")
            tool.schema_for_model()
        except Exception as error:
            raise ToolLoadError(f"Invalid tool definition in {module_path}: {error}") from error
        if tool.name in loaded:
            raise ToolLoadError(f"Duplicate tool name {tool.name!r} in {module_path}")
        loaded[tool.name] = tool
    return loaded


__all__ = ["BaseTool", "ToolLoadError", "ToolResult", "load_all_tools"]
