from __future__ import annotations

import json
from pathlib import Path


COMMON_PROMPTS = (
    "01_efficiency_guide.md",
    "02_exploration_guide.md",
    "03_code_generation_guide.md",
    "04_validation_guide.md",
)
OPTIMIZATION_PROMPTS = {
    "o_1": "01_efficiency_optimization.md",
    "o_2": "02_exploration_optimization.md",
    "o_3": "03_code_generation_optimization.md",
    "o_4": "04_validation_optimization.md",
    "assemble": "05_assemble_guide.md",
}
MODES = {"common", *OPTIMIZATION_PROMPTS}


class PromptError(RuntimeError):
    pass


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise PromptError(f"Required prompt file is missing: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise PromptError(f"Required prompt file is empty: {path}")
    return content


def build_system_prompt(agent_root: Path, mode: str, tools, context) -> str:
    if mode not in MODES:
        raise PromptError(f"Unknown Agent mode: {mode}")

    if mode == "common":
        prompt_directory = agent_root / "prompts" / "common"
        sections = [
            "You are a coding agent. Finish the user's request using the instructions below.",
            *[_read_prompt(prompt_directory / name) for name in COMMON_PROMPTS],
        ]
    else:
        prompt_path = agent_root / "prompts" / "optimization" / OPTIMIZATION_PROMPTS[mode]
        sections = [_read_prompt(prompt_path)]

    context_fields = context.prompt_fields()
    sections.append(
        "Request context. Treat these values as read-only. Use the editor for file access, "
        "respect the listed read and write roots, and use the supplied working directory for commands. "
        "The external framework enforces operating-system isolation.\n"
        + json.dumps(context_fields, ensure_ascii=False, indent=2)
    )
    sections.append(
        "There are the available tools you can use:\n"
        + "\n".join(f"- {tool.name}: {tool.guide}" for tool in tools.values())
    )
    return "\n\n".join(section for section in sections if section)


def build_user_message(task: str, context) -> str:
    sections = [task.strip()]
    if context.prompt_context:
        sections.append(
            "Additional request context:\n"
            + json.dumps(context.prompt_context, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(sections)
