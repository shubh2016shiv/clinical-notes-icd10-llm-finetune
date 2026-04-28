"""
Rendering helpers for centralized prompt specs.
"""

from __future__ import annotations

from collections.abc import Iterable


def render_xml_block(block_name: str, content_lines: Iterable[str], *, indent: int = 2) -> str:
    padding = " " * indent
    rendered_lines = [f"<{block_name}>"]
    for content_line in content_lines:
        if content_line == "":
            rendered_lines.append("")
        else:
            rendered_lines.append(f"{padding}{content_line}")
    rendered_lines.append(f"</{block_name}>")
    return "\n".join(rendered_lines)


def render_list_block(items: Iterable[str], *, default_line: str) -> str:
    normalized_items = [item for item in items if item]
    if not normalized_items:
        return default_line
    return "\n".join(f"- {item}" for item in normalized_items)


def compose_chat_prompt(*, system_prompt: str, user_prompt: str) -> str:
    """
    Keep the client interface unchanged while preserving a stable system/user split.
    """

    return (
        "<system>\n"
        f"{system_prompt.strip()}\n"
        "</system>\n\n"
        "<user>\n"
        f"{user_prompt.strip()}\n"
        "</user>"
    )
