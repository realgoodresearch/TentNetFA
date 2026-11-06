"""A small, config-synchronised drag-and-drop priority-list component."""

from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "tentnetfa_priority_sorter",
    path=str(Path(__file__).parent / "components" / "priority_sorter"),
)


def priority_sorter(items: list[str], *, key: str) -> list[str]:
    """Render draggable rows and return their displayed order."""
    result = _COMPONENT(items=items, default=items, key=key)
    if not isinstance(result, list):
        return items
    ordered = [str(item) for item in result]
    return ordered if set(ordered) == set(items) and len(ordered) == len(items) else items
