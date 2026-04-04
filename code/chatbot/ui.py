"""
UI-specific helpers that build Gradio update payloads.
"""

from typing import Any, Mapping

import gradio as gr

from chatbot.config import OPTIONAL_SPECS, REQUIRED_SPECS, FieldSpec
from chatbot.core import (
    current_field,
    field_by_name,
    field_header,
    normalize_optional_selection,
    optional_choices,
    required_complete,
    section_summary,
)


def input_updates_for_field(field: FieldSpec | None) -> tuple[Any, Any, Any]:
    """
    Return visibility/value updates for text, number, and choice inputs.
    """
    if field is None:
        return (
            gr.update(visible=False, value="", label="Text Input"),
            gr.update(visible=False, value=None, label="Numeric Input"),
            gr.update(visible=False, choices=[], value=None, label="Choice Input"),
        )

    label = field["question"]
    field_type = field["type"]
    return (
        gr.update(visible=(field_type == "text"), value="", label=label),
        gr.update(visible=(field_type in {"int", "float"}), value=None, label=label),
        gr.update(visible=(field_type == "choice"), choices=field.get("choices", []), value=None, label=label),
    )


def tab_update(tab_id: str) -> dict[str, Any]:
    """Return a tab selection update."""
    return gr.update(selected=tab_id)


def required_ui(required_answers: Mapping[str, str], required_index: int, status: str) -> tuple[Any, ...]:
    """Build all UI outputs for the required section."""
    field = current_field(REQUIRED_SPECS, required_index)
    text_update, number_update, choice_update = input_updates_for_field(field)

    return (
        field_header(REQUIRED_SPECS, required_index),
        status,
        section_summary(REQUIRED_SPECS, required_answers),
        text_update,
        number_update,
        choice_update,
        gr.update(interactive=(field is not None)),
    )


def optional_ui(
    required_answers: Mapping[str, str],
    optional_answers: Mapping[str, str],
    optional_selected_name: str | None,
    status: str,
) -> tuple[Any, ...]:
    """Build all UI outputs for the optional section."""
    if not required_complete(required_answers):
        return (
            "Complete all required fields first.",
            status,
            section_summary(OPTIONAL_SPECS, optional_answers),
            gr.update(visible=False, choices=[], value=None, label="Choose optional field"),
            gr.update(visible=False, value="", label="Text Input"),
            gr.update(visible=False, value=None, label="Numeric Input"),
            gr.update(visible=False, choices=[], value=None, label="Choice Input"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    selected_name = normalize_optional_selection(optional_answers, optional_selected_name)
    choices = optional_choices(optional_answers)
    field = field_by_name(OPTIONAL_SPECS, selected_name)
    text_update, number_update, choice_update = input_updates_for_field(field)
    can_submit = field is not None
    can_skip = field is not None

    if not choices:
        header = "All optional fields are already processed."
        selector_update = gr.update(visible=False, choices=[], value=None, label="Choose optional field")
    else:
        header = f"Remaining optional fields: {len(choices)}. Select one to enter."
        selector_update = gr.update(visible=True, choices=choices, value=selected_name, label="Choose optional field")

    return (
        header,
        status,
        section_summary(OPTIONAL_SPECS, optional_answers),
        selector_update,
        text_update,
        number_update,
        choice_update,
        gr.update(interactive=can_submit),
        gr.update(interactive=can_skip),
    )
