"""
Event handlers for required/optional intake interactions.
"""

from typing import Any

import gradio as gr

from config import (
    OPTIONAL_SPECS,
    OPTIONAL_TAB_ID,
    PROCESSING_REPLY,
    REQUIRED_SPECS,
    REQUIRED_TAB_ID,
)
from core import (
    all_summary,
    current_field,
    field_by_name,
    missing_required_names,
    normalize_optional_selection,
    required_complete,
    validate_input,
)
from ui import optional_ui, required_ui, tab_update

def initialize():
    """
    Initialize all states and both UI sections.
    """
    required_answers = {}
    optional_answers = {}
    required_index = 0
    optional_index = 0

    req = required_ui(required_answers, required_index, "Required intake started.")
    optional_selected_name = normalize_optional_selection(optional_answers, None)
    opt = optional_ui(required_answers, optional_answers, optional_selected_name, "Optional fields unlock after required completion.")

    return (
        required_answers,
        required_index,
        optional_answers,
        optional_index,
        optional_selected_name,
        *req,
        *opt,
        gr.update(interactive=False),
        tab_update(REQUIRED_TAB_ID),
        "",
    )


def submit_required(
    required_answers: dict[str, str],
    required_index: int,
    optional_answers: dict[str, str],
    optional_index: int,
    optional_selected_name: str | None,
    text_value: str | None,
    number_value: float | int | None,
    choice_value: str | None,
) -> tuple[Any, ...]:
    """
    Validate and store one required-field input, then refresh all outputs.
    """
    field = current_field(REQUIRED_SPECS, required_index)
    if field is None:
        req = required_ui(required_answers, required_index, "All required fields are already captured.")
        next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
        opt = optional_ui(required_answers, optional_answers, next_optional_selected, "Optional section is available.")
        done_update = gr.update(interactive=required_complete(required_answers))
        selected_tab_update = tab_update(OPTIONAL_TAB_ID if required_complete(required_answers) else REQUIRED_TAB_ID)
        return required_answers, required_index, optional_answers, optional_index, next_optional_selected, *req, *opt, done_update, selected_tab_update, ""

    ok, value, error = validate_input(field, text_value, number_value, choice_value)
    if not ok:
        req = required_ui(required_answers, required_index, f"Invalid input for {field['name']}. {error}")
        next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
        opt = optional_ui(required_answers, optional_answers, next_optional_selected, "Optional fields are available for completion after all required fields are captured.")
        done_update = gr.update(interactive=required_complete(required_answers))
        return required_answers, required_index, optional_answers, optional_index, next_optional_selected, *req, *opt, done_update, tab_update(REQUIRED_TAB_ID), ""

    updated_required = dict(required_answers)
    updated_required[field["name"]] = value
    next_required_index = required_index + 1

    req = required_ui(updated_required, next_required_index, f"Captured {field['name']}.")
    opt_status = "Please fill optional fields." if required_complete(updated_required) else "Optional fields are available for completion after all required fields are captured."
    next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
    opt = optional_ui(updated_required, optional_answers, next_optional_selected, opt_status)
    done_update = gr.update(interactive=required_complete(updated_required))
    selected_tab = OPTIONAL_TAB_ID if required_complete(updated_required) else REQUIRED_TAB_ID

    return (
        updated_required,
        next_required_index,
        optional_answers,
        optional_index,
        next_optional_selected,
        *req,
        *opt,
        done_update,
        tab_update(selected_tab),
        "",
    )


def restart_required(
    optional_answers: dict[str, str],
    optional_index: int,
    optional_selected_name: str | None,
) -> tuple[Any, ...]:
    """
    Reset required.
    """
    required_answers = {}
    required_index = 0

    req = required_ui(required_answers, required_index, "Required section restarted.")
    next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
    opt = optional_ui(required_answers, optional_answers, next_optional_selected, "Optional fields are available for completion after all required fields are captured.")

    return (
        required_answers,
        required_index,
        optional_answers,
        optional_index,
        next_optional_selected,
        *req,
        *opt,
        gr.update(interactive=False),
        tab_update(REQUIRED_TAB_ID),
        "",
    )


def submit_optional(
    required_answers: dict[str, str],
    required_index: int,
    optional_answers: dict[str, str],
    optional_index: int,
    optional_selected_name: str | None,
    text_value: str | None,
    number_value: float | int | None,
    choice_value: str | None,
) -> tuple[Any, ...]:
    """
    Validate and store one optional-field input, then refresh all outputs.
    """
    if not required_complete(required_answers):
        req = required_ui(required_answers, required_index, "Complete required fields first.")
        next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
        opt = optional_ui(required_answers, optional_answers, next_optional_selected, "Please complete required section before submitting optional fields.")
        return required_answers, required_index, optional_answers, optional_index, next_optional_selected, *req, *opt, gr.update(interactive=False), tab_update(REQUIRED_TAB_ID), ""

    selected_name = normalize_optional_selection(optional_answers, optional_selected_name)
    field = field_by_name(OPTIONAL_SPECS, selected_name)
    if field is None:
        req = required_ui(required_answers, required_index, "Required section complete.")
        opt = optional_ui(required_answers, optional_answers, selected_name, "All optional fields already processed.")
        return required_answers, required_index, optional_answers, optional_index, selected_name, *req, *opt, gr.update(interactive=True), tab_update(OPTIONAL_TAB_ID), ""

    ok, value, error = validate_input(field, text_value, number_value, choice_value)
    if not ok:
        req = required_ui(required_answers, required_index, "Required section complete.")
        opt = optional_ui(required_answers, optional_answers, selected_name, f"Invalid input for {field['name']}. {error}")
        return required_answers, required_index, optional_answers, optional_index, selected_name, *req, *opt, gr.update(interactive=True), tab_update(OPTIONAL_TAB_ID), ""

    updated_optional = dict(optional_answers)
    updated_optional[field["name"]] = value
    next_optional_index = len(updated_optional)
    next_optional_selected = normalize_optional_selection(updated_optional, selected_name)

    req = required_ui(required_answers, required_index, "Required section complete.")
    done_msg = "All optional fields captured." if next_optional_selected is None else f"Captured {field['name']}."
    opt = optional_ui(required_answers, updated_optional, next_optional_selected, done_msg)

    return required_answers, required_index, updated_optional, next_optional_index, next_optional_selected, *req, *opt, gr.update(interactive=True), tab_update(OPTIONAL_TAB_ID), ""


def restart_optional(required_answers: dict[str, str], required_index: int) -> tuple[Any, ...]:
    """
    Reset optional.
    """
    optional_answers = {}
    optional_index = 0
    optional_selected_name = normalize_optional_selection(optional_answers, None)

    req = required_ui(required_answers, required_index, "Required section unchanged.")
    opt = optional_ui(required_answers, optional_answers, optional_selected_name, "Optional section restarted.")

    selected_tab = OPTIONAL_TAB_ID if required_complete(required_answers) else REQUIRED_TAB_ID
    return required_answers, required_index, optional_answers, optional_index, optional_selected_name, *req, *opt, gr.update(interactive=required_complete(required_answers)), tab_update(selected_tab), ""


def change_optional_field(
    required_answers: dict[str, str],
    required_index: int,
    optional_answers: dict[str, str],
    optional_index: int,
    optional_selected_name: str | None,
) -> tuple[Any, ...]:
    """
    Handle optional-field dropdown selection changes.
    """
    req_status = "Required section complete." if required_complete(required_answers) else "Complete required fields first."
    req = required_ui(required_answers, required_index, req_status)
    opt = optional_ui(required_answers, optional_answers, optional_selected_name, "Optional field selected.")
    done_update = gr.update(interactive=required_complete(required_answers))
    selected_tab = OPTIONAL_TAB_ID if required_complete(required_answers) else REQUIRED_TAB_ID
    next_optional_selected = normalize_optional_selection(optional_answers, optional_selected_name)
    return required_answers, required_index, optional_answers, optional_index, next_optional_selected, *req, *opt, done_update, tab_update(selected_tab), ""


def done_intake(required_answers: dict[str, str], optional_answers: dict[str, str]) -> str:
    """
    For debugging, print captured input in bottom of screen.
    """
    # if not required_complete(required_answers):
    #     missing = ", ".join(missing_required_names(required_answers))
    #     return f"Cannot process yet. Missing required fields: {missing}."

    # summary = all_summary(required_answers, optional_answers)
    # return f"{PROCESSING_REPLY}\n\nCaptured inputs:\n{summary}"

    return ""
