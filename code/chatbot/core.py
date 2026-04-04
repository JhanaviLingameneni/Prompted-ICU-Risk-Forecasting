"""
Pure business logic helpers for deterministic intake flow.
"""

from typing import Mapping, Sequence

from chatbot.config import FIELD_SPECS, OPTIONAL_SPECS, REQUIRED_SPECS, FieldSpec


def current_field(specs: Sequence[FieldSpec], index: int) -> FieldSpec | None:
    """
    Return the field at index, or None when index is out of range.
    """
    if index < 0 or index >= len(specs):
        return None
    return specs[index]


def field_by_name(specs: Sequence[FieldSpec], field_name: str | None) -> FieldSpec | None:
    """
    Find a field spec by its internal name.
    """
    for field in specs:
        if field["name"] == field_name:
            return field
    return None


def required_complete(answers: Mapping[str, str]) -> bool:
    """
    Check whether all required fields are present in answers.
    """
    for field in REQUIRED_SPECS:
        if field["name"] not in answers:
            return False
    return True


def missing_required_names(answers: Mapping[str, str]) -> list[str]:
    """
    Return missing required field names.
    """
    return [field["name"] for field in REQUIRED_SPECS if field["name"] not in answers]


def section_summary(specs: Sequence[FieldSpec], answers: Mapping[str, str]) -> str:
    """
    Build a printable section summary for UI display.
    """
    lines = []
    for field in specs:
        lines.append(f"<li><b>{field['name']}:</b> {answers.get(field['name'], 'missing')}</li>")
    return f"<ul>{''.join(lines)}</ul>"


def all_summary(required_answers: Mapping[str, str], optional_answers: Mapping[str, str]) -> str:
    """
    Build a printable summary for all known fields.
    """
    merged = dict(required_answers)
    merged.update(optional_answers)
    lines = []
    for field in FIELD_SPECS:
        lines.append(f"- {field['name']}: {merged.get(field['name'], 'missing')}")
    return "\n".join(lines)


def field_header(specs: Sequence[FieldSpec], index: int) -> str:
    """
    Build a human-readable header for current field position.
    """
    field = current_field(specs, index)
    if field is None:
        return "All fields in this section are captured."
    tag = "required" if field["required"] else "optional"
    return f"Field {index + 1}/{len(specs)}: {field['question']} ({tag})"


def numeric_check(value: float | int | None, field: FieldSpec, is_int: bool) -> tuple[bool, str, str]:
    """
    Validate numeric input with optional integer and min/max constraints.
    """
    if value is None:
        return False, "", "Please enter a numeric value."

    if is_int and float(value) != int(value):
        return False, "", "Please enter an integer."

    numeric = int(value) if is_int else float(value)

    if "min" in field and numeric < field["min"]:
        return False, "", f"Value must be >= {field['min']}."
    if "max" in field and numeric > field["max"]:
        return False, "", f"Value must be <= {field['max']}."

    return True, str(numeric), ""


def validate_input(
    field: FieldSpec,
    text_value: str | None,
    number_value: float | int | None,
    choice_value: str | None,
) -> tuple[bool, str, str]:
    """
    Validate and normalize user input for a field spec.
    """
    field_type = field["type"]

    if field_type == "text":
        text = (text_value or "").strip()
        if not text:
            return False, "", "Please provide text."
        return True, text, ""

    if field_type == "choice":
        if choice_value is None or choice_value == "":
            return False, "", "Please choose a value."

        raw_choices = field["choices"]
        lowered = str(choice_value).strip().lower()

        # Support Gradio dropdown choices as (label, value) tuples.
        if raw_choices and isinstance(raw_choices[0], (tuple, list)) and len(raw_choices[0]) == 2:
            value_by_key: dict[str, str] = {}
            allowed_parts: list[str] = []
            for label, value in raw_choices:
                label_str = str(label).strip()
                value_str = str(value).strip()
                value_by_key[label_str.lower()] = value_str
                value_by_key[value_str.lower()] = value_str
                allowed_parts.append(f"{label_str} ({value_str})")

            if lowered not in value_by_key:
                return False, "", f"Allowed values: {', '.join(allowed_parts)}."
            return True, value_by_key[lowered], ""

        allowed = [str(choice).strip().lower() for choice in raw_choices]
        if lowered not in allowed:
            return False, "", f"Allowed values: {', '.join(str(choice) for choice in raw_choices)}."
        return True, lowered, ""

    if field_type == "int":
        return numeric_check(number_value, field, is_int=True)

    return numeric_check(number_value, field, is_int=False)


def optional_choices(optional_answers: Mapping[str, str]) -> list[tuple[str, str]]:
    """
    Return remaining optional fields as dropdown label/value pairs.
    """
    remaining = [field for field in OPTIONAL_SPECS if field["name"] not in optional_answers]
    return [(f"{field['question']} [{field['name']}]", field["name"]) for field in remaining]


def normalize_optional_selection(optional_answers: Mapping[str, str], optional_selected_name: str | None) -> str | None:
    """
    Ensure selected optional field is valid; otherwise pick first remaining.
    """
    choices = optional_choices(optional_answers)
    if not choices:
        return None
    valid_values = [value for _, value in choices]
    if optional_selected_name in valid_values:
        return optional_selected_name
    return valid_values[0]
