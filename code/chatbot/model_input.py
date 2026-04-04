"""Model input row construction from chatbot answers."""

from __future__ import annotations

import pandas as pd

from chatbot.config import AGGREGATE_SUFFIXES, FIELD_SPECS


APP_STATE: dict[str, pd.DataFrame | None] = {"latest_model_input_df": None}


def _build_default_model_row() -> dict[str, float]:
    row: dict[str, float] = {}
    for field in FIELD_SPECS:
        median_raw = field.get("median")
        median = float(median_raw) if median_raw is not None else None
        if median is None:
            continue

        direct_column = field.get("direct_column")
        if direct_column is not None:
            row[direct_column] = median

        onehot_defaults = field.get("onehot_defaults")
        if onehot_defaults is not None:
            for column_name, default_value in onehot_defaults.items():
                row[str(column_name)] = float(default_value)

        feature_base = field.get("feature_base")
        if feature_base is not None:
            for suffix in AGGREGATE_SUFFIXES:
                row[f"{feature_base}_{suffix}"] = median

    return row


DEFAULT_MODEL_ROW = _build_default_model_row()
MODEL_FEATURE_COLUMNS: list[str] = sorted(DEFAULT_MODEL_ROW.keys())


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "skipped":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _gender_to_numeric(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "male":
        return 1
    if normalized == "female":
        return 0
    numeric = _to_float(value)
    return int(numeric) if numeric is not None else None


def build_model_input_df(answers: dict[str, str]) -> pd.DataFrame:
    """
    Build one-row model input with full feature columns and median defaults.
    """
    row = dict(DEFAULT_MODEL_ROW)

    # Apply direct biometrics/categorical fields from FIELD_SPECS metadata.
    for field in FIELD_SPECS:
        field_name = field["name"]
        direct_column = field.get("direct_column")
        if direct_column is None:
            continue
        if direct_column not in row:
            continue

        raw_value = answers.get(field_name)
        if field_name == "gender":
            parsed_value = _gender_to_numeric(raw_value)
        elif field_name == "icu_type":
            numeric_value = _to_float(raw_value)
            parsed_value = int(numeric_value) if numeric_value is not None else None
        else:
            parsed_value = _to_float(raw_value)

        if parsed_value is not None:
            row[direct_column] = parsed_value

            if field_name == "icu_type":
                onehot_defaults = field.get("onehot_defaults")
                if onehot_defaults is not None:
                    for onehot_col in onehot_defaults:
                        if onehot_col in row:
                            row[onehot_col] = 0.0

                    possible_cols = [
                        f"ICUType_{int(parsed_value)}.0",
                        f"ICUType_{int(parsed_value)}",
                    ]
                    for onehot_col in possible_cols:
                        if onehot_col in row:
                            row[onehot_col] = 1.0
                            break

    # Apply aggregate feature columns from FIELD_SPECS metadata.
    for field in FIELD_SPECS:
        field_name = field["name"]
        feature_base = field.get("feature_base")
        if feature_base is None:
            continue

        vital_value = _to_float(answers.get(field_name))
        if vital_value is None:
            continue

        for suffix in AGGREGATE_SUFFIXES:
            column = f"{feature_base}_{suffix}"
            if column in row:
                row[column] = vital_value

    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
