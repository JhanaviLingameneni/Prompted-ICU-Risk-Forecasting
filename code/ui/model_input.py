"""
Model input row construction from UI answers.
"""

from __future__ import annotations

import pandas as pd

from config import AGGREGATE_SUFFIXES, FIELD_SPECS


APP_STATE: dict[str, pd.DataFrame | None] = {"latest_model_input_df": None}


def _build_default_model_row() -> dict[str, float]:
    """
    We are filling missing values with median values from the training set.
    Median values are hard-codeded into the FIELD_SPEC config.
    """
    row = {}
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
                if suffix in {"delta", "range", "std"}:
                    row[f"{feature_base}_{suffix}"] = 0.0 # No delta, range or std if missing
                elif suffix == "missing":
                    row[f"{feature_base}_{suffix}"] = 1 # Default is missing if not provided.
                else:
                    row[f"{feature_base}_{suffix}"] = median

    return row

DEFAULT_MODEL_ROW = _build_default_model_row()
MODEL_FEATURE_COLUMNS = sorted(DEFAULT_MODEL_ROW.keys())

def build_model_input_df(answers: dict[str, str]) -> pd.DataFrame:
    """
    Build one-row model input with full feature columns and median defaults.
    """
    row = dict(DEFAULT_MODEL_ROW)

    # Apply direct values
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

    # Apply aggregate feature columns
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
                if suffix == "missing":
                    row[column] = 0 # We have a reading, so not missing
                elif suffix in {"delta", "range", "std"}: # Delta, range, and std are 0 for a single reading
                    row[column] = 0.0
                else:
                    row[column] = vital_value

    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)


def _to_float(value: str | float | int | None) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
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
