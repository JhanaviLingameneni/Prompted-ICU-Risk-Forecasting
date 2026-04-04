"""
Main entry point for the application.
Used for demoing the Risk Assessment Chatbot.
"""
from chatbot.app import build_app
from chatbot.config import FIELD_SPECS
from chatbot.model_input import APP_STATE, build_model_input_df


def done_output_callback(final_text: str, required_answers: dict[str, str], optional_answers: dict[str, str]) -> str:
    merged_answers: dict[str, str] = dict(required_answers)
    merged_answers.update(optional_answers)

    model_input_df = build_model_input_df(merged_answers)
    APP_STATE["latest_model_input_df"] = model_input_df

    missing_fields = []
    for field in FIELD_SPECS:
        value = merged_answers.get(field["name"])
        if value is None or value == "" or value == "skipped":
            missing_fields.append(field["name"])

    lines = [
        final_text,
        "",
        f"Constructed model input DataFrame with shape: {model_input_df.shape}",
        f"Total columns: {len(model_input_df.columns)}",
    ]

    if missing_fields:
        lines.append(f"Missing chatbot fields filled from hard-coded medians: {', '.join(sorted(missing_fields))}")
    else:
        lines.append("No missing chatbot fields; provided values were used for mapped features.")

    preview = model_input_df.iloc[0].to_string()
    lines.extend(["", "Model input row preview:", preview])
    return "\n".join(lines)


if __name__ == "__main__":
    build_app(done_output_callback=done_output_callback).launch()
