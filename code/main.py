"""
Main entry point for the application.
Used for demoing the Risk Assessment Chatbot.
"""
from chatbot.app import build_app
from chatbot.config import FIELD_SPECS, REQUIRED_SPECS
from chatbot.model_input import APP_STATE, build_model_input_df
from joblib import load as load_scaler
from pathlib import Path
from tensorflow.keras.models import load_model

MODELS_DIR = Path(__file__).resolve().parent / "models"

def _load_model_and_scaler():
    model_path = MODELS_DIR / "ann_model.keras"
    scaler_path = MODELS_DIR / "ann_scaler.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    return load_model(model_path), load_scaler(scaler_path)

def _done_output_callback(final_text: str, required_answers: dict[str, str], optional_answers: dict[str, str]) -> str:
    merged_answers: dict[str, str] = dict(required_answers)
    merged_answers.update(optional_answers)

    model_input_df = build_model_input_df(merged_answers)
    APP_STATE["latest_model_input_df"] = model_input_df

    # missing_fields = []
    # for field in FIELD_SPECS:
    #     value = merged_answers.get(field["name"])
    #     if value is None or value == "" or value == "skipped":
    #         missing_fields.append(field["name"])
    # if len([field for field in missing_fields if field in REQUIRED_SPECS]) > 0:
    #     missing_list = ", ".join(missing_fields)
    #     return f"{final_text}\n\nWarning: Missing or invalid values for fields: {missing_list}. Please review your answers."

    model, scaler = _load_model_and_scaler()
    # Both training (data_loader) and inference (model_input) sort columns alphabetically,
    # so the order is guaranteed to match without any extra alignment step.
    x = scaler.transform(model_input_df)
    prediction = model.predict(x, verbose=0)
    risk_score = float(prediction.ravel()[0])

    if risk_score > 0.3:  # Threshold low. Prefer false positives over false negatives.
        return (
            "<div style='text-align:center; margin-top: 12px;'>"
            "<div style='font-size: 64px; font-weight: 900; color: #d60000; letter-spacing: 2px;'>RISK</div>"
            f"<div style='font-size: 16px; color: #b30000; font-weight: 600;'>Risk score: {risk_score:.3f}</div>"
            f"<div style='margin-top:8px; font-size: 13px; color: #444;'>{final_text}</div>"
            "</div>"
        )

    return (
        "<div style='text-align:center; margin-top: 12px;'>"
        "<div style='font-size: 64px; font-weight: 900; color: #0b8f2a; letter-spacing: 2px;'>NO RISK</div>"
        f"<div style='font-size: 16px; color: #0a6e21; font-weight: 600;'>Risk score: {risk_score:.3f}</div>"
        f"<div style='margin-top:8px; font-size: 13px; color: #444;'>{final_text}</div>"
        "</div>"
    )


if __name__ == "__main__":
    build_app(done_output_callback=_done_output_callback).launch()
