"""
Main entry point for the application.
Used for demoing the Risk Assessment model.
"""

from pathlib import Path
from joblib import load
from tensorflow.keras.models import load_model

from ui.app import build_app
from ui.config import FIELD_SPECS, REQUIRED_SPECS
from ui.model_input import APP_STATE, build_model_input_df

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _load_model_and_scaler():
    model_path = MODELS_DIR / "lstm_model.keras"
    scaler_path = MODELS_DIR / "lstm_scaler.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    model = load_model(model_path)
    scaler = load(scaler_path)
    return model, scaler


def _done_output_callback(final_text: str, required_answers: dict[str, str], optional_answers: dict[str, str]) -> str:
    merged_answers: dict[str, str] = dict(required_answers)
    merged_answers.update(optional_answers)

    model_input_df = build_model_input_df(merged_answers)
    APP_STATE["latest_model_input_df"] = model_input_df

    model, scaler = _load_model_and_scaler()

    # Scale tabular input
    x = scaler.transform(model_input_df)

    # Reshape for LSTM: (samples, features, 1)
    x_lstm = x.reshape((x.shape[0], x.shape[1], 1))

    prediction = model.predict(x_lstm, verbose=0)
    risk_score = float(prediction.ravel()[0])

    if risk_score > 0.4:
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