"""
Entrypoint
"""
from pathlib import Path


if __name__ == "__main__":
    from joblib import dump
    from train import ann, scaler, df_x_train, gradient_boosting

    MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

    with (MODELS_DIR / "columns_spec.txt").open("w", encoding="utf-8") as f:
        f.write("Model input columns:\n")
        f.write("\n".join(df_x_train.columns))
        f.write("\n First 5 columns:\n")
        f.write(df_x_train.head().to_string())

    gradient_boosting()
    ann()
    # # ANN is a Keras model, so we save it in a Keras-compatible format.
    # ann().model_.save(MODELS_DIR / "ann_model.keras")
    # # Save the scaler using joblib
    # dump(scaler, MODELS_DIR / "ann_scaler.bin")
