"""
Entrypoint
"""
from pathlib import Path
from joblib import dump
import pandas as pd

from train import ann, lstm, scaler, df_x_train

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    with (MODELS_DIR / "columns_spec.txt").open("w", encoding="utf-8") as f:
        f.write("Model input columns:\n")
        f.write("\n".join(df_x_train.columns))
        f.write("\n\nFirst 5 rows:\n")
        f.write(df_x_train.head().to_string())

    # Save scaler
    dump(scaler, MODELS_DIR / "lstm_scaler.bin")

    # Train and save ANN if you still want it
    ann_model = ann().model_
    ann_model.save(MODELS_DIR / "ann_model.keras")

    # Train and save LSTM
    lstm_model, lstm_history = lstm()
    lstm_model.save(MODELS_DIR / "lstm_model.keras")

    # Save LSTM history for plotting later
    history_df = pd.DataFrame(lstm_history.history)
    history_df.to_csv(MODELS_DIR / "lstm_history.csv", index=False)

    print("Saved:")
    print(f" - {MODELS_DIR / 'ann_model.keras'}")
    print(f" - {MODELS_DIR / 'lstm_model.keras'}")
    print(f" - {MODELS_DIR / 'lstm_scaler.bin'}")
    print(f" - {MODELS_DIR / 'lstm_history.csv'}")