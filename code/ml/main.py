"""
Entrypoint
"""



if __name__ == "__main__":
    from joblib import dump
    from train import ann, scaler
    # ANN is a Keras model, so we save it in a Keras-compatible format.
    ann().model.save("../models/ann_model.keras")
    # Save the scaler using joblib
    dump(scaler, "../models/ann_scaler.bin")
