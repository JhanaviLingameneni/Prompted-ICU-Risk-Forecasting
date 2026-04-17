import os
import pandas as pd
import matplotlib.pyplot as plt

history_path = "../models/lstm_history.csv"
output_dir = "../../resources"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(history_path)

epochs = range(1, len(df) + 1)

# 1) Like ann_ap.png -> AUC plot
if "auc" in df.columns and "val_auc" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["auc"], label="Train AUC", linewidth=2)
    plt.plot(epochs, df["val_auc"], label="Validation AUC", linewidth=2)
    plt.title("LSTM AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_ap.png"), dpi=150)
    plt.close()

# 2) Like ann_evaluation.png -> Loss plot
if "loss" in df.columns and "val_loss" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, df["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("LSTM Evaluation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_evaluation.png"), dpi=150)
    plt.close()

# 3) Like ann_test_metrics.png -> final metrics bar chart
metric_pairs = [
    ("accuracy", "val_accuracy", "Accuracy"),
    ("auc", "val_auc", "AUC"),
    ("precision", "val_precision", "Precision"),
    ("recall", "val_recall", "Recall"),
]

labels = []
train_vals = []
val_vals = []

for train_col, val_col, label in metric_pairs:
    if train_col in df.columns and val_col in df.columns:
        labels.append(label)
        train_vals.append(df[train_col].iloc[-1])
        val_vals.append(df[val_col].iloc[-1])

if labels:
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], train_vals, width=width, label="Train")
    plt.bar([i + width/2 for i in x], val_vals, width=width, label="Validation")

    for i, v in enumerate(train_vals):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(val_vals):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.xticks(list(x), labels)
    plt.ylim(0, 1.1)
    plt.title("LSTM Test Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_test_metrics.png"), dpi=150)
    plt.close()

print("Saved LSTM plots in resources folder.")