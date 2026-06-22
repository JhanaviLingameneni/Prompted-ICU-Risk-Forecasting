"""
evaluation.py - Generate all visualization plots for model evaluation

Usage:
    python evaluation.py --input ../results/processed.csv --output_dir ../results

Run this AFTER models.py has finished training.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


def undersample(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    died = np.where(y == 1)[0]
    survived = np.where(y == 0)[0]
    survived_sampled = rng.choice(survived, size=len(died), replace=False)
    indices = np.concatenate([died, survived_sampled])
    rng.shuffle(indices)
    return X[indices], y[indices]


def plot_class_distribution(df, target_col, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    bars = ax.bar(["Survived (0)", "Died (1)"], counts.values,
                  color=["#2ecc71", "#e74c3c"], edgecolor="black")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{count}\n({count / len(df) * 100:.1f}%)", ha="center", fontsize=11)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(models, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"RandomForest": "#e74c3c", "LogisticRegression": "#3498db", "GradientBoosting": "#2ecc71"}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors.get(name, "gray"), lw=2,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall_curves(models, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"RandomForest": "#e74c3c", "LogisticRegression": "#3498db", "GradientBoosting": "#2ecc71"}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=colors.get(name, "gray"), lw=2,
                label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(models, X_test, y_test, save_path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Survived", "Died"],
                    yticklabels=["Survived", "Died"])
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_path, save_path):
    df = pd.read_csv(results_path)
    metrics = ["Accuracy", "AUC-ROC", "F1-Score", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.15

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, df[metric], width, label=metric, edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["Model"])
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.1])
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_train_vs_test_auc(models, X_train, y_train, X_test, y_test, save_path):
    names = []
    train_aucs = []
    test_aucs = []

    for name, model in models.items():
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        names.append(name)
        train_aucs.append(roc_auc_score(y_train, train_prob))
        test_aucs.append(roc_auc_score(y_test, test_prob))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, train_aucs, width, label="Train AUC", color="#3498db", edgecolor="black")
    bars2 = ax.bar(x + width / 2, test_aucs, width, label="Test AUC", color="#e74c3c", edgecolor="black")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim([0, 1.15])
    ax.set_title("Overfitting Check: Train vs Test AUC", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(models, feature_names, save_path, top_n=15):
    importance_data = {}
    for name in ["RandomForest", "GradientBoosting"]:
        if name in models:
            importance_data[name] = models[name].feature_importances_

    if not importance_data:
        return

    df = pd.DataFrame(importance_data, index=feature_names)
    df["Average"] = df.mean(axis=1)
    top = df["Average"].sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#e67e22", edgecolor="black")
    ax.set_xlabel("Average Importance")
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_threshold_vs_metrics(models, X_test, y_test, save_path):
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.1, 0.7, 0.01)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))

        axes[i].plot(thresholds, precisions, label="Precision", color="#3498db", lw=2)
        axes[i].plot(thresholds, recalls, label="Recall", color="#e74c3c", lw=2)
        axes[i].plot(thresholds, f1s, label="F1", color="#2ecc71", lw=2)
        axes[i].set_xlabel("Threshold")
        axes[i].set_ylabel("Score")
        axes[i].set_title(name, fontweight="bold")
        axes[i].legend()

    plt.suptitle("Threshold vs Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    target_col = "In-hospital_death"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = undersample(X_train, y_train)

    # Load trained models
    models = {}
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".pkl") and fname != "scaler.pkl":
            name = fname.replace(".pkl", "")
            with open(os.path.join(args.output_dir, fname), "rb") as f:
                models[name] = pickle.load(f)

    print(f"Loaded models: {list(models.keys())}")
    print(f"\nGenerating plots...")

    # Generate all plots
    plot_class_distribution(df, target_col, os.path.join(args.output_dir, "class_distribution.png"))
    plot_roc_curves(models, X_test, y_test, os.path.join(args.output_dir, "roc_curves.png"))
    plot_precision_recall_curves(models, X_test, y_test, os.path.join(args.output_dir, "pr_curves.png"))
    plot_confusion_matrices(models, X_test, y_test, os.path.join(args.output_dir, "confusion_matrices.png"))
    plot_model_comparison(os.path.join(args.output_dir, "model_results.csv"), os.path.join(args.output_dir, "model_comparison.png"))
    plot_train_vs_test_auc(models, X_train, y_train, X_test, y_test, os.path.join(args.output_dir, "overfit_check.png"))
    plot_feature_importance(models, feature_cols, os.path.join(args.output_dir, "feature_importance.png"))
    plot_threshold_vs_metrics(models, X_test, y_test, os.path.join(args.output_dir, "threshold_vs_metrics.png"))

    print("\nAll plots saved!")


if __name__ == "__main__":
    main()