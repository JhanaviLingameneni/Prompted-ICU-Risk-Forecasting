"""
models.py - Train and evaluate Random Forest, Logistic Regression, Gradient Boosting

Usage:
    python models.py --input ../results/processed.csv --output_dir ../results
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")


def undersample(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    died = np.where(y == 1)[0]
    survived = np.where(y == 0)[0]
    survived_sampled = rng.choice(survived, size=len(died), replace=False)
    indices = np.concatenate([died, survived_sampled])
    rng.shuffle(indices)
    return X[indices], y[indices]


def find_best_threshold(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.2, 0.6, 0.01):
        f1 = f1_score(y_true, (y_prob >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def get_models():
    models = {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 8, 10],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [3, 5],
            }
        ),
        "LogisticRegression": (
            LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05],
                "max_depth": [3, 4],
                "subsample": [0.7, 0.8],
                "min_samples_leaf": [5, 10],
            }
        ),
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()
    results = {}

    for name, (estimator, param_grid) in models.items():
        print(f"\nTraining {name}...")

        grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        thresh = find_best_threshold(y_train, y_train_prob)
        y_pred = (y_prob >= thresh).astype(int)

        results[name] = {
            "model": best_model,
            "best_params": grid.best_params_,
            "threshold": thresh,
            "train_auc": roc_auc_score(y_train, y_train_prob),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_curve": roc_curve(y_test, y_prob),
            "y_prob": y_prob,
        }

        with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(best_model, f)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"{'Model':<22} {'Accuracy':>8} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<22} {m['test_accuracy']:>8.4f} {m['test_auc']:>8.4f} {m['test_f1']:>8.4f} {m['test_precision']:>10.4f} {m['test_recall']:>8.4f}")

    # Save results to CSV
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(m["test_accuracy"], 4),
            "AUC-ROC": round(m["test_auc"], 4),
            "F1-Score": round(m["test_f1"], 4),
            "Precision": round(m["test_precision"], 4),
            "Recall": round(m["test_recall"], 4),
        })
    results_df = pd.DataFrame(rows)
    results_path = os.path.join(output_dir, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    return results


# def get_feature_importance(results, feature_names):
#     importance_data = {}
#     for name in ["RandomForest", "GradientBoosting"]:
#         if name in results:
#             importance_data[name] = results[name]["model"].feature_importances_
#     if importance_data:
#         df = pd.DataFrame(importance_data, index=feature_names)
#         df["Average"] = df.mean(axis=1)
#         return df.sort_values("Average", ascending=False)
#     return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

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

    print(f"Before undersampling: {np.bincount(y_train.astype(int))}")
    X_train, y_train = undersample(X_train, y_train)
    print(f"After undersampling:  {np.bincount(y_train.astype(int))}")

    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols, args.output_dir)

    # importance = get_feature_importance(results, feature_cols)
    # if importance is not None:
    #     importance.to_csv(os.path.join(args.output_dir, "feature_importance.csv"))


if __name__ == "__main__":
    main()