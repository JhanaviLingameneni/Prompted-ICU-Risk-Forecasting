"""
Module contains the training code for the ML models.
"""
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from data_loader import DataLoader


df_x_train, df_y_train = DataLoader(data_set="a").process_dataset()
df_x_test, df_y_test = DataLoader(data_set="b").process_dataset()
df_x_val, df_y_val = DataLoader(data_set="c").process_dataset()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df_x_train)
x_test_scaled = scaler.transform(df_x_test)
x_val_scaled = scaler.transform(df_x_val)
y_train = df_y_train.values.ravel()
y_test = df_y_test.values.ravel()
y_val = df_y_val.values.ravel()

def logistic_regression() -> LogisticRegression:
    """
    Trains a logistic regression model with hyperparameter tuning on validation set.
    """
    model = LogisticRegression(random_state=42)

    # Use validation set to tune hyperparamters
    param_grid = [
        {
            "C": [*np.logspace(-4, 4, 20), 1, 10, 100],
            "solver": ["lbfgs", "newton-cholesky", "liblinear", "sag", "saga"],
            "max_iter": [1500, 2000, 3000],
        },
        {
            "C": [*np.logspace(-4, 4, 20), 1, 10, 100],
            "solver": ["saga"],
            "l1_ratio": np.arange(0, 1.1, 0.1),
            "max_iter": [1500, 2000, 3000],
        },
    ]
    best_model = hyperparameter_tuning(
        model,
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        param_grid,
    )

    return best_model

def random_forest() -> RandomForestClassifier:
    """
    Trains a random forest model.
    """
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "min_samples_leaf": [1, 2],
        "n_estimators": range(50, 201, 50),
        "max_features": ["sqrt", "log2"],
        "min_samples_split": range(2, 11, 2),
        "max_depth": [None, 10, 20]
    }

    # Use validation set to tune hyperparamters
    best_model = hyperparameter_tuning(
        model,
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        param_grid,
    )

    return best_model


### HELPERS ###
def compare_models(models: Mapping[str, BaseEstimator]) -> None:
    """
    Compares multiple models by evaluating them on the test set and printing their classification reports.
    Uses global variables.

    Arguments:
        models: A mapping of model names to their corresponding trained estimators.
    """

    reports = []
    for model_name, model in models.items():
        y_pred = model.predict(x_test_scaled)
        prob = model.predict_proba(x_test_scaled)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, labels=[0, 1], target_names=["No Risk", "Risk"])
        reports.append((model_name, report, prob))

    # Little format helper
    def fmt(val):
        return f"{val:.2f}" if isinstance(val, float) else str(int(val))

    class_table = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "Support"],
            **{
                f"{model_name} (No Risk)": [
                    report["No Risk"]["precision"],
                    report["No Risk"]["recall"],
                    report["No Risk"]["f1-score"],
                    report["No Risk"]["support"],
                ]
                for model_name, report, _ in reports
            },
            **{
                f"{model_name} (Risk)": [
                    report["Risk"]["precision"],
                    report["Risk"]["recall"],
                    report["Risk"]["f1-score"],
                    report["Risk"]["support"],
                ]
                for model_name, report, _ in reports
            },
        }
    )
    for col in class_table.columns:
        if col != "Metric":
            class_table[col] = class_table[col].map(fmt)

    overall_table = pd.DataFrame(
        {
            "Metric": ["Overall Accuracy", "ROC-AUC"],
            **{model_name: [report["accuracy"], roc_auc_score(y_test, prob)] for model_name, report, prob in reports},
        }
    )
    for model_name, _, _ in reports:
        overall_table[model_name] = overall_table[model_name].map(fmt)

    avg_table = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "Support"],
            **{
                f"{model_name} (Macro Avg)": [
                    report["macro avg"]["precision"],
                    report["macro avg"]["recall"],
                    report["macro avg"]["f1-score"],
                    report["macro avg"]["support"],
                ]
                for model_name, report, _ in reports
            },
            **{
                f"{model_name} (Weighted Avg)": [
                    report["weighted avg"]["precision"],
                    report["weighted avg"]["recall"],
                    report["weighted avg"]["f1-score"],
                    report["weighted avg"]["support"],
                ]
                for model_name, report, _ in reports
            },
        }
    )
    for col in avg_table.columns:
        if col != "Metric":
            avg_table[col] = avg_table[col].map(fmt)

    print("Model Comparison:\n")
    print(class_table.to_markdown(index=False))
    print()
    print(overall_table.to_markdown(index=False))
    print()
    print(avg_table.to_markdown(index=False))

def _print_classification_report(model_name: str, y: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> None:
    """
    Pretty prints a detailed classification report.

    Arguments:
        model_name: The name of the model being evaluated.
        y: The true labels for the evaluation set.
        y_pred: The predicted labels for the test set.
        prob: The predicted probabilities for the positive class, used for ROC-AUC calculation.
    """
    report = classification_report(y, y_pred, output_dict=True, labels=[0, 1], target_names=["No Risk", "Risk"])

    # Little format helper
    def fmt(val):
        return f"{val:.2f}" if isinstance(val, float) else str(int(val))

    class_table = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "Support"],
            "No Risk": [
                report["No Risk"]["precision"],
                report["No Risk"]["recall"],
                report["No Risk"]["f1-score"],
                report["No Risk"]["support"],
            ],
            "Risk": [
                report["Risk"]["precision"],
                report["Risk"]["recall"],
                report["Risk"]["f1-score"],
                report["Risk"]["support"],
            ],
        }
    )
    class_table["No Risk"] = class_table["No Risk"].map(fmt)
    class_table["Risk"] = class_table["Risk"].map(fmt)

    overall_table = pd.DataFrame(
        {
            "Metric": ["Overall Accuracy", "ROC-AUC"],
            "Score": [report["accuracy"], roc_auc_score(y, prob)],
        }
    )
    overall_table["Score"] = overall_table["Score"].map(fmt)

    avg_table = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "Support"],
            "Macro Avg": [
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
                report["macro avg"]["support"],
            ],
            "Weighted Avg": [
                report["weighted avg"]["precision"],
                report["weighted avg"]["recall"],
                report["weighted avg"]["f1-score"],
                report["weighted avg"]["support"],
            ],
        }
    )
    avg_table["Macro Avg"] = avg_table["Macro Avg"].map(fmt)
    avg_table["Weighted Avg"] = avg_table["Weighted Avg"].map(fmt)


    print(f"Metrics for {model_name}:\n")
    print(class_table.to_markdown(index=False))
    print()
    print(overall_table.to_markdown(index=False))
    print()
    print(avg_table.to_markdown(index=False))

def hyperparameter_tuning(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train_: np.ndarray,
    x_val: np.ndarray,
    y_val_: np.ndarray,
    param_grid: Mapping | Sequence[dict],
) -> BaseEstimator:
    """
    Performs hyperparameter tuning using GridSearchCV with a fixed
    train/validation split.

    Arguments:
        model: The machine learning model to tune.
        x_train: The training data features.
        y_train_: The training data labels.
        x_val: The validation data features.
        y_val_: The validation data labels.
        param_grid: The grid of hyperparameters to search over.

    Returns:
        The best estimator found by GridSearchCV.
    """
    x_all = np.vstack([x_train, x_val])
    y_all = np.concatenate([y_train_, y_val_])

    # -1 means always train,
    # 0 means use the validation fold.
    test_fold = np.concatenate([
        -1 * np.ones(len(y_train_), dtype=int),
        np.zeros(len(y_val_), dtype=int),
    ])
    split = PredefinedSplit(test_fold=test_fold)

    scoring = {
        'recall_pos': make_scorer(recall_score, pos_label=1),
        'precision_pos': make_scorer(precision_score, pos_label=1)
    }

    grid_search = GridSearchCV(
        model, param_grid, cv=split, n_jobs=-1, scoring=scoring, refit='recall_pos')
    grid_search.fit(x_all, y_all)

    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_
