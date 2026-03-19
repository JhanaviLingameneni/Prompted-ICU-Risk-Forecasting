"""
Module contains the training code for the ML models.
"""
from typing import Mapping, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from data_loader import DataLoader


df_x_train, df_y_train = DataLoader(data_set="a").process_dataset()
df_x_test, df_y_test = DataLoader(data_set="b").process_dataset()
df_x_val, df_y_val = DataLoader(data_set="c").process_dataset()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df_x_train)
x_test_scaled = scaler.transform(df_x_test)
x_val_scaled = scaler.transform(df_x_val)


def hyperparameter_tuning(model: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray, param_grid: Mapping | Sequence[dict]) -> BaseEstimator:
    """
    Performs hyperparameter tuning using GridSearchCV.

    Arguments:
        model: The machine learning model to tune.
        x_train: The training data features.
        y_train: The training data labels.
        param_grid: The grid of hyperparameters to search over.

    Returns:
        The best estimator found by GridSearchCV.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best ROC-AUC score: ", grid_search.best_score_)
    return grid_search.best_estimator_


def print_classification_report(model_name: str, y_test: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> None:
    """
    Pretty prints a detailed classification report
    including precision, recall, f1-score, support, overall accuracy and ROC-AUC.

    Arguments:
        model_name: The name of the model being evaluated.
        y_test: The true labels for the test set.
        y_pred: The predicted labels for the test set.
        prob: The predicted probabilities for the positive class, used for ROC-AUC calculation.
    """
    print(f"Metrics for {model_name}:\n")
    classification = classification_report(y_test, y_pred, output_dict=True, labels=[
                                           0, 1], target_names=["Survived", "Died"])
    survived_metrics = classification["Survived"]
    death_metrics = classification["Died"]

    c1, c2, c3 = 32, 10, 10
    header = f"{'Metric':<{c1}} | {'Survived':>{c2}} | {'Died':>{c3}}"
    sep = "-" * len(header)

    def fmt(val):
        return f"{val:.2f}" if isinstance(val, float) else str(int(val))

    def row(label, s_val=None, d_val=None):
        sv = fmt(s_val) if s_val is not None else ""
        dv = fmt(d_val) if d_val is not None else ""
        print(f"{label:<{c1}} | {sv:>{c2}} | {dv:>{c3}}")

    print(header)
    print(sep)
    for metric_value in ["precision", "recall", "f1-score", "support"]:
        row(metric_value.capitalize(),
            survived_metrics[metric_value], death_metrics[metric_value])
    print(sep)
    row("Overall Accuracy", classification["accuracy"])
    row("ROC-AUC", roc_auc_score(y_test, prob))
    print(sep)
    for avg in ["macro avg", "weighted avg"]:
        for sub_metric in ["precision", "recall", "f1-score", "support"]:
            row(f"{sub_metric.capitalize()} ({avg})",
                classification[avg][sub_metric])

def logistic_regression():
    """
    Trains a logistic regression model with hyperparameter tuning on validation set.
    """

    # Best parameters found:
    # {'C': np.float64(0.23357214690901212), 'l1_ratio': 1, 'max_iter': 1000, 'solver': 'saga'}
    y_train = df_y_train.values.ravel()
    y_test = df_y_test.values.ravel()
    y_val = df_y_val.values.ravel()

    model = LogisticRegression()
    # Use training data to fit
    model.fit(x_train_scaled, y_train)

    # Use validation set to tune hyperparamters
    param_grid = [
        {
            "C": [*np.logspace(-4, 4, 20), 1, 10, 100],
            "solver": ["lbfgs", "newton-cholesky", "liblinear", "sag", "saga"],
            "max_iter": [400, 500, 1000, 1500, 2000],
        },
        {
            "C": [*np.logspace(-4, 4, 20), 1, 10, 100],
            "solver": ["saga"],
            "l1_ratio": np.arange(0, 1.1, 0.1),
            "max_iter": [400, 500, 1000, 1500, 2000],
        },
    ]
    best_model = hyperparameter_tuning(model, x_val_scaled, y_val, param_grid)

    # Evaluate
    lr_pred = best_model.predict(x_test_scaled)
    lr_prob = best_model.predict_proba(x_test_scaled)[:, 1]

    print_classification_report("Logistic Regression", y_test, lr_pred, lr_prob)


def random_forest():
    """
    Trains a random forest model.
    """
    model = RandomForestClassifier()

    param_grid = {
        "min_samples_leaf": range(1, 6),
        "n_estimators": range(50, 201, 50),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "min_samples_split": range(2, 11, 2),
        "max_depth": [None, 10, 20, 30]
    }


    y_train = df_y_train.values.ravel()
    y_test = df_y_test.values.ravel()
    y_val = df_y_val.values.ravel()

    model.fit(x_train_scaled, y_train)

    # Use validation set to tune hyperparamters
    best_model = hyperparameter_tuning(model, x_val_scaled, y_val, param_grid)

    # Evaluate
    rf_pred = best_model.predict(x_test_scaled)
    rf_prob = best_model.predict_proba(x_test_scaled)[:, 1]

    print_classification_report("Random Forest", y_test, rf_pred, rf_prob)
