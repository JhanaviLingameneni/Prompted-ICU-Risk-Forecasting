"""
Module contains the training code for the ML models.
"""
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV, StratifiedKFold
try:
    from .data_loader import process_dataset
except ImportError:
    from data_loader import process_dataset
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


df_x_train, df_y_train = process_dataset(data_set="a", undersample=True)
df_x_test, df_y_test = process_dataset(data_set="b", undersample=False)
df_x_val, df_y_val = process_dataset(data_set="c", undersample=False)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df_x_train)
x_test_scaled = scaler.transform(df_x_test)
x_val_scaled = scaler.transform(df_x_val)
y_train = df_y_train.values.ravel()
y_test = df_y_test.values.ravel()
y_val = df_y_val.values.ravel()

# Best params found {'C': np.float64(0.1), 'max_iter': 500, 'solver': 'lbfgs'}
def logistic_regression() -> LogisticRegression:
    """
    Trains a logistic regression model with hyperparameter tuning on validation set.
    """
    model = LogisticRegression(random_state=42)

    # Use validation set to tune hyperparamters
    param_grid = {
        "C": np.logspace(-3, 3, 13),
        "solver": ["lbfgs", "newton-cholesky", "liblinear"],
        "max_iter": [500, 1000, 2000, 3000, 4000],
    }

    return hyperparameter_tuning(
        model,
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        param_grid,
    )

# Best parameters found:  {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 100}
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
        "max_depth": [None, 10, 20, 100]
    }

    # Use validation set to tune hyperparamters
    return hyperparameter_tuning(
        model,
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        param_grid,
    )

def gradient_boosting() -> GradientBoostingClassifier:
    params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 4],
        "subsample": [0.7, 0.8],
        "min_samples_leaf": [5, 10],
    }

    estimator = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator, params, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(x_train_scaled, y_train)
    print(grid.best_params_)


# Best params {'model__neurons': 32, 'model__l2_reg': 0.001, 'model__dropout_rate': 0.7, 'epochs': 100, 'batch_size': 32}
def ann() -> None:
    """
    Trains an Artificial Neural Network (ANN) using Keras with hyperparameter tuning on validation set.
    """

    def create_model(neurons=64, dropout_rate=0.5, l2_reg=0.01):
        model = Sequential([
            Input(shape=(x_train_scaled.shape[1],)),
            Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    clf = KerasClassifier(model=create_model, verbose=0)


    param_dist = {
        # Number of features, so we'll try a range of neurons around that.
        'model__neurons': [10, 16, 32, 64, 128, 256],
        # We avoid overfitting bc we prefer a higher recall. So high dropout rate will prevent this.
        'model__dropout_rate': [0.1, 0.3, 0.5, 0.7],
        # Regularize well across patients by trying a range of L2 regularization strengths.
        'model__l2_reg': [0.001, 0.01, 0.1],
        # Dataset is rather small so batches should not be too high.
        'batch_size': [16, 32, 64, 100],
        'epochs': [50, 100, 200]
    }

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    x_all = np.vstack([x_train_scaled, x_val_scaled])
    y_all = np.concatenate([df_y_train, df_y_val])

    # -1 means always train,
    # 0 means use the validation fold.
    test_fold = np.concatenate([
        -1 * np.ones(len(df_y_train), dtype=int),
        np.zeros(len(df_y_val), dtype=int),
    ])
    split = PredefinedSplit(test_fold=test_fold)

    scoring = {
        'recall_pos': make_scorer(recall_score, pos_label=1),
        'precision_pos': make_scorer(precision_score, pos_label=1)
    }

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=20,
        cv=split,
        scoring=scoring,
        refit='recall_pos',
        n_jobs=-1,
        verbose=0
    )

    search_result = random_search.fit(
        x_all, y_all,
        validation_data=(x_val_scaled, y_val),
        callbacks=[early_stop]
    )

    print("Best parameters found: ", search_result.best_params_)
    # compare_models({"ANN": search_result.best_estimator_})
    return search_result.best_estimator_

def lstm():
    """
    Trains an LSTM model on the same scaled data.
    Since the dataset is tabular, each feature is treated like one timestep.
    Input shape becomes: (samples, features, 1)
    """

    x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
    x_val_lstm = x_val_scaled.reshape((x_val_scaled.shape[0], x_val_scaled.shape[1], 1))

    model = Sequential([
        Input(shape=(x_train_lstm.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        x_train_lstm,
        y_train,
        validation_data=(x_val_lstm, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history

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
