"""
Entrypoint
"""

from train import logistic_regression, random_forest, compare_models

if __name__ == "__main__":
    # Best parameters found:  {'C': np.float64(0.1), 'max_iter': 500, 'solver': 'lbfgs'}
    lr = logistic_regression()

    # Best parameters found:  {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 100}
    rf = random_forest()
    compare_models({"Logistic Regression": lr, "Random Forest": rf})
