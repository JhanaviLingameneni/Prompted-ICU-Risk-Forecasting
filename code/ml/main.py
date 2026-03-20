"""
Entrypoint
"""

from train import logistic_regression, random_forest, compare_models

if __name__ == "__main__":
    lr = logistic_regression()
    rf = random_forest()
    compare_models({"Logistic Regression": lr, "Random Forest": rf})
