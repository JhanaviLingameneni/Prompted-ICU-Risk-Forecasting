"""
Module contains the training code for the ML models.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from data_loader import DataLoader

df_x_train, df_y_train = DataLoader(data_set="a").process_dataset()
df_x_test, df_y_test = DataLoader(data_set="b").process_dataset()


def logistic_regression():
    """
    Trains a logistic regression model.
    """

    y_train = df_y_train.values.ravel()
    y_test = df_y_test.values.ravel()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(df_x_train)
    x_test_scaled = scaler.transform(df_x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

def random_forest():
    """
    Trains a random forest model.
    """
    rf = RandomForestClassifier()

    y_train = df_y_train.values.ravel()
    y_test = df_y_test.values.ravel()

    rf.fit(df_x_train, y_train)
    rf_pred = rf.predict(df_x_test)
    rf_prob = rf.predict_proba(df_x_test)[:, 1]

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

def xgboost():
    """
    Trains an xgboost model.
    """

    xgb = XGBClassifier()


    y_train = df_y_train.values.ravel()
    y_test = df_y_test.values.ravel()

    xgb.fit(df_x_train, y_train)
    xgb_pred = xgb.predict(df_x_test)
    xgb_prob = xgb.predict_proba(df_x_test)[:, 1]

    print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
    print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_prob))
