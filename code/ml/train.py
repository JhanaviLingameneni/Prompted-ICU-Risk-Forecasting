"""
Module contains the training code for the ML models.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import DataLoader


def linear_regression():
    """
    Trains a linear regression model to predict survival days.
    """

    df_x_train, df_y_train = DataLoader(data_set="a").process_dataset()
    df_x_test, df_y_test = DataLoader(data_set="b").process_dataset()

    model = LinearRegression()
    model.fit(df_x_train, df_y_train.values.flatten())

    y_pred = model.predict(df_x_test)

    print("Linear Regression MAE:", mean_absolute_error(df_y_test, y_pred))
    print("Linear Regression RMSE:", mean_squared_error(df_y_test, y_pred))
    print("Linear Regression R²:", r2_score(df_y_test, y_pred))

def random_forest():
    """
    Trains a random forest model to predict survival days.
    """
    rf = RandomForestRegressor(n_estimators=200)
    df_x_train, df_y_train = DataLoader(data_set="a").process_dataset()
    df_x_test, df_y_test = DataLoader(data_set="b").process_dataset()

    rf.fit(df_x_train, df_y_train.values.flatten())
    rf_pred = rf.predict(df_x_test)

    print("Random Forest MAE:", mean_absolute_error(df_y_test, rf_pred))
    print("Random Forest RMSE:", mean_squared_error(df_y_test, rf_pred))
    print("Random Forest R²:", r2_score(df_y_test, rf_pred))
