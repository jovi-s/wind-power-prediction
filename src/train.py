import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from preprocess import region_mapping
from utils import train_data


def process_features(df):
    df["gust"] = np.array(
        [0] + list(df["wind_speed"][1:].values - df["wind_speed"][:-1].values)
    )
    df["region"] = df["wind_direction"].apply(lambda x: region_mapping(x))
    df = df.drop(["wind_direction"], axis=1)

    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["region"]),
        (MinMaxScaler(), ["gust", "wind_speed", "theoretical_power"]),
        remainder="passthrough",
    )
    df = column_trans.fit_transform(df)
    return df


def lstm_features():
    df = train_data()
    df["date_time"] = df["date_time"].apply(
        lambda x: datetime.strptime(x, "%d %m %Y %H:%M")
    )
    df = df.set_index("date_time")

    X = df.drop(["active_power"], axis=1)
    y = df[["active_power"]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train = lstm_feature_engineering(X_train)
    X_test = lstm_feature_engineering(X_test)
    return X_train, X_test, y_train, y_test


def lstm_feature_engineering(df):
    df["region"] = df["wind_direction"].apply(lambda x: region_mapping(x))
    df = df.drop(["wind_direction"], axis=1)

    # Differencing
    diff_cols = ["wind_speed", "theoretical_power"]
    for c in diff_cols:
        df[c + "_diff"] = df[c].diff()
    df.fillna(0, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["region"], prefix="region")

    # Scale
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df


def print_evaluation(y_test, y_pred):
    print("\t\tError Table")
    print(f"Mean Absolute Error      : {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared  Error      : {mean_squared_error(y_test, y_pred)}")
    print(f"Root Mean Squared  Error : {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"R2 Score                 : {r2_score(y_test, y_pred)}")
    print(
        f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred)}"
    )


def feature_importance(model, df, name):
    if name == "LinearRegression":
        importance = model.coef_
        non_extreme_neg_importances = [0 if x < -100 else x for x in importance]
        non_neg_importances = [
            abs(x) if x < 0 else x for x in non_extreme_neg_importances
        ]
        importances = np.array(non_neg_importances)
    else:
        importances = model.feature_importances_
    features = [
        "region_ES",
        "region_NE",
        "region_SW",
        "region_WN",
        "gust",
        "wind_speed",
        "theoretical_power",
    ]
    df = process_features(df)
    df = pd.DataFrame(df, columns=features)
    sorted_idx = importances.argsort()
    plt.barh(df.columns[sorted_idx], importances[sorted_idx])
    plt.title(f"{name} Feature Importance")
    imp = pd.DataFrame(
        {"features": df.columns[sorted_idx], "importance": importances[sorted_idx]}
    )
    imp.to_csv(f"../data/output/{name}_features_importances.csv", index=False)
    return plt.show()
