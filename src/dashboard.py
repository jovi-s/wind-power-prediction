import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from .preprocess import hour_of_day
from .utils import bins_dir, bins_dir_labels


model_preds = {
    "LinearRegression": [
        "./data/output/LinearRegression_predictions.csv",
        "./data/output/LinearRegression_features_importances.csv",
    ],
    "XGBoost": [
        "./data/output/XGBoostRegression_predictions.csv",
        "./data/output/XGBoostRegression_features_importances.csv",
    ],
    "LightGBM": [
        "./data/output/LightGBMRegression_predictions.csv",
        "./data/output/LightGBMRegression_features_importances.csv",
    ],
}


def prediction_plots(name, agg):
    """Initialize and create figure objects for dashboard in main.py.

    Args:
        name (str): Name of model to be used.
        agg (str): Type of aggregation used for main predictions graph.

    Returns:
        compare_fig: Graph of predictions and actual values by date.
        r2: R2 score of predictions and actual values.
        mape: Mean absolute percentage error of predictions and actual values.
    """
    pred_file_path = model_preds[name]
    df = pd.read_csv(pred_file_path[0])

    df["date_time"] = df["date_time"].apply(
        lambda x: datetime.strptime(x, "%d %m %Y %H:%M")
    )

    if agg == "Hourly":
        df = agg_plot(df)
    elif agg == "Daily":
        df = agg_plot(df, "day")

    # Prediction graph
    compare_fig = {
        "data": [
            {
                "x": df["date_time"],
                "y": df["active_power"],
                "type": "line",
                "name": "LV ActivePower (kW)",
                "line": dict(color="green"),
            },
            {
                "x": df["date_time"],
                "y": df["prediction"],
                "type": "line",
                "name": "Power Prediction (kW)",
                "line": dict(color="orange"),
            },
        ],
        "layout": {
            "title": "Actual vs Predicted Power Output for Dec 2018",
            "xaxis": {
                "tickvals": "%d",
                "rangeslider": {"visible": True},
                "rangeselector": {"visible": True},
            },
        },
    }

    # Test set evaluation scores
    rmse = round(np.sqrt(mean_squared_error(df["active_power"], df["prediction"])), 2)
    r2 = round(r2_score(df["active_power"], df["prediction"]), 2)
    mape = round(
        mean_absolute_percentage_error(df["active_power"], df["prediction"]), 2
    )
    _maape = round(maape(df["active_power"], df["prediction"]), 2)
    return compare_fig, rmse, r2, _maape


def explainability_plots(name):
    pred_file_path = model_preds[name]
    # Windrose chart
    fig_windrose = create_windrose(pred_file_path[0])
    # Feature importances
    fig_features = create_features(pred_file_path[1])
    return fig_windrose, fig_features


def create_windrose(file_path):
    df = pd.read_csv(file_path)
    df["dir_binned"] = pd.cut(df["wind_direction"], bins_dir, labels=bins_dir_labels)
    df["wind_speed_rnd"] = df["wind_speed"].round()
    grp = (
        df.groupby(["dir_binned", "wind_speed_rnd"])["prediction"].mean().reset_index()
    )
    grp.dropna(subset=["prediction"], inplace=True)
    fig_windrose = px.bar_polar(
        grp,
        r="wind_speed_rnd",
        theta="dir_binned",
        color="prediction",
        color_continuous_scale=px.colors.sequential.Plasma_r,
        title="Wind Rose Chart of Power Prediction vs Wind Direction vs Wind Speed",
    )
    fig_windrose.update_layout(
        title={"x": 0.5}, polar=dict(radialaxis=dict(showticklabels=False, ticks=""))
    )
    return fig_windrose


def create_features(file_path):
    features_importances = pd.read_csv(file_path)
    fig_features = px.bar(
        features_importances, x="importance", y="features", orientation="h"
    )
    fig_features.update_layout(title={"text": "Feature Importance", "x": 0.5})
    return fig_features


def agg_plot(df, time_frame="hour"):
    """Aggregate data depending on specified time frame.

    Args:
        df (pd.DataFrame): Prediction results.
        time_frame (str, optional): Aggregation type. Defaults to "hour".

    Returns:
        pd.DataFrame: Aggregated dataframe.
    """
    df = df[["date_time", "active_power", "prediction"]]
    if time_frame == "hour":
        df = df.copy()
        df["date_time"] = df["date_time"].apply(lambda x: hour_of_day(x))
        df = df.groupby(["date_time"]).mean().reset_index()
    if time_frame == "day":
        times = df.date_time
        df = df.groupby([times.dt.dayofyear]).mean().reset_index()
        df["date_time"] = df["date_time"].apply(
            lambda x: datetime.strptime("2018" + "-" + str(x), "%Y-%j").strftime(
                "%d-%m-%Y"
            )
        )
    return df


EPSILON = 1e-10


def maape(actual: np.ndarray, predicted: np.ndarray):
    """Mean Arctangent Absolute Percentage Error"""
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))
