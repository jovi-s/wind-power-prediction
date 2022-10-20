import numpy as np
import pandas as pd
from datetime import datetime


bins_dir = [
    0,
    11.25,
    33.75,
    56.25,
    78.75,
    101.25,
    123.75,
    146.25,
    168.75,
    191.25,
    213.75,
    236.25,
    258.75,
    281.25,
    303.75,
    326.25,
    348.75,
]


bins_dir_labels = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


def rename(df):
    df.rename(
        columns={
            "LV ActivePower (kW)": "active_power",
            "Wind Speed (m/s)": "wind_speed",
            "Wind Direction (°)": "wind_direction",
            "Date/Time": "date_time",
            "Theoretical_Power_Curve (KWh)": "theoretical_power",
        },
        inplace=True,
    )
    return df


def all_data(file_path="../data/input/T1.csv"):
    df = pd.read_csv(file_path)
    df = rename(df)
    # df.set_index('date_time')
    return df


def train_data(file_path="../data/input/T1_train.csv"):
    df = pd.read_csv(file_path)
    df = rename(df)
    # df.set_index('date_time')
    return df


def test_data(file_path="../data/input/T1_test.csv"):
    df = pd.read_csv(file_path)
    df = rename(df)
    # df.set_index('date_time')
    return df


def time_power_data(df):
    df["date_time"] = df["date_time"].apply(
        lambda x: datetime.strptime(x, "%d %m %Y %H:%M")
    )
    df = df.set_index("date_time")
    df = df[["active_power"]]
    return df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
