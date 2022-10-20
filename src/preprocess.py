from datetime import datetime


def region_mapping(val):
    if val <= 90:
        return "NE"
    elif val > 90 and val <= 180:
        return "ES"
    elif val > 180 and val <= 270:
        return "SW"
    else:
        return "WN"


def filter_data(df):
    # Remove the data that wind speed is smaller than 3.5 and bigger than 25.5
    # We do that because according to turbine power curve turbine works between these values.
    data2 = df[(df["wind_speed"] > 3.5) & (df["wind_speed"] <= 25.5)]

    # Eliminate datas where wind speed is bigger than 3.5 and active power is zero.
    data3 = data2[
        ((data2["active_power"] != 0) & (data2["wind_speed"] > 3.5))
        | (data2["wind_speed"] <= 3.5)
    ]
    return data3


def hour_of_day(x):
    hour_day = f"{x.year}-{x.month}-{x.day}-{x.hour}"
    hour_day = datetime.strptime(hour_day, "%Y-%m-%d-%H")
    return hour_day


def agg_data(df, time_frame="hour"):
    df = df[["date_time", "active_power"]]
    df["date_time"] = df["date_time"].apply(
        lambda x: datetime.strptime(x, "%d %m %Y %H:%M")
    )
    times = df.date_time
    if time_frame == "hour":
        df["hour"] = df["date_time"].apply(lambda x: hour_of_day(x))
        df = df.groupby(["hour"]).mean()
    if time_frame == "day":
        df = df.groupby([times.dt.dayofyear]).mean()
    if time_frame == "month":
        df = df.groupby([times.dt.month]).mean()
    return df
