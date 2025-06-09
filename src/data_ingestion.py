import pandas as pd
from pathlib import Path


def load_and_merge_openaq_data(
    openaq_dir: str,
    filter_param: str = "pm25",
    save: bool = False,
    output_dir: str = None,
) -> pd.DataFrame:
    """Loads OpenAQ data from directory containing monthly location .csv data, merges and averages the different sensors readings to obtain one location equivalent.

    Args:
        openaq_dir (str): Directory containing the monthly folders with location-wise data
        filter_param (str, optional): Parameter to filter data. Defaults to "pm25".
        save (bool, optional): Saves output df (debugging). Requires output_dir Defaults to False.
        output_dir (str, optional): Location to save output df. Requires save. Defaults to None.

    Raises:
        ValueError: If save is True but no output_dir is set.

    Returns:
        pd.DataFrame: Hourly, single location, averaged sensor readings.
    """
    if save and not output_dir:
        raise ValueError(
            "output_dir cant be None if save is True. Please input output directory to save the merged data."
        )

    merged_dfs = []
    data_dir = Path(openaq_dir)
    print(data_dir)
    for csv_file in data_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if filter_param:
                df = df[df["parameter"] == filter_param]
            merged_dfs.append(df)
        except Exception as e:
            print(f"Failed to parse {csv_file}: {e}")

    # Combine all into one DataFrame
    if merged_dfs:
        df_all = pd.concat(merged_dfs, ignore_index=True)
        print(f"Loaded {len(df_all)} {filter_param} rows from {len(merged_dfs)} files")
    else:
        print(f"No {filter_param} data found.")

    df_all["datetime"] = pd.to_datetime(df_all["datetime"], utc=True)
    # Combine all sensors reading into one signal
    hourly_avg_df = _combine_sensor_data(df_all)

    if save:
        output_path = Path(output_dir)
        hourly_avg_df.to_csv(output_path / "hourly_openaq_data.csv", index=False)
    return hourly_avg_df


def load_weather_data(weather_hourly_file: str) -> pd.DataFrame:
    df_weather = pd.read_csv(weather_hourly_file)
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], utc=True)
    return df_weather


def _combine_sensor_data(merged_openaq_data_df: pd.DataFrame):
    # Smooth out outliars
    merged_openaq_data_df["cutoff"] = merged_openaq_data_df.groupby("sensors_id")[
        "value"
    ].transform(lambda x: x.quantile(0.99))
    df_smoothed = merged_openaq_data_df[
        merged_openaq_data_df["value"] <= merged_openaq_data_df["cutoff"]
    ].copy()

    # Some sensors are sampled at different intervals, floor down to the hour
    df_smoothed["datetime_hour"] = df_smoothed["datetime"].dt.floor("h")
    per_sensor_hourly = (
        df_smoothed.groupby(["sensors_id", "datetime_hour"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"datetime_hour": "datetime", "value": "pm25_hourly"})
    )
    hourly_avg = (
        per_sensor_hourly.groupby("datetime")["pm25_hourly"]
        .mean()
        .reset_index(name="pm25_hourly_avg")
    )
    return hourly_avg
