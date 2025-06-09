import argparse
from pathlib import Path
import joblib
from google.cloud import storage
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_ingestion import load_and_merge_openaq_data, load_weather_data
from src.feature_engineering import (
    make_feature_pipeline,
    get_feature_columns,
    prepare_merged_data,
)
from src import utils_cloud


def main(args):
    client = storage.Client()
    data_root = Path.cwd() / "downloaded_data"

    # Download both folders, preserving structure:
    openaq_local = utils_cloud.download_gcs_prefix(
        client, args.openaq_gcs_path, dest_root=data_root
    )
    weather_local = utils_cloud.download_gcs_prefix(
        client, args.weather_gcs_path, dest_root=data_root
    )
    openaq_df = load_and_merge_openaq_data(openaq_local)
    weather_df = load_weather_data(
        weather_local / "hourly-london-weather-20240101-20250601.csv"
    )

    # Train model
    model, X_val, y_val = train_model(openaq_df, weather_df, target_column=args.target)

    # Export as .joblb
    joblib.dump(model, "model.joblib")
    out_bucket, out_blob = args.output_gcs_path.replace("gs://", "").split("/", 1)
    client.bucket(out_bucket).blob(out_blob).upload_from_filename("model.joblib")
    print("Model training complete and uploaded to", args.output_gcs_path)


def train_model(
    openaq_hourly_df: pd.DataFrame,
    weather_hourly_df: pd.DataFrame,
    target_column: str = "pm25_hourly_avg",
):
    merged_df = prepare_merged_data(openaq_hourly_df, weather_hourly_df)

    split_idx = int(0.8 * len(merged_df))
    df_train = merged_df.iloc[:split_idx]
    df_val = merged_df.iloc[split_idx:]

    # Construct Feature Pipeline
    cyclical_columns = ["hour", "day_of_week", "month", "winddir"]
    weather_cols = ["temperature", "humidity", "windspeed", "precip"]
    feature_pipeline = make_feature_pipeline(
        cyclical_columns=cyclical_columns, weather_cols=weather_cols
    )
    feature_cols = get_feature_columns()

    # Fit Feature pipeline on train data
    X_train = feature_pipeline.fit_transform(df_train.copy())
    X_val = feature_pipeline.transform(df_val.copy())

    X_train_df = pd.DataFrame(
        X_train, index=df_train.index, columns=feature_cols
    ).dropna()
    X_val_df = pd.DataFrame(X_val, index=df_val.index, columns=feature_cols).dropna()

    y_train = df_train.loc[X_train_df.index][target_column]
    y_val = df_val.loc[X_val_df.index][target_column]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_df, y_train)

    return model, X_val_df, y_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--openaq_gcs_path",
        type=str,
        required=True,
        help="gs://bucket/path/to/openaq_data_folder",
    )
    p.add_argument(
        "--weather_gcs_path",
        type=str,
        required=True,
        help="gs://bucket/path/to/weather_hourly.csv",
    )
    p.add_argument(
        "--output_gcs_path",
        type=str,
        required=True,
        help="gs://bucket/path/to/models/folder",
    )
    p.add_argument("--target", type=str, default="pm25_hourly_avg")
    args = p.parse_args()
    main(args)
