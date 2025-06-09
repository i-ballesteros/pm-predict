import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class LagRollingTransformer(BaseEstimator, TransformerMixin):
    """COmputes lag & rolling window features.
    Input: df with datetime index (expected hourly) and feature column
    Output: df with columns:
        feature_lag_i: feature from i hours ago
        feature_rollj: mean over the previous j hours
    """

    def __init__(self, feature_columns: list, lags=[1, 24], rolling_windows=[6, 24]):
        self.feature_columns = feature_columns

        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, df_X: pd.DataFrame):
        if not isinstance(df_X, pd.DataFrame) or not isinstance(
            df_X.index, pd.DatetimeIndex
        ):
            raise ValueError(
                "LagRollingTransformer expects a pandas DataFrame with a DatetimeIndex."
            )
        df_transformed = pd.DataFrame(index=df_X.index)

        for feature in self.feature_columns:
            for lag in self.lags:
                df_transformed[f"{feature}_lag{lag}"] = self._create_lag_feature(
                    df_X[feature], lag
                )

            for window in self.rolling_windows:
                df_transformed[f"{feature}_roll{window}"] = self._create_roll_feature(
                    df_X[feature], window
                )
        return df_transformed.reset_index(drop=True)

    def _create_lag_feature(self, feature: pd.Series, lag: int):
        return feature.shift(lag)

    def _create_roll_feature(self, feature: pd.Series, window: int):
        return feature.rolling(window, min_periods=1).mean().shift(1)


class CyclicalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cyclical_cols: list, periods: list):
        if len(cyclical_cols) != len(periods):
            raise ValueError("cyclical_columns and periods must have the same length.")
        self.cyclical_cols = cyclical_cols
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CyclicalTransformer expects a pandas DataFrame")
        df_res = pd.DataFrame(index=X.index)
        for col, period in zip(self.cyclical_cols, self.periods):
            if col not in X.columns:
                raise ValueError("CyclicalTransformer expects a pandas DataFrame.")
            vals = X[col].astype(float)
            df_res[f"{col}_sin"] = np.sin(2 * np.pi * (vals / period))
            df_res[f"{col}_cos"] = np.cos(2 * np.pi * (vals / period))
        return df_res.reset_index(drop=True)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Utility to select specified columns from a DataFrame.
    Returns a DataFrame with only those columns (converted to numpy array by pipeline if needed).
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].reset_index(drop=True)


def make_feature_pipeline(cyclical_columns: list, weather_cols: list) -> Pipeline:
    lagroll_branch = LagRollingTransformer(
        feature_columns=["pm25_hourly_avg"],
        lags=[1, 24],
        rolling_windows=[6, 24],
    )

    cyclic_branch = CyclicalTransformer(
        cyclical_cols=cyclical_columns, periods=[24, 7, 12, 360]
    )

    weather_branch = Pipeline(
        [
            ("selector", ColumnSelector(weather_cols)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    feature_union = FeatureUnion(
        [
            ("lagroll_features", lagroll_branch),
            ("time_features", cyclic_branch),
            ("weather_features", weather_branch),
        ]
    )

    return feature_union


def merge_data(
    openaq_hourly_df: pd.DataFrame,
    weather_hourly_df: pd.DataFrame,
):
    openaq_hourly_df = openaq_hourly_df.copy()
    weather_hourly_df = weather_hourly_df.copy()
    openaq_hourly_df.set_index("datetime", inplace=True)
    weather_hourly_df.set_index("datetime", inplace=True)
    merged_df = pd.merge(
        openaq_hourly_df, weather_hourly_df, left_index=True, right_index=True
    )
    merged_df.reset_index(inplace=True)
    return merged_df


def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def prepare_merged_data(openaq_hourly_df, weather_hourly_df):
    merged_df = merge_data(openaq_hourly_df, weather_hourly_df)
    merged_df = merged_df.sort_values("datetime")
    merged_df.set_index("datetime", inplace=True)
    merged_df = add_time_features(merged_df)
    return merged_df


def get_feature_columns():
    lagroll_cols = [
        "pm25_hourly_avg_lag1",
        "pm25_hourly_avg_lag24",
        "pm25_hourly_avg_roll6",
        "pm25_hourly_avg_roll24",
    ]
    cyclic_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "winddir_sin",
        "winddir_cos",
    ]
    weather_cols = ["temperature", "humidity", "windspeed", "precip"]
    return lagroll_cols + cyclic_cols + weather_cols
