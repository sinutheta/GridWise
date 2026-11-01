"""Feature engineering, calendar representations, and feature scaling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class ScalerStats:
    """Mean and standard deviation statistics for normalization."""
    mean: float
    std: float

    def transform(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return x * (self.std + 1e-8) + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ScalerStats":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclic calendar and holiday features to dataframe.
    
    Features engineered:
    - sin/cos hour of day
    - sin/cos day of week
    - sin/cos day of year
    - is_weekend
    - is_holiday (approximate UK public holidays)
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True)

    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear

    # Cyclic hour
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

    # Cyclic day of week
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    # Cyclic day of year
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # Binary flags
    df["is_weekend"] = (dow >= 5).astype(np.float32)

    # Basic UK holiday indicator (New Year, Christmas, Boxing Day, May/August bank holidays)
    month = ts.dt.month
    day = ts.dt.day
    is_xmas = (month == 12) & (day.isin([25, 26]))
    is_newyear = (month == 1) & (day == 1)
    # Generic bank holidays approximation: first/last Monday of May, last Monday of Aug
    is_bank_holiday_approx = (dow == 0) & (((month == 5) & (day <= 7)) | ((month == 8) & (day >= 25)))
    df["is_holiday"] = (is_xmas | is_newyear | is_bank_holiday_approx).astype(np.float32)

    return df


def get_feature_columns(include_weather: bool = False) -> Tuple[List[str], List[str]]:
    """Return feature column names for past and future inputs."""
    calendar_cols = [
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "sin_doy",
        "cos_doy",
        "is_weekend",
        "is_holiday",
    ]
    weather_cols = ["wind_speed", "cloud_cover", "temperature", "solar_radiation"] if include_weather else []

    # Past features include past intensity + calendar + optional past weather
    past_cols = ["intensity"] + calendar_cols + weather_cols
    # Future features known in advance: calendar + optional forecast weather
    future_cols = calendar_cols + weather_cols
    return past_cols, future_cols


def fit_intensity_scaler(train_intensity: Union[pd.Series, np.ndarray]) -> ScalerStats:
    """Compute mean and std on training intensity only."""
    values = np.asarray(train_intensity, dtype=np.float32)
    valid_vals = values[~np.isnan(values)]
    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals))
    if std < 1e-4:
        std = 1.0
    return ScalerStats(mean=mean, std=std)
