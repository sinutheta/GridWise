"""Unit tests for data ingestion, cleaning, and windowing."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.data import clean_intensity, generate_synthetic_intensity, parse_intensity_response
from src.features import add_calendar_features, fit_intensity_scaler
from src.dataset import CarbonIntensityDataset, make_chronological_splits


def test_parse_intensity_response():
    sample_response = {
        "data": [
            {
                "from": "2023-01-01T00:00Z",
                "to": "2023-01-01T00:30Z",
                "intensity": {"forecast": 150, "actual": 145, "index": "moderate"},
            },
            {
                "from": "2023-01-01T00:30Z",
                "to": "2023-01-01T01:00Z",
                "intensity": {"forecast": 160, "actual": None, "index": "moderate"},
            },
        ]
    }
    df = parse_intensity_response(sample_response)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert df["intensity"].iloc[0] == 145.0
    # Falls back to forecast when actual is None
    assert df["intensity"].iloc[1] == 160.0


def test_clean_intensity_regular_index_and_imputation():
    # Construct raw data with duplicates, negatives, short gap, and long gap
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 06:00", freq="30min", tz="UTC")
    vals = [100.0, -10.0, 110.0, np.nan, 120.0, np.nan, np.nan, np.nan, 130.0, 140.0, 150.0, 160.0, 170.0]
    df_raw = pd.DataFrame({"timestamp": times[:len(vals)], "intensity": vals})
    # Add a duplicate
    dup_row = pd.DataFrame({"timestamp": [times[0]], "intensity": [999.0]})
    df_raw = pd.concat([df_raw, dup_row])

    df_clean = clean_intensity(df_raw, max_gap_slots=2)

    # 1. No duplicates
    assert len(df_clean) == df_clean["timestamp"].nunique()
    # 2. Dropped duplicate, kept last
    assert df_clean.loc[df_clean["timestamp"] == times[0], "intensity"].iloc[0] == 999.0
    # 3. Negatives clipped
    assert df_clean["intensity"].min() >= 0.0
    # 4. Short gap (1 slot at index 3) imputed and marked valid
    assert not np.isnan(df_clean.loc[3, "intensity"])
    assert df_clean.loc[3, "is_imputed"] == True
    assert df_clean.loc[3, "is_valid"] == True
    # 5. Long gap (3 slots at index 5, 6, 7) marked invalid
    assert not df_clean.loc[6, "is_valid"]


def test_splits_and_dataset_windowing():
    df = generate_synthetic_intensity(start="2023-01-01", end="2023-02-15", seed=42, introduce_gaps=False)
    df_clean = clean_intensity(df)

    train_df, val_df, test_df = make_chronological_splits(df_clean, train_ratio=0.7, val_ratio=0.15, gap_slots=96)

    # Check strictly chronological
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()

    scaler = fit_intensity_scaler(train_df["intensity"])
    ds = CarbonIntensityDataset(train_df, lookback=336, horizon=96, scaler=scaler)

    assert len(ds) > 0
    sample = ds[0]
    assert sample["x_past"].shape == (336, 9)
    assert sample["x_future"].shape == (96, 8)
    assert sample["y"].shape == (96,)
    assert sample["mask"].shape == (96,)
