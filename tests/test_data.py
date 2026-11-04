"""Unit tests for data ingestion, cleaning, and windowing."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.data import clean_intensity, generate_synthetic_intensity, parse_intensity_response
from src.features import add_calendar_features, fit_intensity_scaler
from src.dataset import CarbonIntensityDataset, make_chronological_splits


def test_parse_intensity_response():
    """Test parsing NESO API response: actual intensity is target, not forecast."""
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
    # Uses actual as target
    assert df["intensity"].iloc[0] == 145.0
    # When actual is None, target intensity is NaN (not falling back to forecast)
    assert np.isnan(df["intensity"].iloc[1])
    assert df["forecast_api"].iloc[1] == 160.0


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


def test_no_duplicate_timestamps():
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min", tz="UTC")
    vals = np.ones(len(times)) * 100.0
    df = pd.DataFrame({"timestamp": times, "intensity": vals})
    # Inject duplicated timestamps
    dup = df.iloc[:5].copy()
    dup["intensity"] = 200.0
    df_with_dups = pd.concat([df, dup], ignore_index=True)

    cleaned = clean_intensity(df_with_dups)
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["timestamp"].nunique() == len(cleaned)


def test_regular_30_minute_index():
    times = pd.date_range("2023-01-01 00:00", "2023-01-03 00:00", freq="30min", tz="UTC")
    # Drop some middle slots to create a ragged raw series
    drop_indices = [3, 7, 15, 20]
    kept_times = [t for i, t in enumerate(times) if i not in drop_indices]
    df = pd.DataFrame({"timestamp": kept_times, "intensity": np.ones(len(kept_times)) * 150.0})

    cleaned = clean_intensity(df)
    diffs = cleaned["timestamp"].diff().dropna()
    expected_diff = pd.Timedelta(minutes=30)
    assert all(d == expected_diff for d in diffs)
    assert len(cleaned) == len(times)


def test_split_gaps():
    df = generate_synthetic_intensity(start="2023-01-01", end="2023-06-01", seed=42, introduce_gaps=False)
    cleaned = clean_intensity(df)

    gap_slots = 96  # 48 hours
    train_df, val_df, test_df = make_chronological_splits(cleaned, train_ratio=0.7, val_ratio=0.15, gap_slots=gap_slots)

    # 48-hour gap between train and val
    train_end = train_df["timestamp"].max()
    val_start = val_df["timestamp"].min()
    gap_train_val = (val_start - train_end)
    assert gap_train_val >= pd.Timedelta(hours=48)

    # 48-hour gap between val and test
    val_end = val_df["timestamp"].max()
    test_start = test_df["timestamp"].min()
    gap_val_test = (test_start - val_end)
    assert gap_val_test >= pd.Timedelta(hours=48)

    # No overlap
    assert train_end < val_start
    assert val_end < test_start


def test_imputed_flag_correctness():
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 10:00", freq="30min", tz="UTC")
    vals = np.ones(len(times)) * 100.0
    # Create 1-slot gap at idx 2
    vals[2] = np.nan
    # Create 2-slot gap at idx 5, 6
    vals[5] = np.nan
    vals[6] = np.nan
    # Create 3-slot gap at idx 10, 11, 12
    vals[10] = np.nan
    vals[11] = np.nan
    vals[12] = np.nan

    df = pd.DataFrame({"timestamp": times, "intensity": vals})
    cleaned = clean_intensity(df, max_gap_slots=2)

    # 1-slot gap: imputed=True, valid=True
    assert cleaned.loc[2, "is_imputed"] == True
    assert cleaned.loc[2, "is_valid"] == True
    assert not np.isnan(cleaned.loc[2, "intensity"])

    # 2-slot gap: imputed=True, valid=True
    assert cleaned.loc[5, "is_imputed"] == True
    assert cleaned.loc[5, "is_valid"] == True
    assert cleaned.loc[6, "is_imputed"] == True
    assert cleaned.loc[6, "is_valid"] == True
    assert not np.isnan(cleaned.loc[5, "intensity"])
    assert not np.isnan(cleaned.loc[6, "intensity"])

    # 3-slot gap: imputed=False, valid=False
    assert cleaned.loc[10, "is_imputed"] == False
    assert cleaned.loc[10, "is_valid"] == False
    assert cleaned.loc[11, "is_imputed"] == False
    assert cleaned.loc[11, "is_valid"] == False
    assert cleaned.loc[12, "is_imputed"] == False
    assert cleaned.loc[12, "is_valid"] == False


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
