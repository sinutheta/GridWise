"""Data ingestion, fetching, cleaning, and preparation for UK Carbon Intensity."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_intensity_response(data: dict) -> pd.DataFrame:
    """Parse JSON response from NESO Carbon Intensity API.
    
    Expected response schema:
    {"data": [{"from": "...", "to": "...", "intensity": {"forecast": ..., "actual": ..., "index": ...}}]}
    """
    records = []
    items = data.get("data", [])
    for item in items:
        ts_str = item.get("from")
        intensity_info = item.get("intensity", {})
        # Actual is preferred; fallback to forecast if actual is missing/None
        val = intensity_info.get("actual")
        if val is None:
            val = intensity_info.get("forecast")
        
        if ts_str is not None and val is not None:
            records.append({
                "timestamp": ts_str,
                "intensity": float(val),
                "forecast_api": float(intensity_info.get("forecast", val) or val),
            })

    if not records:
        return pd.DataFrame(columns=["timestamp", "intensity", "forecast_api"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_range(
    start: datetime,
    end: datetime,
    base_url: str = "https://api.carbonintensity.org.uk/intensity",
    session: Optional[requests.Session] = None,
    chunk_days: int = 14,
    pause: float = 0.3,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch historical carbon intensity data from API across consecutive chunks."""
    s = session or requests.Session()
    frames = []
    cur = start

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        from_str = cur.strftime("%Y-%m-%dT%H:%MZ")
        to_str = nxt.strftime("%Y-%m-%dT%H:%MZ")
        url = f"{base_url.rstrip('/')}/{from_str}/{to_str}"
        logger.info(f"Fetching {url}")

        res = None
        for attempt in range(max_retries):
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    res = r
                    break
                logger.warning(f"Attempt {attempt+1} got status {r.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed with error: {e}")
            import time
            time.sleep(2 ** attempt)

        if res is None or res.status_code != 200:
            raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")

        chunk_df = parse_intensity_response(res.json())
        if not chunk_df.empty:
            frames.append(chunk_df)

        cur = nxt
        import time
        time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "intensity"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df


def clean_intensity(df: pd.DataFrame, max_gap_slots: int = 2) -> pd.DataFrame:
    """Clean carbon intensity time series strictly according to architectural rules:
    
    1. Force regular 30-min index (UTC).
    2. Count missing slots. Linearly interpolate gaps <= max_gap_slots; mark longer gaps invalid.
    3. Drop duplicates, keeping last.
    4. Clip physically impossible values (< 0).
    5. Save processed output with `is_imputed` and `is_valid`.
    """
    if df.empty:
        raise ValueError("Cannot clean empty DataFrame.")

    # Drop duplicates
    df = df.drop_duplicates(subset=["timestamp"], keep="last").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Clip negative values
    neg_mask = df["intensity"] < 0
    neg_count = int(neg_mask.sum())
    if neg_count > 0:
        logger.warning(f"Clipped {neg_count} negative intensity values to 0.0")
        df.loc[neg_mask, "intensity"] = 0.0

    # Force 30-min regular index
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="30min",
        tz="UTC",
        name="timestamp",
    )
    df = df.set_index("timestamp").reindex(full_index)

    # Missing slot tracking
    missing_mask = df["intensity"].isna()
    missing_count = int(missing_mask.sum())
    logger.info(f"Total missing half-hour slots: {missing_count} / {len(df)}")

    # Identify gap lengths
    # Group consecutive missing blocks
    is_na = missing_mask.astype(int)
    blocks = (is_na != is_na.shift()).cumsum()
    gap_lengths = is_na.groupby(blocks).transform("sum") * is_na

    # Impute gaps <= max_gap_slots
    interpolated = df["intensity"].interpolate(method="linear", limit=max_gap_slots)

    # Mark imputed vs invalid
    is_imputed = missing_mask & interpolated.notna()
    is_valid = interpolated.notna() & (gap_lengths <= max_gap_slots)

    df["intensity"] = interpolated
    df["is_imputed"] = is_imputed
    df["is_valid"] = is_valid

    df = df.reset_index()
    return df


def generate_synthetic_intensity(
    start: Union[str, datetime] = "2023-01-01T00:00:00Z",
    end: Union[str, datetime] = "2024-01-01T00:00:00Z",
    seed: int = 42,
    introduce_gaps: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic UK half-hourly grid carbon intensity.
    
    Includes diurnal cycles (morning/evening peaks), seasonal trend (higher in winter),
    stochastic wind/solar fluctuations, and realistic missing gaps.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq="30min", tz="UTC", name="timestamp")
    n = len(timestamps)

    # Time variables
    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear
    day_of_week = timestamps.dayofweek

    # 1. Base level: ~180 gCO2/kWh
    base = 180.0

    # 2. Daily pattern: peak at 08:30 and 18:30, dip at night and midday
    daily = -25.0 * np.cos(2 * np.pi * (hour - 3) / 24.0) + 15.0 * np.cos(4 * np.pi * (hour - 8) / 24.0)

    # 3. Weekly pattern: weekends have lower industrial demand
    weekend = np.where(day_of_week >= 5, -15.0, 5.0)

    # 4. Seasonal pattern: higher in winter (Jan/Dec) due to gas heating/lighting
    seasonal = 35.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

    # 5. Stochastic weather/wind fluctuations (red noise / AR(1))
    weather_noise = np.zeros(n)
    phi = 0.985
    innovations = rng.normal(0, 8.0, n)
    for i in range(1, n):
        weather_noise[i] = phi * weather_noise[i - 1] + innovations[i]

    intensity = base + daily + weekend + seasonal + weather_noise
    intensity = np.clip(intensity, 15.0, 450.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "intensity": intensity,
    })

    if introduce_gaps:
        # Occasionally introduce short 1-2 slot gaps and rare longer gaps
        short_gap_indices = rng.choice(n - 5, size=min(20, n // 500), replace=False)
        for idx in short_gap_indices:
            df.loc[idx, "intensity"] = np.nan
        
        long_gap_indices = rng.choice(n - 10, size=min(3, n // 2000), replace=False)
        for idx in long_gap_indices:
            df.loc[idx : idx + 3, "intensity"] = np.nan

    return df


def get_or_create_dataset(
    processed_path: Union[str, Path] = "data/processed/intensity.parquet",
    force_fetch: bool = False,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load existing processed dataset or fetch/synthesize if missing."""
    p = Path(processed_path)
    if p.exists() and not force_fetch:
        logger.info(f"Loading cached dataset from {p}")
        return pd.read_parquet(p)

    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Processed dataset not found. Generating initial dataset...")

    # By default, generate 1 year of realistic synthetic data for initial setup/demo
    # If network/API fetching is enabled, users can run make data
    df_raw = generate_synthetic_intensity(
        start=start or "2023-01-01T00:00:00Z",
        end=end or "2024-01-01T00:00:00Z",
    )
    df_clean = clean_intensity(df_raw)
    df_clean.to_parquet(p, index=False)
    logger.info(f"Saved processed dataset to {p} ({len(df_clean)} records)")
    return df_clean
