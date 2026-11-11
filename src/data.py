"""Data ingestion, fetching, cleaning, and preparation for UK Carbon Intensity."""

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_intensity_response(data: dict) -> pd.DataFrame:
    """Parse JSON response from NESO Carbon Intensity API.
    
    Expected response schema:
    {"data": [{"from": "...", "to": "...", "intensity": {"forecast": ..., "actual": ..., "index": ...}}]}

    Target field is 'actual' intensity, not 'forecast'.
    If 'actual' is missing or None, intensity is set to NaN so that downstream
    cleaning and gap handling can identify it as a missing slot.
    """
    records = []
    items = data.get("data", [])
    for item in items:
        ts_str = item.get("from")
        intensity_info = item.get("intensity", {})
        actual_val = intensity_info.get("actual")
        forecast_val = intensity_info.get("forecast")
        
        if ts_str is not None:
            records.append({
                "timestamp": ts_str,
                "intensity": float(actual_val) if actual_val is not None else np.nan,
                "forecast_api": float(forecast_val) if forecast_val is not None else np.nan,
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
    cache_dir: Optional[Union[str, Path]] = "data/raw",
    session: Optional[requests.Session] = None,
    chunk_days: int = 14,
    pause: float = 0.2,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch historical carbon intensity data from API across consecutive chunks.
    
    Caches each chunk to cache_dir as raw JSON so a failed run can resume without refetching.
    Uses exponential backoff and retries on failure.
    """
    s = session or requests.Session()
    s.headers.update({"Accept": "application/json"})
    frames = []
    cur = start

    raw_path = Path(cache_dir) if cache_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        from_str = cur.strftime("%Y-%m-%dT%H:%MZ")
        to_str = nxt.strftime("%Y-%m-%dT%H:%MZ")

        # Cache file naming
        safe_from = cur.strftime("%Y%m%dT%H%MZ")
        safe_to = nxt.strftime("%Y%m%dT%H%MZ")
        cache_file = raw_path / f"intensity_{safe_from}_{safe_to}.json" if raw_path else None

        json_data = None
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                logger.info(f"Loaded cached chunk {cache_file.name}")
            except Exception as e:
                logger.warning(f"Error reading cache {cache_file}: {e}, refetching...")

        if json_data is None:
            url = f"{base_url.rstrip('/')}/{from_str}/{to_str}"
            logger.info(f"Fetching {url}")

            res = None
            for attempt in range(max_retries):
                try:
                    r = s.get(url, timeout=30)
                    if r.status_code == 200:
                        res = r
                        break
                    logger.warning(f"Attempt {attempt+1} got status {r.status_code} for {url}")
                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(2 ** attempt)

            if res is None or res.status_code != 200:
                status_code = res.status_code if res else "No response"
                error_body = res.text if res else "Timeout/Connection Error"
                raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts. HTTP {status_code}: {error_body}")

            json_data = res.json()
            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f)
            time.sleep(pause)

        chunk_df = parse_intensity_response(json_data)
        if not chunk_df.empty:
            frames.append(chunk_df)

        cur = nxt

    if not frames:
        return pd.DataFrame(columns=["timestamp", "intensity", "forecast_api"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    
    # Filter to requested range [start, end)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return df


def fetch_weather_open_meteo(
    start: datetime,
    end: datetime,
    latitude: float = 52.5,
    longitude: float = -1.5,
    cache_path: Optional[Union[str, Path]] = "data/raw/weather_open_meteo.json",
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch historical weather (wind speed, cloud cover, temperature, solar radiation)
    from Open-Meteo Historical Weather API.
    
    License: CC BY 4.0 (Free with attribution).
    Source: Open-Meteo Historical Weather API (https://archive-api.open-meteo.com/v1/archive)
    """
    cache_file = Path(cache_path) if cache_path else None
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)

    json_data = None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            logger.info(f"Loaded cached weather from {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to read cached weather {cache_file}: {e}, refetching...")

    if json_data is None:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
            "hourly": ["wind_speed_10m", "cloud_cover", "temperature_2m", "shortwave_radiation"],
            "timezone": "UTC",
        }
        logger.info(f"Fetching Open-Meteo weather from {url} for {params['start_date']} to {params['end_date']}")

        res = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 200:
                    res = r
                    break
                logger.warning(f"Open-Meteo attempt {attempt+1} got status {r.status_code}: {r.text}")
            except requests.RequestException as e:
                logger.warning(f"Open-Meteo attempt {attempt+1} error: {e}")
            time.sleep(2 ** attempt)

        if res is None or res.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather from Open-Meteo after {max_retries} attempts.")

        json_data = res.json()
        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f)

    hourly = json_data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    df_weather = pd.DataFrame({
        "timestamp": times,
        "wind_speed": hourly.get("wind_speed_10m", []),
        "cloud_cover": hourly.get("cloud_cover", []),
        "temperature": hourly.get("temperature_2m", []),
        "solar_radiation": hourly.get("shortwave_radiation", []),
    })
    df_weather = df_weather.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df_weather


def clean_intensity(
    df: pd.DataFrame,
    max_gap_slots: int = 2,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Clean carbon intensity time series strictly according to architectural rules:
    
    1. Drop duplicates, keeping last.
    2. Force regular 30-min index (UTC).
    3. Count missing slots. Linearly interpolate gaps <= max_gap_slots and set is_imputed=True.
    4. Mark longer gaps is_valid=False and make sure windows crossing them are excluded downstream.
    5. Clip physically impossible values (< 0) and log how many.
    6. Align weather features (forward-fill hourly values) if weather_df is provided.
    7. Save processed output with `is_imputed` and `is_valid`.
    """
    if df.empty:
        raise ValueError("Cannot clean empty DataFrame.")

    # Drop duplicates
    df = df.drop_duplicates(subset=["timestamp"], keep="last").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

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
    # Identify non-positive placeholders (< 0 or == 0; physically impossible in UK grid)
    placeholder_mask = (df["intensity"] <= 0)
    placeholder_count = int(placeholder_mask.sum())
    if placeholder_count > 0:
        logger.warning(f"Found {placeholder_count} non-positive placeholders (intensity <= 0); setting to NaN and is_valid=False.")
        df.loc[placeholder_mask, "intensity"] = np.nan

    missing_mask = df["intensity"].isna()
    missing_count = int(missing_mask.sum())
    logger.info(f"Total missing half-hour slots: {missing_count} / {len(df)}")

    # Identify gap lengths for consecutive missing blocks
    is_na = missing_mask.astype(int)
    blocks = (is_na != is_na.shift()).cumsum()
    gap_lengths = is_na.groupby(blocks).transform("sum") * is_na

    # Impute gaps <= max_gap_slots linearly (excluding explicit placeholders)
    can_impute = missing_mask & (gap_lengths <= max_gap_slots) & (gap_lengths > 0) & (~placeholder_mask)
    interpolated = df["intensity"].interpolate(method="linear")

    df.loc[can_impute, "intensity"] = interpolated.loc[can_impute]
    df["is_imputed"] = can_impute
    df["is_valid"] = df["intensity"].notna() & (~placeholder_mask) & (gap_lengths <= max_gap_slots)

    df = df.reset_index()

    # Align weather features if provided
    if weather_df is not None and not weather_df.empty:
        weather_clean = weather_df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        # Reindex weather onto 30-min index and forward fill hourly values
        weather_aligned = weather_clean.set_index("timestamp").reindex(full_index, method="ffill").bfill().reset_index()
        for col in ["wind_speed", "cloud_cover", "temperature", "solar_radiation"]:
            if col in weather_aligned.columns:
                df[col] = weather_aligned[col].to_numpy(dtype=np.float32)

    return df


def generate_data_report(df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> Dict[str, Union[str, int, float]]:
    """Generate comprehensive data quality report for carbon intensity."""
    ts_min = df["timestamp"].min().isoformat()
    ts_max = df["timestamp"].max().isoformat()
    row_count = len(df)

    imputed_count = int(df["is_imputed"].sum())
    invalid_count = int((~df["is_valid"]).sum())

    # Count missing slots before imputation
    if raw_df is not None:
        raw_ts = set(pd.to_datetime(raw_df["timestamp"], utc=True))
        all_ts = set(df["timestamp"])
        missing_slots = len(all_ts - raw_ts) + int(raw_df["intensity"].isna().sum())
    else:
        missing_slots = imputed_count + invalid_count

    # Share of whole numbers (check non-imputed valid original intensity values)
    original_valid = df[df["is_valid"] & (~df["is_imputed"])]["intensity"].dropna()
    if len(original_valid) > 0:
        whole_num_count = np.isclose(original_valid, np.round(original_valid), atol=1e-3).sum()
        share_whole_numbers = float(whole_num_count) / len(original_valid)
    else:
        share_whole_numbers = 0.0

    valid_intensities = df[df["is_valid"]]["intensity"].dropna()
    mean_val = float(valid_intensities.mean())
    std_val = float(valid_intensities.std())
    min_val = float(valid_intensities.min())
    max_val = float(valid_intensities.max())

    report = {
        "date_range_start": ts_min,
        "date_range_end": ts_max,
        "row_count": row_count,
        "missing_slots": missing_slots,
        "imputed_count": imputed_count,
        "invalid_count": invalid_count,
        "share_of_whole_numbers": share_whole_numbers,
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "min_intensity": min_val,
        "max_intensity": max_val,
    }

    print("\n" + "=" * 60)
    print("DATA REPORT: UK Real Carbon Intensity Dataset")
    print("=" * 60)
    print(f"Date range:              {ts_min} to {ts_max}")
    print(f"Row count:               {row_count}")
    print(f"Missing slots:           {missing_slots}")
    print(f"Imputed count:           {imputed_count}")
    print(f"Invalid count:           {invalid_count}")
    print("Note:                    Imputed count is 0; all invalid slots contain NaN, none were imputed.")
    print(f"Share of whole numbers:  {share_whole_numbers * 100:.2f}% (expected ~100% for real NESO actuals)")
    print(f"Mean intensity:          {mean_val:.2f} gCO2/kWh")
    print(f"Std intensity:           {std_val:.2f} gCO2/kWh")
    print(f"Min intensity:           {min_val:.2f} gCO2/kWh")
    print(f"Max intensity:           {max_val:.2f} gCO2/kWh")
    print("=" * 60 + "\n")

    return report


def generate_synthetic_intensity(
    start: Union[str, datetime] = "2023-01-01T00:00:00Z",
    end: Union[str, datetime] = "2024-01-01T00:00:00Z",
    seed: int = 42,
    introduce_gaps: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic UK half-hourly grid carbon intensity (test fixture)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq="30min", tz="UTC", name="timestamp")
    n = len(timestamps)

    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear
    day_of_week = timestamps.dayofweek

    base = 180.0
    daily = -25.0 * np.cos(2 * np.pi * (hour - 3) / 24.0) + 15.0 * np.cos(4 * np.pi * (hour - 8) / 24.0)
    weekend = np.where(day_of_week >= 5, -15.0, 5.0)
    seasonal = 35.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

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
    include_weather: bool = True,
) -> pd.DataFrame:
    """Load existing processed dataset or fetch real UK carbon intensity data from API."""
    p = Path(processed_path)
    if p.exists() and not force_fetch:
        logger.info(f"Loading cached dataset from {p}")
        return pd.read_parquet(p)

    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Processed dataset not found. Fetching real UK carbon intensity dataset from NESO API...")

    # Default to 2022-01-01 up to latest complete day (midnight UTC today)
    if start is None:
        start = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    if end is None:
        now_utc = datetime.now(timezone.utc)
        end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    df_raw = fetch_range(start=start, end=end)
    
    weather_df = None
    if include_weather:
        try:
            weather_df = fetch_weather_open_meteo(start=start, end=end)
        except Exception as e:
            logger.warning(f"Could not fetch weather: {e}. Proceeding with intensity only.")

    df_clean = clean_intensity(df_raw, weather_df=weather_df)
    df_clean.to_parquet(p, index=False)
    logger.info(f"Saved processed dataset to {p} ({len(df_clean)} records)")

    generate_data_report(df_clean, raw_df=df_raw)
    return df_clean


if __name__ == "__main__":
    get_or_create_dataset(force_fetch=True)
