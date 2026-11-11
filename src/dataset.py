"""PyTorch Dataset, chronological splitting, and windowing for Carbon-Aware Training."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import ScalerStats, add_calendar_features, fit_intensity_scaler, get_feature_columns


class CarbonIntensityDataset(Dataset):
    """Sliding-window PyTorch Dataset with zero-copy index-based slicing.
    
    Tensors returned per item:
    - x_past: (L, F_past) - normalized past intensity + past features
    - x_future: (H, F_future) - future features (calendar/weather)
    - y: (H,) - true future intensity in gCO2/kWh
    - y_scaled: (H,) - scaled future intensity
    - mask: (H,) - validity mask (1 if valid, 0 if invalid)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 336,
        horizon: int = 96,
        scaler: Optional[ScalerStats] = None,
        include_weather: bool = False,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.include_weather = include_weather

        # Ensure calendar features exist
        df = add_calendar_features(df)
        self.df = df.reset_index(drop=True)

        self.past_cols, self.future_cols = get_feature_columns(include_weather=include_weather)

        # Scale intensity
        raw_intensity = self.df["intensity"].to_numpy(dtype=np.float32)
        if scaler is None:
            self.scaler = fit_intensity_scaler(raw_intensity)
        else:
            self.scaler = scaler

        scaled_intensity = self.scaler.transform(raw_intensity)

        # Prepare feature matrices
        # Past features matrix
        past_matrix_list = [scaled_intensity[:, None]]
        for col in self.past_cols[1:]:
            past_matrix_list.append(self.df[col].to_numpy(dtype=np.float32)[:, None])
        self.past_features = np.hstack(past_matrix_list)  # (N, F_past)

        # Future features matrix
        future_matrix_list = []
        for col in self.future_cols:
            future_matrix_list.append(self.df[col].to_numpy(dtype=np.float32)[:, None])
        self.future_features = np.hstack(future_matrix_list)  # (N, F_future)

        # Target and validity arrays
        self.raw_targets = raw_intensity
        self.scaled_targets = scaled_intensity
        self.is_valid = self.df.get("is_valid", pd.Series(True, index=self.df.index)).to_numpy(dtype=bool)

        # Build valid origin indices
        # Both past [t - lookback, t) and future [t, t + horizon) must be fully valid (no invalid slots)
        n_total = len(self.df)
        valid_indices = []
        for t in range(self.lookback, n_total - self.horizon + 1):
            if np.all(self.is_valid[t - self.lookback : t + self.horizon]):
                valid_indices.append(t)

        self.origin_indices = np.array(valid_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.origin_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.origin_indices[idx]
        L = self.lookback
        H = self.horizon

        # Past window: [t - L, t)
        x_past = torch.from_numpy(self.past_features[t - L : t])

        # Future window: [t, t + H)
        x_future = torch.from_numpy(self.future_features[t : t + H])
        y = torch.from_numpy(self.raw_targets[t : t + H])
        y_scaled = torch.from_numpy(self.scaled_targets[t : t + H])
        mask = torch.from_numpy(self.is_valid[t : t + H].astype(np.float32))

        return {
            "x_past": x_past,
            "x_future": x_future,
            "y": y,
            "y_scaled": y_scaled,
            "mask": mask,
            "origin_idx": torch.tensor(t, dtype=torch.long),
        }


def make_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    gap_slots: int = 96,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe strictly chronologically with gap horizons between splits."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_start = train_end + gap_slots
    val_end = val_start + int(n * val_ratio)
    test_start = val_end + gap_slots

    if test_start >= n:
        raise ValueError(f"Dataset too short for specified splits and {gap_slots}-slot gaps.")

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)

    return train_df, val_df, test_df
