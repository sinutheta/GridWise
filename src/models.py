"""Neural network forecasting models and statistical baselines for carbon intensity."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NHitsBlock(nn.Module):
    """Hierarchical multi-rate pooling block for N-HiTS architecture."""

    def __init__(
        self,
        L: int,
        H: int,
        F_in: int,
        pool: int,
        hidden: int = 256,
        Q: int = 1,
        dropout: float = 0.1,
        F_future: int = 0,
    ):
        super().__init__()
        self.pool = pool
        self.L = L
        self.H = H
        self.Q = Q
        self.F_in = F_in

        # Input dimension after temporal average pooling
        pooled_len = L // pool
        in_dim = pooled_len * F_in

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Optional conditioning on future features (e.g. calendar)
        if F_future > 0:
            self.future_proj = nn.Linear(H * F_future, hidden)
        else:
            self.future_proj = None

        self.back = nn.Linear(hidden, L)
        self.coarse_len = H // pool + 1
        self.fore = nn.Linear(hidden, self.coarse_len * Q)

    def forward(
        self, x: torch.Tensor, x_future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Past tensor (B, L, F_in)
            x_future: Optional future feature tensor (B, H, F_future)
            
        Returns:
            backcast: (B, L)
            forecast: (B, H, Q)
        """
        B = x.size(0)

        # Multi-rate pooling across time: (B, F_in, L) -> pool -> (B, F_in, L//pool)
        z = F.avg_pool1d(x.transpose(1, 2), kernel_size=self.pool, stride=self.pool)
        z = z.transpose(1, 2)  # (B, L//pool, F_in)

        h = self.mlp(z.flatten(1))

        if self.future_proj is not None and x_future is not None:
            h = h + self.future_proj(x_future.flatten(1))

        backcast = self.back(h)  # (B, L)

        coarse = self.fore(h).view(B, self.coarse_len, self.Q)  # (B, coarse_len, Q)
        # Interpolate coarse forecast to full horizon H
        forecast = F.interpolate(
            coarse.transpose(1, 2),
            size=self.H,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # (B, H, Q)

        return backcast, forecast


class NHitsForecaster(nn.Module):
    """N-HiTS architecture with hierarchical multi-rate pooling blocks and backcast residuals."""

    def __init__(
        self,
        lookback: int = 336,
        horizon: int = 96,
        in_features: int = 9,
        future_features: int = 8,
        pools: List[int] = [8, 4, 1],
        hidden: int = 256,
        Q: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features
        self.future_features = future_features
        self.pools = pools
        self.Q = Q

        self.blocks = nn.ModuleList([
            NHitsBlock(
                L=lookback,
                H=horizon,
                F_in=in_features,
                pool=p,
                hidden=hidden,
                Q=Q,
                dropout=dropout,
                F_future=future_features,
            )
            for p in pools
        ])

    def forward(
        self, x_past: torch.Tensor, x_future: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through residual stack.
        
        Args:
            x_past: (B, L, F_in) - channel 0 is target intensity
            x_future: (B, H, F_future) - future calendar/weather
            
        Returns:
            forecast: (B, H, Q)
        """
        curr_past = x_past.clone()
        total_forecast = None

        for block in self.blocks:
            backcast, forecast = block(curr_past, x_future)
            # Subtract backcast from target channel (channel 0)
            curr_past[:, :, 0] = curr_past[:, :, 0] - backcast

            if total_forecast is None:
                total_forecast = forecast
            else:
                total_forecast = total_forecast + forecast

        return total_forecast


# --- Baseline Forecasting Functions ---

def persistence(x_past: torch.Tensor, H: int) -> torch.Tensor:
    """Repeat last observed intensity across horizon H.
    
    Args:
        x_past: (B, L, F) where channel 0 is intensity
    Returns:
        (B, H)
    """
    return x_past[:, -1:, 0].repeat(1, H)


def same_time_yesterday(x_past: torch.Tensor, H: int) -> torch.Tensor:
    """Repeat intensity from 24h (48 half-hours) ago.
    
    Args:
        x_past: (B, L, F)
    Returns:
        (B, H)
    """
    repeat_count = H // 48 + 1
    yesterday_slice = x_past[:, -48:, 0]
    return yesterday_slice.repeat(1, repeat_count)[:, :H]


def same_time_last_week(x_past: torch.Tensor, H: int) -> torch.Tensor:
    """Lookback intensity from 7 days (336 half-hours) ago.
    
    Args:
        x_past: (B, L, F) with L >= 336
    Returns:
        (B, H)
    """
    if x_past.size(1) < 336:
        raise ValueError(f"Lookback L={x_past.size(1)} is less than 336 (7 days).")
    return x_past[:, -336 : -336 + H, 0]
