"""Carbon-aware job scheduling policies, decision rules, and emission computations."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np


@dataclass
class Job:
    """Compute job specification for carbon-aware scheduling."""
    d: int          # duration in slots (1 slot = 30 minutes)
    power_kw: float # average power consumption in kW
    slack: int      # maximum allowed delay in slots


def compute_emissions(
    intensity_slice: np.ndarray,
    power_kw: float = 1.0,
    slot_hours: float = 0.5,
) -> float:
    """Calculate total emissions in gCO2.
    
    Formula: E = power_kw * slot_hours * sum(intensity)
    where intensity is in gCO2/kWh.
    """
    return float(power_kw * slot_hours * np.sum(intensity_slice))


def pick_start(
    forecast: np.ndarray,
    d: int,
    slack: int,
    mode: str = "mean",
    upper: Optional[np.ndarray] = None,
    tie_break_eps: float = 1.0e-4,
) -> int:
    """Pick optimal start slot s in [0, slack] minimizing window emissions.
    
    Args:
        forecast: (H,) predicted mean or median intensity path
        d: job length in half-hour slots
        slack: maximum allowed delay in slots
        mode: "mean" or "cautious"
        upper: (H,) optional upper bound array for cautious scheduling
        tie_break_eps: epsilon tolerance to prefer earlier starts if costs are within eps
    Returns:
        Chosen start slot offset (integer)
    """
    series = upper if (mode == "cautious" and upper is not None) else forecast
    if len(series) < d:
        return 0

    costs = np.convolve(series, np.ones(d, dtype=np.float32), mode="valid")
    # Restrict to allowed slack
    allowed_costs = costs[: slack + 1]

    if len(allowed_costs) == 0:
        return 0

    min_cost = np.min(allowed_costs)
    # Prefer earlier start if cost is within small epsilon to avoid pointless delays
    candidates = np.where(allowed_costs <= min_cost + tie_break_eps)[0]
    return int(candidates[0])


# Policy Dispatcher
def schedule_job(
    policy_name: str,
    job: Job,
    forecast: Optional[np.ndarray] = None,
    true_series: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    tie_break_eps: float = 1.0e-4,
) -> int:
    """Schedule job start time under specified policy."""
    policy_name = policy_name.lower()

    if policy_name == "run_now":
        return 0

    if policy_name == "oracle":
        if true_series is None:
            raise ValueError("Oracle policy requires true_series.")
        return pick_start(true_series, job.d, job.slack, mode="mean", tie_break_eps=tie_break_eps)

    if policy_name == "m1_cautious":
        if upper_bound is None:
            raise ValueError("m1_cautious policy requires upper_bound.")
        return pick_start(forecast if forecast is not None else upper_bound, job.d, job.slack, mode="cautious", upper=upper_bound, tie_break_eps=tie_break_eps)

    # Standard forecasters (m0, m1_median, m2, persistence, yesterday, last_week)
    if forecast is None:
        raise ValueError(f"Policy '{policy_name}' requires forecast series.")

    return pick_start(forecast, job.d, job.slack, mode="mean", tie_break_eps=tie_break_eps)
