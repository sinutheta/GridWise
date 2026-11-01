"""Unit tests for scheduling policies, pick_start, and emission calculations."""

import numpy as np
import pytest

from src.scheduler import Job, compute_emissions, pick_start, schedule_job


def test_run_now_returns_zero():
    job = Job(d=4, power_kw=1.0, slack=12)
    s = schedule_job("run_now", job)
    assert s == 0


def test_start_never_exceeds_slack():
    rng = np.random.default_rng(42)
    forecast = rng.uniform(50, 300, size=96)
    job = Job(d=8, power_kw=1.0, slack=16)

    s = pick_start(forecast, d=job.d, slack=job.slack)
    assert 0 <= s <= job.slack


def test_oracle_never_has_negative_regret():
    rng = np.random.default_rng(123)
    for _ in range(20):
        true_series = rng.uniform(50, 400, size=96)
        job = Job(d=4, power_kw=2.0, slack=24)

        oracle_start = pick_start(true_series, job.d, job.slack)
        oracle_emissions = compute_emissions(true_series[oracle_start : oracle_start + job.d], job.power_kw)

        # Test against arbitrary start
        for s in range(job.slack + 1):
            e = compute_emissions(true_series[s : s + job.d], job.power_kw)
            assert e >= oracle_emissions - 1e-6, f"Oracle had higher emissions: {oracle_emissions} vs {e}"


def test_tie_breaking_prefers_earlier_start():
    # Constant series where all windows have identical emissions
    flat_series = np.ones(96) * 150.0
    s = pick_start(flat_series, d=4, slack=24, tie_break_eps=1e-4)
    assert s == 0
