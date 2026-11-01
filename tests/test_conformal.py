"""Unit tests for conformal quantile calibrator and coverage guarantees."""

import numpy as np
import pytest

from src.conformal import ConformalCalibrator


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(42)
    N_cal, N_test, H = 2000, 2000, 10
    alpha = 0.1  # Target 90% coverage

    # Synthetic ground truth
    y_cal = rng.normal(150.0, 20.0, size=(N_cal, H))
    y_test = rng.normal(150.0, 20.0, size=(N_test, H))

    # Uncalibrated quantiles (under-covering)
    q_lo_cal = y_cal - 15.0
    q_hi_cal = y_cal + 15.0
    q_lo_test = y_test - 15.0
    q_hi_test = y_test + 15.0

    calibrator = ConformalCalibrator(alpha=alpha)
    offsets = calibrator.fit(q_lo_cal, q_hi_cal, y_cal)

    lo_calib, hi_calib = calibrator.predict(q_lo_test, q_hi_test)
    metrics = calibrator.evaluate_coverage(lo_calib, hi_calib, y_test)

    # Empirical coverage should be close to 1 - alpha (>= 0.88)
    assert metrics["overall_coverage"] >= 0.88, f"Coverage was {metrics['overall_coverage']}"
    assert np.all(metrics["coverage_per_horizon"] >= 0.86)


def test_window_upper_bound():
    calibrator = ConformalCalibrator(alpha=0.1)
    hi = np.array([10.0, 20.0, 30.0, 40.0])
    upper_bounds = calibrator.compute_window_upper_bound(hi, d=2)
    expected = np.array([30.0, 50.0, 70.0])
    assert np.allclose(upper_bounds, expected)
