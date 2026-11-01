"""Conformalized Quantile Regression (CQR) and uncertainty quantification."""

from typing import Dict, Optional, Tuple, Union

import numpy as np


class ConformalCalibrator:
    """Conformalized Quantile Regression (CQR) for multi-horizon forecasts."""

    def __init__(self, alpha: float = 0.1, method: str = "higher"):
        self.alpha = alpha
        self.method = method
        self.offsets: Optional[np.ndarray] = None  # (H,)

    def fit(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute horizon-specific conformal non-conformity offsets on calibration data.
        
        Args:
            q_lo: Predicted lower quantile array (N, H)
            q_hi: Predicted upper quantile array (N, H)
            y: Ground truth array (N, H)
            
        Returns:
            offsets: (H,) calibration quantiles Q_h
        """
        # Conformity scores: s_{i,h} = max(q_lo - y, y - q_hi)
        scores = np.maximum(q_lo - y, y - q_hi)
        n = scores.shape[0]

        # Empirical quantile level with finite-sample correction
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.offsets = np.quantile(scores, level, axis=0, method=self.method)
        return self.offsets

    def predict(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        offsets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibrated conformal offsets to produce guaranteed prediction intervals.
        
        Returns:
            calibrated_lo, calibrated_hi (both same shape as q_lo)
        """
        if offsets is None:
            if self.offsets is None:
                raise ValueError("Calibrator has not been fitted yet and no offsets provided.")
            offsets = self.offsets

        calibrated_lo = q_lo - offsets
        calibrated_hi = q_hi + offsets
        return calibrated_lo, calibrated_hi

    def evaluate_coverage(
        self,
        q_lo_cal: np.ndarray,
        q_hi_cal: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate empirical coverage and average interval width on test set."""
        covered = (y_test >= q_lo_cal) & (y_test <= q_hi_cal)
        coverage_per_horizon = np.mean(covered, axis=0)  # (H,)
        overall_coverage = float(np.mean(covered))
        width_per_horizon = np.mean(q_hi_cal - q_lo_cal, axis=0)
        overall_width = float(np.mean(q_hi_cal - q_lo_cal))

        return {
            "overall_coverage": overall_coverage,
            "overall_width": overall_width,
            "coverage_per_horizon": coverage_per_horizon,
            "width_per_horizon": width_per_horizon,
        }

    def compute_window_upper_bound(
        self,
        calibrated_hi: np.ndarray,
        d: int,
    ) -> np.ndarray:
        """Compute conservative upper bound on cumulative window emissions of length d.
        
        Args:
            calibrated_hi: 1D array of horizon upper endpoints (H,)
            d: duration in slots
        Returns:
            window_costs: (H - d + 1,)
        """
        return np.convolve(calibrated_hi, np.ones(d), mode="valid")
