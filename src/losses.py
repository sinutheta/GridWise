"""Loss functions for point forecasting, quantile regression, and decision-focused optimization."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    quantiles: List[float] = [0.05, 0.5, 0.95],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pinball / tilted loss for quantile forecasting.
    
    Args:
        pred: Predicted quantiles (B, H, Q)
        y: Ground truth target (B, H)
        quantiles: List of quantile levels Q
        mask: Optional binary validity mask (B, H)
    Returns:
        Scalar pinball loss
    """
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred  # (B, H, Q)
    loss = torch.maximum(q * e, (q - 1.0) * e)  # (B, H, Q)

    if mask is not None:
        mask_exp = mask.unsqueeze(-1)
        loss = (loss * mask_exp).sum() / (mask_exp.sum() * len(quantiles) + 1e-8)
    else:
        loss = loss.mean()

    return loss


def window_sums(x: torch.Tensor, d: int) -> torch.Tensor:
    """Calculate moving window sums of length d using avg_pool1d * d.
    
    Args:
        x: (B, T)
        d: Duration window size in slots
    Returns:
        (B, T - d + 1)
    """
    return F.avg_pool1d(x.unsqueeze(1), kernel_size=d, stride=1).squeeze(1) * float(d)


def start_mask(T: int, d: int, slack: int, device: torch.device) -> torch.Tensor:
    """Generate boolean mask for allowed start slots s in [0, min(slack, T - d)]."""
    n = T - d + 1
    idx = torch.arange(n, device=device)
    return idx <= slack


def decision_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    d_list: List[int] = [2, 4, 8, 16],
    slack_list: List[int] = [12, 24, 48, 96],
    tau: float = 5.0,
    lam: float = 0.1,
    scale_factor: float = 100.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decision-focused soft-argmin scheduling regret loss.
    
    Optimizes the model directly for the downstream scheduling task:
    minimizing expected carbon emissions under soft-argmin policy selection.
    
    Args:
        pred: Predicted intensity (B, H) or (B, H, 1)
        y: True intensity (B, H)
        d_list: Grid of job durations in slots
        slack_list: Grid of maximum allowed delay in slots
        tau: Softmax temperature parameter (in scaled intensity units)
        lam: Weight on the auxiliary MSE regularizer
        scale_factor: Normalization divisor for stable temperature behavior
        mask: Optional validity mask (B, H)
    Returns:
        Combined decision regret + lambda * MSE loss
    """
    if pred.ndim == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if y.ndim == 3 and y.size(-1) == 1:
        y = y.squeeze(-1)

    # Scale intensities for meaningful tau temperature scaling
    pred_scaled = pred / scale_factor
    y_scaled = y / scale_factor

    total_regret = 0.0
    n_terms = 0

    T = pred_scaled.size(1)

    for d in d_list:
        if d > T:
            continue
        W_hat = window_sums(pred_scaled, d)  # (B, T - d + 1)
        W = window_sums(y_scaled, d)          # (B, T - d + 1)

        for slack in slack_list:
            m = start_mask(T, d, slack, pred.device)
            if not m.any():
                continue

            # Mask out forbidden start slots beyond deadline/slack
            Wh = W_hat.masked_fill(~m, float("inf"))
            Wt = W.masked_fill(~m, float("inf"))

            # Soft-argmin probability distribution over starts
            p = torch.softmax(-Wh / tau, dim=1)

            # Expected true emissions under predicted policy
            exp_cost = (p * W.masked_fill(~m, 0.0)).sum(dim=1)

            # True oracle minimal emissions
            oracle_cost = Wt.min(dim=1).values

            regret = (exp_cost - oracle_cost).mean()
            total_regret = total_regret + regret
            n_terms += 1

    mean_regret = total_regret / max(n_terms, 1)
    # Multiply regret back to original units for logging clarity or keep scaled
    # Add auxiliary MSE term to maintain point-accuracy anchoring
    mse = F.mse_loss(pred_scaled, y_scaled)
    return mean_regret + lam * mse


def compute_oracle_regret(
    pred: torch.Tensor,
    y: torch.Tensor,
    d: int,
    slack: int,
) -> torch.Tensor:
    """Compute hard argmin scheduling regret vs true oracle for evaluation.
    
    Returns:
        (B,) regret in true emission units
    """
    if pred.ndim == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if y.ndim == 3 and y.size(-1) == 1:
        y = y.squeeze(-1)

    T = pred.size(1)
    if d > T:
        return torch.zeros(pred.size(0), device=pred.device)

    W_hat = window_sums(pred, d)
    W = window_sums(y, d)

    m = start_mask(T, d, slack, pred.device)
    Wh = W_hat.masked_fill(~m, float("inf"))
    Wt = W.masked_fill(~m, float("inf"))

    chosen_start = Wh.argmin(dim=1)
    chosen_cost = W.gather(1, chosen_start.unsqueeze(1)).squeeze(1)
    oracle_cost = Wt.min(dim=1).values

    regret = chosen_cost - oracle_cost
    return regret
