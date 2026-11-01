"""Unit tests for pinball and decision-focused regret losses."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses import compute_oracle_regret, decision_loss, pinball_loss, window_sums


def test_pinball_loss_manual_calculation():
    # Setup simple pred and y
    # y = 10.0, pred = 8.0 -> error e = y - pred = 2.0
    # For q = 0.5: loss = max(0.5*2, -0.5*2) = 1.0
    # For q = 0.9: loss = max(0.9*2, -0.1*2) = 1.8
    # For q = 0.1: loss = max(0.1*2, -0.9*2) = 0.2
    y = torch.tensor([[10.0]])
    pred = torch.tensor([[[8.0, 8.0, 8.0]]])  # (B=1, H=1, Q=3)
    quantiles = [0.1, 0.5, 0.9]

    loss = pinball_loss(pred, y, quantiles=quantiles)
    expected_mean = (0.2 + 1.0 + 1.8) / 3.0
    assert torch.isclose(loss, torch.tensor(expected_mean), atol=1e-5)


def test_window_sums_convolution():
    # Moving sum of length 3 on [1, 2, 3, 4, 5]
    # Window sums: [1+2+3=6, 2+3+4=9, 3+4+5=12]
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sums = window_sums(x, d=3)
    expected = torch.tensor([[6.0, 9.0, 12.0]])
    assert torch.allclose(sums, expected, atol=1e-5)


def test_decision_loss_zero_regret_when_pred_equals_y():
    B, H = 4, 96
    y = torch.sin(torch.linspace(0, 10, H)).unsqueeze(0).repeat(B, 1) + 2.0
    pred = y.clone()

    # When pred == y, the soft-argmin probability matches the true cost curve
    # Regret should be strictly non-negative and very small (close to 0)
    # With lam = 0, the loss should only reflect soft-regret
    loss_val = decision_loss(
        pred=pred,
        y=y,
        d_list=[4, 8],
        slack_list=[12, 24],
        tau=0.01,  # Low temperature approaches hard argmin
        lam=0.0,
        scale_factor=1.0,
    )
    assert loss_val.item() >= 0.0
    assert loss_val.item() < 0.05


def test_decision_loss_slack_zero_gives_zero_regret():
    B, H = 2, 48
    y = torch.randn(B, H)
    pred = torch.randn(B, H)

    # When slack = 0, slot 0 is the only option, so regret vs oracle is identically 0.0
    loss_val = decision_loss(
        pred=pred,
        y=y,
        d_list=[4],
        slack_list=[0],
        tau=1.0,
        lam=0.0,
    )
    assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-6)


def test_decision_loss_gradients_flow_without_nan():
    B, H = 2, 96
    pred = torch.randn(B, H, requires_grad=True)
    y = torch.randn(B, H)

    loss = decision_loss(
        pred=pred,
        y=y,
        d_list=[2, 4],
        slack_list=[12, 24],
        tau=5.0,
        lam=0.1,
    )
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()
