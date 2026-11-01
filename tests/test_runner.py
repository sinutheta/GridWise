"""Unit tests for pause/resume runner, deadline adherence, and checkpoint roundtrips."""

import os
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from src.runner import (
    CheckpointManager,
    PauseResumeRunner,
    RunnerState,
    SimulatedClock,
)


def test_deadline_always_met():
    # 24 slot deadline, 10 slots of work required
    intensity_series = np.ones(48) * 150.0
    clock = SimulatedClock(intensity_series)

    runner = PauseResumeRunner(
        clock=clock,
        forecast=intensity_series,
        total_steps=1000,
        deadline_slots=20,
        steps_per_slot=100,  # 10 slots needed
        safety_margin_slots=2,
        policy="planned_slots",
    )

    def dummy_train_fn(step):
        return 0.1

    # Step runner until done
    for _ in range(30):
        done = runner.step_execution(dummy_train_fn)
        if done:
            break

    assert runner.current_step == 1000
    assert runner.state == RunnerState.DONE
    # Check that job finished on or before deadline
    assert clock.now_slot() <= 20


def test_checkpoint_roundtrip_restores_rng(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path)

    # Set specific RNG seeds and draw samples
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)

    # Save state
    dummy_model = torch.nn.Linear(2, 2)
    dummy_opt = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

    mgr.save_checkpoint(
        step=50,
        model_state=dummy_model.state_dict(),
        optimizer_state=dummy_opt.state_dict(),
        metric_log=[{"step": 50, "loss": 0.5}],
    )

    # Advance RNGs
    val_py_before = random.random()
    val_np_before = np.random.rand()
    val_torch_before = torch.rand(1).item()

    # Reset seeds to something different
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)

    # Now load checkpoint
    loaded = mgr.load_checkpoint()
    assert loaded is not None
    assert loaded["step"] == 50

    # The next random numbers generated should match val_*_before exactly!
    val_py_after = random.random()
    val_np_after = np.random.rand()
    val_torch_after = torch.rand(1).item()

    assert val_py_before == val_py_after
    assert val_np_before == val_np_after
    assert val_torch_before == val_torch_after
