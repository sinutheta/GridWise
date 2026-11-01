"""Pause/resume job runner with state machine, atomic checkpointing, and clock abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.energy import EnergyAccountant

logger = logging.getLogger(__name__)


class RunnerState(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    PAUSED = "PAUSED"
    DONE = "DONE"


class BaseClock(ABC):
    """Abstract clock interface allowing simulation replay or real wall-clock timing."""

    @abstractmethod
    def now_slot(self) -> int:
        """Current half-hour slot index."""
        pass

    @abstractmethod
    def advance_slot(self) -> None:
        """Advance time by one slot."""
        pass

    @abstractmethod
    def get_current_intensity(self) -> float:
        """Return intensity for current slot."""
        pass


class SimulatedClock(BaseClock):
    """Simulated clock replaying pre-recorded or synthetic half-hourly intensity series."""

    def __init__(self, intensity_series: np.ndarray, start_slot: int = 0):
        self.series = intensity_series
        self.slot = start_slot

    def now_slot(self) -> int:
        return self.slot

    def advance_slot(self) -> None:
        if self.slot < len(self.series) - 1:
            self.slot += 1

    def get_current_intensity(self) -> float:
        return float(self.series[self.slot])


class RealClock(BaseClock):
    """Real wall-clock for live demo execution."""

    def __init__(self, start_slot: int = 0, fetch_fn: Optional[Callable[[], float]] = None):
        self.start_time = time.time()
        self.start_slot = start_slot
        self.fetch_fn = fetch_fn

    def now_slot(self) -> int:
        elapsed_seconds = time.time() - self.start_time
        return self.start_slot + int(elapsed_seconds // 1800)  # 30 min slots

    def advance_slot(self) -> None:
        pass

    def get_current_intensity(self) -> float:
        if self.fetch_fn:
            return self.fetch_fn()
        return 180.0


class CheckpointManager:
    """Atomic checkpointing for model weights, optimizer, step counters, and all RNG states."""

    def __init__(self, checkpoint_dir: Union[str, Path] = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.checkpoint_dir / "runner_checkpoint.pt"
        self.temp_path = self.checkpoint_dir / "runner_checkpoint.pt.tmp"

    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        metric_log: List[Dict[str, Any]],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Atomic write to temporary file then atomic rename. Returns overhead duration in seconds."""
        t0 = time.time()
        payload = {
            "step": step,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metric_log": metric_log,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "extra_data": extra_data or {},
        }

        # Write to temporary file
        torch.save(payload, self.temp_path)
        # Atomic rename
        if self.ckpt_path.exists():
            os.replace(self.temp_path, self.ckpt_path)
        else:
            self.temp_path.rename(self.ckpt_path)

        overhead_s = time.time() - t0
        logger.info(f"Saved atomic checkpoint at step {step} in {overhead_s:.3f}s")
        return overhead_s

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Restore checkpoint and restore all RNG states."""
        if not self.ckpt_path.exists():
            return None

        t0 = time.time()
        payload = torch.load(self.ckpt_path, weights_only=False)

        # Restore RNG states for strict reproducibility
        rng = payload.get("rng_states", {})
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if "numpy" in rng and rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

        logger.info(f"Loaded checkpoint from {self.ckpt_path} in {time.time() - t0:.3f}s")
        return payload


class PauseResumeRunner:
    """State machine runner orchestrating paused/resumed execution based on carbon forecasts."""

    def __init__(
        self,
        clock: BaseClock,
        forecast: np.ndarray,
        total_steps: int = 1000,
        deadline_slots: int = 24,
        steps_per_slot: int = 100,
        safety_margin_slots: int = 2,
        policy: str = "planned_slots",
        threshold_quantile: float = 0.35,
        checkpoint_manager: Optional[CheckpointManager] = None,
        energy_accountant: Optional[EnergyAccountant] = None,
    ):
        self.clock = clock
        self.forecast = forecast
        self.total_steps = total_steps
        self.deadline_slots = deadline_slots
        self.steps_per_slot = steps_per_slot
        self.safety_margin_slots = safety_margin_slots
        self.policy = policy
        self.threshold_quantile = threshold_quantile

        self.ckpt_mgr = checkpoint_manager or CheckpointManager()
        self.energy = energy_accountant or EnergyAccountant()

        self.state = RunnerState.WAITING
        self.current_step = 0
        self.metric_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    def should_run(self, current_slot: int) -> bool:
        """Evaluate carbon-aware decision rule:
        
        remaining_work = steps_left / steps_per_slot (in slots)
        time_left = deadline - current_slot
        must_run = (remaining_work >= time_left - safety_margin)
        """
        steps_left = self.total_steps - self.current_step
        remaining_slots_needed = math.ceil(steps_left / self.steps_per_slot)
        time_left = self.deadline_slots - current_slot

        # Deadline pressure override: MUST RUN
        if remaining_slots_needed >= (time_left - self.safety_margin_slots):
            return True

        # Carbon-aware slot selection policy
        available_slots = range(current_slot, min(self.deadline_slots, len(self.forecast)))
        if not available_slots:
            return True

        forecast_window = self.forecast[current_slot: self.deadline_slots]

        if self.policy == "threshold":
            cutoff = np.quantile(forecast_window, self.threshold_quantile)
            return bool(self.forecast[current_slot] <= cutoff)
        elif self.policy == "planned_slots":
            # Pick cheapest k slots before deadline
            k = min(remaining_slots_needed, len(forecast_window))
            cheapest_slots = np.argsort(forecast_window)[:k]
            # Is current slot in the cheapest k slots?
            rel_slot = 0  # current slot relative to start of window
            return bool(rel_slot in cheapest_slots)
        else:
            return True

    def step_execution(
        self,
        train_fn: Callable[[int], float],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """Execute one simulation cycle. Returns True when training is complete."""
        if self.current_step >= self.total_steps:
            self.state = RunnerState.DONE
            return True

        current_slot = self.clock.now_slot()
        run_decision = self.should_run(current_slot)
        current_intensity = self.clock.get_current_intensity()

        if run_decision:
            if self.state in [RunnerState.WAITING, RunnerState.PAUSED]:
                self.state = RunnerState.RUNNING
                self.energy.start_segment(segment_id=current_slot)

            # Perform steps for this slot
            steps_to_run = min(self.steps_per_slot, self.total_steps - self.current_step)
            for _ in range(steps_to_run):
                loss = train_fn(self.current_step)
                self.metric_log.append({"step": self.current_step, "loss": loss})
                self.current_step += 1

            self.energy.stop_segment(
                segment_id=current_slot,
                real_intensity_slots=[current_intensity],
            )

            if self.current_step >= self.total_steps:
                self.state = RunnerState.DONE
        else:
            if self.state == RunnerState.RUNNING:
                # State transition: Running -> Checkpointing -> Paused
                self.state = RunnerState.CHECKPOINTING
                overhead = 0.0
                if model is not None and optimizer is not None:
                    overhead = self.ckpt_mgr.save_checkpoint(
                        step=self.current_step,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        metric_log=self.metric_log,
                    )
                self.state = RunnerState.PAUSED
            else:
                self.state = RunnerState.PAUSED

        self.history.append({
            "slot": current_slot,
            "state": self.state.value,
            "step": self.current_step,
            "intensity": current_intensity,
            "run_decision": run_decision,
        })

        self.clock.advance_slot()
        return self.state == RunnerState.DONE
