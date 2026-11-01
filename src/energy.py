"""Energy accounting and time-resolved emissions tracking using CodeCarbon."""

from dataclasses import dataclass
import logging
import platform
import time
from typing import Dict, List, Optional, Union

import numpy as np
import psutil

logger = logging.getLogger(__name__)

# CodeCarbon import with fallback
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    logger.warning("CodeCarbon not installed. Using fallback energy tracker.")


@dataclass
class EnergySegment:
    """Represents an active compute segment and its energy/carbon measurements."""
    segment_id: int
    duration_s: float
    energy_kwh: float
    mean_intensity_gco2: float
    time_resolved_gco2: float
    codecarbon_gco2: Optional[float] = None
    checkpoint_overhead_s: float = 0.0


class EnergyAccountant:
    """Manages active energy measurement and time-resolved grid emissions attribution."""

    def __init__(
        self,
        measure_power_secs: int = 5,
        default_power_watts: float = 150.0,
        output_dir: str = "results/logs",
    ):
        self.measure_power_secs = measure_power_secs
        self.default_power_watts = default_power_watts
        self.output_dir = output_dir
        self.segments: List[EnergySegment] = []
        self._current_tracker = None
        self._start_time = None

    def start_segment(self, segment_id: int) -> None:
        """Start tracking energy for an active compute slice."""
        self._start_time = time.time()
        if CODECARBON_AVAILABLE:
            try:
                self._current_tracker = EmissionsTracker(
                    measure_power_secs=self.measure_power_secs,
                    output_dir=self.output_dir,
                    save_to_file=False,
                    log_level="warning",
                )
                self._current_tracker.start()
            except Exception as e:
                logger.warning(f"Could not start CodeCarbon tracker: {e}. Falling back to estimate.")
                self._current_tracker = None
        else:
            self._current_tracker = None

    def stop_segment(
        self,
        segment_id: int,
        real_intensity_slots: Union[List[float], np.ndarray],
        checkpoint_overhead_s: float = 0.0,
    ) -> EnergySegment:
        """Stop tracking, compute kWh and time-resolved emissions."""
        elapsed_s = time.time() - (self._start_time or time.time())
        codecarbon_emissions = None
        energy_kwh = 0.0

        if self._current_tracker is not None:
            try:
                emissions_kg = self._current_tracker.stop()
                if emissions_kg is not None:
                    codecarbon_emissions = float(emissions_kg * 1000.0)  # kg to gCO2
                # In CodeCarbon, total energy consumed is in tracker._total_energy.kWh
                if hasattr(self._current_tracker, "_total_energy"):
                    energy_kwh = float(self._current_tracker._total_energy.kWh)
            except Exception as e:
                logger.warning(f"Error stopping CodeCarbon tracker: {e}")

        # Fallback energy calculation if hardware measurement was unavailable or 0
        if energy_kwh <= 0.0:
            energy_kwh = (self.default_power_watts / 1000.0) * (elapsed_s / 3600.0)

        # Time-resolved grid carbon calculation:
        # CO2 = kWh_segment * mean(intensity over that segment's real slots)
        intensities = np.asarray(real_intensity_slots, dtype=np.float32)
        mean_intensity = float(np.mean(intensities)) if len(intensities) > 0 else 180.0
        time_resolved_gco2 = energy_kwh * mean_intensity

        segment = EnergySegment(
            segment_id=segment_id,
            duration_s=elapsed_s,
            energy_kwh=energy_kwh,
            mean_intensity_gco2=mean_intensity,
            time_resolved_gco2=time_resolved_gco2,
            codecarbon_gco2=codecarbon_emissions,
            checkpoint_overhead_s=checkpoint_overhead_s,
        )
        self.segments.append(segment)
        self._current_tracker = None
        return segment

    def get_summary(self) -> Dict[str, float]:
        """Aggregate total energy, emissions, and overheads across all active segments."""
        if not self.segments:
            return {"total_energy_kwh": 0.0, "total_emissions_gco2": 0.0, "total_overhead_s": 0.0}

        total_kwh = sum(s.energy_kwh for s in self.segments)
        total_gco2 = sum(s.time_resolved_gco2 for s in self.segments)
        total_overhead = sum(s.checkpoint_overhead_s for s in self.segments)
        return {
            "total_energy_kwh": total_kwh,
            "total_emissions_gco2": total_gco2,
            "total_overhead_s": total_overhead,
            "num_segments": len(self.segments),
        }

    @staticmethod
    def get_system_hardware_info() -> Dict[str, str]:
        """Capture host CPU, RAM, and GPU platform information for reproducibility."""
        info = {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "ram_gb": f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB",
            "gpu": "None",
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return info
