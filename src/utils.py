"""Common utilities for seed management, config loading, and logging."""

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch RNGs for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash or return placeholder."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unversioned"


def get_metadata(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Collect runtime environment metadata for checkpoint provenance."""
    import pandas as pd

    return {
        "git_commit": get_git_commit_hash(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config or {},
    }
