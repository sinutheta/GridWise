# Project Contribution Log

This document records the distribution of responsibilities, project milestones, and individual contributions according to the course specifications.

## Team Members & Responsibilities

| Contributor | Focus Areas | Key Deliverables |
|---|---|---|
| Contributor 1 | Data Pipeline & Uncertainty Calibration | NESO API ingestion, cleaning pipeline, windowing dataset, Conformalized Quantile Regression (`conformal.py`). |
| Contributor 2 | Deep Learning Architectures & Loss Functions | Hierarchical N-HiTS forecaster, Pinball loss (M1), Decision-focused soft-argmin regret loss (M2). |
| Contributor 3 | Decision Simulation & Policy Optimization | Policy evaluation engine (`simulate.py`), statistical baselines, bootstrap confidence intervals, grid ablation. |
| Contributor 4 | Execution Infrastructure & Streamlit Demo | Pause/resume state machine (`runner.py`), CodeCarbon energy accounting (`energy.py`), Streamlit frontend (`streamlit_app.py`). |

---

## Milestone Timeline

- **Milestone 1 (Scaffold & Pipeline)**: Architectural repository layout, data fetching/cleaning, feature engineering, unit tests.
- **Milestone 2 (Forecasting Models)**: M0 (MSE), M1 (Quantile pinball), and M2 (Decision-focused) training across 5 random seeds.
- **Milestone 3 (Uncertainty & Decision Layer)**: Calibrated conformal prediction intervals, cautious scheduling policy, full simulation grid.
- **Milestone 4 (Hardware & Execution Verification)**: Simulated and wall-clock pause/resume runner, CodeCarbon measurements, metric parity check.
- **Milestone 5 (Interactive Demo & Report)**: Streamlit horizon planner, figures, tables, and academic report preparation.
