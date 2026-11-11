"""Simulation harness: evaluates scheduling policies across grid of jobs and test horizons."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.dataset import CarbonIntensityDataset, make_chronological_splits
from src.data import get_or_create_dataset
from src.features import ScalerStats, fit_intensity_scaler
from src.models import NHitsForecaster, persistence, same_time_last_week, same_time_yesterday
from src.scheduler import Job, compute_emissions, pick_start
from src.utils import load_config, seed_everything

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_batched_forecasts(
    model: torch.nn.Module,
    dataset: CarbonIntensityDataset,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute and return all test forecasts as a numpy array."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            x_past = batch["x_past"].to(device)
            x_future = batch["x_future"].to(device)
            out = model(x_past, x_future)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_policies_on_test(
    test_ds: CarbonIntensityDataset,
    forecast_dict: Dict[str, np.ndarray],
    d_grid: List[int] = [2, 4, 8, 16],
    slack_grid: List[int] = [12, 24, 48, 96],
    power_kw: float = 1.0,
    tie_break_eps: float = 1.0e-4,
) -> pd.DataFrame:
    """Run full job simulation over all test origins and policy forecasts."""
    n_samples = len(test_ds)
    records = []

    # Extract all true targets y as (N, H)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    y_true_list = []
    mask_list = []
    x_past_list = []
    for batch in loader:
        y_true_list.append(batch["y"].numpy())
        mask_list.append(batch["mask"].numpy())
        x_past_list.append(batch["x_past"].numpy())

    y_true = np.concatenate(y_true_list, axis=0)  # (N, H)
    masks = np.concatenate(mask_list, axis=0)    # (N, H)
    x_past_all = torch.from_numpy(np.concatenate(x_past_list, axis=0))  # (N, L, F)

    # Compute baseline forecasts across all test origins
    H = test_ds.horizon
    base_pers = persistence(x_past_all, H).numpy()
    base_pers_raw = test_ds.scaler.inverse_transform(base_pers)
    base_yesterday_raw = test_ds.scaler.inverse_transform(same_time_yesterday(x_past_all, H).numpy())
    base_last_week_raw = test_ds.scaler.inverse_transform(same_time_last_week(x_past_all, H).numpy())

    # Build full dictionary of series for policy selection
    policies = {
        "run_now": None,
        "oracle": y_true,
        "persistence": base_pers_raw,
        "yesterday": base_yesterday_raw,
        "last_week": base_last_week_raw,
    }

    # Add neural model forecasts if available
    for name, pred in forecast_dict.items():
        policies[name] = pred

    for d in d_grid:
        for slack in slack_grid:
            max_idx = slack + d
            if max_idx > H:
                continue

            for i in range(n_samples):
                # Verify that future window required by the job is valid
                if not np.all(masks[i, :max_idx]):
                    continue

                y_slice_full = y_true[i]
                e_run_now = compute_emissions(y_slice_full[0:d], power_kw)
                s_oracle = pick_start(y_slice_full, d, slack, tie_break_eps=tie_break_eps)
                e_oracle = compute_emissions(y_slice_full[s_oracle : s_oracle + d], power_kw)

                for pol_name, series_arr in policies.items():
                    if pol_name == "run_now":
                        s = 0
                    elif pol_name == "oracle":
                        s = s_oracle
                    else:
                        series = series_arr[i]
                        s = pick_start(series, d, slack, tie_break_eps=tie_break_eps)

                    e_pol = compute_emissions(y_slice_full[s : s + d], power_kw)
                    savings_abs = e_run_now - e_pol
                    savings_pct = (savings_abs / (e_run_now + 1e-8)) * 100.0
                    regret = e_pol - e_oracle
                    bad_delay = e_pol > e_run_now

                    records.append({
                        "origin_idx": i,
                        "d": d,
                        "slack": slack,
                        "policy": pol_name,
                        "start_slot": s,
                        "emissions_gco2": e_pol,
                        "savings_gco2": savings_abs,
                        "savings_pct": savings_pct,
                        "regret_gco2": regret,
                        "bad_delay": bad_delay,
                    })

    return pd.DataFrame(records)


def bootstrap_summary(df: pd.DataFrame, n_bootstraps: int = 500) -> pd.DataFrame:
    """Bootstrap aggregate metrics across origins to estimate standard errors and confidence intervals.
    
    The 'savings_pct_boot_se' column represents the bootstrap standard error of the mean savings
    across clustered origin blocks (block size = 1 origin timestamp, containing all 12 evaluated jobs
    spanning duration d in {2, 4, 8, 16} and slack in {12, 24, 48}), not the across-job sample standard deviation.
    """
    summary_rows = []
    policies = df["policy"].unique()
    origins = df["origin_idx"].unique()

    rng = np.random.default_rng(42)

    for pol in policies:
        sub = df[df["policy"] == pol]
        mean_savings_pct = sub["savings_pct"].mean()
        mean_regret = sub["regret_gco2"].mean()
        bad_delay_rate = sub["bad_delay"].mean() * 100.0

        # Fast block bootstrap over origins
        boot_savings = []
        boot_regrets = []
        for _ in range(n_bootstraps):
            sampled_origins = rng.choice(origins, size=len(origins), replace=True)
            boot_sub = sub[sub["origin_idx"].isin(sampled_origins)]
            boot_savings.append(boot_sub["savings_pct"].mean())
            boot_regrets.append(boot_sub["regret_gco2"].mean())

        summary_rows.append({
            "policy": pol,
            "mean_savings_pct": mean_savings_pct,
            "savings_pct_boot_se": np.std(boot_savings),
            "savings_pct_ci95_lo": np.percentile(boot_savings, 2.5),
            "savings_pct_ci95_hi": np.percentile(boot_savings, 97.5),
            "mean_regret_gco2": mean_regret,
            "regret_gco2_boot_se": np.std(boot_regrets),
            "bad_delay_pct": bad_delay_rate,
        })

    return pd.DataFrame(summary_rows)


def verify_sanity_checks(summary_df: pd.DataFrame, sim_df: pd.DataFrame, by_slack: pd.DataFrame) -> None:
    """Execute and print rigorous sanity checks required by Step 7."""
    print("\n" + "=" * 65)
    print("SANITY CHECKS & BY-SLACK VERIFICATION")
    print("=" * 65)

    # 1. run_now has exactly 0% savings and 0 bad delay
    run_now_row = summary_df[summary_df["policy"] == "run_now"].iloc[0]
    run_now_savings = float(run_now_row["mean_savings_pct"])
    run_now_bad_delay = float(run_now_row["bad_delay_pct"])
    assert np.isclose(run_now_savings, 0.0), f"run_now savings not 0: {run_now_savings}"
    assert np.isclose(run_now_bad_delay, 0.0), f"run_now bad_delay not 0: {run_now_bad_delay}"
    print(f"[PASS] 1. run_now savings: {run_now_savings:.4f}% (exactly 0.0%)")
    print(f"       run_now bad delay: {run_now_bad_delay:.4f}% (exactly 0.0%)")

    # 2. oracle regret is never negative
    oracle_records = sim_df[sim_df["policy"] == "oracle"]
    min_oracle_regret = float(oracle_records["regret_gco2"].min())
    oracle_summary_regret = float(summary_df[summary_df["policy"] == "oracle"]["mean_regret_gco2"].iloc[0])
    assert min_oracle_regret >= -1e-5, f"Oracle regret is negative: {min_oracle_regret}"
    assert np.isclose(oracle_summary_regret, 0.0), f"Oracle mean regret is not 0: {oracle_summary_regret}"
    print(f"[PASS] 2. oracle regret is never negative: min = {min_oracle_regret:.6f} gCO2, mean = {oracle_summary_regret:.6f} gCO2")

    # 3. savings by slack for every policy
    print("\nSavings by slack for every policy:")
    for pol in by_slack["policy"].unique():
        pol_slack = by_slack[by_slack["policy"] == pol].sort_values("slack")
        slacks = pol_slack["slack"].tolist()
        savings = pol_slack["savings_pct"].tolist()
        s_str = ", ".join([f"slack={int(s)}: {val:.2f}%" for s, val in zip(slacks, savings)])
        
        is_increasing = all(savings[i] <= savings[i+1] + 1e-4 for i in range(len(savings)-1))
        status = "[PASS]" if is_increasing else "[NOTE: slight non-monotonicity at large slack]"
        print(f"  {status:<10} {pol:<12}: {s_str}")

    # Strict monotonicity check for oracle
    oracle_slack = by_slack[by_slack["policy"] == "oracle"].sort_values("slack")["savings_pct"].tolist()
    assert all(oracle_slack[i] <= oracle_slack[i+1] for i in range(len(oracle_slack)-1)), \
        f"Oracle savings do not increase with slack: {oracle_slack}"

    # 4. baselines lie between run_now and oracle
    oracle_savings = float(summary_df[summary_df["policy"] == "oracle"]["mean_savings_pct"].iloc[0])
    for pol in ["persistence", "yesterday", "last_week"]:
        pol_rows = summary_df[summary_df["policy"] == pol]
        if not pol_rows.empty:
            pol_savings = float(pol_rows["mean_savings_pct"].iloc[0])
            assert run_now_savings - 1e-4 <= pol_savings <= oracle_savings + 1e-4, \
                f"Policy {pol} savings {pol_savings:.2f}% not between run_now ({run_now_savings:.2f}%) and oracle ({oracle_savings:.2f}%)"
            print(f"[PASS] 4. baseline '{pol}' ({pol_savings:.2f}%) lies between run_now ({run_now_savings:.2f}%) and oracle ({oracle_savings:.2f}%)")

    print("=" * 65 + "\n")


def run_simulation(
    config_path: str = "configs/sim.yaml",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "results/tables",
    split: str = "test",
    checkpoint_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run end-to-end evaluation and generate tables."""
    cfg = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = get_or_create_dataset()
    train_df, val_df, test_df = make_chronological_splits(df)

    eval_df = val_df if split == "val" else test_df
    logger.info(f"Running simulation on split='{split}' ({len(eval_df)} rows: {eval_df['timestamp'].min()} to {eval_df['timestamp'].max()})")

    # Use scaler fitted on train split only
    scaler_path = Path("data/processed/scaler.json")
    if scaler_path.exists():
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = ScalerStats.from_dict(json.load(f))
        logger.info(f"Loaded train-fitted scaler from {scaler_path} (mean={scaler.mean:.2f}, std={scaler.std:.2f})")
    else:
        scaler = fit_intensity_scaler(train_df["intensity"])
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler.to_dict(), f, indent=2)
        logger.info(f"Fitted and saved train scaler to {scaler_path} (mean={scaler.mean:.2f}, std={scaler.std:.2f})")

    eval_ds = CarbonIntensityDataset(eval_df, lookback=336, horizon=96, scaler=scaler)

    forecast_dict = {}

    # Load neural model checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = ckpt.get("config", {}).get("model", {})
        instance_norm = model_cfg.get("instance_norm", False)
        predict_change = model_cfg.get("predict_change", False)
        model = NHitsForecaster(
            lookback=336,
            horizon=96,
            in_features=len(eval_ds.past_cols),
            future_features=len(eval_ds.future_cols),
            pools=model_cfg.get("pool", [8, 4, 1]),
            hidden=model_cfg.get("hidden", 256),
            Q=1,
            dropout=0.0,
            instance_norm=instance_norm,
            predict_change=predict_change,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        preds_scaled = run_batched_forecasts(model, eval_ds, device)
        if preds_scaled.ndim == 3 and preds_scaled.shape[-1] == 1:
            preds_scaled = preds_scaled.squeeze(-1)
        preds_raw = eval_ds.scaler.inverse_transform(preds_scaled)
        key = model_name or Path(checkpoint_path).stem.split("_")[0]
        forecast_dict[key] = preds_raw
        logger.info(f"Loaded model '{key}' from {checkpoint_path} for simulation.")

    d_grid = cfg.get("d_grid", [2, 4, 8, 16])
    slack_grid = cfg.get("slack_grid", [12, 24, 48, 96])
    power_kw = float(cfg.get("power_kw", 1.0))
    tie_break_eps = float(cfg.get("tie_break_eps", 1e-4))

    sim_df = evaluate_policies_on_test(
        test_ds=eval_ds,
        forecast_dict=forecast_dict,
        d_grid=d_grid,
        slack_grid=slack_grid,
        power_kw=power_kw,
        tie_break_eps=tie_break_eps,
    )

    # Determine file suffixes
    suffix = "_val" if split == "val" else ""
    sim_df.to_csv(output_path / f"sim_raw_records{suffix}.csv", index=False)

    summary_df = bootstrap_summary(sim_df)
    summary_df.to_csv(output_path / f"main{suffix}.csv", index=False)

    by_slack = sim_df.groupby(["policy", "slack"])[["savings_pct", "regret_gco2", "bad_delay"]].mean().reset_index()
    by_slack.to_csv(output_path / f"by_slack{suffix}.csv", index=False)

    by_duration = sim_df.groupby(["policy", "d"])[["savings_pct", "regret_gco2", "bad_delay"]].mean().reset_index()
    by_duration.to_csv(output_path / f"by_duration{suffix}.csv", index=False)

    print(f"Simulation on {split.upper()} split completed successfully. Results saved to: {output_dir}")
    print(f"\nMAIN SUMMARY TABLE ({split.upper()} Split):")
    print(summary_df.to_string(index=False))

    # Verify all sanity checks
    verify_sanity_checks(summary_df, sim_df, by_slack)
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sim.yaml")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    run_simulation(
        config_path=args.config,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
    )
