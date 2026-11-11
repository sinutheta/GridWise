"""Training loop for M0 (MSE), M1 (Quantile), and M2 (Decision-focused) models."""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CarbonIntensityDataset, make_chronological_splits
from src.data import get_or_create_dataset
from src.features import fit_intensity_scaler
from src.losses import compute_oracle_regret, decision_loss, pinball_loss, start_mask, window_sums
from src.models import NHitsForecaster, persistence, same_time_last_week, same_time_yesterday
from src.utils import get_metadata, load_config, seed_everything

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    scaler_std: float,
) -> float:
    """Run one epoch of gradient descent updates."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    tau_init = float(loss_cfg.get("tau", 5.0))
    if loss_cfg.get("anneal_tau", False) and total_epochs > 1:
        tau_min = 1.0
        tau = tau_init * (tau_min / tau_init) ** (epoch / total_epochs)
    else:
        tau = tau_init

    lam = float(loss_cfg.get("lam", 0.1))
    d_list = loss_cfg.get("d_list", [2, 4, 8, 16])
    slack_list = loss_cfg.get("slack_list", [12, 24, 48, 96])
    quantiles = loss_cfg.get("quantiles", [0.05, 0.5, 0.95])

    for batch in loader:
        x_past = batch["x_past"].to(device)
        x_future = batch["x_future"].to(device)
        y = batch["y"].to(device)
        y_scaled = batch["y_scaled"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        out = model(x_past, x_future)

        if model_type == "m0":
            # MSE on scaled predictions
            pred = out.squeeze(-1) if out.ndim == 3 else out
            loss = torch.mean((pred - y_scaled) ** 2 * mask)
        elif model_type == "m1":
            # Pinball loss on scaled values
            loss = pinball_loss(out, y_scaled, quantiles=quantiles, mask=mask)
        elif model_type == "m2":
            # Decision regret loss
            # Scale intensities by scaler_std for meaningful tau
            pred = out.squeeze(-1) if out.ndim == 3 else out
            loss = decision_loss(
                pred=pred,
                y=y_scaled,
                d_list=d_list,
                slack_list=slack_list,
                tau=tau,
                lam=lam,
                scale_factor=1.0,  # Already normalized via scaler
                mask=mask,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    scaler: Any,
) -> Dict[str, float]:
    """Evaluate model on validation or test set."""
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    regret_sum = 0.0
    n_samples = 0

    for batch in loader:
        x_past = batch["x_past"].to(device)
        x_future = batch["x_future"].to(device)
        y = batch["y"].to(device)
        y_scaled = batch["y_scaled"].to(device)
        mask = batch["mask"].to(device)

        out = model(x_past, x_future)

        if model_type == "m1":
            # Use median (index 1 of [0.05, 0.5, 0.95]) as point forecast
            pred_scaled = out[:, :, 1]
        else:
            pred_scaled = out.squeeze(-1) if out.ndim == 3 else out

        # Invert to raw gCO2/kWh units for evaluation
        pred_raw = scaler.inverse_transform(pred_scaled.cpu()).to(device)

        # MAE and RMSE in gCO2/kWh
        abs_err = torch.abs(pred_raw - y) * mask
        sq_err = ((pred_raw - y) ** 2) * mask
        mae_sum += abs_err.sum().item()
        mse_sum += sq_err.sum().item()

        # Decision regret across full simulator job grid matching simulate.py identically:
        # d in [2, 4, 8, 16], slack in [12, 24, 48], power_kw=1.0, slot_hours=0.5
        B, H = pred_raw.shape
        d_grid = [2, 4, 8, 16]
        slack_grid = [12, 24, 48]
        tie_break_eps = 1e-4

        for d in d_grid:
            W_hat = window_sums(pred_raw, d)
            W = window_sums(y, d)
            n_starts = H - d + 1
            idx_pen = torch.arange(n_starts, dtype=pred_raw.dtype, device=device) * tie_break_eps
            for slack in slack_grid:
                m = start_mask(H, d, slack, device)
                Wh = (W_hat + idx_pen).masked_fill(~m, float("inf"))
                Wt = (W + idx_pen).masked_fill(~m, float("inf"))

                s_pred = Wh.argmin(dim=1, keepdim=True)
                s_oracle = Wt.argmin(dim=1, keepdim=True)

                e_pred = W.gather(1, s_pred).squeeze(1) * 0.5
                e_oracle = W.gather(1, s_oracle).squeeze(1) * 0.5

                reg = e_pred - e_oracle
                regret_sum += reg.sum().item()

        n_samples += mask.sum().item()

    mae = mae_sum / max(n_samples, 1)
    rmse = np.sqrt(mse_sum / max(n_samples, 1))
    total_eval_jobs = len(loader.dataset) * len(d_grid) * len(slack_grid)
    regret = regret_sum / max(total_eval_jobs, 1)

    return {"mae": mae, "rmse": rmse, "regret": regret}


def compute_mae_by_horizon(
    preds: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    horizons_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute Mean Absolute Error at specified lookahead horizons (in slots).
    
    Default horizons:
      1h  -> slot 2 (1 hour = 2 half-hours)
      6h  -> slot 12
      12h -> slot 24
      24h -> slot 48
      48h -> slot 96
    """
    if horizons_dict is None:
        horizons_dict = {
            "1h": 2,
            "6h": 12,
            "12h": 24,
            "24h": 48,
            "48h": 96,
        }

    p = np.asarray(preds)
    y = np.asarray(y_true)

    results = {}
    for name, slot in horizons_dict.items():
        idx = slot - 1
        if idx < p.shape[1] and idx < y.shape[1]:
            err = np.abs(p[:, idx] - y[:, idx])
            results[name] = float(np.nanmean(err))
    return results


def evaluate_horizon_and_leakage(
    model: nn.Module,
    val_loader: DataLoader,
    val_ds: CarbonIntensityDataset,
    device: torch.device,
    scaler: Any,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute MAE by horizon vs baselines and perform target-shuffling leakage test."""
    model.eval()

    all_y = []
    all_x_past = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            x_p = batch["x_past"].to(device)
            x_f = batch["x_future"].to(device)
            y = batch["y"].to(device)
            out = model(x_p, x_f)
            pred_scaled = out.squeeze(-1) if out.ndim == 3 else out
            pred_raw = scaler.inverse_transform(pred_scaled.cpu())

            all_preds.append(pred_raw.numpy())
            all_y.append(y.cpu().numpy())
            all_x_past.append(batch["x_past"].cpu())

    y_true = np.concatenate(all_y, axis=0)
    preds_m0 = np.concatenate(all_preds, axis=0)
    x_past_all = torch.cat(all_x_past, dim=0)

    # Compute baseline predictions
    H = val_ds.horizon
    preds_pers = scaler.inverse_transform(persistence(x_past_all, H).numpy())
    preds_yest = scaler.inverse_transform(same_time_yesterday(x_past_all, H).numpy())
    preds_last_week = scaler.inverse_transform(same_time_last_week(x_past_all, H).numpy())

    horizons = {"1h": 2, "6h": 12, "12h": 24, "24h": 48, "48h": 96}
    mae_m0 = compute_mae_by_horizon(preds_m0, y_true, horizons)
    mae_pers = compute_mae_by_horizon(preds_pers, y_true, horizons)
    mae_yest = compute_mae_by_horizon(preds_yest, y_true, horizons)
    mae_last_week = compute_mae_by_horizon(preds_last_week, y_true, horizons)

    horizon_rows = []
    for h_name in horizons.keys():
        horizon_rows.append({
            "horizon": h_name,
            "persistence": mae_pers[h_name],
            "yesterday": mae_yest[h_name],
            "last_week": mae_last_week[h_name],
            "m0": mae_m0[h_name],
        })
    horizon_df = pd.DataFrame(horizon_rows)

    print("\n" + "=" * 65)
    print("VALIDATION MAE BY HORIZON (gCO2/kWh)")
    print("=" * 65)
    print(horizon_df.to_string(index=False))
    print("=" * 65)

    # PART E: Leakage check - shuffle targets across windows and re-score M0
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_true))
    y_shuffled = y_true[perm]

    orig_mae = float(np.mean(np.abs(preds_m0 - y_true)))
    shuffled_mae = float(np.mean(np.abs(preds_m0 - y_shuffled)))

    leakage_results = {
        "original_val_mae": orig_mae,
        "shuffled_val_mae": shuffled_mae,
        "degradation_pct": ((shuffled_mae - orig_mae) / orig_mae) * 100.0,
    }

    print("\n" + "=" * 65)
    print("PART E: LEAKAGE CHECK (Target Shuffling Test)")
    print("=" * 65)
    print(f"Original Validation MAE:  {orig_mae:.2f} gCO2/kWh")
    print(f"Shuffled Validation MAE:  {shuffled_mae:.2f} gCO2/kWh")
    print(f"Degradation:              +{leakage_results['degradation_pct']:.1f}% error increase")
    print("=" * 65 + "\n")

    return horizon_df, leakage_results


def run_training(
    config_path: str,
    model_type: str = "m0",
    seed: int = 42,
    device_name: Optional[str] = None,
    save_dir: str = "checkpoints",
    log_dir: str = "results/logs",
    save_name: Optional[str] = None,
) -> Path:
    """Run full training pipeline for a given model type and random seed."""
    cfg = load_config(config_path)
    seed_everything(seed)

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training {model_type.upper()} with seed={seed} on {device}")

    # Load and split dataset
    df = get_or_create_dataset()
    data_cfg = cfg.get("data", {})
    lookback = data_cfg.get("lookback", 336)
    horizon = data_cfg.get("horizon", 96)
    include_weather = data_cfg.get("future_weather", False)

    train_df, val_df, test_df = make_chronological_splits(df)
    scaler = fit_intensity_scaler(train_df["intensity"])

    train_ds = CarbonIntensityDataset(train_df, lookback, horizon, scaler, include_weather)
    val_ds = CarbonIntensityDataset(val_df, lookback, horizon, scaler, include_weather)

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 256)
    epochs = train_cfg.get("epochs", 50)
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = train_cfg.get("early_stop_patience", 5)
    early_metric = train_cfg.get("early_stop_metric", "val_mae")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_cfg = cfg.get("model", {})
    pools = model_cfg.get("pool", [8, 4, 1])
    hidden = model_cfg.get("hidden", 256)
    dropout = model_cfg.get("dropout", 0.1)
    quantiles = model_cfg.get("quantiles", None)
    Q = len(quantiles) if quantiles else 1

    F_past = len(train_ds.past_cols)
    F_future = len(train_ds.future_cols)

    instance_norm = model_cfg.get("instance_norm", False)
    predict_change = model_cfg.get("predict_change", False)

    model = NHitsForecaster(
        lookback=lookback,
        horizon=horizon,
        in_features=F_past,
        future_features=F_future,
        pools=pools,
        hidden=hidden,
        Q=Q,
        dropout=dropout,
        instance_norm=instance_norm,
        predict_change=predict_change,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Logging setup
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    save_stem = save_name or f"{model_type}_seed_{seed}"
    log_file = Path(log_dir) / f"{save_stem}.csv"
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mae", "val_rmse", "val_regret", "lr", "epoch_time_s"])

    best_metric = float("inf")
    patience_counter = 0
    best_checkpoint_path = Path(save_dir) / f"{save_stem}_best.pt"
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    loss_cfg = cfg.get("loss", {})
    if quantiles:
        loss_cfg["quantiles"] = quantiles

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_type=model_type,
            device=device,
            loss_cfg=loss_cfg,
            epoch=epoch,
            total_epochs=epochs,
            scaler_std=scaler.std,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, model_type, device, scaler)
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_metrics['mae']:.2f} | Val Regret: {val_metrics['regret']:.2f} | Time: {epoch_time:.1f}s"
        )

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_metrics["mae"], val_metrics["rmse"], val_metrics["regret"], curr_lr, epoch_time])

        current_val = val_metrics["regret"] if early_metric == "val_regret" else val_metrics["mae"]
        if current_val < best_metric:
            best_metric = current_val
            patience_counter = 0
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "scaler": scaler.to_dict(),
                "seed": seed,
                "config": cfg,
                "metadata": get_metadata(cfg),
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved new best checkpoint to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # Load best checkpoint for post-training validation evaluations
    logger.info(f"Loading best checkpoint from {best_checkpoint_path} for evaluation...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Part C & Part E: Validation MAE by horizon and Leakage check
    evaluate_horizon_and_leakage(
        model=model,
        val_loader=val_loader,
        val_ds=val_ds,
        device=device,
        scaler=scaler,
    )

    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train carbon intensity forecaster.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model-type", type=str, default="m0", choices=["m0", "m1", "m2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        model_type=args.model_type,
        seed=args.seed,
        device_name=args.device,
        save_name=args.save_name,
    )
