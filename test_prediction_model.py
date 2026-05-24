# ===== test_prediction_model.py =====
"""
Prediction Model Test & Evaluation
===================================
Tests the trained trajectory prediction model on validation/test data.
Evaluates prediction quality and visualises results.

Prerequisites:
  • models/trajectory_predictor.pkl (trained model from main.py)
  • dataset/val/ and dataset/test/ folders (including obs_past.npy)

Usage:
    python test_prediction_model.py
    python test_prediction_model.py  # uses DEFAULT_CONFIG

Output:
  • Performance metrics (MSE, RMSE, MAE, per-feature breakdown)
  • Prediction visualisations (trajectories, errors over time)
  • Per-sample error analysis
  • Saved plots to outputs/
"""

from __future__ import annotations

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from config import DEFAULT_CONFIG
from trajectory_predictor import TrajectoryPredictor, compute_prediction_metrics
from utils import print_section, load_npy


# ─────────────────────────────────────────────────────────────────────────────
# Public API (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    """Compute full-feature and position-only evaluation metrics."""
    position_dim = 3 if predictions.shape[-1] >= 9 else 2
    return compute_prediction_metrics(predictions, ground_truth, position_dim=position_dim)


def visualize_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    split: str = "val",
    n_samples: int = 3,
    sample_indices: np.ndarray | list[int] | None = None,
    filename_prefix: str = "predictions",
) -> None:
    """
    Visualise predictions vs ground truth for a selection of samples.

    Parameters
    ----------
    predictions   : [M, future_len, features]
    ground_truth  : [M, future_len, features]
    split         : "val" or "test"
    n_samples     : number of samples to plot
    sample_indices: explicit sample indices (overrides n_samples)
    """
    os.makedirs("outputs", exist_ok=True)

    if sample_indices is None:
        sample_indices = np.arange(min(n_samples, len(predictions)))
    else:
        sample_indices = np.asarray(sample_indices, dtype=int)[:n_samples]

    n_samples = len(sample_indices)
    if n_samples == 0:
        return

    fig = plt.figure(figsize=(16, 4 * n_samples))
    fig.patch.set_facecolor("#0d1117")

    for row_idx, sample_idx in enumerate(sample_indices):
        pred = predictions[sample_idx]    # [future_len, features]
        true = ground_truth[sample_idx]
        pos_error = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)

        # Plot 1: XY trajectory
        ax = fig.add_subplot(n_samples, 3, row_idx * 3 + 1)
        ax.set_facecolor("#161b22")
        ax.plot(true[:, 0], true[:, 1], "o-", color="#58a6ff", lw=2,
                markersize=6, label="Ground truth")
        ax.plot(pred[:, 0], pred[:, 1], "s--", color="#3fb950", lw=1.5,
                markersize=5, alpha=0.8, label="Predicted")
        ax.set_xlabel("X (m)", color="#8b949e")
        ax.set_ylabel("Y (m)", color="#8b949e")
        ax.set_title(f"Sample {sample_idx}: XY Projection", color="#c9d1d9")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.2, color="#30363d")
        _style_ax(ax)

        # Plot 2: Position components over time
        t = np.arange(len(true))
        ax = fig.add_subplot(n_samples, 3, row_idx * 3 + 2)
        ax.set_facecolor("#161b22")
        ax.plot(t, true[:, 0], "o-",  color="#58a6ff", label="X true")
        ax.plot(t, pred[:, 0], "s--", color="#3fb950", alpha=0.8, label="X pred")
        ax.plot(t, true[:, 2], "^-",  color="#a371f7", label="Z true")
        ax.plot(t, pred[:, 2], "^--", color="#f85149", alpha=0.8, label="Z pred")
        ax.set_xlabel("Timestep", color="#8b949e")
        ax.set_ylabel("Position (m)", color="#8b949e")
        ax.set_title("Position Components", color="#c9d1d9")
        ax.legend(fontsize=7, framealpha=0.3)
        ax.grid(True, alpha=0.2, color="#30363d")
        _style_ax(ax)

        # Plot 3: Position error over time
        ax = fig.add_subplot(n_samples, 3, row_idx * 3 + 3)
        ax.set_facecolor("#161b22")
        ax.fill_between(t, 0, pos_error, color="#f85149", alpha=0.5)
        ax.plot(t, pos_error, "o-", color="#f85149", lw=2, markersize=5)
        mean_err = float(np.mean(pos_error))
        ax.axhline(mean_err, color="#d29922", linestyle="--", lw=1,
                   label=f"Mean: {mean_err:.3f} m")
        ax.set_xlabel("Timestep", color="#8b949e")
        ax.set_ylabel("Error (m)", color="#8b949e")
        ax.set_title(f"Position Error (Max: {np.max(pos_error):.3f} m)", color="#c9d1d9")
        ax.legend(fontsize=7, framealpha=0.3)
        ax.grid(True, alpha=0.2, color="#30363d")
        _style_ax(ax)

    plt.tight_layout()
    save_path = f"outputs/{filename_prefix}_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.close()


def get_worst_sample_indices(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_samples: int = 5,
) -> np.ndarray:
    """Return sample indices with the largest mean position error over the horizon."""
    pos_error    = np.linalg.norm(predictions[:, :, :3] - ground_truth[:, :, :3], axis=2)
    sample_error = np.mean(pos_error, axis=1)
    return np.argsort(sample_error)[-n_samples:][::-1]


def visualize_worst_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    split: str = "val",
    n_samples: int = 5,
) -> None:
    """Create plots for the worst-performing samples."""
    worst = get_worst_sample_indices(predictions, ground_truth, n_samples=n_samples)
    print(f"  Worst {len(worst)} {split} sample indices: {worst.tolist()}")
    visualize_predictions(
        predictions, ground_truth,
        split=split,
        n_samples=len(worst),
        sample_indices=worst,
        filename_prefix="worst_predictions",
    )


def plot_error_distribution(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    split: str = "val",
) -> None:
    """Plot error distribution histograms (overall + cumulative)."""
    os.makedirs("outputs", exist_ok=True)

    pos_pred  = predictions[:, :, :3]
    pos_true  = ground_truth[:, :, :3]
    pos_error = np.linalg.norm(pos_pred - pos_true, axis=2).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")

    # Histogram
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.hist(pos_error, bins=50, color="#58a6ff", alpha=0.7, edgecolor="#30363d")
    ax.axvline(np.mean(pos_error),   color="#f85149", linestyle="--", lw=2,
               label=f"Mean: {np.mean(pos_error):.3f} m")
    ax.axvline(np.median(pos_error), color="#3fb950", linestyle="--", lw=2,
               label=f"Median: {np.median(pos_error):.3f} m")
    ax.set_xlabel("Position Error (m)", color="#8b949e")
    ax.set_ylabel("Frequency", color="#8b949e")
    ax.set_title("Error Distribution", color="#c9d1d9")
    ax.legend(fontsize=9, framealpha=0.3)
    _style_ax(ax)

    # Cumulative distribution
    ax = axes[1]
    ax.set_facecolor("#161b22")
    sorted_errors = np.sort(pos_error)
    cumsum = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumsum, lw=2, color="#58a6ff")
    ax.set_xlabel("Position Error (m)", color="#8b949e")
    ax.set_ylabel("Cumulative (%)", color="#8b949e")
    ax.set_title("Cumulative Error Distribution", color="#c9d1d9")
    ax.grid(True, alpha=0.2, color="#30363d")
    _style_ax(ax)

    plt.tight_layout()
    save_path = f"outputs/error_distribution_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.close()


def _style_ax(ax) -> None:
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")


def _interpret_position_rmse(rmse: float, workspace_size: float = 10.0) -> str:
    pct = (rmse / workspace_size) * 100
    if pct < 2:
        return f"Excellent ({pct:.1f}% of workspace)"
    elif pct < 5:
        return f"Very Good ({pct:.1f}% of workspace)"
    elif pct < 10:
        return f"Good ({pct:.1f}% of workspace)"
    else:
        return f"Fair ({pct:.1f}% of workspace)"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(config: dict | None = None) -> None:
    """
    Run the full evaluation pipeline.

    Parameters
    ----------
    config : optional config dict (e.g. CUSTOM_CONFIG from main.py).
             Falls back to DEFAULT_CONFIG so the script works standalone.
    """
    # Merge caller config over DEFAULT_CONFIG so save_dir and n_obstacles are correct.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        # Shallow-merge top-level keys; for nested dicts do a deep merge.
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v

    dataset_dir = cfg["dataset"]["save_dir"]
    model_path  = "models/trajectory_predictor.pkl"

    print_section("Prediction Model Evaluation")

    # Load model
    if not os.path.exists(model_path):
        print(f"  ✗ Model not found at {model_path}")
        print(f"  Run: python main.py  (to generate dataset and train)")
        return

    print("  Loading model…")
    try:
        predictor = TrajectoryPredictor.load(model_path, device="cpu")
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return

    # ── Validation set ────────────────────────────────────────────────────────
    print_section("Validation Set Results")

    try:
        drone_past_val   = load_npy(f"{dataset_dir}/val/drone_past.npy")
        obs_past_val     = load_npy(f"{dataset_dir}/val/obs_past.npy")
        drone_future_val = load_npy(f"{dataset_dir}/val/drone_future.npy")
    except FileNotFoundError as e:
        print(f"  ✗ Dataset not found: {e}")
        return

    print(f"  Samples: {len(drone_past_val)}")
    print("\n  Making predictions…")

    predictions_val = predictor.predict(drone_past_val, obs_past_val)
    metrics_val     = evaluate_predictions(predictions_val, drone_future_val)

    _print_full_metrics(metrics_val)

    print("\n  Per-timestep full-feature RMSE:")
    for t, rmse_t in enumerate(metrics_val["full_feature_rmse_per_timestep"]):
        print(f"    Step {t+1:2d}: {rmse_t:.6f}")

    print("\n  Creating visualizations…")
    visualize_predictions(predictions_val, drone_future_val, split="val", n_samples=3)
    visualize_worst_predictions(predictions_val, drone_future_val, split="val", n_samples=5)
    plot_error_distribution(predictions_val, drone_future_val, split="val")

    # ── Test set ──────────────────────────────────────────────────────────────
    print_section("Test Set Results")

    try:
        drone_past_test   = load_npy(f"{dataset_dir}/test/drone_past.npy")
        obs_past_test     = load_npy(f"{dataset_dir}/test/obs_past.npy")
        drone_future_test = load_npy(f"{dataset_dir}/test/drone_future.npy")
    except FileNotFoundError:
        print("  ⚠️  Test set not found.")
        return

    print(f"  Samples: {len(drone_past_test)}")
    print("\n  Making predictions…")

    predictions_test = predictor.predict(drone_past_test, obs_past_test)
    metrics_test     = evaluate_predictions(predictions_test, drone_future_test)

    _print_full_metrics(metrics_test)

    print("\n  Per-timestep full-feature RMSE:")
    for t, rmse_t in enumerate(metrics_test["full_feature_rmse_per_timestep"]):
        print(f"    Step {t+1:2d}: {rmse_t:.6f}")

    print("\n  Creating visualizations…")
    visualize_predictions(predictions_test, drone_future_test, split="test", n_samples=3)
    visualize_worst_predictions(predictions_test, drone_future_test, split="test", n_samples=5)
    plot_error_distribution(predictions_test, drone_future_test, split="test")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("Summary & Interpretation")

    for label, m in (("Validation", metrics_val), ("Test", metrics_test)):
        print(f"\n  {label} Performance (Position-Only):")
        print(f"    Position RMSE  : {m['position_rmse_m']:.6f} m")
        print(f"    Mean pos error : {m['mean_position_error_m']:.6f} m")
        print(f"    P95 pos error  : {m['p95_position_error_m']:.6f} m")
        print(f"    P99 pos error  : {m['p99_position_error_m']:.6f} m")
        print(f"    Interpretation : {_interpret_position_rmse(m['position_rmse_m'])}")

    print(f"\n  Full-feature RMSE Growth (Validation):")
    step1  = metrics_val["full_feature_rmse_per_timestep"][0]
    step10 = metrics_val["full_feature_rmse_per_timestep"][-1]
    growth = metrics_val["full_feature_rmse_growth_pct"]
    print(f"    Step 1  : {step1:.6f}")
    print(f"    Step 10 : {step10:.6f}")
    print(f"    Growth  : {growth:.1f}%")
    if "velocity_rmse_growth_pct" in metrics_val:
        print(f"    Velocity growth     : {metrics_val['velocity_rmse_growth_pct']:.1f}%")
    if "acceleration_rmse_growth_pct" in metrics_val:
        print(f"    Acceleration growth : {metrics_val['acceleration_rmse_growth_pct']:.1f}%")

    if growth > 25:
        print("    ⚠️  Full-feature RMSE growth is still high")
    else:
        print("    ✓ Full-feature RMSE growth is controlled")

    print("\n✅ Evaluation complete!")
    print("   Visualizations saved to outputs/")


def _print_full_metrics(m: dict) -> None:
    print(f"\n  Metrics:")
    print(f"    Full-feature RMSE       : {m['full_feature_rmse']:.6f}")
    print(f"    Full-feature MSE        : {m['full_feature_mse']:.6f}")
    print(f"    Full-feature MAE        : {m['full_feature_mae']:.6f}")
    print(f"    Position RMSE           : {m['position_rmse_m']:.4f} m")
    print(f"    Mean Position Error     : {m['mean_position_error_m']:.4f} m")
    print(f"    First-step Pos Error    : {m['first_step_position_error_m']:.4f} m")
    print(f"    Last-step Pos Error     : {m['last_step_position_error_m']:.4f} m")
    print(f"    P90 Position Error      : {m['p90_position_error_m']:.4f} m")
    print(f"    P95 Position Error      : {m['p95_position_error_m']:.4f} m")
    print(f"    P99 Position Error      : {m['p99_position_error_m']:.4f} m")
    print(f"    Max Position Error      : {m['max_position_error_m']:.4f} m")
    print(f"    Min Position Error      : {m['min_position_error_m']:.4f} m")
    print(f"    Worst sample index      : {m['worst_sample_idx']}")
    print(f"    Worst sample mean error : {m['worst_sample_mean_position_error_m']:.4f} m")
    print(f"    Worst point sample/step : {m['worst_point_sample_idx']} / step {m['worst_timestep']+1}")
    if "velocity_rmse" in m:
        print(f"    Velocity RMSE           : {m['velocity_rmse']:.6f}")
        print(f"    Velocity Growth         : {m['velocity_rmse_growth_pct']:.1f}%")
    if "acceleration_rmse" in m:
        print(f"    Acceleration RMSE       : {m['acceleration_rmse']:.6f}")
        print(f"    Acceleration Growth     : {m['acceleration_rmse_growth_pct']:.1f}%")


if __name__ == "__main__":
    main()
