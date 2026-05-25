<<<<<<< HEAD
import sys
import os

# STEP 1: Tell Python WHERE to look (The "Bridge")
# This MUST happen before you try to import from simulation
project_root = os.path.dirname(__file__)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# STEP 2: Now that the bridge is built, import your function
# Use the function name you actually defined in drone_env.py
try:
    from simulation.drone_env import start_sim
except ImportError as e:
    print(f"Import Error: {e}")
    print("Check if src/simulation/__init__.py exists!")
    sys.exit(1)

# STEP 3: The Start Button
if __name__ == "__main__":
    print("--- Project: Drone Trajectory Prediction & Control ---")
    print("Initializing Simulation...")
    start_sim()
=======
# ===== main.py =====
"""
Main Entry Point — Drone Simulation & Dataset Pipeline
=======================================================
Demonstrates the full pipeline end-to-end:

    1. Generate a single drone trajectory
    2. Simulate dynamic obstacles
    3. Build a labelled dataset (train / val / test)
    4. Save dataset to disk (.npy + .csv)
    5. Format model inputs
    6. Run the lightweight simulator & visualise
    7. Train trajectory predictor (model v6)
    8. Evaluate the trained model

Run with:
    python main.py
"""

import os
import sys
import pickle
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_CONFIG
from trajectory_generator import generate_trajectory, get_feature_names, TRAJECTORY_GENERATOR_VERSION
from obstacle_simulation import simulate_obstacles, ObstacleSimulator
from dataset_generator import generate_dataset, save_dataset, get_model_input
from simulator import run_simulation, visualise_simulation
from trajectory_predictor import TrajectoryPredictor
from utils import print_section, set_seed, load_npy


# ─────────────────────────────────────────────────────────────────────────────
# Custom config
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CONFIG = {
    "simulation_time": 8.0,
    "time_step"      : 0.05,
    "random_seed"    : 42,

    "trajectory": {
        "mode"        : "3d",
        "n_waypoints" : 8,
        "max_velocity": 3.0,
    },

    "obstacles": {
        "n_obstacles": 4,
        "types"      : ["linear", "circular", "random_walk"],
        "max_speed"  : 1.5,
    },

    "dataset": {
        "seq_len"       : 20,
        "future_len"    : 10,
        "n_trajectories": 500,
        "train_ratio"   : 0.8,
        "val_ratio"     : 0.1,
        "save_dir"      : "dataset",
    },
}

MODEL_PATH = "models/trajectory_predictor.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility guards
# ─────────────────────────────────────────────────────────────────────────────

def dataset_exists() -> bool:
    save_dir  = CUSTOM_CONFIG["dataset"]["save_dir"]
    meta_path = f"{save_dir}/meta.json"
    required  = [
        f"{save_dir}/train/drone_past.npy",
        f"{save_dir}/train/drone_future.npy",
        f"{save_dir}/train/obs_past.npy",
        f"{save_dir}/val/drone_past.npy",
        f"{save_dir}/val/drone_future.npy",
        f"{save_dir}/test/drone_past.npy",
        f"{save_dir}/test/drone_future.npy",
        meta_path,
    ]
    if not all(os.path.exists(p) for p in required):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return (
            meta.get("generation_version") == TRAJECTORY_GENERATOR_VERSION
            and meta.get("seq_len")        == CUSTOM_CONFIG["dataset"]["seq_len"]
            and meta.get("future_len")     == CUSTOM_CONFIG["dataset"]["future_len"]
            and meta.get("n_obstacles")    == CUSTOM_CONFIG["obstacles"]["n_obstacles"]
        )
    except Exception:
        return False


def model_exists() -> bool:
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            state = pickle.load(f)
        return (
            state.get("format_version") == 12
            and state.get("target_mode") == (
                "relative_position_normalized_state_smooth_dynamic_labels_v12"
            )
            and state.get("seq_len")     == CUSTOM_CONFIG["dataset"]["seq_len"]
            and state.get("future_len")  == CUSTOM_CONFIG["dataset"]["future_len"]
            and state.get("features")    == 9
            and state.get("n_obstacles") == CUSTOM_CONFIG["obstacles"]["n_obstacles"]
        )
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Single trajectory
# ─────────────────────────────────────────────────────────────────────────────

def demo_trajectory() -> None:
    print_section("Step 1 · Trajectory Generator")

    set_seed(42)
    traj = generate_trajectory(CUSTOM_CONFIG)

    print(f"  Trajectory shape : {traj.shape}")
    print(f"  Features         : {get_feature_names('3d')}")
    print(f"  First step       : {traj[0]}")
    print(f"  Last  step       : {traj[-1]}")
    print(f"  Pos range  X     : [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}]")
    print(f"  Pos range  Y     : [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}]")
    print(f"  Pos range  Z     : [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Obstacle simulation
# ─────────────────────────────────────────────────────────────────────────────

def demo_obstacles() -> None:
    print_section("Step 2 · Obstacle Simulator")

    obs_history = simulate_obstacles(CUSTOM_CONFIG)
    T, N, feat  = obs_history.shape

    print(f"  History shape    : {obs_history.shape}  (T={T}, N_obs={N}, features={feat})")
    print(f"  Obstacle 0 at t=0: pos={obs_history[0, 0, :3]}  vel={obs_history[0, 0, 3:]}")
    print(f"  Obstacle 0 at t=T: pos={obs_history[-1, 0, :3]}  vel={obs_history[-1, 0, 3:]}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def demo_dataset() -> dict:
    print_section("Step 3 · Dataset Generator")

    dataset = generate_dataset(CUSTOM_CONFIG)

    print(f"\n  Metadata:")
    for k, v in dataset["meta"].items():
        print(f"    {k:30s}: {v}")

    sample = dataset["train"][0]
    print(f"\n  Sample keys      : {list(sample.keys())}")
    print(f"  drone_past shape : {sample['drone_past'].shape}")
    print(f"  obs_past shape   : {sample['obs_past'].shape}")
    print(f"  drone_future shp : {sample['drone_future'].shape}")
    print(f"  action           : {sample['action']}")
    print(f"  had_collision    : {sample['had_collision']}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Save dataset
# ─────────────────────────────────────────────────────────────────────────────

def demo_save(dataset: dict) -> None:
    print_section("Step 4 · Save Dataset")

    save_dataset(dataset, CUSTOM_CONFIG)

    reloaded = np.load("dataset/train/drone_past.npy", allow_pickle=True)
    print(f"  Reloaded drone_past shape: {reloaded.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Model input formatting
# ─────────────────────────────────────────────────────────────────────────────

def demo_model_input(dataset: dict) -> None:
    print_section("Step 5 · Model Input Formatter")

    sample    = dataset["train"][0]
    formatted = get_model_input(sample)

    print("  ── Member 2 (Prediction model) input ──")
    print(f"    x_pred   shape : {formatted['x_pred'].shape}")
    print(f"    y_future shape : {formatted['y_future'].shape}")

    print("  ── Member 3 (Control / Imitation model) input ──")
    print(f"    x_ctrl   shape : {formatted['x_ctrl'].shape}")
    print(f"    y_action shape : {formatted['y_action'].shape}")
    print(f"    action         : {formatted['y_action']}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Lightweight simulator + visualisation
# ─────────────────────────────────────────────────────────────────────────────

def demo_simulation() -> None:
    print_section("Step 6 · Lightweight Simulator")

    record = run_simulation(CUSTOM_CONFIG)

    print(f"  Drone states shape     : {record.drone_states.shape}")
    print(f"  Obstacle history shape : {record.obstacle_history.shape}")
    print(f"  Distances shape        : {record.distances.shape}")
    print(f"  Total collision steps  : {len(record.collision_steps)}")

    os.makedirs("outputs", exist_ok=True)
    visualise_simulation(record, mode="3d", save_path="outputs/simulation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Trajectory prediction (training)
# ─────────────────────────────────────────────────────────────────────────────

def demo_prediction() -> None:
    print_section("Step 7 · Trajectory Prediction (Member 2)")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("  ⚠️  PyTorch not installed. Skipping prediction demo.")
        print("     Install with: pip install torch")
        return

    seq_len     = CUSTOM_CONFIG["dataset"]["seq_len"]
    future_len  = CUSTOM_CONFIG["dataset"]["future_len"]
    n_obstacles = CUSTOM_CONFIG["obstacles"]["n_obstacles"]

    predictor = TrajectoryPredictor(
        seq_len     = seq_len,
        future_len  = future_len,
        features    = 9,
        n_obstacles = n_obstacles,
        hidden_size = 128,       # increased from 64 to match larger encoder input
        num_layers  = 2,
        device      = device,
        dt          = CUSTOM_CONFIG["time_step"],   # used to rederive acc at inference
    )
    print(f"  Model initialized (device: {device})")

    dataset_dir = CUSTOM_CONFIG["dataset"]["save_dir"]
    train_count = len(load_npy(os.path.join(dataset_dir, "train/drone_past.npy")))
    print(f"  Training on {train_count} samples…")

    predictor.train(
        config       = CUSTOM_CONFIG,
        epochs       = 50,    # 50 epochs — extra 10 vs v11 to help velocity convergence
        batch_size   = 256,
        learning_rate= 0.001,
        verbose      = True,
    )

    val_metrics  = predictor.evaluate(config=CUSTOM_CONFIG, split="val",  verbose=False)
    test_metrics = predictor.evaluate(config=CUSTOM_CONFIG, split="test", verbose=False)

    print(f"\n  Validation metrics:")
    print(f"    Full-feature RMSE   : {val_metrics['full_feature_rmse']:.6f}")
    print(f"    RMSE growth         : {val_metrics['full_feature_rmse_growth_pct']:.1f}%")
    print(f"    Position RMSE       : {val_metrics['position_rmse_m']:.6f} m")
    print(f"    Mean position error : {val_metrics['mean_position_error_m']:.6f} m")
    print(f"    P95 position error  : {val_metrics['p95_position_error_m']:.6f} m")
    if "velocity_rmse_growth_pct" in val_metrics:
        print(f"    Velocity growth     : {val_metrics['velocity_rmse_growth_pct']:.1f}%")
    if "acceleration_rmse_growth_pct" in val_metrics:
        print(f"    Acceleration growth : {val_metrics['acceleration_rmse_growth_pct']:.1f}%")

    print(f"\n  Test metrics:")
    print(f"    Full-feature RMSE   : {test_metrics['full_feature_rmse']:.6f}")
    print(f"    RMSE growth         : {test_metrics['full_feature_rmse_growth_pct']:.1f}%")
    print(f"    Position RMSE       : {test_metrics['position_rmse_m']:.6f} m")
    print(f"    Mean position error : {test_metrics['mean_position_error_m']:.6f} m")
    print(f"    P95 position error  : {test_metrics['p95_position_error_m']:.6f} m")

    os.makedirs("models", exist_ok=True)
    predictor.save(MODEL_PATH)

    # Sample predictions — include obstacle context
    print(f"\n  Sample predictions (first 3 validation samples):")
    try:
        drone_past_val = load_npy(os.path.join(dataset_dir, "val/drone_past.npy"))
        obs_past_val   = load_npy(os.path.join(dataset_dir, "val/obs_past.npy"))
        drone_future_val = load_npy(os.path.join(dataset_dir, "val/drone_future.npy"))

        if len(drone_past_val) == 0:
            print("    No validation samples available.")
            return

        for i in range(min(3, len(drone_past_val))):
            pred  = predictor.predict(drone_past_val[i], obs_past_val[i])
            true  = drone_future_val[i]
            error = np.mean(np.linalg.norm(pred[:, :3] - true[:, :3], axis=1))
            print(f"    Sample {i+1}: mean position error = {error:.4f} m")
            print(f"      Pred pos[0] = {pred[0, :3]}")
            print(f"      True pos[0] = {true[0, :3]}")
    except Exception as e:
        print(f"    ⚠️  Could not load validation data: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Evaluate trained model
# ─────────────────────────────────────────────────────────────────────────────

def demo_test_prediction() -> None:
    print_section("Step 8 · Prediction Model Evaluation")

    from test_prediction_model import (
        evaluate_predictions,
        visualize_predictions,
        plot_error_distribution,
        _interpret_position_rmse,
    )

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ Model not found at {MODEL_PATH}")
        return

    print("  Loading model…")
    try:
        predictor = TrajectoryPredictor.load(MODEL_PATH, device="cpu")
        print(f"  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return

    dataset_dir = CUSTOM_CONFIG["dataset"]["save_dir"]

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

    _print_metrics(metrics_val)

    print(f"\n  Per-timestep full-feature RMSE (growth over prediction horizon):")
    for t, rmse_t in enumerate(metrics_val["full_feature_rmse_per_timestep"]):
        print(f"    Step {t+1:2d}: {rmse_t:.6f}")

    print(f"\n  Creating visualizations…")
    visualize_predictions(predictions_val, drone_future_val, split="val", n_samples=3)
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

    _print_metrics(metrics_test)

    print(f"\n  Per-timestep full-feature RMSE (growth over prediction horizon):")
    for t, rmse_t in enumerate(metrics_test["full_feature_rmse_per_timestep"]):
        print(f"    Step {t+1:2d}: {rmse_t:.6f}")

    print(f"\n  Creating visualizations…")
    visualize_predictions(predictions_test, drone_future_test, split="test", n_samples=3)
    plot_error_distribution(predictions_test, drone_future_test, split="test")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("Summary & Interpretation")

    print(f"\n  Validation Performance (Full Features):")
    print(f"    Full-feature RMSE : {metrics_val['full_feature_rmse']:.6f}")
    print("    Note: mixes position / velocity / acceleration units.")

    print(f"\n  Validation Performance (Position-Only):")
    print(f"    Position RMSE     : {metrics_val['position_rmse_m']:.6f} m")
    print(f"    Mean pos error    : {metrics_val['mean_position_error_m']:.6f} m")
    print(f"    P95 pos error     : {metrics_val['p95_position_error_m']:.6f} m")
    print(f"    P99 pos error     : {metrics_val['p99_position_error_m']:.6f} m")
    print(f"    Interpretation    : {_interpret_position_rmse(metrics_val['position_rmse_m'])}")

    print(f"\n  Test Performance (Full Features):")
    print(f"    Full-feature RMSE : {metrics_test['full_feature_rmse']:.6f}")
    print("    Note: mixes position / velocity / acceleration units.")

    print(f"\n  Test Performance (Position-Only):")
    print(f"    Position RMSE     : {metrics_test['position_rmse_m']:.6f} m")
    print(f"    Mean pos error    : {metrics_test['mean_position_error_m']:.6f} m")
    print(f"    P95 pos error     : {metrics_test['p95_position_error_m']:.6f} m")
    print(f"    P99 pos error     : {metrics_test['p99_position_error_m']:.6f} m")
    print(f"    Interpretation    : {_interpret_position_rmse(metrics_test['position_rmse_m'])}")

    # ── Full growth breakdown ─────────────────────────────────────────────────
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
        print(f"    ⚠️  Full-feature RMSE growth is still high")
    else:
        print(f"    ✓ Full-feature RMSE growth is controlled")

    print("\n✅ Evaluation complete!")
    print("   Visualizations saved to outputs/")


def _print_metrics(m: dict) -> None:
    """Print the standard metric block."""
    print(f"\n  Metrics:")
    print(f"    Full-feature RMSE       : {m['full_feature_rmse']:.6f}")
    print(f"    Full-feature MSE        : {m['full_feature_mse']:.6f}")
    print(f"    Full-feature MAE        : {m['full_feature_mae']:.6f}")
    print(f"    Position RMSE           : {m['position_rmse_m']:.4f} m")
    print(f"    Mean Position Error     : {m['mean_position_error_m']:.4f} m")
    print(f"    P90 Position Error      : {m['p90_position_error_m']:.4f} m")
    print(f"    P95 Position Error      : {m['p95_position_error_m']:.4f} m")
    print(f"    P99 Position Error      : {m['p99_position_error_m']:.4f} m")
    print(f"    Max Position Error      : {m['max_position_error_m']:.4f} m")
    print(f"    Worst sample index      : {m['worst_sample_idx']}")
    print(f"    Worst sample mean error : {m['worst_sample_mean_position_error_m']:.4f} m")
    if "velocity_rmse" in m:
        print(f"    Velocity RMSE           : {m['velocity_rmse']:.6f}")
        print(f"    Velocity Growth         : {m['velocity_rmse_growth_pct']:.1f}%")
    if "acceleration_rmse" in m:
        print(f"    Acceleration RMSE       : {m['acceleration_rmse']:.6f}")
        print(f"    Acceleration Growth     : {m['acceleration_rmse_growth_pct']:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_trajectory()
    demo_obstacles()

    if not dataset_exists():
        dataset = demo_dataset()
        demo_save(dataset)
        demo_model_input(dataset)
    else:
        print("\n✓ Dataset already exists. Skipping dataset generation.")

    demo_simulation()

    if not model_exists():
        print("\nModel missing or incompatible. Training a new model…")
        demo_prediction()
    else:
        print("\n✓ Compatible model already exists. Skipping training.")

    demo_test_prediction()

    print("\n✅  All steps completed successfully.")
    print("   Dataset      →  dataset/")
    print("   Model        →  models/trajectory_predictor.pkl")
    print("   Viz          →  outputs/simulation.png")
    print("   Eval plots   →  outputs/predictions_*.png")
>>>>>>> teammate/main
