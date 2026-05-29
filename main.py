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
import matplotlib.pyplot as plt

# Allow internal pipeline scripts to find each other cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.join(current_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, "src"))

# Clean, explicit package imports
from src.pipeline.config import DEFAULT_CONFIG
from src.pipeline.trajectory_generator import generate_trajectory, get_feature_names, TRAJECTORY_GENERATOR_VERSION
from src.pipeline.obstacle_simulation import simulate_obstacles, ObstacleSimulator
from src.pipeline.dataset_generator import generate_dataset, save_dataset, get_model_input
from src.pipeline.simulator import run_simulation, visualise_simulation
from models.trajectory_predictor import TrajectoryPredictor
from src.pipeline.utils import print_section, set_seed, load_npy

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
        "n_obstacles": 25,
        "max_tracked_obstacles": 25, 
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
# Step Modules
# ─────────────────────────────────────────────────────────────────────────────

def demo_trajectory() -> None:
    print_section("Step 1 · Trajectory Generator")
    set_seed(42)
    traj = generate_trajectory(CUSTOM_CONFIG)
    print(f"  Trajectory shape : {traj.shape}")
    print(f"  Features         : {get_feature_names('3d')}")


def demo_obstacles() -> None:
    print_section("Step 2 · Obstacle Simulator")
    obs_history = simulate_obstacles(CUSTOM_CONFIG)
    T, N, feat  = obs_history.shape
    print(f"  History shape    : {obs_history.shape}  (T={T}, N_obs={N}, features={feat})")


def demo_dataset() -> dict:
    print_section("Step 3 · Dataset Generator")
    dataset = generate_dataset(CUSTOM_CONFIG)
    sample = dataset["train"][0]
    print(f"  drone_past shape : {sample['drone_past'].shape}")
    return dataset


def demo_save(dataset: dict) -> None:
    print_section("Step 4 · Save Dataset")
    save_dataset(dataset, CUSTOM_CONFIG)
    reloaded = np.load("dataset/train/drone_past.npy", allow_pickle=True)
    print(f"  Reloaded drone_past shape: {reloaded.shape}")


def demo_model_input(dataset: dict) -> None:
    print_section("Step 5 · Model Input Formatter")
    sample    = dataset["train"][0]
    formatted = get_model_input(sample)
    print(f"    x_pred   shape : {formatted['x_pred'].shape}")


def demo_simulation() -> None:
    print_section("Step 6 · Lightweight Simulator")
    record = run_simulation(CUSTOM_CONFIG)
    print(f"  Total collision steps  : {len(record.collision_steps)}")
    os.makedirs("outputs", exist_ok=True)
    visualise_simulation(record, mode="3d", save_path="outputs/simulation.png")


def demo_prediction() -> None:
    print_section("Step 7 · Trajectory Prediction (Member 2)")

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        print("  ⚠️  PyTorch not installed. Skipping prediction demo.")
        return

    seq_len     = CUSTOM_CONFIG["dataset"]["seq_len"]
    future_len  = CUSTOM_CONFIG["dataset"]["future_len"]
    n_obstacles = CUSTOM_CONFIG["obstacles"]["n_obstacles"]

    predictor = TrajectoryPredictor(
        seq_len     = seq_len,
        future_len  = future_len,
        features    = 9,
        n_obstacles = n_obstacles,
        hidden_size = 384,
        num_layers  = 2,
        device      = device,
        dt          = CUSTOM_CONFIG["time_step"],
    )
    print(f"  Model initialized (device: {device})")

    dataset_dir = CUSTOM_CONFIG["dataset"]["save_dir"]
    train_count = len(load_npy(os.path.join(dataset_dir, "train/drone_past.npy")))
    print(f"  Training on {train_count} samples…")

    predictor.train(
        config       = CUSTOM_CONFIG,
        epochs       = 50,
        batch_size   = 256,
        learning_rate= 0.001,
        verbose      = True,
    )

    val_metrics  = predictor.evaluate(config=CUSTOM_CONFIG, split="val",  verbose=False)
    test_metrics = predictor.evaluate(config=CUSTOM_CONFIG, split="test", verbose=False)

    print(f"\n  Validation Position RMSE: {val_metrics['position_rmse_m']:.6f} m")
    print(f"  Test Position RMSE: {test_metrics['position_rmse_m']:.6f} m")

    os.makedirs("models", exist_ok=True)
    predictor.save(MODEL_PATH)


def demo_test_prediction() -> None:
    print_section("Step 8 · Prediction Model Evaluation")

    from src.pipeline.test_prediction_model import (
        evaluate_predictions,
        visualize_predictions,
        plot_error_distribution,
        _interpret_position_rmse,
    )

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ Model not found at {MODEL_PATH}")
        return

    print("  Loading model…")
    try:
        predictor = TrajectoryPredictor.load(MODEL_PATH, device=device)
        print(f"  ✓ Model loaded dynamically onto: {device}")
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

    predictions_val = predictor.predict(drone_past_val, obs_past_val)
    metrics_val     = evaluate_predictions(predictions_val, drone_future_val)
    _print_metrics(metrics_val)

    # ── Test set ──────────────────────────────────────────────────────────────
    print_section("Test Set Results")
    try:
        drone_past_test   = load_npy(f"{dataset_dir}/test/drone_past.npy")
        obs_past_test     = load_npy(f"{dataset_dir}/test/obs_past.npy")
        drone_future_test = load_npy(f"{dataset_dir}/test/drone_future.npy")
    except FileNotFoundError:
        print("  ⚠️  Test set not found.")
        return

    predictions_test = predictor.predict(drone_past_test, obs_past_test)
    metrics_test     = evaluate_predictions(predictions_test, drone_future_test)
    _print_metrics(metrics_test)

    # ====================== AJOUT ICI ======================
    print_section("Step 8.5 · Rapport Membre 5")
    generate_member5_report(predictions_test, drone_future_test, output_folder="outputs/member5")
    # =======================================================

def generate_member5_report(predictions, ground_truth, output_folder="outputs"):
    """
    PARTIE ÉVALUATION - MEMBRE 5
    Analyse les tableaux Numpy de prédiction et génère les graphiques.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # On prend le premier test du dataset pour dessiner la trajectoire
    # On suppose que x = feature 0 et y = feature 1
    ia_x = predictions[0, :, 0]
    ia_y = predictions[0, :, 1]

    reel_x = ground_truth[0, :, 0]
    reel_y = ground_truth[0, :, 1]

    # --- Graphique 1 : Trajectoires ---
    plt.figure(figsize=(10, 6))
    plt.plot(ia_x, ia_y, linestyle='-', label="IA (Prédiction)", color='blue', linewidth=2)
    plt.plot(reel_x, reel_y, linestyle='--', label="Réel (Ground Truth)", color='red', linewidth=2)

    plt.title("Évaluation Membre 5 : Comparaison de la trajectoire")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/m5_trajectoires.png")
    plt.close()

    # --- Graphique 2 : Fluidité (Jerk estimé) ---
    # Calcul basique de la variation d'accélération pour évaluer la fluidité
    jerk_ia = np.mean(np.abs(np.diff(np.diff(ia_x)))) + np.mean(np.abs(np.diff(np.diff(ia_y))))
    jerk_reel = np.mean(np.abs(np.diff(np.diff(reel_x)))) + np.mean(np.abs(np.diff(np.diff(reel_y))))

    plt.figure(figsize=(8, 5))
    plt.bar(["IA (Prédiction)", "Réel (Vérité terrain)"], [jerk_ia, jerk_reel], color=['blue', 'red'])
    plt.title("Évaluation Membre 5 : Analyse de la fluidité (Jerk)")
    plt.ylabel("Jerk estimé (m/s³)")
    plt.savefig(f"{output_folder}/m5_fluidite.png")
    plt.close()

    print(f"  ✅ Graphiques Membre 5 sauvegardés dans /{output_folder}")


def _print_metrics(m: dict) -> None:
    """Print the standard metric block."""
    print(f"\n  Metrics:")
    print(f"    Full-feature RMSE       : {m['full_feature_rmse']:.6f}")
    print(f"    Position RMSE           : {m['position_rmse_m']:.4f} m")
    print(f"    Mean Position Error     : {m['mean_position_error_m']:.4f} m")


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
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
    print("\n✅ All steps completed successfully.")
    