# ===== dataset_generator.py =====
"""
Dataset Generator
=================
Builds labeled datasets for both Member 2 (trajectory prediction) and
Member 3 (reactive control / imitation learning).

Dataset sample structure
------------------------
Each sample is a Python dict:

    {
        # ── Inputs ────────────────────────────────────────────────────
        "drone_past"    : np.ndarray  [seq_len, drone_features]
                          Past drone states (pos, vel, acc)

        "obs_past"      : np.ndarray  [seq_len, N_obs, 6]
                          Past obstacle states for each time-step

        # ── Targets ───────────────────────────────────────────────────
        "drone_future"  : np.ndarray  [future_len, drone_features]
                          Ground-truth future trajectory (Member 2 target)

        "action"        : np.ndarray  [4]
                          Safe next action  (vx, vy, vz, yaw_rate)
                          (Member 3 imitation learning target)

        # ── Meta ──────────────────────────────────────────────────────
        "had_collision" : bool
                          Whether raw trajectory had a collision in future window
                          (before avoidance correction was applied)
    }

Sliding-window extraction
-------------------------
From a single trajectory of length T we extract:
    T - seq_len - future_len
windows, each offset by 1 time-step.
"""

from __future__ import annotations

import os
import copy
import numpy as np

from config import DEFAULT_CONFIG
from trajectory_generator import generate_trajectory
from obstacle_simulation import ObstacleSimulator
from utils import (
    check_collision,
    repulsion_velocity,
    save_npy,
    save_csv,
    save_json,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(config: dict | None = None) -> dict:
    """
    Generate the full labeled dataset.

    Parameters
    ----------
    config : dict (optional) — merged with DEFAULT_CONFIG

    Returns
    -------
    dataset : dict with keys
        "train"   → list[sample_dict]
        "val"     → list[sample_dict]
        "test"    → list[sample_dict]
        "meta"    → dict with shapes & feature names
    """
    cfg = _merge(config)
    set_seed(cfg["random_seed"])

    ds_cfg = cfg["dataset"]
    seq_len    = ds_cfg["seq_len"]
    future_len = ds_cfg["future_len"]
    n_trajs    = ds_cfg["n_trajectories"]

    drone_cfg = cfg["drone"]
    obs_cfg   = cfg["obstacles"]

    all_samples: list[dict] = []

    print(f"[DatasetGenerator] Generating {n_trajs} trajectories …")

    for traj_idx in range(n_trajs):
        # 1. Generate drone trajectory  [T, drone_feat]
        traj = generate_trajectory(cfg)                    # [T, 9] or [T, 6]
        T = traj.shape[0]

        # 2. Simulate obstacles for the SAME duration  [T, N, 6].
        # Each trajectory receives a deterministic but different obstacle seed.
        obs_cfg_for_traj = copy.deepcopy(cfg)
        obs_cfg_for_traj["random_seed"] = int(cfg["random_seed"]) + 1009 * (traj_idx + 1)
        obs_sim = ObstacleSimulator(obs_cfg_for_traj)
        obs_history = _run_obstacle_sim(obs_sim, T, cfg["time_step"])  # [T, N, 6]

        # 3. Sliding-window extraction
        max_start = T - seq_len - future_len
        if max_start <= 0:
            continue

        for start in range(max_start):
            end_past   = start + seq_len
            end_future = end_past + future_len

            drone_past   = traj[start:end_past]            # [seq_len, drone_feat]
            drone_future = traj[end_past:end_future]       # [future_len, drone_feat]
            obs_past     = obs_history[start:end_past]     # [seq_len, N, 6]

            # 4. Labelling — collision detection across the future window
            obs_pos_now = obs_history[end_past - 1, :, :3]  # [N, 3]
            colliding = _future_window_has_collision(
                drone_future=drone_future,
                obs_future=obs_history[end_past:end_future],
                drone_radius=drone_cfg["collision_radius"],
                obstacle_radius=obs_cfg["radius"],
            )

            # 5. Compute safe action
            action = _compute_safe_action(
                drone_past=drone_past,
                obs_pos_now=obs_pos_now,
                drone_cfg=drone_cfg,
                dt=cfg["time_step"],
                max_velocity=cfg["trajectory"]["max_velocity"],
            )

            sample = {
                "drone_past"    : drone_past.astype(np.float32),
                "obs_past"      : obs_past.astype(np.float32),
                "drone_future"  : drone_future.astype(np.float32),
                "action"        : action.astype(np.float32),
                "had_collision" : bool(colliding),
            }
            all_samples.append(sample)

        if (traj_idx + 1) % max(1, n_trajs // 10) == 0:
            print(f"  [{traj_idx + 1}/{n_trajs}]  samples so far: {len(all_samples)}")

    # 6. Train / val / test split
    np.random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * ds_cfg["train_ratio"])
    n_val   = int(n * ds_cfg["val_ratio"])

    dataset = {
        "train" : all_samples[:n_train],
        "val"   : all_samples[n_train:n_train + n_val],
        "test"  : all_samples[n_train + n_val:],
        "meta"  : _build_meta(cfg, all_samples[0] if all_samples else {}),
    }

    print(
        f"[DatasetGenerator] Done.  "
        f"train={len(dataset['train'])}  "
        f"val={len(dataset['val'])}  "
        f"test={len(dataset['test'])}"
    )
    return dataset


def save_dataset(dataset: dict, config: dict | None = None) -> None:
    """
    Persist a dataset returned by generate_dataset() to disk.

    Saves:
        dataset/<split>/drone_past.npy
        dataset/<split>/obs_past.npy
        dataset/<split>/drone_future.npy
        dataset/<split>/actions.npy
        dataset/<split>/had_collision.npy
        dataset/meta.json
        dataset/<split>/drone_past.csv   (first 2000 rows, flattened)

    Parameters
    ----------
    dataset : dict — returned by generate_dataset()
    config  : dict (optional) — used to read save_dir
    """
    cfg = _merge(config)
    save_dir = cfg["dataset"]["save_dir"]

    for split in ("train", "val", "test"):
        samples = dataset[split]
        if not samples:
            continue

        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Stack arrays across samples
        drone_past    = np.stack([s["drone_past"]   for s in samples])   # [M, seq, feat]
        obs_past      = np.stack([s["obs_past"]     for s in samples])   # [M, seq, N, 6]
        drone_future  = np.stack([s["drone_future"] for s in samples])   # [M, fut, feat]
        actions       = np.stack([s["action"]       for s in samples])   # [M, 4]
        collisions    = np.array([s["had_collision"] for s in samples])  # [M]

        save_npy(drone_past,   os.path.join(split_dir, "drone_past.npy"))
        save_npy(obs_past,     os.path.join(split_dir, "obs_past.npy"))
        save_npy(drone_future, os.path.join(split_dir, "drone_future.npy"))
        save_npy(actions,      os.path.join(split_dir, "actions.npy"))
        save_npy(collisions,   os.path.join(split_dir, "had_collision.npy"))

        # CSV export (flattened drone_past for easy inspection, capped at 2000)
        flat = drone_past.reshape(len(samples), -1)[:2000]
        save_csv(flat, os.path.join(split_dir, "drone_past.csv"))

        print(f"  Saved {split}: {len(samples)} samples → {split_dir}/")

    save_json(dataset["meta"], os.path.join(save_dir, "meta.json"))
    print(f"  Metadata → {save_dir}/meta.json")


def get_model_input(sample: dict) -> dict:
    """
    Format a single dataset sample into model-ready tensors.

    Member 2 (prediction) input
    ---------------------------
    "x_pred" : [seq_len, drone_feat]  ← drone_past

    Member 3 (control) input
    ------------------------
    "x_ctrl" : [seq_len, drone_feat + N_obs * 6]
                  drone_past and flattened obstacle states concatenated per step

    Member 2 / 3 targets
    ---------------------
    "y_future" : [future_len, drone_feat]
    "y_action" : [4]

    Parameters
    ----------
    sample : dict — one element from dataset["train"] (or val / test)

    Returns
    -------
    formatted : dict with keys x_pred, x_ctrl, y_future, y_action
    """
    drone_past  = sample["drone_past"]   # [seq_len, drone_feat]
    obs_past    = sample["obs_past"]     # [seq_len, N, 6]
    seq_len     = drone_past.shape[0]

    # Flatten obstacles: [seq_len, N*6]
    obs_flat = obs_past.reshape(seq_len, -1)

    # Concatenated input for control model: [seq_len, drone_feat + N*6]
    x_ctrl = np.concatenate([drone_past, obs_flat], axis=1)

    return {
        "x_pred"   : drone_past,               # Member 2
        "x_ctrl"   : x_ctrl,                   # Member 3
        "y_future" : sample["drone_future"],    # Member 2 target
        "y_action" : sample["action"],          # Member 3 target
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_obstacle_sim(
    sim: ObstacleSimulator,
    T: int,
    dt: float,
) -> np.ndarray:
    """Run obstacle simulator for T steps, return [T, N, 6]."""
    history = []
    for _ in range(T):
        states = sim.step(dt)          # [N, 6]
        history.append(states)
    return np.stack(history, axis=0)   # [T, N, 6]


def _future_window_has_collision(
    drone_future: np.ndarray,
    obs_future: np.ndarray,
    drone_radius: float,
    obstacle_radius: float,
) -> bool:
    """Return True if any future drone step collides with any obstacle step."""
    for k in range(min(len(drone_future), len(obs_future))):
        colliding, _ = check_collision(
            drone_pos=drone_future[k, :3],
            obstacle_positions=obs_future[k, :, :3],
            drone_radius=drone_radius,
            obstacle_radius=obstacle_radius,
        )
        if colliding:
            return True
    return False


def _compute_safe_action(
    drone_past: np.ndarray,
    obs_pos_now: np.ndarray,
    drone_cfg: dict,
    dt: float,
    max_velocity: float,
) -> np.ndarray:
    """
    Compute a safe next action (vx, vy, vz, yaw_rate) using a potential-field
    approach (repulsion from obstacles + continuation of current velocity).

    This creates "expert" demonstrations for Member 3's imitation learning.

    Returns
    -------
    action : (4,) float32  →  [vx, vy, vz, yaw_rate]
    """
    pos_now = drone_past[-1, :3]   # current position
    vel_now = drone_past[-1, 3:6]  # current velocity

    # Repulsion from nearby obstacles
    v_rep = repulsion_velocity(
        drone_pos=pos_now,
        obstacle_positions=obs_pos_now,
        safe_distance=drone_cfg["safe_distance"],
        strength=2.0,
    )

    # Blend current velocity with repulsion
    v_safe = vel_now + v_rep

    # Clamp magnitude to the active trajectory configuration.
    max_v = float(max_velocity)
    speed = np.linalg.norm(v_safe)
    if speed > max_v:
        v_safe = v_safe / speed * max_v

    # Simple yaw: angle of horizontal velocity in xy-plane
    yaw_rate = float(np.arctan2(v_safe[1], v_safe[0])) if speed > 1e-4 else 0.0

    return np.array([v_safe[0], v_safe[1], v_safe[2], yaw_rate], dtype=np.float32)


def _build_meta(cfg: dict, sample: dict) -> dict:
    """Build dataset metadata dict (shapes, feature names, config summary)."""
    from trajectory_generator import get_feature_names, TRAJECTORY_GENERATOR_VERSION
    from obstacle_simulation import OBSTACLE_FEATURE_NAMES

    mode = cfg["trajectory"]["mode"]
    drone_features = get_feature_names(mode)
    n_obs = cfg["obstacles"]["n_obstacles"]

    meta = {
        "drone_features"       : drone_features,
        "obstacle_features"    : OBSTACLE_FEATURE_NAMES,
        "n_obstacles"          : n_obs,
        "seq_len"              : cfg["dataset"]["seq_len"],
        "future_len"           : cfg["dataset"]["future_len"],
        "drone_past_shape"     : f"[seq_len={cfg['dataset']['seq_len']}, features={len(drone_features)}]",
        "obs_past_shape"       : f"[seq_len, n_obstacles={n_obs}, 6]",
        "drone_future_shape"   : f"[future_len={cfg['dataset']['future_len']}, features={len(drone_features)}]",
        "action_shape"         : "[4]  →  vx, vy, vz, yaw_rate",
        "x_ctrl_feature_dim"   : len(drone_features) + n_obs * 6,
        "generation_version"   : TRAJECTORY_GENERATOR_VERSION,
        "config_summary"       : {
            k: cfg[k]
            for k in ("simulation_time", "time_step", "random_seed")
        },
    }
    return meta


def _merge(user_config: dict | None) -> dict:
    base = copy.deepcopy(DEFAULT_CONFIG)
    if user_config:
        _deep_update(base, user_config)
    return base


def _deep_update(base: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
