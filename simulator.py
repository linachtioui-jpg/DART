# ===== simulator.py =====
"""
Lightweight Test Simulator
===========================
A self-contained simulation loop that:
    • Flies the drone along a pre-generated trajectory
    • Moves dynamic obstacles concurrently
    • Detects collisions at each step
    • Logs the full simulation record
    • Visualises everything with Matplotlib

Design notes
------------
This module is intentionally decoupled from PyBullet / ROS so that it can be
dropped-in replaced by Member 4's physics engine later. The public interface
(SimulationRecord, run_simulation) will remain stable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from config import DEFAULT_CONFIG
from trajectory_generator import generate_trajectory
from obstacle_simulation import ObstacleSimulator
from utils import (
    check_collision,
    distances_to_obstacles,
    print_section,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationRecord:
    """
    Full log of a simulation run.  Passed to visualise_simulation().

    Attributes
    ----------
    drone_states     : [T, drone_feat]  full drone trajectory used
    obstacle_history : [T, N_obs, 6]   obstacle states at each step
    distances        : [T, N_obs]       drone-to-obstacle distances
    collision_flags  : [T]              True where a collision occurred
    collision_steps  : list[int]        time-step indices with collisions
    dt               : float            time step (seconds)
    config           : dict             config used for this run
    """
    drone_states     : np.ndarray
    obstacle_history : np.ndarray
    distances        : np.ndarray
    collision_flags  : np.ndarray
    collision_steps  : list[int]
    dt               : float
    config           : dict


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(config: dict | None = None) -> SimulationRecord:
    """
    Execute the lightweight simulation loop.

    Parameters
    ----------
    config : dict (optional) — merged with DEFAULT_CONFIG

    Returns
    -------
    SimulationRecord
        Complete log of the run, ready for visualisation or further analysis.
    """
    cfg = _merge(config)
    dt  = cfg["time_step"]
    T   = int(cfg["simulation_time"] / dt)

    drone_cfg = cfg["drone"]
    obs_cfg   = cfg["obstacles"]

    print_section("Running Lightweight Simulation")
    print(f"  Steps    : {T}")
    print(f"  dt       : {dt} s")
    print(f"  Obstacles: {obs_cfg['n_obstacles']}")

    # ── Generate drone trajectory ─────────────────────────────────────────────
    drone_states = generate_trajectory(cfg)   # [T_gen, feat]
    # Trim / pad to exactly T steps
    drone_states = _adjust_length(drone_states, T)

    # ── Initialise obstacle simulator ─────────────────────────────────────────
    obs_sim = ObstacleSimulator(cfg)

    # ── Simulation loop ───────────────────────────────────────────────────────
    obstacle_history : list[np.ndarray] = []  # each: [N, 6]
    distances_log    : list[np.ndarray] = []  # each: [N]
    collision_flags  : list[bool]        = []
    collision_steps  : list[int]         = []

    for t in range(T):
        obs_states = obs_sim.step(dt)          # [N, 6]
        obstacle_history.append(obs_states)

        drone_pos = drone_states[t, :3]
        obs_pos   = obs_states[:, :3]

        dists = distances_to_obstacles(drone_pos, obs_pos)
        distances_log.append(dists)

        colliding, hit_ids = check_collision(
            drone_pos=drone_pos,
            obstacle_positions=obs_pos,
            drone_radius=drone_cfg["collision_radius"],
            obstacle_radius=obs_cfg["radius"],
        )
        collision_flags.append(colliding)
        if colliding:
            collision_steps.append(t)

    # ── Pack results ─────────────────────────────────────────────────────────
    record = SimulationRecord(
        drone_states     = drone_states,
        obstacle_history = np.stack(obstacle_history, axis=0),  # [T, N, 6]
        distances        = np.stack(distances_log,    axis=0),  # [T, N]
        collision_flags  = np.array(collision_flags,  dtype=bool),
        collision_steps  = collision_steps,
        dt               = dt,
        config           = cfg,
    )

    n_col = len(collision_steps)
    print(f"  Collisions detected: {n_col} / {T} steps "
          f"({100.0 * n_col / T:.1f}%)")
    return record


def visualise_simulation(
    record: SimulationRecord,
    mode: str = "3d",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise the simulation with Matplotlib.

    Plots
    -----
    1. Drone and obstacle trajectories (3-D or 2-D projection)
    2. Distance-to-obstacles over time
    3. Collision timeline

    Parameters
    ----------
    record    : SimulationRecord returned by run_simulation()
    mode      : "3d" or "2d"
    save_path : if provided, save figure to this path instead of showing
    """
    drone_states  = record.drone_states
    obs_history   = record.obstacle_history   # [T, N, 6]
    dists         = record.distances          # [T, N]
    col_flags     = record.collision_flags    # [T]
    col_steps     = record.collision_steps
    T, N          = obs_history.shape[:2]
    time_axis     = np.arange(T) * record.dt

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    _style = {"color": "#c9d1d9", "fontsize": 10}

    # ── Plot 1: Trajectories ─────────────────────────────────────────────────
    if mode == "3d":
        ax1 = fig.add_subplot(2, 2, (1, 2), projection="3d")
        ax1.set_facecolor("#161b22")

        # Drone path
        ax1.plot(
            drone_states[:, 0], drone_states[:, 1], drone_states[:, 2],
            color="#58a6ff", linewidth=1.5, label="Drone"
        )
        # Collision points
        if col_steps:
            cx = drone_states[col_steps, 0]
            cy = drone_states[col_steps, 1]
            cz = drone_states[col_steps, 2]
            ax1.scatter(cx, cy, cz, color="#f85149", s=30, zorder=5, label="Collision")

        # Obstacles
        colours = plt.cm.Set2(np.linspace(0, 1, N))
        for n in range(N):
            ax1.plot(
                obs_history[:, n, 0],
                obs_history[:, n, 1],
                obs_history[:, n, 2],
                color=colours[n], linewidth=0.8, alpha=0.7,
                label=f"Obs {n}",
            )

        ax1.set_xlabel("X (m)", **_style)
        ax1.set_ylabel("Y (m)", **_style)
        ax1.set_zlabel("Z (m)", **_style)
        ax1.set_title("Drone & Obstacle Trajectories (3-D)", color="#c9d1d9", pad=8)
        ax1.legend(fontsize=8, framealpha=0.3, loc="upper left")

    else:  # 2-D top-down
        ax1 = fig.add_subplot(2, 2, (1, 2))
        ax1.set_facecolor("#161b22")
        ax1.plot(drone_states[:, 0], drone_states[:, 1],
                 color="#58a6ff", lw=1.5, label="Drone")
        if col_steps:
            ax1.scatter(drone_states[col_steps, 0], drone_states[col_steps, 1],
                        c="#f85149", s=30, zorder=5, label="Collision")
        colours = plt.cm.Set2(np.linspace(0, 1, N))
        for n in range(N):
            ax1.plot(obs_history[:, n, 0], obs_history[:, n, 1],
                     color=colours[n], lw=0.8, alpha=0.7, label=f"Obs {n}")
        ax1.set_xlabel("X (m)", **_style)
        ax1.set_ylabel("Y (m)", **_style)
        ax1.set_title("Drone & Obstacle Trajectories (top-down)", color="#c9d1d9")
        ax1.legend(fontsize=8, framealpha=0.3)

    _apply_dark_ax(ax1)

    # ── Plot 2: Distances ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor("#161b22")
    for n in range(N):
        ax2.plot(time_axis, dists[:, n],
                 color=colours[n], lw=0.9, alpha=0.8, label=f"Obs {n}")
    # safe-distance line
    safe_d = record.config["drone"]["safe_distance"]
    ax2.axhline(safe_d, color="#d29922", lw=1, linestyle="--", label="Safe dist")
    col_thresh = record.config["drone"]["collision_radius"] + record.config["obstacles"]["radius"]
    ax2.axhline(col_thresh, color="#f85149", lw=1, linestyle="--", label="Collision thresh")
    ax2.set_xlabel("Time (s)", **_style)
    ax2.set_ylabel("Distance (m)", **_style)
    ax2.set_title("Drone → Obstacle Distances", color="#c9d1d9")
    ax2.legend(fontsize=7, framealpha=0.3)
    _apply_dark_ax(ax2)

    # ── Plot 3: Collision timeline ────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor("#161b22")
    ax3.fill_between(time_axis, col_flags.astype(float),
                     color="#f85149", alpha=0.6, step="mid")
    ax3.set_ylim(-0.05, 1.2)
    ax3.set_xlabel("Time (s)", **_style)
    ax3.set_ylabel("Collision (1=yes)", **_style)
    ax3.set_title("Collision Events", color="#c9d1d9")
    _apply_dark_ax(ax3)

    plt.tight_layout(pad=2.0)
    plt.suptitle(
        "Drone Simulation — Lightweight Test Environment",
        color="#58a6ff", fontsize=13, y=1.01, fontweight="bold"
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _adjust_length(arr: np.ndarray, T: int) -> np.ndarray:
    """Trim or tile *arr* so that arr.shape[0] == T."""
    if arr.shape[0] >= T:
        return arr[:T]
    # Repeat to fill (edge case: very short sim)
    reps = int(np.ceil(T / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:T]


def _apply_dark_ax(ax) -> None:
    """Apply consistent dark-theme styling to a Matplotlib axes."""
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    try:
        ax.zaxis.label.set_color("#8b949e")
    except AttributeError:
        pass
