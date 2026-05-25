# ===== utils.py =====
"""
Shared utility functions used across all pipeline modules.
"""

import numpy as np
import os
import csv
import json


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two points (any dimensionality)."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns zero vector if norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# Collision detection
# ─────────────────────────────────────────────────────────────────────────────

def check_collision(
    drone_pos: np.ndarray,
    obstacle_positions: np.ndarray,
    drone_radius: float,
    obstacle_radius: float,
) -> tuple[bool, list[int]]:
    """
    Check whether the drone collides with any obstacle.

    Parameters
    ----------
    drone_pos          : (3,) array — drone position [x, y, z]
    obstacle_positions : (N, 3) array — obstacle positions
    drone_radius       : bounding sphere of the drone (metres)
    obstacle_radius    : bounding sphere of each obstacle (metres)

    Returns
    -------
    colliding  : True if at least one collision detected
    hit_ids    : list of obstacle indices that collide with the drone
    """
    threshold = drone_radius + obstacle_radius
    hit_ids = []
    for i, obs_pos in enumerate(obstacle_positions):
        if euclidean_distance(drone_pos, obs_pos) < threshold:
            hit_ids.append(i)
    return len(hit_ids) > 0, hit_ids


def distances_to_obstacles(
    drone_pos: np.ndarray,
    obstacle_positions: np.ndarray,
) -> np.ndarray:
    """
    Compute distance from drone to every obstacle.

    Returns
    -------
    distances : (N,) array of floats
    """
    return np.array([
        euclidean_distance(drone_pos, obs) for obs in obstacle_positions
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Repulsion / avoidance
# ─────────────────────────────────────────────────────────────────────────────

def repulsion_velocity(
    drone_pos: np.ndarray,
    obstacle_positions: np.ndarray,
    safe_distance: float,
    strength: float = 2.0,
) -> np.ndarray:
    """
    Compute a repulsion velocity vector that pushes the drone away from
    nearby obstacles (simple potential-field approach).

    Parameters
    ----------
    drone_pos          : (3,) current drone position
    obstacle_positions : (N, 3) obstacle positions
    safe_distance      : influence radius (metres)
    strength           : scalar multiplier

    Returns
    -------
    v_rep : (3,) repulsion velocity correction
    """
    v_rep = np.zeros(3)
    for obs_pos in obstacle_positions:
        diff = drone_pos - obs_pos
        dist = np.linalg.norm(diff)
        if 0 < dist < safe_distance:
            # Inverse-square repulsion, capped
            magnitude = strength * (1.0 / dist - 1.0 / safe_distance) / (dist ** 2)
            v_rep += magnitude * normalize(diff)
    return v_rep


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_npy(array: np.ndarray, path: str) -> None:
    """Save a NumPy array to *path* (.npy)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, array)


def save_csv(array: np.ndarray, path: str, header: list[str] | None = None) -> None:
    """
    Save a 2-D NumPy array to *path* (.csv).

    Parameters
    ----------
    array  : 2-D numpy array  [rows, cols]
    path   : destination file path
    header : optional list of column names
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(array.tolist())


def save_json(data: dict, path: str) -> None:
    """Persist a dict as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_npy(path: str) -> np.ndarray:
    """Load a .npy file from *path*."""
    return np.load(path, allow_pickle=True)


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix NumPy random seed for reproducibility."""
    np.random.seed(seed)


def print_section(title: str) -> None:
    """Pretty-print a section header."""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
