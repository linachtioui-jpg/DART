# ===== trajectory_generator.py =====
"""
Trajectory Generator
====================
Generates bounded, physically plausible drone trajectories in 2-D or 3-D.

Output shape per trajectory:
    [T, 9]   (3-D mode)  →  [x, y, z, vx, vy, vz, ax, ay, az]
    [T, 6]   (2-D mode)  →  [x, y, vx, vy, ax, ay]

The generator keeps positions inside the configured workspace and caps velocity
magnitude to trajectory.max_velocity.
"""

from __future__ import annotations

import copy
import numpy as np
from scipy.interpolate import splprep, splev

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback for minimal scipy builds
    savgol_filter = None

from .config import DEFAULT_CONFIG


TRAJECTORY_GENERATOR_VERSION = "smooth_dynamics_v1"


def generate_trajectory(config: dict | None = None) -> np.ndarray:
    """
    Generate a single smooth drone trajectory.

    Parameters
    ----------
    config : dict (optional)
        Accepts DEFAULT_CONFIG or any subset with keys:
        simulation_time, time_step, trajectory.

    Returns
    -------
    traj : np.ndarray
        3-D mode → [T, 9]  = position, velocity, acceleration
        2-D mode → [T, 6]  = position, velocity, acceleration
    """
    cfg = _merge_config(config)
    traj_cfg = cfg["trajectory"]
    mode = traj_cfg["mode"]

    if mode not in {"2d", "3d"}:
        raise ValueError(f"Unsupported trajectory mode: {mode!r}. Expected '2d' or '3d'.")

    dt = float(cfg["time_step"])
    if dt <= 0:
        raise ValueError("time_step must be positive.")

    T = int(float(cfg["simulation_time"]) / dt)
    if T < 2:
        raise ValueError("simulation_time/time_step must produce at least two steps.")

    max_velocity = float(traj_cfg["max_velocity"])
    if max_velocity <= 0:
        raise ValueError("trajectory.max_velocity must be positive.")

    waypoints = _sample_waypoints(traj_cfg, mode)
    dense_pos = _sample_spline_path(waypoints, traj_cfg.get("spline_degree", 3), max(5 * T, 250))
    dense_pos = _clip_positions(dense_pos, traj_cfg, mode)

    pos = _resample_with_speed_cap(dense_pos, T, dt, max_velocity)
    pos = _clip_positions(pos, traj_cfg, mode)

    # Keep the position target accurate, but smooth velocity/acceleration labels.
    # Full-feature RMSE growth is often driven by noisy derivative labels, not by
    # bad position prediction. Smoothing the dynamic labels gives the model a
    # cleaner physical target without changing the predicted path.
    vel = _velocity_from_positions(pos, dt, max_velocity)
    vel = _smooth_signal(vel, preferred_window=9, polyorder=3)
    vel = _cap_velocity(vel, max_velocity)
    acc = np.gradient(vel, dt, axis=0)
    acc = _smooth_signal(acc, preferred_window=9, polyorder=3)

    traj = np.concatenate([pos, vel, acc], axis=1)
    return traj.astype(np.float32)


def generate_trajectory_batch(n: int, config: dict | None = None) -> list[np.ndarray]:
    """Generate n independent trajectories."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    return [generate_trajectory(config) for _ in range(n)]


def _merge_config(user_config: dict | None) -> dict:
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


def _sample_waypoints(traj_cfg: dict, mode: str) -> np.ndarray:
    n = int(traj_cfg["n_waypoints"])
    if n < 2:
        raise ValueError("trajectory.n_waypoints must be at least 2.")

    # Sample waypoints slightly inside the official workspace. This avoids
    # clipped/saturated boundary trajectories that create artificial velocity
    # and acceleration labels and inflate full-feature growth.
    x_range = _inner_range(traj_cfg["x_range"], margin_ratio=0.04)
    y_range = _inner_range(traj_cfg["y_range"], margin_ratio=0.04)
    xs = np.random.uniform(*x_range, size=n)
    ys = np.random.uniform(*y_range, size=n)

    if mode == "3d":
        z_range = _inner_range(traj_cfg["z_range"], margin_ratio=0.08)
        zs = np.random.uniform(*z_range, size=n)
        return np.stack([xs, ys, zs], axis=1)
    return np.stack([xs, ys], axis=1)


def _sample_spline_path(waypoints: np.ndarray, degree: int, n_samples: int) -> np.ndarray:
    coords = [waypoints[:, i] for i in range(waypoints.shape[1])]
    k = max(1, min(int(degree), len(waypoints) - 1))
    try:
        tck, _ = splprep(coords, k=k, s=0)
        u = np.linspace(0.0, 1.0, n_samples)
        return np.array(splev(u, tck)).T
    except Exception:
        idx = np.linspace(0, len(waypoints) - 1, n_samples)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(waypoints) - 1)
        alpha = (idx - lo)[:, None]
        return waypoints[lo] * (1.0 - alpha) + waypoints[hi] * alpha


def _clip_positions(pos: np.ndarray, traj_cfg: dict, mode: str) -> np.ndarray:
    clipped = pos.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], *traj_cfg["x_range"])
    clipped[:, 1] = np.clip(clipped[:, 1], *traj_cfg["y_range"])
    if mode == "3d":
        clipped[:, 2] = np.clip(clipped[:, 2], *traj_cfg["z_range"])
    return clipped


def _inner_range(bounds: tuple[float, float], margin_ratio: float) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    width = hi - lo
    if width <= 0:
        raise ValueError(f"Invalid range: {bounds!r}")
    margin = width * float(margin_ratio)
    return lo + margin, hi - margin


def _smooth_signal(values: np.ndarray, preferred_window: int = 9, polyorder: int = 3) -> np.ndarray:
    if values.shape[0] < 5:
        return values.astype(np.float32)

    window = min(int(preferred_window), values.shape[0])
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > values.shape[0]:
        return values.astype(np.float32)

    if savgol_filter is not None:
        return savgol_filter(values, window_length=window, polyorder=polyorder, axis=0, mode="interp").astype(np.float32)

    # Fallback: small centered moving average.
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.stack([
        np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[1])
    ], axis=1)
    return smoothed.astype(np.float32)


def _cap_velocity(vel: np.ndarray, max_velocity: float) -> np.ndarray:
    capped = vel.copy()
    speed = np.linalg.norm(capped, axis=1)
    mask = speed > max_velocity
    if np.any(mask):
        capped[mask] *= (max_velocity / np.maximum(speed[mask], 1e-12))[:, None]
    return capped.astype(np.float32)


def _resample_with_speed_cap(path: np.ndarray, T: int, dt: float, max_velocity: float) -> np.ndarray:
    segment = np.diff(path, axis=0)
    segment_len = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_len)])
    total = cumulative[-1]

    if total <= 1e-9:
        return np.repeat(path[:1], T, axis=0)

    max_distance = max_velocity * dt * (T - 1)
    end_distance = min(total, max_distance)
    target_distance = np.linspace(0.0, end_distance, T)

    dims = []
    for d in range(path.shape[1]):
        dims.append(np.interp(target_distance, cumulative, path[:, d]))
    return np.stack(dims, axis=1)


def _velocity_from_positions(pos: np.ndarray, dt: float, max_velocity: float) -> np.ndarray:
    vel = np.zeros_like(pos)
    vel[:-1] = (pos[1:] - pos[:-1]) / dt
    vel[-1] = vel[-2]

    speed = np.linalg.norm(vel, axis=1)
    mask = speed > max_velocity
    if np.any(mask):
        vel[mask] *= (max_velocity / np.maximum(speed[mask], 1e-12))[:, None]
    return vel


FEATURE_NAMES_3D = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
FEATURE_NAMES_2D = ["x", "y", "vx", "vy", "ax", "ay"]


def get_feature_names(mode: str = "3d") -> list[str]:
    if mode == "3d":
        return FEATURE_NAMES_3D
    if mode == "2d":
        return FEATURE_NAMES_2D
    raise ValueError(f"Unsupported trajectory mode: {mode!r}.")
