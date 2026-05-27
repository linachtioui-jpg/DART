# ===== config.py =====pipeline
"""
Central configuration module for the drone simulation and dataset pipeline.
All parameters are configurable here — no hardcoded values elsewhere.
"""

DEFAULT_CONFIG = {
    # ── Simulation ──────────────────────────────────────────────────────────
    "simulation_time": 100.0,        # Total simulation duration (seconds)
    "time_step": 0.05,              # dt between steps (seconds) → 20 Hz

    # ── Trajectory ───────────────────────────────────────────────────────────
    "trajectory": {
        "mode": "3d",               # "2d" or "3d"
        "n_waypoints": 8,           # Number of random control waypoints
        "spline_degree": 3,         # Cubic spline (k=3)
        "max_velocity": 3.0,        # m/s — soft cap on waypoint spread
        "x_range": (-10.0, 10.0),   # Workspace bounds (metres)
        "y_range": (-10.0, 10.0),
        "z_range": (0.5, 5.0),      # z always positive (above ground)
    },

    # ── Obstacles ─────────────────────────────────────────────────────────────
    "obstacles": {
        "n_obstacles": 4,
        "types": ["linear", "circular", "random_walk"],
        "radius": 0.5,              # Obstacle bounding sphere (metres)
        "max_speed": 1.5,           # m/s
        "x_range": (-8.0, 8.0),
        "y_range": (-8.0, 8.0),
        "z_range": (0.5, 4.0),
    },

    # ── Dataset ───────────────────────────────────────────────────────────────
    "dataset": {
        "seq_len": 20,              # Past time-steps fed to models
        "future_len": 10,           # Future steps to predict
        "n_trajectories": 500,      # Trajectories generated per dataset build (increased from 200)
        "train_ratio": 0.8,
        "val_ratio": 0.1,           # remainder → test
        "save_dir": "dataset",
    },

    # ── Drone physics (lightweight) ───────────────────────────────────────────
    "drone": {
        "collision_radius": 0.3,    # metres
        "safe_distance": 1.5,       # repulsion kicks in inside this range
    },

    # ── Misc ─────────────────────────────────────────────────────────────────
    "random_seed": 42,
}
