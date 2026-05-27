# ===== obstacle_simulation.py =====
"""
Obstacle Simulation
===================
Simulates N dynamic obstacles with three behaviour types:
    • linear      — constant velocity, bounces off workspace walls
    • circular    — orbits a fixed centre point
    • random_walk — Brownian-motion-style random direction changes

Output shape at each timestep:
    [N_obstacles, 6]  →  [x, y, z, vx, vy, vz]

This matches what Member 2's prediction model expects as contextual input.
"""

from __future__ import annotations

import numpy as np
from .config import DEFAULT_CONFIG
from src.pipeline.utils import clamp


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle data structure
# ─────────────────────────────────────────────────────────────────────────────

class Obstacle:
    """
    A single simulated obstacle.

    Attributes
    ----------
    pos      : (3,) current position [x, y, z]
    vel      : (3,) current velocity [vx, vy, vz]
    obs_type : one of {"linear", "circular", "random_walk"}
    radius   : bounding sphere radius (metres)
    """

    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        obs_type: str,
        radius: float,
        max_speed: float,
        workspace: dict,
        rng: np.random.Generator,
    ) -> None:
        self.pos = pos.astype(np.float32)
        self.vel = vel.astype(np.float32)
        self.obs_type = obs_type
        self.radius = radius
        self.max_speed = max_speed
        self.workspace = workspace          # {"x": (lo,hi), "y": ..., "z": ...}
        self._rng = rng

        # Extra state for circular motion
        if obs_type == "circular":
            self._centre = pos.copy()
            self._orbit_radius = float(rng.uniform(0.5, 3.0))
            self._angular_speed = float(rng.uniform(0.3, 1.0))
            self._phase = float(rng.uniform(0, 2 * np.pi))
            self._t = 0.0                   # internal time counter

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, dt: float) -> None:
        """Advance obstacle state by one time-step."""
        if self.obs_type == "linear":
            self._step_linear(dt)
        elif self.obs_type == "circular":
            self._step_circular(dt)
        elif self.obs_type == "random_walk":
            self._step_random_walk(dt)

    def _step_linear(self, dt: float) -> None:
        self.pos += self.vel * dt
        # Bounce off workspace walls
        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = self.workspace[axis]
            if self.pos[i] < lo or self.pos[i] > hi:
                self.vel[i] *= -1.0
                self.pos[i] = clamp(float(self.pos[i]), lo, hi)

    def _step_circular(self, dt: float) -> None:
        self._t += dt
        angle = self._angular_speed * self._t + self._phase
        self.pos[0] = self._centre[0] + self._orbit_radius * np.cos(angle)
        self.pos[1] = self._centre[1] + self._orbit_radius * np.sin(angle)
        # Velocity is tangent to orbit
        self.vel[0] = -self._orbit_radius * self._angular_speed * np.sin(angle)
        self.vel[1] =  self._orbit_radius * self._angular_speed * np.cos(angle)
        self.vel[2] = 0.0

    def _step_random_walk(self, dt: float) -> None:
        # Add small Gaussian noise to velocity, then clamp speed
        noise = self._rng.normal(0, 0.5, size=3).astype(np.float32)
        self.vel += noise * dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        self.pos += self.vel * dt
        # Soft boundary: reflect if outside workspace
        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = self.workspace[axis]
            if self.pos[i] < lo or self.pos[i] > hi:
                self.vel[i] *= -1.0
                self.pos[i] = clamp(float(self.pos[i]), lo, hi)

    # ── State vector ──────────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Return [x, y, z, vx, vy, vz] as float32 array."""
        return np.concatenate([self.pos, self.vel]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleSimulator:
    """
    Manages a collection of Obstacle objects for the full simulation horizon.

    Parameters
    ----------
    config : dict — accepts DEFAULT_CONFIG or partial override
    """

    def __init__(self, config: dict | None = None) -> None:
        import copy
        base = copy.deepcopy(DEFAULT_CONFIG)
        if config:
            _deep_update(base, config)
        self.cfg = base
        self.obs_cfg = base["obstacles"]
        self._rng = np.random.default_rng(base.get("random_seed", 42))
        self.obstacles: list[Obstacle] = []
        self._reset()

    def _reset(self) -> None:
        """(Re-)initialise all obstacles."""
        self.obstacles.clear()
        n = self.obs_cfg["n_obstacles"]
        types_pool = self.obs_cfg["types"]
        workspace = {
            "x": self.obs_cfg["x_range"],
            "y": self.obs_cfg["y_range"],
            "z": self.obs_cfg["z_range"],
        }
        for i in range(n):
            obs_type = types_pool[i % len(types_pool)]
            pos = np.array([
                float(self._rng.uniform(*self.obs_cfg["x_range"])),
                float(self._rng.uniform(*self.obs_cfg["y_range"])),
                float(self._rng.uniform(*self.obs_cfg["z_range"])),
            ], dtype=np.float32)
            speed = float(self._rng.uniform(0.3, self.obs_cfg["max_speed"]))
            direction = self._rng.normal(size=3).astype(np.float32)
            norm = np.linalg.norm(direction)
            vel = direction / norm * speed if norm > 1e-8 else np.zeros(3, dtype=np.float32)
            self.obstacles.append(
                Obstacle(
                    pos=pos,
                    vel=vel,
                    obs_type=obs_type,
                    radius=self.obs_cfg["radius"],
                    max_speed=self.obs_cfg["max_speed"],
                    workspace=workspace,
                    rng=self._rng,
                )
            )

    def step(self, dt: float) -> np.ndarray:
        """
        Advance all obstacles by one time-step.

        Returns
        -------
        states : [N_obstacles, 6] float32
        """
        for obs in self.obstacles:
            obs.step(dt)
        return self.get_states()

    def get_states(self) -> np.ndarray:
        """
        Return current states of all obstacles.

        Returns
        -------
        np.ndarray, shape [N_obstacles, 6]  →  [x, y, z, vx, vy, vz]
        """
        return np.stack([obs.state for obs in self.obstacles], axis=0)

    def get_positions(self) -> np.ndarray:
        """Return [N_obstacles, 3] position array."""
        return np.stack([obs.pos for obs in self.obstacles], axis=0)

    def reset(self) -> None:
        """Reset all obstacles to new random initial conditions."""
        self._reset()


def simulate_obstacles(config: dict | None = None) -> np.ndarray:
    """
    Convenience function: run the full obstacle simulation and return
    the complete state history.

    Parameters
    ----------
    config : dict (optional) — see DEFAULT_CONFIG

    Returns
    -------
    history : np.ndarray, shape [T, N_obstacles, 6]
        Full time-series of all obstacle states.
    """
    import copy
    base = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        _deep_update(base, config)

    T = int(base["simulation_time"] / base["time_step"])
    dt = base["time_step"]

    sim = ObstacleSimulator(config)
    history = []
    for _ in range(T):
        states = sim.step(dt)   # [N, 6]
        history.append(states)

    return np.stack(history, axis=0).astype(np.float32)  # [T, N, 6]


# ─────────────────────────────────────────────────────────────────────────────
# Feature metadata
# ─────────────────────────────────────────────────────────────────────────────

OBSTACLE_FEATURE_NAMES = ["x", "y", "z", "vx", "vy", "vz"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (private)
# ─────────────────────────────────────────────────────────────────────────────

def _deep_update(base: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
