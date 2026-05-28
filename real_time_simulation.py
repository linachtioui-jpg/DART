import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

# ── Model imports ──────────────────────────────────────────────────────────────
try:
    from models.trajectory_predictor import TrajectoryPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("⚠️  trajectory_predictor not found → running without predictor")

try:
    from control.controller import DroneController
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    print("⚠️  control/controller not found → running without NN controller")

# ── Tuning ─────────────────────────────────────────────────────────────────────
TARGET_ALT   = 2.5
DODGE_RADIUS = 4.5
PANIC_RADIUS = 2.0
ARENA_HALF   = 12.0        # wall distance from center
WALL_PUSH    = 14.0
SEQ_LEN      = 20
N_OBS        = 4           # how many obstacle states the NN expects
LATENCY_COMP = 0.06        # seconds to predict obstacle position ahead
# ──────────────────────────────────────────────────────────────────────────────


class RealDroneSciFiArena:
    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Floor
        floor = p.loadURDF("plane.urdf")
        p.changeVisualShape(floor, -1, rgbaColor=[0.05, 0.05, 0.08, 1.0])

        self._build_walls()
        self.drone_id      = self._load_custom_drone()
        self.obstacle_ids  = self._create_moving_rocks()

        # Shadows
        self.drone_shadow     = self._make_shadow(r=0.35, color=[0,0,0,0.35])
        self.obstacle_shadows = [self._make_shadow(r=0.25, color=[0,0,0,0.22])
                                 for _ in self.obstacle_ids]
        
        # In __init__, after creating obstacles:
        self._launched = set()
        self._active_rocks = {}   # oid -> target_pos (np.array)       # tracks which rocks have been fired
        self._next_launch_time = time.time()
        self._launch_index = 0       # which rock fires next

        # ── Load trajectory predictor ──────────────────────────────────────────
        self.predictor = None
        if PREDICTOR_AVAILABLE:
            try:
                path = os.path.join(current_dir, "models", "trajectory_predictor.pkl")
                self.predictor = TrajectoryPredictor.load(path, device="cpu")
                print("✅ Trajectory Predictor loaded")
            except Exception as e:
                print(f"⚠️  Predictor load failed: {e}")

        # ── Load reaction controller ───────────────────────────────────────────
        self.controller = None
        if CONTROLLER_AVAILABLE:
            try:
                self.controller = DroneController()
                path = os.path.join(current_dir, "models", "reaction_controller.pth")
                self.controller.load(path)
                print("✅ Reaction Controller loaded")
            except Exception as e:
                print(f"⚠️  Controller load failed: {e}")

        # History buffers for NN
        self.drone_history = []
        self.obs_history   = []

        # Smooth command state
        self.vx_old = self.vy_old = self.vz_old = self.yaw_old = 0.0

        print(f"✅ Obstacles : {len(self.obstacle_ids)}")
        print(f"   Predictor : {'ON' if self.predictor   else 'OFF'}")
        print(f"   Controller: {'ON' if self.controller  else 'OFF'}")

    # ── Arena construction ─────────────────────────────────────────────────────

    def _build_walls(self):
        h  = 8.0          # wall height
        t  = 0.4          # wall thickness
        sz = ARENA_HALF * 2 + t
        color = [0.05, 0.15, 0.35, 0.85]

        walls = [
            ([-ARENA_HALF, 0,    h/2], [t,  sz, h]),
            ([ ARENA_HALF, 0,    h/2], [t,  sz, h]),
            ([0, -ARENA_HALF,    h/2], [sz, t,  h]),
            ([0,  ARENA_HALF,    h/2], [sz, t,  h]),
        ]
        for pos, size in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
            vis = p.createVisualShape( p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
            p.createMultiBody(0, col, vis, pos)

        # Ceiling
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ARENA_HALF, ARENA_HALF, 0.2])
        vis = p.createVisualShape( p.GEOM_BOX, halfExtents=[ARENA_HALF, ARENA_HALF, 0.2],
                                   rgbaColor=[0.05, 0.15, 0.35, 0.3])
        p.createMultiBody(0, col, vis, [0, 0, 8.0])

    def _load_custom_drone(self):
        corrective = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        try:
            visual = p.createVisualShape(
                p.GEOM_MESH,
                fileName=os.path.join("src", "models", "drone.obj"),
                meshScale=[0.003]*3,
                visualFrameOrientation=corrective,
                rgbaColor=[0.7, 0.85, 1.0, 1.0])
        except Exception:
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1],
                                         rgbaColor=[0.7, 0.85, 1.0, 1.0])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1])
        drone = p.createMultiBody(0.7, collision, visual, [0, 0, TARGET_ALT], [0,0,0,1])
        p.changeDynamics(drone, -1, linearDamping=0.2, angularDamping=12.0)
        return drone

    def _create_moving_rocks(self):
        obs_ids = []
        try:
            visual_base = p.createVisualShape(
                p.GEOM_MESH,
                fileName=os.path.join("src", "models", "rock.obj"),
                meshScale=[0.5]*3)
        except Exception:
            visual_base = None

        colors = [[1.0,0.3,0.3,1], [0.3,0.8,1.0,1], [1.0,0.6,0.2,1], [0.8,0.3,1.0,1]]

        for i in range(25):
            angle = np.random.uniform(0, 2*np.pi)
            dist  = np.random.uniform(6, 10)
            z     = np.random.uniform(1.5, 5.5)
            pos   = [dist * np.cos(angle), dist * np.sin(angle), z]

            if visual_base is not None:
                vis = visual_base
            else:
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3,
                                          rgbaColor=colors[i % len(colors)])

            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            oid = p.createMultiBody(0, col, vis, pos)
            p.changeDynamics(oid, -1, mass=1.0, linearDamping=0.02)
            obs_ids.append(oid)

        return obs_ids

    def _make_shadow(self, r=0.3, color=[0,0,0,0.3]):
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.02, rgbaColor=color)
        return p.createMultiBody(0, -1, vis, [0, 0, -10])

    # ── Obstacle helpers ───────────────────────────────────────────────────────

    def _get_obstacle_states(self):
        """Return list of [px,py,pz, vx,vy,vz] with latency compensation."""
        states = []
        for oid in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            vel, _ = p.getBaseVelocity(oid)
            lc = LATENCY_COMP
            states.append([
                pos[0] + vel[0]*lc, pos[1] + vel[1]*lc, pos[2] + vel[2]*lc,
                vel[0], vel[1], vel[2]
            ])
        return np.array(states, dtype=np.float32)

    def _launch_next_rock(self, drone_pos, drone_vel):
        """Fire one rock at a time toward predicted drone intercept position."""
        t = time.time()
        if t < self._next_launch_time:
            return

        oid = self.obstacle_ids[self._launch_index % len(self.obstacle_ids)]
        self._launch_index += 1
        self._next_launch_time = t + 3.5

        # Spawn outside arena
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(8, 11)
        z     = np.random.uniform(1.5, 5.5)
        spawn = [dist * np.cos(angle), dist * np.sin(angle), z]
        p.resetBasePositionAndOrientation(oid, spawn, [0, 0, 0, 1])
        p.resetBaseVelocity(oid, [0, 0, 0], [0, 0, 0])   # ← reset velocity first

        # Predict intercept
        rock_speed   = 6.0
        dist_to_drone = np.linalg.norm(np.array(drone_pos) - np.array(spawn)) + 1e-6
        travel_time  = dist_to_drone / rock_speed
        predicted    = np.array(drone_pos) + np.array(drone_vel) * travel_time * 0.6
        predicted[2] = np.clip(predicted[2], 0.5, 7.0)

        # Store target so we can keep pushing the rock toward it
        self._active_rocks[oid] = predicted

        p.changeDynamics(oid, -1, mass=1.0, linearDamping=0.01)

        print(f"🪨 Rock launched → intercept "
            f"({predicted[0]:.1f}, {predicted[1]:.1f}, {predicted[2]:.1f})")

    # ── Reactive dodge (geometry, runs every frame) ───────────────────────────

    def _drive_active_rocks(self, drone_pos):
        """Apply steering force to in-flight rocks each step so they don't just fall."""
        ROCK_SPEED   = 6.0      # m/s target speed
        GRAVITY_COMP = 9.81     # counteract gravity (rock mass=1 kg)
        done = []

        for oid, target in self._active_rocks.items():
            pos, _ = p.getBasePositionAndOrientation(oid)
            pos    = np.array(pos)

            # Direction toward locked-in target
            to_target = target - pos
            dist      = np.linalg.norm(to_target)

            if dist < 0.5:          # close enough — let physics take over
                done.append(oid)
                continue

            direction  = to_target / dist
            force      = direction * ROCK_SPEED * 12.0   # proportional push
            force[2]  += GRAVITY_COMP                    # cancel gravity so it flies level

            p.applyExternalForce(oid, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)

        for oid in done:
            del self._active_rocks[oid]

    def _reactive_dodge(self, pos, vel, obs_states):
        dodge = np.zeros(3)
        panic = False
        for obs in obs_states:
            diff    = np.array(pos) - obs[:3]
            dist    = np.linalg.norm(diff) + 1e-8
            if dist > DODGE_RADIUS:
                continue
            rel_vel = obs[3:6] - np.array(vel)
            closing = max(0.0, -np.dot(diff/dist, rel_vel))
            weight  = (closing * 1.5 + 1.0) / (dist ** 2)
            dodge  += (diff / dist) * weight
            if dist < PANIC_RADIUS:
                panic = True

        scale = 20.0 if panic else 10.0
        return dodge * scale, panic

    # ── Trajectory predictor helper ────────────────────────────────────────────

    def _predict_obs_future(self, obs_states):
        """
        If predictor is available, use it to get predicted future positions.
        Falls back to linear extrapolation if not.
        Returns array of shape (N, 3) — predicted positions.
        """
        if self.predictor is not None and len(self.obs_history) >= SEQ_LEN:
            try:
                obs_seq = np.array(self.obs_history[-SEQ_LEN:], dtype=np.float32)
                predicted = self.predictor.predict(obs_seq)
                return predicted
            except Exception:
                pass
        # Linear fallback: predict 0.3s ahead
        return obs_states[:, :3] + obs_states[:, 3:6] * 0.3

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("🚀 Running...")
        step = 0

        while True:
            t0 = time.perf_counter()
            p.stepSimulation()

            # ── State ──────────────────────────────────────────────────────────
            pos, orn    = p.getBasePositionAndOrientation(self.drone_id)
            vel, ang_vel = p.getBaseVelocity(self.drone_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            pos = np.array(pos)
            vel = np.array(vel)

            # ── Camera ─────────────────────────────────────────────────────────
            p.resetDebugVisualizerCamera(3.5, 40, -28, pos.tolist())

            # ── Steer rocks & get their states ─────────────────────────────────
            self._launch_next_rock(pos, vel)
            self._drive_active_rocks(pos)
            obs_states = self._get_obstacle_states()   # shape (N, 6)

            # ── Update history buffers ─────────────────────────────────────────
            drone_state = np.concatenate([pos, vel, [0.0, 0.0, 0.0]]).astype(np.float32)

            # Always keep exactly N_OBS obstacle entries for the NN
            if len(obs_states) >= N_OBS:
                dists      = [np.linalg.norm(o[:3] - pos) for o in obs_states]
                nn_obs     = obs_states[np.argsort(dists)[:N_OBS]]
            else:
                nn_obs     = np.zeros((N_OBS, 6), dtype=np.float32)
                nn_obs[:len(obs_states)] = obs_states

            self.drone_history.append(drone_state)
            self.obs_history.append(nn_obs)
            if len(self.drone_history) > SEQ_LEN:
                self.drone_history.pop(0)
                self.obs_history.pop(0)

            # ── Base commands ──────────────────────────────────────────────────
            vz_cmd       = (TARGET_ALT - pos[2]) * 3.5
            vx_cmd       = 0.0
            vy_cmd       = 0.0
            yaw_rate_cmd = 0.0

            # ── Reaction controller (NN) ───────────────────────────────────────
            nn_action = np.zeros(4)
            if self.controller is not None and len(self.drone_history) == SEQ_LEN:
                try:
                    drone_arr = np.array(self.drone_history, dtype=np.float32)
                    obs_arr   = np.array(self.obs_history,   dtype=np.float32)
                    input_vec = np.concatenate([
                        np.concatenate([drone_arr[t], obs_arr[t].flatten()])
                        for t in range(SEQ_LEN)
                    ])
                    nn_action = self.controller.predict_action(input_vec)
                    # Blend NN into commands
                    vx_cmd       += nn_action[0] * 8.0
                    vy_cmd       += nn_action[1] * 8.0
                    vz_cmd       += nn_action[3] * 5.0
                    yaw_rate_cmd += nn_action[2] * 3.0
                except Exception as e:
                    pass   # NN failed this frame — geometry dodge still runs

            # ── Trajectory predictor: dodge predicted future positions ──────────
            if len(obs_states) > 0:
                future_pos = self._predict_obs_future(obs_states)
                # Build fake "stationary" states at predicted positions for dodge
                future_states = np.concatenate(
                    [future_pos, np.zeros_like(future_pos)], axis=1)
                pred_dodge, _ = self._reactive_dodge(pos, vel, future_states)
                vx_cmd += pred_dodge[0] * 0.4
                vy_cmd += pred_dodge[1] * 0.4
                vz_cmd += pred_dodge[2] * 0.3

            # ── Reactive geometry dodge (current positions, highest priority) ───
            dodge_vec, panic = self._reactive_dodge(pos, vel, obs_states)
            if panic:
                # Override everything in panic
                vx_cmd = dodge_vec[0]
                vy_cmd = dodge_vec[1]
                vz_cmd = dodge_vec[2]
            else:
                vx_cmd += dodge_vec[0]
                vy_cmd += dodge_vec[1]
                vz_cmd += dodge_vec[2]

            # ── Wall repulsion ─────────────────────────────────────────────────
            limit = ARENA_HALF - 2.5
            if pos[0] >  limit: vx_cmd -= (pos[0] -  limit) * WALL_PUSH
            if pos[0] < -limit: vx_cmd += (-limit - pos[0]) * WALL_PUSH
            if pos[1] >  limit: vy_cmd -= (pos[1] -  limit) * WALL_PUSH
            if pos[1] < -limit: vy_cmd += (-limit - pos[1]) * WALL_PUSH

            CEIL =5.0   # your ceiling is at 8.0, stay well below it
            FLOOR = 0.5
            if pos[2] > CEIL:  vz_cmd -= (pos[2] - CEIL)  * WALL_PUSH
            if pos[2] < FLOOR: vz_cmd += (FLOOR - pos[2]) * WALL_PUSH

            # ── Smooth commands ────────────────────────────────────────────────
            a = 0.5   # smoothing (lower = snappier)
            vx_cmd       = a * self.vx_old  + (1-a) * vx_cmd
            vy_cmd       = a * self.vy_old  + (1-a) * vy_cmd
            vz_cmd       = a * self.vz_old  + (1-a) * vz_cmd
            yaw_rate_cmd = a * self.yaw_old + (1-a) * yaw_rate_cmd
            self.vx_old, self.vy_old = vx_cmd, vy_cmd
            self.vz_old, self.yaw_old = vz_cmd, yaw_rate_cmd

            # ── World-frame rotation ───────────────────────────────────────────
            c, s   = np.cos(yaw), np.sin(yaw)
            vx_w   = vx_cmd * c - vy_cmd * s
            vy_w   = vx_cmd * s + vy_cmd * c

            t_roll  = np.clip(-vy_w * 1.6, -0.85, 0.85)
            t_pitch = np.clip( vx_w * 1.6, -0.85, 0.85)

            thrust_z = 9.81*0.7 + (vz_cmd - vel[2])*90.0 + (TARGET_ALT - pos[2])*20.0
            thrust   = [0, 0, max(0.0, thrust_z)]
            torque   = [
                (t_roll  - roll)  * 900.0 - ang_vel[0] * 45.0,
                (t_pitch - pitch) * 900.0 - ang_vel[1] * 45.0,
                (yaw_rate_cmd - ang_vel[2]) * 90.0,
            ]

            p.applyExternalForce( self.drone_id, -1, thrust, [0,0,0], p.WORLD_FRAME)
            p.applyExternalTorque(self.drone_id, -1, torque,           p.WORLD_FRAME)

            # ── Shadows ────────────────────────────────────────────────────────
            p.resetBasePositionAndOrientation(
                self.drone_shadow, [pos[0], pos[1], 0.02], [0,0,0,1])
            for oid, sid in zip(self.obstacle_ids, self.obstacle_shadows):
                opos = p.getBasePositionAndOrientation(oid)[0]
                p.resetBasePositionAndOrientation(sid, [opos[0], opos[1], 0.02], [0,0,0,1])

            # ── Logging ────────────────────────────────────────────────────────
            if step % 60 == 0 and len(obs_states) > 0:
                min_d  = min(np.linalg.norm(o[:3] - pos) for o in obs_states)
                status = "🚨PANIC" if panic else ("⚠️ DODGE" if min_d < DODGE_RADIUS else "   OK  ")
                print(f"Step {step:5d} | {status} | alt={pos[2]:.2f} | closest={min_d:.2f}m "
                      f"| NN={'ON' if self.controller else 'OFF'} "
                      f"| Pred={'ON' if self.predictor else 'OFF'}")

            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 1/240 - elapsed))


if __name__ == "__main__":
    RealDroneSciFiArena().run()