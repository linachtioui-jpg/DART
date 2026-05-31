import time
import numpy as np
import pybullet as p
import pybullet_data
import os
from src.models.trajectory_predictor import TrajectoryPredictor
# ── Simulation constants ───────────────────────────────────────────────────
SEQ_LEN    = 20
N_OBS_FULL = 25
N_OBS_CTRL = 25

# ── Boundary avoidance buffers ─────────────────────────────────────────────
BOUNDARY_BUFFER   = 5.0
BOUNDARY_STRENGTH = 5.0
CRUISE_SPEED = 2.0


# Define a target waypoint sequence (or load from trajectory_generator)
WAYPOINTS = [
    np.array([5.0,  0.0, 2.5]),
    np.array([5.0,  5.0, 3.0]),
    np.array([0.0,  5.0, 2.5]),
    np.array([-5.0, 0.0, 2.0]),
]
WAYPOINT_THRESH = 1.0  # switch to next when within 1 m

TARGET_ALT = 2.2
ARENA_HALF = 30.0
CEIL       = 6.0
FLOOR      = 0.7

# ── Drone physical model ───────────────────────────────────────────────────
MASS        = 0.7
GRAVITY     = 9.81
ARM_LEN     = 0.23
MAX_THRUST  = MASS * GRAVITY * 2.2
MOTOR_TAU   = 0.04

# ── Cascaded controller gains ──────────────────────────────────────────────
VEL_P        = 2.0         # softer velocity tracking
VEL_MAX_ACC  = 4.0         # was 9.0 — limits how hard it pushes
MAX_TILT     = 0.18        # rad ≈ 10° — was 30°, very flat now
ATT_P        = 5.0         # was 9.0 — slower attitude response
ATT_MAX_RATE = 2.5         # was 7.0 — gentle rotation speed
RATE_P       = 0.10        # was 0.15
RATE_D       = 0.008       # more derivative damping to kill wobble
ALT_VEL_P   = 0.40
ALT_POS_P   = 2.5
ALT_I_GAIN  = 0.20
ALT_I_MAX   = 1.5
HOVER_THR   = 0.68

# ── Evasion parameters ────────────────────────────────────────────────────
EVASION_RADIUS  = 5.0
EVASION_SPEED   = 6.0
EVASION_BLEND   = 0.72
TTC_THRESHOLD   = 2.8
MIN_CLOSING     = 0.4
DETECTION_RADIUS = 7.0

# ── Combat (evasion) gains — unlocked when urgency > 0 ───────────────────
# Normal flight stays glassy-smooth (10° max tilt).
# The moment a threat is detected these scale in so dodges are snappy.
COMBAT_VEL_P       = 7.0   # vs cruise 2.0  — tracks velocity setpoint hard
COMBAT_VEL_MAX_ACC = 19.0  # vs cruise 4.0  — allows fierce acceleration
COMBAT_MAX_TILT    = 0.50  # rad ≈ 27°       — real lean during a dodge
COMBAT_ATT_P       = 14.0  # vs cruise 5.0  — snaps to angle fast
COMBAT_ATT_MAX_RATE = 9.0  # vs cruise 2.5  — rapid rotation
COMBAT_RATE_P      = 0.22  # vs cruise 0.10
COMBAT_EVASION_SPEED = 26.0 # vs 12.0        — how fast it throws itself sideways

# ── Emergency safety layer ────────────────────────────────────────────────
EMERGENCY_RADIUS  = 1.5    # m  — hard override trigger distance
EMERGENCY_SPEED   = 14.0    # m/s — safe push-away (reduced to prevent flip)
EMERGENCY_THR     = 0.78   # slightly above hover; full throttle caused flips
EMERGENCY_MAX_TILT = 0.38  # rad ≈ 22° — hard dodge, still stable

# ── Attitude recovery ─────────────────────────────────────────────────────
RECOVERY_TILT_THRESHOLD = np.radians(40)   # if |roll| or |pitch| exceeds this → recover
RECOVERY_THR            = 0.72             # gentle throttle while levelling

from src.control.controller import DroneController


class MLDroneArena:

    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        # In __init__, after p.connect():
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(1.0 / 240.0)

        floor_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(floor_id, -1, rgbaColor=[0.05, 0.05, 0.08, 1.0])

        self._build_walls()
        self.drone_id     = self._load_drone()
        self.obstacle_ids = self._create_obstacles()
        self.drone_shadow     = self._make_shadow(r=0.35, color=[0,0,0,0.35])
        self.obstacle_shadows = [self._make_shadow(r=0.25, color=[0,0,0,0.22])
                                 for _ in self.obstacle_ids]

        self._active_obstacles = {}
        self._next_launch_time = time.time()
        self._launch_index     = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        ctrl_path   = os.path.join(current_dir, "models", "reaction_controller.pth")
        self.controller = DroneController()
        self.controller.load(ctrl_path)
        print("✅ reaction_controller.pth loaded")
        self.predictor = None
        try:
            pred_path = os.path.join(current_dir, "models", "trajectory_predictor.pkl")
            self.predictor = TrajectoryPredictor.load(pred_path, device="cpu")
            print("✅ trajectory_predictor.pkl loaded")
        except Exception as e:
            print(f"⚠️  Predictor not loaded: {e}")

        self.drone_history    = []
        self.obs_history      = []
        self.ctrl_obs_history = []
        self.prev_vel         = np.zeros(3)
        self.prev_rate_des    = np.zeros(3)

        self.motor_thr    = np.full(4, HOVER_THR)
        self.alt_integral = 0.0
        self.prev_ang_vel = np.zeros(3)

        self._evade_vec  = np.zeros(3)
        self._evade_mag  = 0.0
        self._evade_lock = 0.0
        self.current_waypoint = 0
        
        # In __init__:
        self._trail_positions = []
        self._trail_line_ids  = []
        TRAIL_LEN = 80
        # Closest obstacle predicted path
        self._pred_line_ids = []

        

    # ── Scene helpers ──────────────────────────────────────────────────────

    def _build_walls(self):
        h, t  = 8.0, 0.4
        sz    = ARENA_HALF * 2 + t
        color = [0.05, 0.15, 0.35, 0.85]
        for pos, size in [
            ([-ARENA_HALF, 0, h/2], [t, sz, h]),
            ([ ARENA_HALF, 0, h/2], [t, sz, h]),
            ([0, -ARENA_HALF, h/2], [sz, t, h]),
            ([0,  ARENA_HALF, h/2], [sz, t, h]),
        ]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
            vis = p.createVisualShape (p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
            p.createMultiBody(0, col, vis, pos)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ARENA_HALF, ARENA_HALF, 0.2])
        vis = p.createVisualShape (p.GEOM_BOX, halfExtents=[ARENA_HALF, ARENA_HALF, 0.2],
                                   rgbaColor=[0.05, 0.15, 0.35, 0.3])
        p.createMultiBody(0, col, vis, [0, 0, 8.0])

    def _load_drone(self):
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
        col   = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1])
        drone = p.createMultiBody(MASS, col, visual, [0, 0, TARGET_ALT], [0, 0, 0, 1])
        p.changeDynamics(drone, -1, linearDamping=0.1, angularDamping=4.0)
        return drone
    

    def _create_obstacles(self):
        ids = []
        SCALE = [0.25, 0.25, 0.25] # Adjust this to resize your obstacle
        
        # Pre-load the Visual Shape
        vbase = None
        cbase = None
            

        colors = [[1.0, 0.3, 0.3, 1], [0.3, 0.8, 1.0, 1], [1.0, 0.6, 0.2, 1], [0.8, 0.3, 1.0, 1]]
        
        for i in range(N_OBS_FULL):
            angle = np.random.uniform(0, 2*np.pi)
            dist  = np.random.uniform(6, 10)
            z     = np.random.uniform(1.5, 5.5)
            pos   = [dist*np.cos(angle), dist*np.sin(angle), z]
            
            if vbase and cbase:
                # Use the loaded mesh
                oid = p.createMultiBody(baseMass=1.0, 
                                        baseCollisionShapeIndex=cbase, 
                                        baseVisualShapeIndex=vbase, 
                                        basePosition=pos)
            else:
                # Fallback to sphere
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=colors[i % 4])
                oid = p.createMultiBody(1.0, col, vis, pos)
                
            p.changeDynamics(oid, -1, linearDamping=0.02)
            ids.append(oid)
        return ids

    def _make_shadow(self, r=0.3, color=None):
        color = color or [0, 0, 0, 0.3]
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.02, rgbaColor=color)
        return p.createMultiBody(0, -1, vis, [0, 0, -10])

    # ── Obstacle management ────────────────────────────────────────────────

    def _get_obstacle_states(self):
        states = []
        for oid in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            vel, _ = p.getBaseVelocity(oid)
            states.append([*pos, *vel])
        return np.array(states, dtype=np.float32)

    def _closest_obs_for_controller(self, obs_states, drone_pos):
        if len(obs_states) == 0:
            return np.zeros((12, 6), dtype=np.float32)
        dists = np.linalg.norm(obs_states[:, :3] - drone_pos, axis=1)
        return obs_states[np.argsort(dists)].copy()

    # ── Obstacle management (Modified for coherent movement) ────────────────

    def _launch_next_obstacle(self, drone_pos, drone_vel):
        if time.time() < self._next_launch_time:
            return
        oid = self.obstacle_ids[self._launch_index % N_OBS_FULL]
        self._launch_index += 1
        self._next_launch_time = time.time() + 2.5 # Slightly faster frequency

        # Create "Corridors": obstacles travel across the center of the arena
        # Path 0: X-axis sweep, Path 1: Y-axis sweep
        path_type = self._launch_index % 2
        z = np.random.uniform(2.0, 5.0)
        
        if path_type == 0: # Crossing X
            start = [np.random.choice([-15, 15]), np.random.uniform(-5, 5), z]
            end   = [-start[0], np.random.uniform(-5, 5), z]
        else: # Crossing Y
            start = [np.random.uniform(-5, 5), np.random.choice([-15, 15]), z]
            end   = [np.random.uniform(-5, 5), -start[1], z]

        p.resetBasePositionAndOrientation(oid, start, [0,0,0,1])
        p.resetBaseVelocity(oid, [0,0,0], [0,0,0])

        # Store the path
        self._active_obstacles[oid] = np.array(end)
        p.changeDynamics(oid, -1, mass=1.0, linearDamping=0.01)

    def _drive_active_obs(self):
        # Move obstacles at a constant, non-chaotic velocity
        speed = 5.0
        done = []
        for oid, target in list(self._active_obstacles.items()):
            pos = np.array(p.getBasePositionAndOrientation(oid)[0])
            to_target = target - pos
            dist = np.linalg.norm(to_target)
            
            if dist < 0.5:
                done.append(oid)
                continue
            
            # Constant velocity vector towards target
            vel_vec = (to_target / dist) * speed
            p.resetBaseVelocity(oid, vel_vec.tolist(), [0,0,0])
            
        for oid in done:
            del self._active_obstacles[oid]

    # ── Emergency safety layer ─────────────────────────────────────────────

    def _check_emergency(self, drone_pos, obs_states):
        """
        Returns (is_emergency, push_vel_3d).
        Triggers when any obstacle is within EMERGENCY_RADIUS.
        Velocity is directed straight away from the closest threat.
        Kept moderate (EMERGENCY_SPEED) so the attitude loop can follow
        without flipping the drone.
        """
        if len(obs_states) == 0:
            return False, np.zeros(3)

        dists   = np.linalg.norm(obs_states[:, :3] - drone_pos, axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist > EMERGENCY_RADIUS:
            return False, np.zeros(3)

        away = drone_pos - obs_states[min_idx, :3]
        n    = np.linalg.norm(away)
        if n < 1e-6:
            away = np.array([0.0, 0.0, 1.0])
        else:
            away /= n

        # Slight upward bias so the drone doesn't pancake into the floor
        away[2] = max(away[2], 0.15)
        away   /= np.linalg.norm(away)

        return True, away * EMERGENCY_SPEED

    def _is_attitude_unsafe(self, roll, pitch):
        """True when the drone is tilted past the recovery threshold."""
        return (abs(roll) > RECOVERY_TILT_THRESHOLD or
                abs(pitch) > RECOVERY_TILT_THRESHOLD)

    # ── Shared wrench application ──────────────────────────────────────────

    def _apply_wrench(self, base_thr, m_roll, m_pitch, m_yaw, dt):
        """Motor mixing → lag → forces.  Shared by all control modes."""
        motor_cmd = np.array([
            base_thr + m_roll - m_pitch - m_yaw,   # FL
            base_thr - m_roll - m_pitch + m_yaw,   # FR
            base_thr + m_roll + m_pitch + m_yaw,   # RL
            base_thr - m_roll + m_pitch - m_yaw,   # RR
        ])
        alpha_m = dt / (MOTOR_TAU + dt)
        self.motor_thr = ((1.0 - alpha_m) * self.motor_thr
                          + alpha_m * np.clip(motor_cmd, 0.0, 1.0))

        f_per_motor  = MAX_THRUST / 4.0
        thrust_total = float(np.sum(self.motor_thr)) * f_per_motor
        tau_roll  = float((self.motor_thr[0]+self.motor_thr[2])
                         -(self.motor_thr[1]+self.motor_thr[3])) * f_per_motor * ARM_LEN
        tau_pitch = float((self.motor_thr[2]+self.motor_thr[3])
                         -(self.motor_thr[0]+self.motor_thr[1])) * f_per_motor * ARM_LEN
        YAW_DRAG  = 0.06
        tau_yaw   = float((self.motor_thr[1]+self.motor_thr[2])
                         -(self.motor_thr[0]+self.motor_thr[3])) * f_per_motor * YAW_DRAG

        p.applyExternalForce (self.drone_id, -1, [0, 0, thrust_total], [0,0,0], p.LINK_FRAME)
        p.applyExternalTorque(self.drone_id, -1, [tau_roll, tau_pitch, tau_yaw], p.LINK_FRAME)

    # ── Shared inner attitude + rate loops ────────────────────────────────

    def _attitude_to_wrench(self, roll, pitch, yaw,
                            ang_vel, dt,
                            roll_des, pitch_des, yaw_rate_des,
                            base_thr, max_tilt=None,
                            att_p=None, att_max_rate=None, rate_p=None):
        """
        Runs loops 3 & 4 (attitude → rates → motor differential).
        Returns (m_roll, m_pitch, m_yaw) without modifying motor state.
        Optional gain overrides let the combat path run hotter than cruise.
        """
        _max_tilt    = max_tilt     if max_tilt     is not None else MAX_TILT
        _att_p       = att_p        if att_p        is not None else ATT_P
        _att_max_rate= att_max_rate if att_max_rate is not None else ATT_MAX_RATE
        _rate_p      = rate_p       if rate_p       is not None else RATE_P

        roll_des  = np.clip(roll_des,  -_max_tilt, _max_tilt)
        pitch_des = np.clip(pitch_des, -_max_tilt, _max_tilt)

        roll_rate_des  = np.clip((roll_des  - roll)  * _att_p, -_att_max_rate, _att_max_rate)
        pitch_rate_des = np.clip((pitch_des - pitch) * _att_p, -_att_max_rate, _att_max_rate)

        SLEW_LIMIT = 0.2
        roll_rate_des  = np.clip(roll_rate_des,
                                 self.prev_rate_des[0] - SLEW_LIMIT,
                                 self.prev_rate_des[0] + SLEW_LIMIT)
        pitch_rate_des = np.clip(pitch_rate_des,
                                 self.prev_rate_des[1] - SLEW_LIMIT,
                                 self.prev_rate_des[1] + SLEW_LIMIT)
        self.prev_rate_des = np.array([roll_rate_des, pitch_rate_des, yaw_rate_des])

        d_roll  = (ang_vel[0] - self.prev_ang_vel[0]) / dt
        d_pitch = (ang_vel[1] - self.prev_ang_vel[1]) / dt
        self.prev_ang_vel = ang_vel.copy()

        m_roll  = (roll_rate_des  - ang_vel[0]) * _rate_p - d_roll  * RATE_D
        m_pitch = (pitch_rate_des - ang_vel[1]) * _rate_p - d_pitch * RATE_D
        m_yaw   = (yaw_rate_des   - ang_vel[2]) * 0.06

        return m_roll, m_pitch, m_yaw

    # ── Evasion layer ──────────────────────────────────────────────────────

    def _compute_evasion(self, drone_pos, drone_vel, obs_states, dt):
        """obstacle evasion + boundary only"""
        best_urgency = 0.0
        best_evade   = np.zeros(3)

        # obstacle Evasion
        for obs in obs_states:
            obs_pos  = obs[:3].astype(float)
            obs_vel  = obs[3:6].astype(float)
            rel_pos  = obs_pos - drone_pos
            dist     = np.linalg.norm(rel_pos)
            
            if dist > EVASION_RADIUS or dist < 1e-6:
                continue
                
            threat_dir = rel_pos / dist
            rel_vel    = obs_vel - drone_vel
            closing    = -np.dot(threat_dir, rel_vel)
            
            if closing < MIN_CLOSING:
                continue
                
            ttc = dist / (closing + 1e-6)
            if ttc > TTC_THRESHOLD:
                continue
                
            urgency = np.clip(
                (1.0 - ttc / TTC_THRESHOLD) * (1.0 - dist / EVASION_RADIUS), 
                0.0, 1.0)
            
            if urgency > best_urgency:
                horiz = np.array([threat_dir[0], threat_dir[1], 0.0])
                hlen  = np.linalg.norm(horiz)
                if hlen < 0.1:
                    horiz = np.array([1.0, 0.0, 0.0])
                else:
                    horiz /= hlen

                perp_a = np.array([-horiz[1],  horiz[0], 0.0])
                perp_b = np.array([ horiz[1], -horiz[0], 0.0])
                obs_h  = np.array([obs_vel[0], obs_vel[1], 0.0])
                perp   = perp_a if -np.dot(perp_a, obs_h) >= -np.dot(perp_b, obs_h) else perp_b

                raw = 0.85 * perp + 0.15 * (-horiz)
                n   = np.linalg.norm(raw)
                best_evade   = raw / n if n > 1e-6 else raw
                best_urgency = urgency

        # Boundary avoidance
        margin = BOUNDARY_BUFFER
        boundary_vel = np.zeros(3)
        
        for dim in [0, 1]:
            pos_dim = drone_pos[dim]
            dist_pos = ARENA_HALF - pos_dim
            dist_neg = pos_dim - (-ARENA_HALF)
            
            if dist_pos < margin:
                boundary_vel[dim] -= BOUNDARY_STRENGTH / (dist_pos + 0.1)**2
            if dist_neg < margin:
                boundary_vel[dim] += BOUNDARY_STRENGTH / (dist_neg + 0.1)**2

        if drone_pos[2] < FLOOR + margin:
            boundary_vel[2] += BOUNDARY_STRENGTH / ((drone_pos[2] - FLOOR) + 0.1)**2
        elif drone_pos[2] > CEIL - margin:
            boundary_vel[2] -= BOUNDARY_STRENGTH / ((CEIL - drone_pos[2]) + 0.1)**2

        # Final combination
        final_evade = best_evade * best_urgency + boundary_vel * 0.7

        n = np.linalg.norm(final_evade)
        if n > 1e-6:
            final_evade /= n

        urgency = best_urgency

        # Update internal state
        if best_urgency > 0.12:
            self._evade_vec  = best_evade
            self._evade_mag  = best_urgency
            self._evade_lock = 0.35
        else:
            self._evade_lock = max(0.0, self._evade_lock - dt)
            if self._evade_lock <= 0.0:
                self._evade_mag = max(0.0, self._evade_mag - dt * 2.5)

        return final_evade * EVASION_SPEED, urgency, boundary_vel

    def _predict_future_obs(self, drone_history, obs_history):
        """Return predicted future obstacle positions (shape: N_obs, future_len, 3)."""
        if self.predictor is None or len(drone_history) < SEQ_LEN:
            return None
        try:
            drone_seq = np.array(drone_history[-SEQ_LEN:], dtype=np.float32)  # (20, 9)
            obs_seq   = np.array(obs_history[-SEQ_LEN:], dtype=np.float32)    # (20, N_obs, 6)
            # predictor.predict() expects batched input: (1, seq_len, features)
            pred = self.predictor.predict(drone_seq[np.newaxis], obs_seq[np.newaxis])
            return pred[0]  # (future_len, 9) — predicted drone future states
        except Exception as e:
            return None

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        print("🚀 Realistic drone physics (cascaded PID + motor dynamics)")
        step = 0
        dt   = 1.0 / 240.0

        while True:
            t0 = time.perf_counter()
            p.stepSimulation()

            # ── State readout ──────────────────────────────────────────────
            pos, orn         = p.getBasePositionAndOrientation(self.drone_id)
            vel, ang_vel     = p.getBaseVelocity(self.drone_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)

            pos     = np.array(pos,     dtype=float)
            vel     = np.array(vel,     dtype=float)
            ang_vel = np.array(ang_vel, dtype=float)


            acc = (vel - self.prev_vel) / dt
            self.prev_vel = vel.copy()
            
            obs_all  = self._get_obstacle_states()
            obs_ctrl = self._closest_obs_for_controller(obs_all, pos)

            p.resetDebugVisualizerCamera(3.5, 40, -28, pos.tolist())

            # Motion trail — remove oldest line, add newest segment
            self._trail_positions.append(pos.copy())
            if len(self._trail_positions) > 60:
                self._trail_positions.pop(0)
                if self._trail_line_ids:
                    p.removeUserDebugItem(self._trail_line_ids.pop(0))
            if len(self._trail_positions) >= 2:
                alpha = 1.0
                fade  = alpha * (len(self._trail_positions) / 60)
                lid   = p.addUserDebugLine(
                    self._trail_positions[-2].tolist(),
                    self._trail_positions[-1].tolist(),
                    lineColorRGB=[0.2 + 0.6*fade, 0.5*fade, 1.0],
                    lineWidth=1.5 + fade)
                self._trail_line_ids.append(lid)
            self._launch_next_obstacle(pos, vel)
            self._drive_active_obs()

            # Predicted path of closest obstacle — full redraw every 6 steps
            if step % 6 == 0 and len(obs_all) > 0:
                for lid in self._pred_line_ids:
                    p.removeUserDebugItem(lid)
                self._pred_line_ids.clear()

                dists   = np.linalg.norm(obs_all[:, :3] - pos, axis=1)
                closest = obs_all[int(np.argmin(dists))]
                o_pos   = closest[:3].astype(float)
                o_vel   = closest[3:6].astype(float)
                dist_now = float(np.linalg.norm(o_pos - pos))

                if dist_now < EVASION_RADIUS and np.linalg.norm(o_vel) > 0.3:
                    heat      = float(np.clip(1.0 - dist_now / EVASION_RADIUS, 0, 1))
                    predicted = o_pos + o_vel * 2.0   # where it'll be in 2 seconds
                    self._pred_line_ids.append(p.addUserDebugLine(
                        o_pos.tolist(), predicted.tolist(),
                        lineColorRGB=[1.0, 1.0 - heat, 1.0 - heat],
                        lineWidth=2.5))

        

            # History for trained controller
            drone_state = np.concatenate([pos, vel, acc]).astype(np.float32)
            self.drone_history.append(drone_state)
            self.obs_history.append(obs_all.copy())
            self.ctrl_obs_history.append(obs_ctrl.copy())
            if len(self.drone_history) > SEQ_LEN:
                self.drone_history.pop(0)
                self.obs_history.pop(0)
                self.ctrl_obs_history.pop(0)

            # ══════════════════════════════════════════════════════════════
            # PRIORITY 1 — Attitude recovery
            #   If the drone is badly tilted (e.g. from a dodge that went
            #   wrong) level it first before anything else. Full throttle
            #   while inverted pushes you into the ground; gentle hover
            #   throttle + hard level command is safer.
            # ══════════════════════════════════════════════════════════════
            if self._is_attitude_unsafe(roll, pitch):
                if step % 60 == 0:
                    print(f"Step {step:5d} | 🔄 RECOVERY "
                          f"roll={np.degrees(roll):+.1f}° pitch={np.degrees(pitch):+.1f}°")
                m_roll, m_pitch, m_yaw = self._attitude_to_wrench(
                    roll, pitch, yaw, ang_vel, dt,
                    roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0,
                    base_thr=RECOVERY_THR)
                self._apply_wrench(RECOVERY_THR, m_roll, m_pitch, m_yaw, dt)
                step += 1
                time.sleep(max(0, dt - (time.perf_counter() - t0)))
                continue

            # ══════════════════════════════════════════════════════════════
            # PRIORITY 2 — Emergency dodge (≤ 1 m)
            #   Uses moderate speed + tighter tilt limit so the inner loops
            #   can actually follow the command without flipping.
            # ══════════════════════════════════════════════════════════════
            is_emergency, emergency_vel = self._check_emergency(pos, obs_all)

            if is_emergency:
                if step % 60== 0:
                    print(f"Step {step:5d} | ⚠️  EMERGENCY DODGE")

                vx_err = emergency_vel[0] - vel[0]
                vy_err = emergency_vel[1] - vel[1]
                ax_des = np.clip(vx_err * VEL_P, -VEL_MAX_ACC, VEL_MAX_ACC)
                ay_des = np.clip(vy_err * VEL_P, -VEL_MAX_ACC, VEL_MAX_ACC)

                c_y, s_y  = np.cos(yaw), np.sin(yaw)
                ax_body   =  ax_des * c_y + ay_des * s_y
                ay_body   = -ax_des * s_y + ay_des * c_y

                # Tighter angle limit prevents the flip
                pitch_des = np.clip(-np.arctan2(ax_body, GRAVITY),
                                    -EMERGENCY_MAX_TILT, EMERGENCY_MAX_TILT)
                roll_des  = np.clip( np.arctan2(ay_body, GRAVITY),
                                    -EMERGENCY_MAX_TILT, EMERGENCY_MAX_TILT)

                m_roll, m_pitch, m_yaw = self._attitude_to_wrench(
                    roll, pitch, yaw, ang_vel, dt,
                    roll_des=roll_des, pitch_des=pitch_des, yaw_rate_des=0.0,
                    base_thr=EMERGENCY_THR, max_tilt=EMERGENCY_MAX_TILT)

                # Tilt-compensated throttle — keeps altitude during the bank
                tilt_comp = 1.0 / max(np.cos(roll) * np.cos(pitch), 0.4)
                base_thr  = float(np.clip(EMERGENCY_THR * tilt_comp, 0.05, 0.95))
                if not (np.isfinite(m_roll) and np.isfinite(m_pitch) and np.isfinite(m_yaw) and np.isfinite(base_thr)):
                    print(f"⚠️ NaN or Inf detected! roll={m_roll}, pitch={m_pitch}, yaw={m_yaw}, thr={base_thr}")
                    # Reset to safe defaults to prevent crash
                    m_roll, m_pitch, m_yaw = 0, 0, 0
                    base_thr = HOVER_THR
                self._apply_wrench(base_thr, m_roll, m_pitch, m_yaw, dt)
                step += 1
                time.sleep(max(0, dt - (time.perf_counter() - t0)))
                continue

            # ══════════════════════════════════════════════════════════════
            # PRIORITY 3 — Normal flight (ML controller + evasion blend)
            # ══════════════════════════════════════════════════════════════

            # ── Waypoint pursuit (replaces the ML controller or supplements it) ──
            target   = WAYPOINTS[self.current_waypoint]
            to_goal  = target - pos
            dist     = np.linalg.norm(to_goal)

            if dist < WAYPOINT_THRESH:
                self.current_waypoint = (self.current_waypoint + 1) % len(WAYPOINTS)
                target  = WAYPOINTS[self.current_waypoint]
                to_goal = target - pos
                dist    = np.linalg.norm(to_goal) + 1e-6

            direction = to_goal / dist
            vx_des       = direction[0] * CRUISE_SPEED
            vy_des       = direction[1] * CRUISE_SPEED
            vz_des       = direction[2] * CRUISE_SPEED
            yaw_rate_des = 0.0

            # ML controller overrides waypoint velocity when buffer is full
            if len(self.drone_history) == SEQ_LEN:
                x_ctrl = np.concatenate([
                    np.concatenate([self.drone_history[t],
                                    self.ctrl_obs_history[t].flatten()])
                    for t in range(SEQ_LEN)
                ]).astype(np.float32)
                action       = self.controller.predict_action(x_ctrl)
                vx_des       = float(np.clip(action[0] * 3.0, -7.0, 7.0))
                vy_des       = float(np.clip(action[1] * 3.0, -7.0, 7.0))
                vz_des       = float(np.clip(action[2] * 1.6, -5.0, 5.0))
                yaw_rate_des = float(action[3] * 1.5)

            evade_vel, urgency, boundary_vel = self._compute_evasion(pos, vel, obs_all, dt)
            blend = np.clip(urgency * 2.5, 0.0, 1.0)
            # After: evade_vel, urgency, boundary_vel = self._compute_evasion(...)

            # ── Predictive layer: dodge obstacles' future positions ──────────────────
            predicted_future = self._predict_future_obs(self.drone_history, self.obs_history)
            if predicted_future is not None:
                # predicted_future shape: (future_len, 9) — columns 0:3 are x,y,z
                future_pos = predicted_future[:, :3]           # next N steps of drone prediction
                # Use first predicted step to warn of upcoming collision
                pred_threat = future_pos[0]                    # where drone is predicted to be in 1 step
                pred_dist_to_obs = np.linalg.norm(obs_all[:, :3] - pred_threat, axis=1).min()
                if pred_dist_to_obs < EVASION_RADIUS * 1.2:    # widen evasion when predictor warns
                    urgency = min(1.0, urgency + 0.2)
                    blend   = np.clip(urgency * 2.5, 0.0, 1.0)

            # ── Interpolate cruise ↔ combat gains based on urgency ─────────
            # At urgency=0 the drone is glassy-smooth.
            # At urgency=0.5+ it's operating at full combat aggressiveness.
            u = blend   # shorthand
            eff_vel_p       = VEL_P       + u * (COMBAT_VEL_P       - VEL_P)
            eff_vel_max_acc = VEL_MAX_ACC + u * (COMBAT_VEL_MAX_ACC  - VEL_MAX_ACC)
            eff_max_tilt    = MAX_TILT    + u * (COMBAT_MAX_TILT     - MAX_TILT)
            eff_att_p       = ATT_P       + u * (COMBAT_ATT_P        - ATT_P)
            eff_att_max_rate= ATT_MAX_RATE+ u * (COMBAT_ATT_MAX_RATE - ATT_MAX_RATE)
            eff_rate_p      = RATE_P      + u * (COMBAT_RATE_P       - RATE_P)
            eff_evade_speed = 12.0        + u * (COMBAT_EVASION_SPEED - 12.0)

            # boundary_vel is always added — walls repel even with zero obstacle urgency
            vx_des_w = (1.0 - blend) * vx_des + blend * evade_vel[0] * eff_evade_speed + boundary_vel[0]
            vy_des_w = (1.0 - blend) * vy_des + blend * evade_vel[1] * eff_evade_speed + boundary_vel[1]
            vz_des   = vz_des + boundary_vel[2]

            # ── LOOP 1 — Velocity → desired acceleration ───────────────────
            vx_err = vx_des_w - vel[0]
            vy_err = vy_des_w - vel[1]
            ax_des = np.clip(vx_err * eff_vel_p, -eff_vel_max_acc, eff_vel_max_acc)
            ay_des = np.clip(vy_err * eff_vel_p, -eff_vel_max_acc, eff_vel_max_acc)

            # ── LOOP 2 — Desired acceleration → tilt angles ────────────────
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            ax_body  =  ax_des * c_y + ay_des * s_y
            ay_body  = -ax_des * s_y + ay_des * c_y

            pitch_des = np.clip(-np.arctan2(ax_body, GRAVITY), -eff_max_tilt, eff_max_tilt)
            roll_des  = np.clip( np.arctan2(ay_body, GRAVITY), -eff_max_tilt, eff_max_tilt)

            # ── LOOPS 3 & 4 — Attitude + rate (combat-scaled) ─────────────
            m_roll, m_pitch, m_yaw = self._attitude_to_wrench(
                roll, pitch, yaw, ang_vel, dt,
                roll_des=roll_des, pitch_des=pitch_des,
                yaw_rate_des=yaw_rate_des,
                base_thr=HOVER_THR,
                att_p=eff_att_p,
                att_max_rate=eff_att_max_rate,
                rate_p=eff_rate_p)

            # ── Altitude PID → base throttle ───────────────────────────────
            if pos[2] > CEIL - 0.4:
                vz_des = min(vz_des, -4.0)

            vz_err = vz_des - vel[2]
            self.alt_integral += (TARGET_ALT - pos[2]) * dt
            self.alt_integral  = np.clip(self.alt_integral, -ALT_I_MAX, ALT_I_MAX)

            tilt_comp    = 1.0 / max(np.cos(roll) * np.cos(pitch), 0.3)
            urgency_boost = urgency * 0.25

            base_thr = (
                (HOVER_THR + urgency_boost) * tilt_comp
                + vz_err * ALT_VEL_P
                + (TARGET_ALT - pos[2]) * (ALT_POS_P / MAX_THRUST)
                + self.alt_integral * (ALT_I_GAIN / MAX_THRUST)
            )

            if pos[2] < FLOOR + 0.8:
                cushion_boost = (FLOOR + 0.8 - pos[2]) * 15.0
                base_thr += cushion_boost / MAX_THRUST

            base_thr = float(np.clip(base_thr, 0.05, 1.0))

            self._apply_wrench(base_thr, m_roll, m_pitch, m_yaw, dt)

            if step % 60 == 0:
                min_d     = float(min(np.linalg.norm(o[:3] - pos) for o in obs_all))
                evade_str = f"EVADE urgency={urgency:.2f}" if urgency > 0.05 else "nominal"
                print(f"Step {step:5d} | alt={pos[2]:.2f} | "
                      f"roll={np.degrees(roll):+.1f}° pitch={np.degrees(pitch):+.1f}° | "
                      f"closest={min_d:.2f}m | {evade_str}")

            step += 1
            time.sleep(max(0, dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    MLDroneArena().run()