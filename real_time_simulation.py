# real_time_simulation.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.join(current_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, "src"))

import pybullet as p
import pybullet_data
import numpy as np
import time
import torch

from models.trajectory_predictor import TrajectoryPredictor
from control.controller import DroneController
from safety import SafetyFilter


class RealDroneSciFiArena:
    def __init__(self, gui=True):
        print("🌌 Initializing Clean Sci-Fi Drone Arena...")

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        self._setup_arena()
        self.drone_id = self._load_custom_drone("src/models/drone.obj")

        self.seq_len = 20
        self.n_obstacles = 12
        self.drone_history = []
        self.obs_history = []

        self._load_models()
        self.safety = SafetyFilter(min_clearance=0.8, max_speed=3.0)
        self.obstacle_ids = self._create_dynamic_objects()

        # Force clean start
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 3.0], 
                                         p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])

    def _setup_arena(self):
        plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane, -1, rgbaColor=[0.01, 0.01, 0.06, 1.0])

        self._create_wall([-12, 0, 3], [0.3, 25, 6], [0, 0.9, 1, 0.6])
        self._create_wall([12, 0, 3], [0.3, 25, 6], [0, 0.9, 1, 0.6])
        self._create_wall([0, -12, 3], [25, 0.3, 6], [0, 0.9, 1, 0.6])
        self._create_wall([0, 12, 3], [25, 0.3, 6], [0, 0.9, 1, 0.6])

    def _create_wall(self, pos, size, color):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
        p.createMultiBody(0, col, vis, pos)

    def _load_custom_drone(self, obj_path):
        """Load any .obj file as a drone with a collision box underneath"""

        corrective_quat = p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0])  # Rotate model to face forward

        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[0.002, 0.002, 0.002],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            visualFrameOrientation=corrective_quat
        )

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.06]
        )

        drone_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, 2.5],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

        p.changeDynamics(drone_id, -1, 
                        linearDamping=0.15, 
                        angularDamping=40.0,
                        lateralFriction=0.9,
                        rollingFriction=0.1, 
                        spinningFriction=0.1, 
                        restitution=0.0,
                        localInertiaDiagonal=[0.012, 0.012, 0.018])

        return drone_id    

    def _create_dynamic_objects(self):
        obs_ids = []
        colors = [[1,0.3,0.6,1], [0.2,1,0.9,1], [1,0.8,0.1,1], [0.7,0.3,1,1]]
        for i in range(self.n_obstacles):
            shape = p.GEOM_SPHERE if i % 2 == 0 else p.GEOM_BOX
            size = 0.42
            if shape == p.GEOM_SPHERE:
                col = p.createCollisionShape(shape, radius=size)
                vis = p.createVisualShape(shape, radius=size, rgbaColor=colors[i % len(colors)])
            else:
                col = p.createCollisionShape(shape, halfExtents=[size]*3)
                vis = p.createVisualShape(shape, halfExtents=[size]*3, rgbaColor=colors[i % len(colors)])
            pos = [np.random.uniform(-9,9), np.random.uniform(-9,9), np.random.uniform(1.5,6)]
            oid = p.createMultiBody(0.6, col, vis, pos)
            obs_ids.append(oid)
        return obs_ids

    def _move_obstacles(self):
        t = time.time()
        for i, oid in enumerate(self.obstacle_ids):
            angle = t * 0.6 + i * 1.4
            speed = 1.1
            vx = np.cos(angle) * speed
            vy = np.sin(angle * 0.8) * speed
            p.resetBaseVelocity(oid, [vx, vy, 0.2 * np.sin(t + i)])

    def _draw_predicted_path(self, predicted):
        if predicted is None or len(predicted) < 2:
            return
        for i in range(len(predicted)-1):
            p.addUserDebugLine(
                predicted[i][:3], 
                predicted[i+1][:3],
                lineColorRGB=[1.0, 0.2, 0.0],
                lineWidth=4.0,
                lifeTime=0.15
            )

    def _load_models(self):
        try:
            self.predictor = TrajectoryPredictor.load("src/models/trajectory_predictor.py", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Predictor loaded")
        except:
            self.predictor = None

        try:
            self.controller = DroneController()
            self.controller.load("models/reaction_controller.pth")
            print("✅ Controller loaded")
        except:
            print("Creating fresh controller...")
            self.controller = self._create_fresh_controller()

    def _create_fresh_controller(self):
        controller = DroneController()
        input_dim = self.seq_len * (9 + self.n_obstacles * 6)
        controller.model.network[0] = torch.nn.Linear(input_dim, 128)
        print(f"✅ Fresh controller created (input_dim={input_dim})")
        return controller

    def run(self, max_steps=10000):
        print("🌌 Simulation Running...")

        for step in range(max_steps):
            p.stepSimulation()
            self._move_obstacles()
            time.sleep(1./240.)

            # Get current state
            pos, orn = p.getBasePositionAndOrientation(self.drone_id)
            vel, ang_vel = p.getBaseVelocity(self.drone_id)
            
            drone_state = np.array([*pos, *vel, 0.,0.,0.], dtype=np.float32)

            self.drone_history.append(drone_state)
            if len(self.drone_history) > self.seq_len:
                self.drone_history.pop(0)

            obs_states = self.get_obstacle_states()
            self.obs_history.append(obs_states)
            if len(self.obs_history) > self.seq_len:
                self.obs_history.pop(0)

            # Check orientation to evaluate stability (threshold around ~23 degrees)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            is_unstable = abs(roll) > 0.4 or abs(pitch) > 0.4

            # ===================== MUTUALLY EXCLUSIVE CONTROL =====================
            if len(self.drone_history) >= self.seq_len:
                
                # Condition A: Run strong stabilization if drone tilts excessively OR if ML agent is unavailable
                if is_unstable or not self.controller:
                    target_z = 3.0
                    height_error = target_z - pos[2]

                    thrust = np.array([0.0, 0.0, 0.5 * 9.81 + height_error * 18.0])

                    torque = np.array([
                        -roll * 80.0 - ang_vel[0] * 25.0,     # Strong on X axis (roll)
                        -pitch * 60.0 - ang_vel[1] * 18.0,    # Strong on Y axis (pitch)
                        -ang_vel[2] * 22.0
                    ])

                    p.applyExternalForce(self.drone_id, -1, thrust, [0,0,0], p.WORLD_FRAME)
                    p.applyExternalTorque(self.drone_id, -1, torque, p.WORLD_FRAME)
                
                # Condition B: ML tracking control takes full priority during normal, upright flight
                elif self.controller:
                    drone_past = np.array(self.drone_history)
                    obs_past = np.array(self.obs_history)

                    predicted = None
                    if self.predictor:
                        try:
                            predicted = self.predictor.predict(drone_past, obs_past)
                            self._draw_predicted_path(predicted)
                        except:
                            pass

                    obs_flat = obs_past.reshape(self.seq_len, -1)
                    x_ctrl = np.concatenate([drone_past, obs_flat], axis=1)

                    raw = self.controller.predict_action(x_ctrl)
                    final = self.safety.filter_action(raw, drone_state, obs_states[0] if len(obs_states)>0 else None)

                    # Exclusive target trajectory force calculation
                    force = (np.array(final[:3]) - vel) * 25.0
                    p.applyExternalForce(self.drone_id, -1, force, [0,0,0], p.WORLD_FRAME)
                    
                    # Mild passive stabilization torque to counter minor external drifts without overriding the network
                    passive_torque = np.array([-roll * 15.0, -pitch * 15.0, -ang_vel[2] * 5.0])
                    p.applyExternalTorque(self.drone_id, -1, passive_torque, p.WORLD_FRAME)
            # ======================================================================

            if step % 25 == 0:
                print(f"Step {step:4d} | Drone @ [{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}]")

        p.disconnect()

    def get_obstacle_states(self):
        states = []
        for oid in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            vel, _ = p.getBaseVelocity(oid)
            states.append([*pos, *vel])
        return np.array(states, dtype=np.float32)


if __name__ == "__main__":
    sim = RealDroneSciFiArena(gui=True)
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n🌌 Simulation ended.")
        p.disconnect()
    except Exception as e:
        print(f"Error: {e}")
        p.disconnect()