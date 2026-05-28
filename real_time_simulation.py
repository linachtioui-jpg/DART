import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from control.controller import DroneController
except ImportError:
    DroneController = None

class RealDroneSciFiArena:
    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        # REMOVE GUI WINDOWS
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        p.loadURDF("plane.urdf")
        self.drone_id = self._load_custom_drone()
        self.obstacle_ids = self._create_moving_rocks()
        self.controller = DroneController() if DroneController else None
        print("Controller:", self.controller)
        print("DroneController import:", DroneController)
        print(f"✅ Obstacles created: {len(self.obstacle_ids)}")

    def _load_custom_drone(self):
        corrective = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        visual = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join("src", "models", "drone.obj"), 
                                     meshScale=[0.003]*3, visualFrameOrientation=corrective)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1])
        # Drone at 2.5m
        drone = p.createMultiBody(0.7, collision, visual, [0, 0, 2.5], [0,0,0,1])
        return drone

    def _create_moving_rocks(self):
        obs_ids = []
        visual = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join("src", "models", "rock.obj"), meshScale=[0.5]*3)
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)

        for _ in range(12):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(5, 9)
            # Spawn at varied heights, not just 2.0
            z = np.random.uniform(1.5, 4.5)
            pos = [dist * np.cos(angle), dist * np.sin(angle), z]

            oid = p.createMultiBody(0, collision, visual, pos)
            p.changeDynamics(oid, -1, mass=0)
            obs_ids.append(oid)

        return obs_ids

    def run(self):
        print("🚀 Running...")
        step = 0
        while True:
            p.stepSimulation()

            pos, orn = p.getBasePositionAndOrientation(self.drone_id)
            vel, ang_vel = p.getBaseVelocity(self.drone_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)

            # --- Obstacle states ---
            obs_states = []
            # --- Re-aim rocks at drone every frame ---
            for oid in self.obstacle_ids:
                opos, _ = p.getBasePositionAndOrientation(oid)
                target = np.array(pos)   # drone's current position
                diff = target - np.array(opos)
                dist = np.linalg.norm(diff) + 1e-6
                direction = diff / dist
                speed = 2.5
                p.resetBaseVelocity(oid, (direction * speed).tolist())

            # --- Reactive dodge (no NN needed, runs every frame) ---
            vx_cmd = vy_cmd = vz_cmd = 0.0
            for obs in obs_states:
                diff = np.array(pos) - obs[:3]
                dist = np.linalg.norm(diff) + 1e-6
                if dist < 4.0:
                    weight = 1.0 / (dist ** 2)
                    vx_cmd += (diff[0] / dist) * weight * 10.0
                    vy_cmd += (diff[1] / dist) * weight * 10.0
                    vz_cmd += (diff[2] / dist) * weight * 5.0

            # --- Altitude hold ---
            vz_cmd += (2.5 - pos[2]) * 3.0

            # --- Apply forces ---
            c, s = np.cos(yaw), np.sin(yaw)
            vx_w = vx_cmd * c - vy_cmd * s
            vy_w = vx_cmd * s + vy_cmd * c

            target_roll  = np.clip(-vy_w * 1.5, -0.8, 0.8)
            target_pitch = np.clip( vx_w * 1.5, -0.8, 0.8)

            thrust_z = 9.81 * 0.7 + (vz_cmd - vel[2]) * 80.0 + (2.5 - pos[2]) * 15.0
            thrust = [0, 0, max(0.0, thrust_z)]
            torque = [
                (target_roll  - roll)  * 800.0 - ang_vel[0] * 40.0,
                (target_pitch - pitch) * 800.0 - ang_vel[1] * 40.0,
                -ang_vel[2] * 80.0
            ]

            p.applyExternalForce(self.drone_id, -1, thrust, [0,0,0], p.WORLD_FRAME)
            p.applyExternalTorque(self.drone_id, -1, torque, p.WORLD_FRAME)

            if step % 60 == 0:
                if len(obs_states) > 0:
                    min_d = min(np.linalg.norm(np.array(pos) - o[:3]) for o in obs_states)
                    print(f"Step {step:5d} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | closest={min_d:.2f}m")
                else:
                    print(f"Step {step:5d} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | no obstacles found!")

            step += 1
            time.sleep(1/240)


if __name__ == "__main__":
    RealDroneSciFiArena().run()