import numpy as np

class SafetyFilter:
    def __init__(self, min_clearance=0.5, max_speed=3.0):
        self.min_clearance = min_clearance
        self.max_speed = max_speed

    def filter_action(self, action, current_state, obstacle_state):
        vx, vy, vz, yaw_rate = action
        
        # Extrait les positions 3D (x, y, z) 
        # Dans votre dataset à 9 colonnes, les 3 premières sont généralement la position
        drone_pos = np.array(current_state[:3])
        obs_pos = np.array(obstacle_state[:3])
        
        # Calcul de la distance
        distance_vector = obs_pos - drone_pos
        distance = np.linalg.norm(distance_vector)
        
        # 1. Enforce max speed clipping
        vx = np.clip(vx, -self.max_speed, self.max_speed)
        vy = np.clip(vy, -self.max_speed, self.max_speed)
        vz = np.clip(vz, -self.max_speed, self.max_speed)
        
        # 2. Collision avoidance override
        if distance < self.min_clearance:
            print(f"⚠️ SAFETY FILTER TRIGGERED! Obstacle at {distance:.2f}m!")
            if distance > 0.01:
                # Direction opposée à l'obstacle
                escape_direction = -distance_vector / distance
                push_factor = (self.min_clearance - distance) * 3.0
                vx = escape_direction[0] * push_factor
                vy = escape_direction[1] * push_factor
                vz = escape_direction[2] * push_factor
            else:
                vx, vy, vz = 0.0, 0.0, 0.0 # Emergency freeze
                
        return np.array([vx, vy, vz, yaw_rate])