import numpy as np
import torch
import os
from models.trajectory_predictor import TrajectoryPredictor
from control.controller import DroneController
from safety import SafetyFilter

def main():
    print(" Initializing unified PPP Flight System...")
    
    # 1. Init Member 2 Predictor 
    predictor = TrajectoryPredictor()
    # Note: Si votre collègue vous a fourni un fichier de poids pour le modèle ML, 
    # chargez-le ici via predictor.load_state_dict(...) si sa classe le supporte.
    
    # 2. Init Your Trained Controller
    controller = DroneController()
    controller.load("models/reaction_controller.pth")
    print(" Loaded trained Behavioral Cloning weights.")
    
    # 3. Init Safety Filter
    safety = SafetyFilter(min_clearance=0.8, max_speed=3.0)
    
    print("\n Starting runtime orchestration loop...")
    
    # Simulate 3 control frames matching your shapes
    for step in range(3):
        print(f"\n--- [ TIMESTEP {step+1} ] ---")
        
        # Fake tracking window matching (20 timesteps, 9 features)
        mock_history = np.random.randn(20, 9)
        
        # Fake current positioning positions
        current_drone_state = np.array([0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]) 
        current_obstacle_state = np.array([0.2, 0.2, 1.1]) # Deliberately placed dangerously close
        
        # Step A: Generate raw reactive intent via your trained network
        raw_action = controller.predict_action(mock_history)
        print(f" MLP Engine Action: vx={raw_action[0]:.2f}, vy={raw_action[1]:.2f}, vz={raw_action[2]:.2f}")
        
        # Step B: Run the actions through safety rules to enforce physical limits
        final_action = safety.filter_action(raw_action, current_drone_state, current_obstacle_state)
        print(f" Final Safe Action: vx={final_action[0]:.2f}, vy={final_action[1]:.2f}, vz={final_action[2]:.2f}")

if __name__ == "__main__":
    main()