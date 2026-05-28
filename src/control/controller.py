import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Inject it into Python's search paths
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =========================================================
# Reaction Controller Network
# =========================================================

from src.pipeline.config import DEFAULT_CONFIG

class ReactionController(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=4):
        super().__init__()
        
        # Calculate dimensions dynamically from config:
        seq_len = DEFAULT_CONFIG["dataset"]["seq_len"]                       # 20
        # ✨ CHANGE THIS: Track fixed subset instead of entire simulation count
        max_obs = DEFAULT_CONFIG["obstacles"]["max_tracked_obstacles"]       # 4 
        
        drone_feat = 9                                                       # x, y, z, vx, vy, vz, ax, ay, az
        obs_feat = 6                                                         # x, y, z, vx, vy, vz
        
        # Math is locked strictly at: 20 * (9 + (4 * 6)) = 660 elements
        input_dim = seq_len * (drone_feat + (max_obs * obs_feat)) 

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.network(x)


# =========================================================
# Controller Wrapper
# =========================================================

class DroneController:
    def __init__(self, model_path=None):

        self.device = torch.device("cpu")

        self.model = ReactionController().to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    # -----------------------------------------------------
    # Predict drone action
    # -----------------------------------------------------
    def predict_action(self, x_ctrl):
        self.model.eval()
        with torch.no_grad():
            # If your model was trained on flattened input, use:
            # x = torch.from_numpy(x_ctrl).float().unsqueeze(0).to(self.device)
            
            # If your model is an LSTM/RNN, use:
            # x = torch.from_numpy(x_ctrl).float().view(1, self.seq_len, -1).to(self.device)
            
            x = torch.from_numpy(x_ctrl).float().unsqueeze(0).to(self.device)
            
            action = self.model(x)
            
            # Debug: Check if the model is actually outputting movement
            act_np = action.cpu().numpy()[0]
            # print(f"DEBUG: Model Output: {act_np}") # Uncomment to verify
            
            return act_np

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()