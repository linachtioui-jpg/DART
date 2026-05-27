import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# Reaction Controller Network
# =========================================================

from src.pipeline.config import DEFAULT_CONFIG

class ReactionController(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=4):
        super().__init__()
        
        # Calculate dimensions dynamically from config:
        seq_len = DEFAULT_CONFIG["dataset"]["seq_len"]          # 20
        n_obs = DEFAULT_CONFIG["obstacles"]["n_obstacles"]      # 4
        drone_feat = 9                                          # x, y, z, vx, vy, vz, ax, ay, az
        obs_feat = 6                                            # x, y, z, vx, vy, vz
        input_dim = seq_len * (drone_feat + (n_obs * obs_feat)) 

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Now dynamically equals 660!
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

            # flatten input
            x = x_ctrl.flatten()

            x = torch.tensor(
                x,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            action = self.model(x)

            return action.cpu().numpy()[0]

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