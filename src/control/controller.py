import numpy as np
import torch
import torch.nn as nn
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ReactionController(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 384, output_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DroneController:
    def __init__(self, input_dim=3180, hidden_size=384, model_path=None):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ReactionController(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=4
        ).to(self.device)

        if model_path:
            self.load(model_path)

    def predict_action(self, x_ctrl):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x_ctrl).float().unsqueeze(0).to(self.device)
            return self.model(x).cpu().numpy()[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()