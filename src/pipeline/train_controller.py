import sys
import os
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # PPP-drone root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added: {project_root}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Direct import — no importlib, no stale cache
from src.control.controller import DroneController
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# Create missing directories
# =========================================================
os.makedirs("models", exist_ok=True)

# =========================================================
# Load dataset (Updated to track 3180 features)
# =========================================================
print("📦 Loading dataset components...")

# Paths assume running from project root. Adjust if files are inside src/pipeline/dataset
drone_path = "dataset/train/drone_past.npy"
obs_path = "dataset/train/obs_past.npy"     # Verify your team's exact file name here!
actions_path = "dataset/train/actions.npy"

if not os.path.exists(drone_path):
    # Fallback to check local directory if executed inside src/pipeline/
    drone_path = os.path.join(project_root, drone_path)
    obs_path = os.path.join(project_root, obs_path)
    actions_path = os.path.join(project_root, actions_path)

x_drone = np.load(drone_path)        # Expected Shape: (N, 20, 9)
x_obstacles = np.load(obs_path)      # Expected Shape: (N, 20, N_obs, 6)
y_action = np.load(actions_path)     # Expected Shape: (N, 4)

print("-> Initial drone_past shape:", x_drone.shape)
print("-> Initial objects_past shape:", x_obstacles.shape)
print("-> Initial actions shape:", y_action.shape)

# --- GLOBAL MATRICES CONSTANTS ---
N = x_drone.shape[0]
seq_len = 20
max_tracked_obs =25

print("🔄 Processing raw logs into unified 3180-element sequences...")
combined_samples = []

for idx in range(N):
    timestep_windows = []
    for t in range(seq_len):
        t_drone = x_drone[idx, t] 
        t_obs = x_obstacles[idx, t, :max_tracked_obs].flatten()

        # Merge drone + obstacles for this timestep
        step_features = np.concatenate([t_drone, t_obs])
        timestep_windows.append(step_features)
        
    # Flatten all timesteps into one vector
    combined_samples.append(np.array(timestep_windows).flatten())

x_ctrl = np.array(combined_samples, dtype=np.float32)
print("-> Unified training input shape:", x_ctrl.shape)
input_dim = x_ctrl.shape[1]
print(f"-> Final input_dim = {input_dim}")

# =========================================================
# Global Tensors & Variables Definition (Clears Pylance Warnings)
# =========================================================
X = torch.tensor(x_ctrl, dtype=torch.float32)
Y = torch.tensor(y_action, dtype=torch.float32)

epochs = 20
batch_size = 256

# =========================================================
# Model Setup
# =========================================================
print(f"-> Network will use input_dim={input_dim}, hidden_size=384")

# Création propre du modèle (on ne touche plus manuellement aux layers)
controller = DroneController(
    input_dim=3180,      # Doit être 660 ou 3180 selon ta logique
    hidden_size=384
)

criterion = nn.MSELoss()
optimizer = optim.Adam(controller.model.parameters(), lr=0.001)

# =========================================================
# Training loop
# =========================================================
print(f"\n🚀 Training reaction network over {epochs} epochs...")

for epoch in range(epochs):
    permutation = torch.randperm(X.size(0))
    epoch_loss = 0.0

    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X[indices].to(controller.device)
        batch_y = Y[indices].to(controller.device)
        optimizer.zero_grad()
        predictions = controller.model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.6f}")

# =========================================================
# Save model
# =========================================================
# Outputs directly to project root models directory
output_model_path = os.path.join(project_root, "models", "reaction_controller.pth")
controller.save(output_model_path)
print(f"\n✅ Controller weights trained and successfully saved to '{output_model_path}'!")