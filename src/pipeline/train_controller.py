import sys
import os

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.control.controller import DroneController
from src.models.trajectory_predictor import TrajectoryPredictor

os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

# =========================================================
# Load dataset
# =========================================================
print("📦 Loading dataset...")

drone_path   = os.path.join(project_root, "dataset", "train", "drone_past.npy")
obs_path     = os.path.join(project_root, "dataset", "train", "obs_past.npy")
actions_path = os.path.join(project_root, "dataset", "train", "actions.npy")

x_drone     = np.load(drone_path)
x_obstacles = np.load(obs_path)
y_action    = np.load(actions_path).astype(np.float32)

print(f"   drone_past shape    : {x_drone.shape}")
print(f"   obs_past shape      : {x_obstacles.shape}")
print(f"   actions shape       : {y_action.shape}")

N               = x_drone.shape[0]
seq_len         = 20
max_tracked_obs = 25

# =========================================================
# Build 3180-dim past input vectors
# =========================================================
print("🔄 Building 3180-element input vectors...")
combined_samples = []
for idx in range(N):
    window = []
    for t in range(seq_len):
        t_drone = x_drone[idx, t]
        t_obs   = x_obstacles[idx, t, :max_tracked_obs].flatten()
        window.append(np.concatenate([t_drone, t_obs]))
    combined_samples.append(np.array(window).flatten())

x_ctrl = np.array(combined_samples, dtype=np.float32)
print(f"   Past input shape    : {x_ctrl.shape}")

# =========================================================
# Append GRU predictions (90 values) to each input
# =========================================================
print("🔮 Generating GRU predictions for training data...")

pred_path = os.path.join(project_root, "models", "trajectory_predictor.pkl")
predictor = TrajectoryPredictor.load(pred_path, device="cpu")

x_future_list = []
for i in range(N):
    drone_seq = x_drone[i].astype(np.float32)
    obs_seq   = x_obstacles[i].astype(np.float32)
    try:
        pred = predictor.predict(
            drone_seq[np.newaxis], obs_seq[np.newaxis])
        x_future_list.append(pred[0].flatten().astype(np.float32))
    except Exception:
        x_future_list.append(np.zeros(90, dtype=np.float32))
    if i % 5000 == 0:
        print(f"   {i}/{N} samples processed")

x_future = np.array(x_future_list, dtype=np.float32)
x_ctrl   = np.concatenate([x_ctrl, x_future], axis=1)
print(f"   Final input shape   : {x_ctrl.shape}")

# =========================================================
# Tensors
# =========================================================
X = torch.tensor(x_ctrl,   dtype=torch.float32)
Y = torch.tensor(y_action, dtype=torch.float32)

# =========================================================
# Model setup
# =========================================================
epochs     = 20
batch_size = 256

print(f"\n-> Network input_dim=3270, hidden_size=384")

controller = DroneController(input_dim=3270, hidden_size=384)
criterion  = nn.MSELoss()
optimizer  = optim.Adam(controller.model.parameters(), lr=0.001)

# =========================================================
# Training loop
# =========================================================
print(f"🚀 Training over {epochs} epochs...\n")

for epoch in range(epochs):
    permutation = torch.randperm(X.size(0))
    epoch_loss  = 0.0

    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X[indices].to(controller.device)
        batch_y = Y[indices].to(controller.device)
        optimizer.zero_grad()
        loss = criterion(controller.model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.6f}")

# =========================================================
# Save
# =========================================================
output_path = os.path.join(project_root, "models", "reaction_controller.pth")
controller.save(output_path)
print(f"\n✅ Model saved to '{output_path}'")
print(f"   Input dim: 3270 (3180 past states + 90 GRU prediction)")