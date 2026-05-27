import sys
import os

# Path fix
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # go up to PPP-drone root

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Project root added: {project_root}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.control.controller import DroneController

# =========================================================
# Create missing directories
# =========================================================
os.makedirs("models", exist_ok=True)

# =========================================================
# Load dataset (Updated to match your actual filenames)
# =========================================================
print("📦 Loading dataset components...")
x_ctrl = np.load("dataset/train/drone_past.npy")
y_action = np.load("dataset/train/actions.npy")

print("-> Initial drone_past shape:", x_ctrl.shape)
print("-> Initial actions shape:", y_action.shape)

# Flatten the time-steps and features into a single continuous window
N = x_ctrl.shape[0]
x_ctrl = x_ctrl.reshape(N, -1)
print("-> Flattened training input shape:", x_ctrl.shape)

# Ensure our neural network input matches the dataset dimension dynamically
input_dim = x_ctrl.shape[1]
print(f"-> Network will auto-configure to input_dim = {input_dim}")

# =========================================================
# Convert to tensors
# =========================================================
X = torch.tensor(x_ctrl, dtype=torch.float32)
Y = torch.tensor(y_action, dtype=torch.float32)

# =========================================================
# Model Setup
# =========================================================
controller = DroneController()

# Dynamically adjust the controller's underlying model layers if shape differs from 660
if input_dim != 660:
    controller.model.network[0] = nn.Linear(input_dim, 128)

criterion = nn.MSELoss()
optimizer = optim.Adam(controller.model.parameters(), lr=0.001)

# =========================================================
# Training loop
# =========================================================
epochs = 20
batch_size = 256

print(f"\n🚀 Training reaction network over {epochs} epochs...")

for epoch in range(epochs):
    permutation = torch.randperm(X.size(0))
    epoch_loss = 0.0

    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X[indices]
        batch_y = Y[indices]

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
controller.save("models/reaction_controller.pth")
print("\n✅ Controller weights trained and successfully saved to 'models/reaction_controller.pth'!")