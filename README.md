# Drone Trajectory Prediction & Reactive Control System (PPP)

This repository contains the integrated pipeline for a closed-loop drone navigation system. The architecture relies on an advanced **Sense-Plan-Act** design splits across three primary core modules: Simulation Environment, Trajectory Forecasting, and Reactive Neural Control.

## System Architecture Diagram



1. **Environment (Member 1)**: Tracks live physics telemetry vectors inside PyBullet.
2. **Trajectory Predictor (Member 2)**: Processes position tracking data with Savitzky-Golay smoothing to generate trajectory estimates.
3. **Reactive Controller & Safety Filter (Member 3 - Your Contribution)**: Uses Behavioral Cloning to map historical timelines into flight velocities, wrapped in an axiomatic safety override envelope.

---

## Core Component Overview (Control & Reaction Module)

* `controller.py`: An MLP Neural Network mapping historical multi-modal states ($20 \text{ timesteps} \times 9 \text{ features} = 180\text{-dim}$) directly to directional execution velocities ($[v_x, v_y, v_z, \text{yaw\_rate}]$).
* `safety.py`: A non-ML deterministic protection boundary that intercepts flight coordinates if a clearance breach ($< 0.8\text{m}$) is detected, enforcing reactive evasion vectors.
* `train_controller.py`: The imitation learning loop utilizing behavioral cloning across $52,000$ telemetry frames to fit the weights configuration.
* `run_loop.py`: The live integration framework showcasing how inputs pipe from perception to physical application frames.

---

## Getting Started

### 1. Installation & Environment Setup
Clone the repository and install the dependencies inside your virtual environment:
```bash
pip install -r requirements.txt