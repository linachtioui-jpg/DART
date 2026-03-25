import sys
import os

# STEP 1: Tell Python WHERE to look (The "Bridge")
# This MUST happen before you try to import from simulation
project_root = os.path.dirname(__file__)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# STEP 2: Now that the bridge is built, import your function
# Use the function name you actually defined in drone_env.py
try:
    from simulation.drone_env import start_sim
except ImportError as e:
    print(f"Import Error: {e}")
    print("Check if src/simulation/__init__.py exists!")
    sys.exit(1)

# STEP 3: The Start Button
if __name__ == "__main__":
    print("--- Project: Drone Trajectory Prediction & Control ---")
    print("Initializing Simulation...")
    start_sim()