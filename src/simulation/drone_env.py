import pybullet as p
import pybullet_data
import time

def start_sim():
    # This line MUST happen first
    client = p.connect(p.GUI) 
    
    # Check if connection actually worked
    if client < 0:
        print("Error: Could not connect to PyBullet GUI.")
        return

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    print("Simulation Initialized and Connected.")
    
    # The Loop
    for _ in range(500):
        try:
            p.stepSimulation()
            time.sleep(1./240.)
        except p.error:
            print("Physics server disconnected.")
            break
            
    p.disconnect()