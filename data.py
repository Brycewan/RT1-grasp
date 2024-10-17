import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from panda_sim import PandaSim

p.connect(p.GUI)

# simulator configuration
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(enableConeFriction=0)
p.setTimeStep(1./240.)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # disable GUI
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.5])

# output settings
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pose_data_list = []

# initialize enviroment
sim = PandaSim(p)
sim.load_assets()

while True:
    # reset robot and randomly reset the target objects
    sim.reset_robot()
    sim.reset_target()
    
    target_pos, target_ori = sim.bullet_client.getBasePositionAndOrientation(sim.target_obj)
    target_ori = p.getQuaternionFromEuler([0, -np.pi, 0])
    grasp_pos = list(target_pos)
    grasp_pos[2] += 0.1
    
    # inverse kinematics
    joint_poses = sim.inverse_kinematics(grasp_pos, target_ori)
    sim.move_robot(joint_poses)
    for i in range(24):
        sim.bullet_client.stepSimulation()
        time.sleep(1./24.)
    p.stepSimulation()
p.disconnect()
