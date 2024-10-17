import pybullet as p
import pybullet_data
import time
import random
import numpy as np

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client):
        self.bullet_client = bullet_client
        # Set solverResidualThreshold to 0 for physics engine parameter
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0) 
        # enable cached, improve rendering performance
        self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES 

        # Load assets
        self.robot = None
        self.target_obj = None
        
        # robot parameters
        self.initial_joint_poses = [1.22, -0.458, 0.31, -2.0, 0.20, 1.56, 2.32, 0.04, 0.04]
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.DOF = 7
        self.EndEffectorIndex = 11
        self.gripper_range = [0.01, 0.04]
        self.finger_index = [9, 10]

    def load_assets(self):
        self.target_obj = self.bullet_client.loadURDF("assets/cube/cube.urdf", [0, 0.65, 0.82], globalScaling=0.04, flags=self.flags)
        self.bullet_client.changeVisualShape(self.target_obj, -1, rgbaColor=[1, 0, 0, 1])
        self.bullet_client.loadURDF("assets/table/table.urdf", [0, 0.35, 0], [0, 0, 0, 1], flags=self.flags)
        self.robot = self.bullet_client.loadURDF("assets/franka_panda/panda.urdf", [0, 0, 0.62], [0, 0, 0, 1], useFixedBase=True, flags=self.flags)
        # create a constraint to keep the fingers centered, 9 and 10 for finger indices
        c = self.bullet_client.createConstraint(
            self.robot,
            9,
            self.robot,
            10,
            jointType=self.bullet_client.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def reset_robot(self, random_robot=False):
        if random_robot:
            for i in range(self.arm_dof):
                self.initial_joint_poses[i] += random.uniform(-np.pi / 10, np.pi / 10)
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.robot)):
            self.bullet_client.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_type = self.bullet_client.getJointInfo(self.robot, j)[2]
            if joint_type in [
                self.bullet_client.JOINT_PRISMATIC,
                self.bullet_client.JOINT_REVOLUTE,
            ]:
                self.bullet_client.resetJointState(self.robot, j, self.initial_joint_poses[index])
                index = index + 1
        
        # time.sleep(1)
        
    def reset_target(self, target_pos=None, target_orn=None, random_target=True):
        tar_obj_range = [-0.2, 0.2, 0.3, 0.7, -np.pi, np.pi]  # [x_min, x_max, y_min, y_max, rot_min, rot_max]
        if random_target:
            # Generate random position and rotation within the specified range
            x = random.uniform(tar_obj_range[0], tar_obj_range[1])
            y = random.uniform(tar_obj_range[2], tar_obj_range[3])
            r = random.uniform(tar_obj_range[4], tar_obj_range[5])
            pos = [x, y, 0.665]  # Set the z-coordinate (height) of the object
            rot = p.getQuaternionFromEuler(
                [0, -np.pi, 0]
            )  # Convert Euler angles to quaternion
        else:
            x, y = target_pos[0], target_pos[1]
            pos = [x, y, 0.645]
            
        # Reset the target object's position and orientation
        self.bullet_client.resetBasePositionAndOrientation(self.target_obj, pos, rot)
        
        # time.sleep(1)
        
    def inverse_kinematics(self, pos, orn=None):
        if orn is None:
            joint_poses = self.bullet_client.calculateInverseKinematics(
                self.robot, 
                self.EndEffectorIndex, 
                pos, 
                ll, 
                ul, 
                jr, 
                self.initial_joint_poses, 
                maxNumIterations=100,
                solver=p.IK_DLS
            )
        else:
            joint_poses = self.bullet_client.calculateInverseKinematics(
                self.robot, 
                self.EndEffectorIndex, 
                pos, 
                orn, 
                ll, 
                ul, 
                jr, 
                self.initial_joint_poses, 
                maxNumIterations=100,
                solver=p.IK_DLS
            )
        return joint_poses
    
    def move_robot(self, joint_poses):
        for i in range(self.DOF):
            self.bullet_client.setJointMotorControl2(
                bodyUniqueId=self.robot, 
                jointIndex=i, 
                controlMode=self.bullet_client.POSITION_CONTROL, 
                targetPosition=joint_poses[i], 
                force=500
            )
            
    def move_gripper(self, gripper_state):
        gripper_pos = (
            gripper_state * (self.gripper_range[1] - self.gripper_range[0])
            + self.gripper_range[0]
        )
        for i, j_idx in enumerate(self.finger_index):
            self.bullet_client.setJointMotorControl2(
                self.robot,
                j_idx,
                self.bullet_client.POSITION_CONTROL,
                gripper_pos,
                maxVelocity=0.1,
                force=1000,
            )


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.5])

    # initialize enviroment
    panda = PandaSim(p)
    panda.load_assets()
    
    while True:
        panda.reset_robot()
        panda.reset_target()
        
        for i in range(240):
            panda.bullet_client.stepSimulation()
            time.sleep(1./240.)        
    p.disconnect()

