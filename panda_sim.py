import pybullet as p
import pybullet_data
import time


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
        self.load_assets()

    def load_assets(self):
        self.target_obj = self.bullet_client.loadURDF("assets/cube/cube.urdf", [0, 0.65, 0.82], globalScaling=0.04, flags=self.flags)
        self.bullet_client.changeVisualShape(self.target_obj, -1, rgbaColor=[1, 0, 0, 1])
        self.bullet_client.loadURDF("assets/table/table.urdf", [0, 0.42, 0], [0, 0, 0, 1], flags=self.flags)
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


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.5])

    panda = PandaSim(p)
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
    p.disconnect()

