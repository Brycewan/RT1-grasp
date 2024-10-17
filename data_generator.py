import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from panda_sim import PandaSim
from PIL import Image


# Dictionary defining various camera views along with their configuration parameters
CAM_INFO = {
    "front": [
        [0, 0, 0.7],
        1.8,
        180,
        -20,
        0,
        40,
    ],  # Front view: [position], distance, angles, fov
    "fronttop": [
        [0, 0.5, 0.7],
        1.5,
        180,
        -60,
        0,
        35,
    ],  # Front-top view: [position], distance, angles, fov
    "topdown": [
        [0, 0.35, 0],
        2.0,
        0,
        -90,
        0,
        45,
    ],  # Top-down view: [position], distance, angles, fov
    "side": [
        [0, 0.35, 0.9],
        1.5,
        90,
        0,
        0,
        40,
    ],  # Side view: [position], distance, angles, fov
    "root": [
        [0, 0.6, 0.75],
        1.3,
        -35,
        -5,
        0,
        40,
    ],  # Root view: [position], distance, angles, fov
    "wrist": [],  # Placeholder for the 'wrist', since wrist view goes with the end effector, so no predefined camera parameters required
}

# Tuple defining the resolution of the camera (width x height)
cam_resolution = (1080, 864)


def get_cam_projection_matrix(cam_view):
    """
    Calculates the camera projection matrix based on the given camera view.

    Parameters:
    - cam_view (str): Specifies the camera view.

    Returns:
    - cam_projection_matrix (list): Projection matrix for the specified camera view.
    """

    # Calculate the aspect ratio based on camera resolution
    aspect = float(cam_resolution[0]) / cam_resolution[1]
    nearVal = 0.1  # Default near clipping plane value
    farVal = 100  # Default far clipping plane value

    if cam_view == "wrist":
        # Adjust parameters for wrist camera view
        fov = 100  # Field of view for wrist camera
        nearVal = 0.018  # Adjusted near clipping plane value for wrist camera
    else:
        # Use field of view based on the specified camera view
        fov = CAM_INFO[cam_view][-1]  # Get field of view for the specified camera view

    # Compute the camera projection matrix using PyBullet's function
    cam_projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
    )

    # Return the calculated camera projection matrix
    return cam_projection_matrix


def get_view_matrix(cam_view, robot_id, ee_index):
    """
    Generates the view matrix for a specified camera view relative to a robot's end-effector.

    Parameters:
    - cam_view (str): Specifies the camera view.
    - robot_id (int): Identifier for the robot.
    - ee_index (int): Index of the end-effector on the robot.

    Returns:
    - cam_view_matrix (list): View matrix for the specified camera view.
    """

    if cam_view == "wrist":
        # Calculate view matrix for wrist camera view
        eye_pos, eye_ori = p.getLinkState(
            robot_id,
            ee_index,
            computeForwardKinematics=True,
        )[0:2]
        eye_pos = list(eye_pos)
        eye_pos = p.multiplyTransforms(eye_pos, eye_ori, [0, 0, -0.05], [0, 0, 0, 1])[0]
        r_mat = p.getMatrixFromQuaternion(eye_ori)
        tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])
        ty_vec = np.array([r_mat[1], r_mat[4], r_mat[7]])
        tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])
        camera_position = np.array(eye_pos)
        target_position = eye_pos + 0.001 * tz_vec

        # Compute view matrix for wrist camera using PyBullet's function
        cam_view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=ty_vec,
        )
    else:
        # Calculate view matrix for non-wrist camera views using yaw, pitch, and roll
        cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            CAM_INFO[cam_view][0],
            CAM_INFO[cam_view][1],
            CAM_INFO[cam_view][2],
            CAM_INFO[cam_view][3],
            CAM_INFO[cam_view][4],
            2,
        )

    # Return the computed camera view matrix
    return cam_view_matrix


def get_cam_view_img(cam_view, robot_id=None, ee_index=None):
    """
    Captures an image from a specified camera view using PyBullet.

    Parameters:
    - cam_view (str): Specifies the camera view.
    - robot_id (int, optional): Identifier for the robot.
    - ee_index (int, optional): Index of the end-effector on the robot.

    Returns:
    - img (numpy.ndarray): Captured image from the specified camera view.
    """

    # Obtain the view matrix for the camera view
    cam_view_matrix = get_view_matrix(cam_view, robot_id, ee_index)

    # Obtain the projection matrix for the camera view
    cam_projection_matrix = get_cam_projection_matrix(cam_view)

    # Capture the camera image using PyBullet
    (width, height, rgb_pixels, _, _) = p.getCameraImage(
        cam_resolution[0],
        cam_resolution[1],
        viewMatrix=cam_view_matrix,
        projectionMatrix=cam_projection_matrix,
    )

    # Reshape and process the image data
    rgb_array = np.array(rgb_pixels).reshape((height, width, 4)).astype(np.uint8)
    img = np.array(resize_and_crop(rgb_array[:, :, :3]))  # Process the image

    # Return the captured and processed image
    return img


def resize_and_crop(input_image):
    """
    Crop the image to a 5:4 aspect ratio and resize it to 320x256 pixels.

    Parameters:
    - input_image (numpy.ndarray): Input image data in array format.

    Returns:
    - input_image (PIL.Image.Image): Cropped and resized image in PIL Image format.
    """

    # Convert the input image array to a PIL Image
    input_image = Image.fromarray(input_image)

    # Get the width and height of the input image
    width, height = input_image.size

    # Define target and current aspect ratios
    target_aspect = 5 / 4
    current_aspect = width / height

    if current_aspect > target_aspect:
        # If the image is too wide, crop its width
        new_width = int(target_aspect * height)
        left_margin = (width - new_width) / 2
        input_image = input_image.crop((left_margin, 0, width - left_margin, height))
    elif current_aspect < target_aspect:
        # If the image is too tall, crop its height
        new_height = int(width / target_aspect)
        top_margin = (height - new_height) / 2
        input_image = input_image.crop((0, top_margin, width, height - top_margin))

    # Resize the cropped image to 320x256 pixels
    input_image = input_image.resize((320, 256))

    # Return the cropped and resized image as a PIL Image
    return input_image

create_video = False  
fps = 60
time_step = 1. / fps

if create_video:
    p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"simulation_output.mp4\" --mp4fps=" + str(fps))
else:
    p.connect(p.GUI)

# simulator configuration
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(enableConeFriction=0)
p.setTimeStep(1./240.)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # disable GUI
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-45, cameraTargetPosition=[0.5, 0, 0.5])

# output settings
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pose_data_list = []

# initialize enviroment
sim = PandaSim(p)
sim.load_assets() 
    

if not create_video:

    for iteration in range(100):
        # reset robot and randomly reset the target objects
        sim.reset_robot()
        sim.move_gripper(1)
        sim.reset_target()
        
        target_pos, target_ori = sim.bullet_client.getBasePositionAndOrientation(sim.target_obj)
        # target_ori = p.getQuaternionFromEuler([0, -np.pi, 0])
        grasp_pos = list(target_pos)
        grasp_pos[2] += 0.1
        
        # inverse kinematics
        joint_poses = sim.inverse_kinematics(grasp_pos, target_ori)
        sim.move_robot(joint_poses)
        for i in range(60):
            sim.bullet_client.stepSimulation()
            time.sleep(time_step)
            
            # #capture image
            # img = get_cam_view_img('front', sim.robot, sim.EndEffectorIndex)
            # img_file_path = os.path.join(output_dir, f"iteration_{iteration}_frame_{i}.png")
            # Image.fromarray(img).save(img_file_path)
            # # get pose info
            # ee_pos, ee_ori = sim.bullet_client.getLinkState(sim.robot, sim.EndEffectorIndex)[0:2]
            # pose_data_list.append({
            #     "iteration": iteration,
            #     "frame": i,
            #     "x": ee_pos[0],
            #     "y": ee_pos[1],
            #     "z": ee_pos[2],
            #     "qx": ee_ori[0],
            #     "qy": ee_ori[1],
            #     "qz": ee_ori[2],
            #     "qw": ee_ori[3]
            # })
            
        # lower the end-effector in preparation for gripping
        grasp_pos[2] -= 0.1
        joint_poses = sim.inverse_kinematics(grasp_pos, target_ori)
        sim.move_robot(joint_poses)
        for i in range(60):
            sim.bullet_client.stepSimulation()
            time.sleep(time_step)
            
            # #capture image
            # img = get_cam_view_img('front', sim.robot, sim.EndEffectorIndex)
            # img_file_path = os.path.join(output_dir, f"iteration_{iteration}_frame_{i+240}.png")
            # Image.fromarray(img).save(img_file_path)
            # # get pose info
            # ee_pos, ee_ori = sim.bullet_client.getLinkState(sim.robot, sim.EndEffectorIndex)[0:2]
            # pose_data_list.append({
            #     "iteration": iteration,
            #     "frame": i+240,
            #     "x": ee_pos[0],
            #     "y": ee_pos[1],
            #     "z": ee_pos[2],
            #     "qx": ee_ori[0],
            #     "qy": ee_ori[1],
            #     "qz": ee_ori[2],
            #     "qw": ee_ori[3]
            # })
            
        # close gripper
        sim.move_gripper(0)
        for i in range(60):
            sim.bullet_client.stepSimulation()
            time.sleep(time_step)
            
            # #capture image
            # img = get_cam_view_img('front', sim.robot, sim.EndEffectorIndex)
            # img_file_path = os.path.join(output_dir, f"iteration_{iteration}_frame_{i+480}.png")
            # Image.fromarray(img).save(img_file_path)
            # # get pose info
            # ee_pos, ee_ori = sim.bullet_client.getLinkState(sim.robot, sim.EndEffectorIndex)[0:2]
            # pose_data_list.append({
            #     "iteration": iteration,
            #     "frame": i+480,
            #     "x": ee_pos[0],
            #     "y": ee_pos[1],
            #     "z": ee_pos[2],
            #     "qx": ee_ori[0],
            #     "qy": ee_ori[1],
            #     "qz": ee_ori[2],
            #     "qw": ee_ori[3]
            # })
            
        # # up object
        grasp_pos[2] += 0.2
        joint_poses = sim.inverse_kinematics(grasp_pos, target_ori)
        sim.move_robot(joint_poses)
        for i in range(120):
            sim.bullet_client.stepSimulation()
            time.sleep(time_step)
            
            # #capture image
            # img = get_cam_view_img('front', sim.robot, sim.EndEffectorIndex)
            # img_file_path = os.path.join(output_dir, f"iteration_{iteration}_frame_{i+720}.png")
            # Image.fromarray(img).save(img_file_path)
            # # get pose info
            # ee_pos, ee_ori = sim.bullet_client.getLinkState(sim.robot, sim.EndEffectorIndex)[0:2]
            # pose_data_list.append({
            #     "iteration": iteration,
            #     "frame": i+720,
            #     "x": ee_pos[0],
            #     "y": ee_pos[1],
            #     "z": ee_pos[2],
            #     "qx": ee_ori[0],
            #     "qy": ee_ori[1],
            #     "qz": ee_ori[2],
            #     "qw": ee_ori[3]
            # })
            
    # pose_data = pd.DataFrame(pose_data_list)
    # excel_file_path = os.path.join(output_dir, "end_effector_poses.xlsx")
    # pose_data.to_excel(excel_file_path, index=False)
        
        
    p.disconnect()
