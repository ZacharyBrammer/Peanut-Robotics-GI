import cv2
import numpy as np
import os
import torch
from Utils import rotateImage


def load_camera_info(camera_info_file):
    camera_info = []
    with open(camera_info_file, 'r') as file:
        for line in file:
            value = line.strip().split(' ')
            camera_info.append(value)
    return camera_info


# Function to load a depth image
def load_depth_image(image_path):
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return depth_image

# Function to get distance data from a depth image


def get_distance_data(depth_image, scale_factor=0.001):
    distance_data = depth_image.astype(np.float32) * scale_factor
    return distance_data


def load_img_info(dbname, img):
    img_name = os.path.splitext(img)[0]
    cam_depth = load_depth_image(os.path.join(
        './uploads', dbname, dbname+'_frames', 'lowres_depth', img_name+'.png'))

    cam_int = load_camera_info(os.path.join(
        './uploads', dbname, dbname+'_frames', 'lowres_wide_intrinsics', img_name+'.pincam'))
    cam_int = cam_int[0]

    cam_wide = cv2.imread(os.path.join(
        './uploads', dbname, dbname+'_frames', 'lowres_wide', img_name+'.png'))
    image = cam_wide

    if image is None:
        raise ValueError(f"Failed to load image: {img}")

    # units are in pixels
    fx = float(cam_int[2])  # Focal length in x
    fy = float(cam_int[3])  # Focal length in y
    cx = float(cam_int[4])  # Principal point x
    cy = float(cam_int[5])  # Principal point y
    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

    config_path = os.path.abspath('Utils/yolov3.cfg')
    weights_path = os.path.abspath('Utils/yolov3.weights')
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(
        416, 416), swapRB=True, crop=False)  # YOLO-specific parameters
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >= 0.5:
                center_x = int(object_detection[0] * image.shape[1])
                center_y = int(object_detection[1] * image.shape[0])
                cam_depth = get_distance_data(cam_depth)
                depth_value = cam_depth[center_y, center_x]

                X_c = (center_x - cx) * depth_value / fx
                Y_c = (center_y - cy) * depth_value / fy
                Z_c = depth_value

                point_camera = np.array(
                    [X_c, Y_c, Z_c], dtype=np.float32).reshape(3, 1)
                print("Original 3D Point in camera coordinates:\n", point_camera)

                R_z = np.array([
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                point_camera_rotated = np.dot(R_z, point_camera)
                rotated_x = point_camera_rotated[0, 0]
                rotated_y = point_camera_rotated[1, 0]
                rotated_z = point_camera_rotated[2, 0]

                print(
                    f"Rotated 3D Coordinates in Camera Frame: X_c={rotated_x}, Y_c={rotated_y}, Z_c={rotated_z}")
                return rotated_x, rotated_y

    if (confidence < 0.5):
        return None, None
