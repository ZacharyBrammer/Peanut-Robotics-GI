
def load_camera_info(camera_info_file):
    camera_info = pandas.read_csv(camera_info_file)
    return camera_info

# Function to load a depth image
def load_depth_image(image_path):
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Preserve depth information
    if depth_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return depth_image

# Function to get distance data from a depth image
def get_distance_data(depth_image, scale_factor=0.001):
    # Convert depth image to meters (assuming depth image is in millimeters)
    distance_data = depth_image.astype(np.float32) * scale_factor
    return distance_data

intrinsic_path = nullptr
# Camera Calibration // parameters will change depending on camera so need to be adjusted
fx = 800  # Focal length in x
fy = 800  # Focal length in y
cx = 320  # Principal point x
cy = 240  # Principal point y
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

# Load Image
image = cv2.imread('image.jpg') # substitute with the path to the image

# Object Detection (using a pre-trained YOLO model)
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Prepare the image for the network
blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False) # YOLO-specific parameters
net.setInput(blob) # Set the input to the network
detections = net.forward(output_layers)

# Extract the first detected object's 2D coordinates (example)
for detection in detections: 
    for object_detection in detection: 
        scores = object_detection[5:] 
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x = int(object_detection[0] * image.shape[1])
            center_y = int(object_detection[1] * image.shape[0])
            depth_value = 5  # Placeholder for depth value

            # Back-Projection to 3D Camera Coordinates
            X_c = (center_x - cx) * depth_value / fx
            Y_c = (center_y - cy) * depth_value / fy
            Z_c = depth_value

            # Transformation to World Coordinates (example values)
            rvec = np.array([0, 0, 0], dtype=np.float32)  # Rotation vector
            tvec = np.array([0, 0, 0], dtype=np.float32)  # Translation vector
            R, _ = cv2.Rodrigues(rvec)
            point_camera = np.array([X_c, Y_c, Z_c], dtype=np.float32).reshape(3, 1)
            point_world = np.dot(R, point_camera) + tvec.reshape(3, 1)

            print(f"3D Coordinates in Camera Frame: X_c={X_c}, Y_c={Y_c}, Z_c={Z_c}")
            print(f"3D Coordinates in World Frame: X_w={point_world[0][0]}, Y_w={point_world[1][0]}, Z_w={point_world[2][0]}")

# Visualization 
cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 

