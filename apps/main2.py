"""
Main script for head pose estimation challenge.
"""

import pandas as pd
import cv2
import time
from services.Detect_and_RecogFace import detect_and_recognize_face
from sklearn.preprocessing import Normalizer
from services.headPoseEstimation import HeadPoseEstimation

# Constants
EMBEDDING_FILE = './data/embedding_with_label.csv'
WEBCAM_INDEX = 0
CAMERA_DELAY = 1

# Load embeddings and initialize normalizer
embedding_df = pd.read_csv(EMBEDDING_FILE)
l2_normalizer = Normalizer('l2')

# Initialize webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Delay for camera stabilization
time.sleep(CAMERA_DELAY)

# Run head pose estimation challenge
success, image = HeadPoseEstimation(cap)

if image is None:
    print("Error: Cannot read image from camera")
    exit()

# Perform face recognition on the captured image
processed_image = detect_and_recognize_face(image, l2_normalizer, embedding_df)

# Display the result
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
