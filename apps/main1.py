"""
Main script for face recognition from webcam.
"""

import pandas as pd
import cv2
from services.Detect_and_RecogFace import detect_and_recognize_face
from sklearn.preprocessing import Normalizer

# Constants
EMBEDDING_FILE = './data/embedding_with_label.csv'
WEBCAM_INDEX = 0
EXIT_KEY = ord('q')

# Load embeddings and initialize normalizer
embedding_df = pd.read_csv(EMBEDDING_FILE)
l2_normalizer = Normalizer('l2')

# Initialize webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Starting face recognition. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot receive frame from webcam")
            break

        # Perform face detection and recognition
        processed_frame = detect_and_recognize_face(frame, l2_normalizer, embedding_df)

        # Display the frame
        cv2.imshow('Webcam Face Recognition', processed_frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()