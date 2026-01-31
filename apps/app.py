"""
Streamlit app for face recognition with head pose estimation.
"""

import streamlit as st
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
APP_TITLE = "Face Recognition with Head Pose Estimation"
APP_DESCRIPTION = "Ứng dụng nhận diện khuôn mặt sử dụng webcam và head pose estimation trên local host."


def load_embeddings(file_path):
    """
    Load embeddings from CSV file with caching.

    Args:
        file_path (str): Path to the embedding CSV file.

    Returns:
        pandas.DataFrame: Loaded embeddings DataFrame.
    """
    return pd.read_csv(file_path)


def main():
    """
    Main function for the Streamlit app.
    """
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    # Load embeddings and initialize normalizer with caching
    embedding_df = st.cache_data(load_embeddings)(EMBEDDING_FILE)
    l2_normalizer = Normalizer('l2')

    # Button to start face recognition
    if st.button("Start Face Recognition"):
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            st.error("Error: Camera not accessible")
            return

        # Delay for camera stabilization
        time.sleep(CAMERA_DELAY)

        try:
            # Run head pose estimation challenge
            success, image = HeadPoseEstimation(cap)
            if not success or image is None:
                st.error("Error: Cannot read image from camera")
                return

            # Perform face recognition
            processed_image = detect_and_recognize_face(image, l2_normalizer, embedding_df)

            # Convert to RGB for display
            display_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption="Processed Image", use_column_width=True)

        finally:
            # Ensure camera is released
            cap.release()
            time.sleep(CAMERA_DELAY)  # Additional delay for camera reset


if __name__ == '__main__':
    main()
