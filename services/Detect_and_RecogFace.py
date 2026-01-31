
import cv2
import numpy as np
from utils.DetectFace import detect_face
from utils.Embedding import extract_feature
from models.facenet import recognize_face


def normalize_image(image):
    mean, std = image.mean(), image.std()
    return (image - mean) / std


def preprocess_face_crop(face_crop):
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = normalize_image(face_crop)
    face_crop = cv2.resize(face_crop, (160, 160))
    return face_crop


def detect_and_recognize_face(frame, l2_normalizer, embedding_df):
    
    bounding_boxes = detect_face(frame)

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 4)

        # Crop and preprocess face
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue  # Skip if crop is empty

        face_crop = preprocess_face_crop(face_crop)

        # Extract and normalize embedding
        embedding = extract_feature(face_crop)
        embedding = l2_normalizer.transform(embedding.reshape(1, -1))[0]

        # Recognize face
        person_name = recognize_face(embedding, embedding_df)

        # Draw label
        cv2.putText(frame, person_name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

