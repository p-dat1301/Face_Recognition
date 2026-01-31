
import os
import cv2
import numpy as np
import pandas as pd
import csv

from tensorflow.keras.models import load_model
from models.Inception_Resnet import InceptionResNetV2
from utils.DetectFace import detect_face

# Constants
MODEL_WEIGHTS_PATH = './models/facenet_keras_weights.h5'
DATA_FOLDER = './data'
EMBEDDING_SIZE = 128  # Assuming FaceNet output size
FACE_SIZE = (160, 160)

# Load and initialize the FaceNet model
facenet_model = InceptionResNetV2()
facenet_model.load_weights(MODEL_WEIGHTS_PATH)


def normalize_image(image):
    return image / 255.0


def extract_feature(image):
    image = np.expand_dims(image, axis=0)
    return facenet_model.predict(image)


def process_single_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bounding_boxes = detect_face(image)

    if not bounding_boxes:
        print(f"No face detected in {image_path}")
        return None

    # Use the first detected face
    x1, y1, x2, y2 = bounding_boxes[0]
    face_crop = image[y1:y2, x1:x2]
    face_crop = normalize_image(face_crop)
    face_crop = cv2.resize(face_crop, FACE_SIZE)

    embedding = extract_feature(face_crop)
    return embedding.flatten()


def create_embeddings_for_person(person_folder_path, person_name):
    embeddings = []

    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)
        if not os.path.isfile(image_path):
            continue

        embedding = process_single_face(image_path)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        print(f"No valid embeddings for {person_name}")
        return None

    # Return mean embedding
    return np.mean(embeddings, axis=0)


def create_embedding(data_folder_path=DATA_FOLDER):
    embeddings = []
    labels = []

    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_folder_path):
            continue

        embedding = create_embeddings_for_person(person_folder_path, person_name)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(person_name)

    return np.array(embeddings), labels


def embeddings_to_csv(embeddings, labels, output_path='./data/embedding_with_label.csv'):
    # Format embeddings as strings
    formatted_embeddings = ['[' + ' '.join(map(str, emb)) + ']' for emb in embeddings]

    df = pd.DataFrame({'embedding': formatted_embeddings, 'person_name': labels})
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')


# Main execution to generate embeddings
if __name__ == "__main__":
    embeddings, labels = create_embedding()
    embeddings_to_csv(embeddings, labels)
















































