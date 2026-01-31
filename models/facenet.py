"""
FaceNet recognition module.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# Constants
SIMILARITY_THRESHOLD = 0.3
UNKNOWN_LABEL = "Khong co du lieu ve khuon mat nay"


def parse_embedding_string(input_str):
    """
    Parse embedding string to numpy array.

    Args:
        input_str: String or array-like input.

    Returns:
        numpy.ndarray: Parsed embedding array.
    """
    if isinstance(input_str, str):
        input_str = input_str.strip('[]')
        number_list = [float(num) for num in input_str.split()]
    elif isinstance(input_str, (np.ndarray, list)):
        number_list = np.array(input_str, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported input type: {type(input_str)}")

    return np.array(number_list, dtype=np.float32)


def recognize_face(query_embedding, embedding_df):
    """
    Recognize face by finding the closest embedding in the database.

    Args:
        query_embedding (numpy.ndarray): Embedding of the query face.
        embedding_df (pandas.DataFrame): DataFrame with embeddings and labels.

    Returns:
        str: Recognized person name or unknown label.
    """
    # Parse embeddings in DataFrame
    embedding_df = embedding_df.copy()
    embedding_df['embedding'] = embedding_df['embedding'].apply(parse_embedding_string)

    distances = []
    query_embedding = np.ravel(query_embedding)

    for _, row in embedding_df.iterrows():
        stored_embedding = row['embedding']
        distance = cosine(stored_embedding, query_embedding)
        distances.append(distance)

    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    if min_distance > SIMILARITY_THRESHOLD:
        person_name = UNKNOWN_LABEL
        print(f"Min threshold {min_distance:.4f}")
    else:
        person_name = embedding_df.iloc[min_distance_index]['person_name']
        print(f"Min threshold {min_distance:.4f} for {person_name}")

    return person_name


# Alias for backward compatibility
FaceNet = recognize_face


