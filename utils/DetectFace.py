"""
Face detection utilities using YOLO model.
"""

from ultralytics import YOLO
import cv2

# Constants
MODEL_PATH = '../models/yolov8n-face.pt'
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Load the YOLO model globally for efficiency
yolo_model = YOLO(MODEL_PATH)


def detect_face(frame):
    """
    Detect faces in the given frame using YOLO model.

    Args:
        frame (numpy.ndarray): Input image frame.

    Returns:
        list: List of bounding boxes as [x1, y1, x2, y2].
    """
    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
    boxes = results[0].boxes
    bounding_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0][:4]
        bounding_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return bounding_boxes


