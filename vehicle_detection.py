"""
vehicle_detection.py
----------------------
Module 1: Vehicle Detection
Detects cars, bikes, buses, and trucks from a live video frame using YOLOv8.
"""

from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VEHICLE_MODEL_PATH, CONFIDENCE_THRESHOLD, VEHICLE_CLASSES


class VehicleDetector:
    def __init__(self, model_path=VEHICLE_MODEL_PATH):
        print("[VehicleDetector] Loading model...")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs detection on a single frame.

        Returns a list of dicts:
        [{"bbox": (x1, y1, x2, y2), "class": "car", "conf": 0.91, "track_id": 3}, ...]
        """
        results = self.model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": VEHICLE_CLASSES[cls_id],
                    "conf": conf,
                    "track_id": track_id,
                })
        return detections

    @staticmethod
    def crop(frame, bbox):
        x1, y1, x2, y2 = bbox
        return frame[max(0, y1):y2, max(0, x1):x2]
