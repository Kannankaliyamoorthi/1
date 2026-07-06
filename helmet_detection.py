"""
helmet_detection.py
----------------------
Module 3: Helmet Detection
Detects whether a motorcycle rider is wearing a helmet.

This uses a custom-trained YOLOv8 model with two classes: "helmet" and
"no_helmet". You need to train this yourself on a labeled dataset
(e.g. the public "Helmet Detection" dataset on Roboflow) since a stock
COCO model has no helmet class.

Training example (run once, separately):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data="helmet_dataset/data.yaml", epochs=50, imgsz=640)
    # then copy runs/detect/train/weights/best.pt -> models/helmet_best.pt
"""

from ultralytics import YOLO
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HELMET_MODEL_PATH, CONFIDENCE_THRESHOLD


class HelmetDetector:
    def __init__(self, model_path=HELMET_MODEL_PATH):
        if not os.path.exists(model_path):
            print(f"[HelmetDetector] WARNING: model not found at {model_path}. "
                  f"Train a custom model first (see module docstring). "
                  f"Falling back to 'unknown' results.")
            self.model = None
        else:
            self.model = YOLO(model_path)

    def detect(self, rider_crop):
        """
        rider_crop: cropped image of the motorcycle rider region (numpy array).
        Returns: "helmet", "no_helmet", or "unknown"
        """
        if self.model is None or rider_crop is None or rider_crop.size == 0:
            return "unknown"

        results = self.model(rider_crop, conf=CONFIDENCE_THRESHOLD, verbose=False)
        best_label, best_conf = "unknown", 0.0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]
                if conf > best_conf:
                    best_label, best_conf = label, conf

        return best_label
