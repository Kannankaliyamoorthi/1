"""
config.py
----------
Central configuration file for the Automatic Traffic Rule Violation
Detection System. Change these values according to your setup.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Video / Camera Source ----------------
# Set to 0 for webcam, or a file path / RTSP URL for CCTV camera
VIDEO_SOURCE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------- Model Paths ----------------
# Pre-trained YOLOv8 model for general vehicle detection (COCO classes)
VEHICLE_MODEL_PATH = "yolov8n.pt"

# Custom-trained YOLOv8 model for helmet / no-helmet detection.
# You must train this on a helmet dataset (e.g. Roboflow "Helmet Detection")
# and place the weights file here.
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "models", "helmet_best.pt")

# Custom-trained YOLOv8 model for traffic light color detection.
SIGNAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "signal_best.pt")

# ---------------- Detection Thresholds ----------------
CONFIDENCE_THRESHOLD = 0.5
SPEED_LIMIT_KMPH = 40          # vehicles above this are flagged as overspeeding
STOP_LINE_Y = 500              # y-pixel coordinate of the stop line in the frame
PIXELS_PER_METER = 10          # calibration value: how many pixels = 1 metre
                                # (measure this for your specific camera setup)

# ---------------- Vehicle classes from COCO (used by yolov8n.pt) ----------------
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------------- Output Folders ----------------
VIOLATION_IMAGES_DIR = os.path.join(BASE_DIR, "static", "violations")
os.makedirs(VIOLATION_IMAGES_DIR, exist_ok=True)

# ---------------- MySQL Database ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",   # change this
    "database": "traffic_violations",
}
