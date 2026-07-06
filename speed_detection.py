"""
speed_detection.py
----------------------
Module 4: Speed Detection
Estimates vehicle speed by tracking how far its bounding box centroid
moves across frames, using the known frame rate (FPS) and a pixel-to-metre
calibration value.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PIXELS_PER_METER, SPEED_LIMIT_KMPH


class SpeedDetector:
    def __init__(self, pixels_per_meter=PIXELS_PER_METER):
        self.pixels_per_meter = pixels_per_meter
        # track_id -> {"position": (x, y), "time": timestamp}
        self.history = {}

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update_and_estimate(self, track_id, bbox):
        """
        Call this once per frame per tracked vehicle.
        Returns speed in km/h, or None if not enough data yet.
        """
        now = time.time()
        cx, cy = self._centroid(bbox)

        if track_id not in self.history:
            self.history[track_id] = {"position": (cx, cy), "time": now}
            return None

        prev = self.history[track_id]
        dt = now - prev["time"]
        if dt <= 0:
            return None

        px_dist = ((cx - prev["position"][0]) ** 2 + (cy - prev["position"][1]) ** 2) ** 0.5
        meters = px_dist / self.pixels_per_meter
        speed_mps = meters / dt
        speed_kmph = speed_mps * 3.6

        self.history[track_id] = {"position": (cx, cy), "time": now}
        return round(speed_kmph, 1)

    @staticmethod
    def is_overspeeding(speed_kmph, limit=SPEED_LIMIT_KMPH):
        return speed_kmph is not None and speed_kmph > limit
