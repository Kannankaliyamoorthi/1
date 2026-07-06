"""
signal_detection.py
----------------------
Module 2: Traffic Signal Detection
Identifies the current traffic light color (red / yellow / green) and
checks whether a vehicle crossed the stop line while the signal was red.
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOP_LINE_Y


class SignalDetector:
    """
    Simple HSV color-based traffic light detector.
    For higher accuracy, replace `detect_color_hsv` with a custom-trained
    YOLOv8 model (see config.SIGNAL_MODEL_PATH) that directly classifies the
    traffic light region.
    """

    def __init__(self, light_roi=None):
        # light_roi = (x1, y1, x2, y2) region of the frame where the traffic
        # light is located. Must be calibrated per camera installation.
        self.light_roi = light_roi

    def detect_color_hsv(self, frame):
        if self.light_roi:
            x1, y1, x2, y2 = self.light_roi
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        red_mask = red_mask1 | red_mask2

        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

        counts = {
            "red": cv2.countNonZero(red_mask),
            "yellow": cv2.countNonZero(yellow_mask),
            "green": cv2.countNonZero(green_mask),
        }

        color = max(counts, key=counts.get)
        if counts[color] < 50:  # not enough colored pixels detected
            return "unknown"
        return color

    @staticmethod
    def crossed_stop_line(bbox, stop_line_y=STOP_LINE_Y):
        """A vehicle 'crosses' the line if its front (bottom edge) passes it."""
        _, _, _, y2 = bbox
        return y2 > stop_line_y

    def check_violation(self, frame, vehicle_bbox):
        """
        Returns True if signal is red AND the vehicle has crossed the stop line.
        """
        signal_color = self.detect_color_hsv(frame)
        crossed = self.crossed_stop_line(vehicle_bbox)
        return signal_color == "red" and crossed, signal_color
