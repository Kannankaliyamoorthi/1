"""
main.py
--------
Entry point for the Automatic Traffic Rule Violation Detection System.

Pipeline (matches the "Working Process" in the project synopsis):
Camera -> Video Capture -> Vehicle Detection -> Violation Detection
       -> Number Plate Recognition -> Database Storage -> Report Generation

Run:
    python main.py
Press 'q' to quit the live view.
"""

import cv2
import time
import os

import config
from modules.vehicle_detection import VehicleDetector
from modules.signal_detection import SignalDetector
from modules.helmet_detection import HelmetDetector
from modules.speed_detection import SpeedDetector
from modules.number_plate import NumberPlateReader
from database.db_manager import DBManager


class TrafficViolationSystem:
    def __init__(self):
        print("[System] Initializing modules...")
        self.vehicle_detector = VehicleDetector()
        self.signal_detector = SignalDetector()
        self.helmet_detector = HelmetDetector()
        self.speed_detector = SpeedDetector()
        self.plate_reader = NumberPlateReader()
        self.db = DBManager()

        # avoid logging the same track_id + violation repeatedly every frame
        self.already_flagged = set()

    def _save_violation_image(self, frame, bbox, violation_type, track_id):
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        filename = f"{violation_type}_{track_id}_{int(time.time())}.jpg"
        path = os.path.join(config.VIOLATION_IMAGES_DIR, filename)
        cv2.imwrite(path, crop if crop.size else frame)
        return path

    def process_frame(self, frame):
        detections = self.vehicle_detector.detect(frame)

        for det in detections:
            bbox = det["bbox"]
            vehicle_class = det["class"]
            track_id = det["track_id"]
            vehicle_crop = self.vehicle_detector.crop(frame, bbox)

            # ---- Speed check ----
            speed = self.speed_detector.update_and_estimate(track_id, bbox)
            if self.speed_detector.is_overspeeding(speed):
                self._handle_violation(frame, bbox, track_id, "Overspeeding",
                                        vehicle_class, vehicle_crop, speed)

            # ---- Signal jump / stop line check ----
            is_signal_violation, signal_color = self.signal_detector.check_violation(frame, bbox)
            if is_signal_violation:
                self._handle_violation(frame, bbox, track_id, "Signal Jump",
                                        vehicle_class, vehicle_crop)

            # ---- Helmet check (motorcycles only) ----
            if vehicle_class == "motorcycle":
                helmet_status = self.helmet_detector.detect(vehicle_crop)
                if helmet_status == "no_helmet":
                    self._handle_violation(frame, bbox, track_id, "No Helmet",
                                            vehicle_class, vehicle_crop)

            # ---- draw overlay for live view ----
            x1, y1, x2, y2 = bbox
            label = f"{vehicle_class} #{track_id}"
            if speed:
                label += f" {speed}km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.line(frame, (0, config.STOP_LINE_Y), (frame.shape[1], config.STOP_LINE_Y),
                  (0, 0, 255), 2)
        return frame

    def _handle_violation(self, frame, bbox, track_id, violation_type,
                           vehicle_class, vehicle_crop, speed=None):
        key = f"{track_id}_{violation_type}"
        if key in self.already_flagged:
            return
        self.already_flagged.add(key)

        plate_number = self.plate_reader.read_plate(vehicle_crop)
        image_path = self._save_violation_image(frame, bbox, violation_type, track_id)

        self.db.insert_violation(
            vehicle_number=plate_number,
            violation_type=violation_type,
            vehicle_type=vehicle_class,
            speed_kmph=speed,
            image_path=image_path,
        )
        print(f"[VIOLATION] {violation_type} | vehicle={vehicle_class} "
              f"plate={plate_number} speed={speed}")

    def run(self):
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("[System] ERROR: Could not open video source.")
            return

        print("[System] Running. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Traffic Violation Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.db.close()


if __name__ == "__main__":
    system = TrafficViolationSystem()
    system.run()
