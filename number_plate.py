"""
number_plate.py
----------------------
Module 5: Number Plate Recognition
Locates the number plate region in a vehicle crop and reads the
registration number using EasyOCR.
"""

import cv2
import re
import easyocr


class NumberPlateReader:
    def __init__(self, languages=("en",)):
        print("[NumberPlateReader] Loading OCR model...")
        self.reader = easyocr.Reader(list(languages), gpu=False)

    @staticmethod
    def _locate_plate_region(vehicle_crop):
        """
        Basic heuristic plate localisation using edge detection + contours.
        For production use, replace this with a dedicated YOLOv8
        number-plate-detection model for far higher accuracy.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  # rectangular shape -> likely a plate
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h else 0
                if 2 <= aspect_ratio <= 6:  # plates are wider than tall
                    return vehicle_crop[y:y + h, x:x + w]

        return vehicle_crop  # fallback: run OCR on the whole crop

    @staticmethod
    def _clean_text(text):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        return text

    def read_plate(self, vehicle_crop):
        """
        Returns the cleaned plate string, or None if nothing readable was found.
        """
        plate_region = self._locate_plate_region(vehicle_crop)
        if plate_region is None or plate_region.size == 0:
            return None

        results = self.reader.readtext(plate_region)
        if not results:
            return None

        # pick the text with the highest OCR confidence
        best = max(results, key=lambda r: r[2])
        text, conf = best[1], best[2]
        cleaned = self._clean_text(text)

        if len(cleaned) < 4 or conf < 0.3:
            return None
        return cleaned
