# Automatic Traffic Rule Violation Detection System

AI + Computer Vision based system that detects helmet violations, signal
jumping, overspeeding, and stop-line crossing from camera footage, reads
number plates with OCR, and logs everything to a MySQL database with a
Flask dashboard for traffic authorities.

Matches the project synopsis modules:
1. Vehicle Detection (YOLOv8)
2. Traffic Signal Detection (HSV color detection)
3. Helmet Detection (custom YOLOv8 model)
4. Speed Detection (centroid tracking + calibration)
5. Number Plate Recognition (EasyOCR)
6. Database Management (MySQL)

## Folder Structure
```
traffic_violation_system/
│
├── main.py                  # Live detection pipeline (run this with a camera)
├── app.py                   # Flask dashboard for viewing violations
├── config.py                # All settings: paths, thresholds, DB credentials
├── requirements.txt
│
├── modules/
│   ├── vehicle_detection.py
│   ├── signal_detection.py
│   ├── helmet_detection.py
│   ├── speed_detection.py
│   └── number_plate.py
│
├── database/
│   ├── schema.sql            # Run once in MySQL to create tables
│   └── db_manager.py
│
├── templates/
│   └── dashboard.html
│
└── static/violations/        # Auto-saved evidence images
```

## Setup

1. Install Python 3.9+ and MySQL Server.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create the database:
   ```
   mysql -u root -p < database/schema.sql
   ```

4. Edit `config.py`:
   - Set your MySQL password in `DB_CONFIG`.
   - Set `VIDEO_SOURCE` (0 for webcam, or a video file / RTSP URL).
   - Calibrate `PIXELS_PER_METER` and `STOP_LINE_Y` for your camera angle.

5. Train the helmet detection model (required — no public stock model has
   this class):
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="helmet_dataset/data.yaml", epochs=50, imgsz=640)
   ```
   Copy the resulting `best.pt` to `models/helmet_best.pt`. A good public
   starter dataset is the "Helmet Detection" dataset on Roboflow Universe.

   Until you train this, `helmet_detection.py` will safely return
   `"unknown"` instead of crashing, so the rest of the system still runs.

## Running

Start live detection (needs a camera/video):
```
python main.py
```
Press `q` to quit the preview window.

Start the dashboard (in a separate terminal) to view logged violations:
```
python app.py
```
Then open http://localhost:5000

## Notes / Things to Calibrate for Real Deployment
- **Number plate localisation** in `number_plate.py` uses a simple contour
  heuristic. For higher accuracy, train a dedicated YOLOv8 plate-detector
  and swap it in.
- **Signal color detection** uses HSV thresholds and assumes a fixed
  `light_roi` region — measure this once for your camera and pass it to
  `SignalDetector(light_roi=(x1,y1,x2,y2))`.
- **Speed detection** accuracy depends entirely on correct
  `PIXELS_PER_METER` calibration for your camera's field of view and
  mounting height/angle.
- Indian number plates: the OCR cleanup regex keeps only A-Z and 0-9,
  which fits the standard `SS00SS0000` format.
