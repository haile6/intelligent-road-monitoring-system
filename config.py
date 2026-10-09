"""
Settings for the Road Monitoring System.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Files
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "test_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "outputs", "annotated_test.mp4")

# Road damage model
EXPECTED_CLASSES = {0: "alligator_crack", 1: "pothole"}
CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640

# Display
SHOW_PREVIEW = False
BOX_THICKNESS = 3
CLASS_COLORS = {
    "alligator_crack": (0, 165, 255),
    "pothole": (0, 0, 255),
}

# Phase 2
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_MISSED_FRAMES = 5
MIN_CONFIRMATION_FRAMES = 3
OUTPUT_VIDEO_PHASE2 = os.path.join(BASE_DIR, "outputs", "phase2_tracked.mp4")
EVENTS_CSV_PATH = os.path.join(BASE_DIR, "outputs", "phase2_events.csv")

# Phase 3: vehicle detection
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11s.pt")
VEHICLE_CONFIDENCE_THRESHOLD = 0.40
VEHICLE_IMAGE_SIZE = 640

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_COLORS = {
    "car": (255, 100, 0),
    "motorcycle": (0, 255, 255),
    "bus": (255, 255, 0),
    "truck": (180, 0, 255),
}

VEHICLE_TRACK_IOU_THRESHOLD = 0.30
VEHICLE_TRACK_MAX_MISSED_FRAMES = 15
VEHICLE_MIN_CONFIRMATION_FRAMES = 3

IGNORE_OWN_VEHICLE = True
OWN_VEHICLE_BOTTOM_MARGIN = 0.02
OWN_VEHICLE_MIN_WIDTH_RATIO = 0.55

OUTPUT_VIDEO_PHASE3 = os.path.join(BASE_DIR, "outputs", "annotated_phase3.mp4")
PHASE3_DAMAGE_EVENTS_CSV = os.path.join(BASE_DIR, "outputs", "phase3_damage_events.csv")
VEHICLES_CSV_PATH = os.path.join(BASE_DIR, "outputs", "phase3_vehicles.csv")

DAMAGE_CSV_PHASE3 = os.path.join(BASE_DIR, 'outputs', 'phase3_damage_events.csv')

DAMAGE_CSV_PHASE3 = os.path.join(BASE_DIR, 'outputs', 'phase3_damage_events.csv')
