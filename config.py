
"""
Settings for the Road Monitoring System.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# FILES
# ============================================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "test_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "outputs", "annotated_test.mp4")

# ============================================================
# PHASE 1: ROAD DAMAGE DETECTION
# ============================================================
EXPECTED_CLASSES = {
    0: "alligator_crack",
    1: "pothole",
}

CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640

# ============================================================
# DISPLAY
# ============================================================
SHOW_PREVIEW = False
BOX_THICKNESS = 3

CLASS_COLORS = {
    "alligator_crack": (0, 165, 255),
    "pothole": (0, 0, 255),
}

# ============================================================
# PHASE 2: DAMAGE TRACKING
# ============================================================
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_MISSED_FRAMES = 5
MIN_CONFIRMATION_FRAMES = 3

OUTPUT_VIDEO_PHASE2 = os.path.join(
    BASE_DIR, "outputs", "phase2_tracked.mp4"
)
EVENTS_CSV_PATH = os.path.join(
    BASE_DIR, "outputs", "phase2_events.csv"
)

# ============================================================
# PHASE 3: VEHICLE DETECTION AND TRACKING
# ============================================================
VEHICLE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "yolo11s.pt"
)

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

OUTPUT_VIDEO_PHASE3 = os.path.join(
    BASE_DIR, "outputs", "annotated_phase3.mp4"
)
PHASE3_DAMAGE_EVENTS_CSV = os.path.join(
    BASE_DIR, "outputs", "phase3_damage_events.csv"
)
VEHICLES_CSV_PATH = os.path.join(
    BASE_DIR, "outputs", "phase3_vehicles.csv"
)
DAMAGE_CSV_PHASE3 = os.path.join(
    BASE_DIR, "outputs", "phase3_damage_events.csv"
)

# ============================================================
# PHASE 4: VEHICLE-PATH ANALYSIS
# ============================================================
PATH_WIDTH_RATIO = 0.35
PATH_TOP_WIDTH_RATIO = 0.12
PATH_TOP_Y_RATIO = 0.45
PATH_CENTER_X_RATIO = 0.50
PATH_MIN_OVERLAP_RATIO = 0.30

PHASE4_USE_VEHICLES = True

OUTPUT_VIDEO_PHASE4 = os.path.join(
    BASE_DIR, "outputs", "annotated_phase4.mp4"
)
DAMAGE_CSV_PHASE4 = os.path.join(
    BASE_DIR, "outputs", "phase4_damage_events.csv"
)

# ============================================================
# PHASE 5: DISTANCE ESTIMATION
# ============================================================
CAMERA_HEIGHT_M = 1.2

# Temporary camera calibration values.
# Calibrate these for your actual camera before relying
# on estimated distances.
CAMERA_TILT_DEG = 5.0
CAMERA_HORIZONTAL_FOV_DEG = 70.0
FOCAL_LENGTH_PX = None

DISTANCE_METHOD = "perspective"
DISTANCE_MAX_M = 80
DISTANCE_SMOOTHING = 0.5
SHOW_HORIZON_LINE = True

OUTPUT_VIDEO_PHASE5 = os.path.join(
    BASE_DIR, "outputs", "annotated_phase5.mp4"
)
DAMAGE_CSV_PHASE5 = os.path.join(
    BASE_DIR, "outputs", "phase5_damage_events.csv"
)

# ============================================================
# PHASE 6: GPS SYNCHRONIZATION
# ============================================================
GPS_PATH = os.path.join(BASE_DIR, "data", "gps_track.csv")

VIDEO_START_TIME = "2026-10-08 10:30:00"
GPS_TIME_OFFSET_S = 0
GPS_MAX_GAP_S = 5

HEADING_LOOKBACK_S = 2
HEADING_MIN_MOVE_M = 1.0

# ============================================================
# PHASE 7: SPEED
# ============================================================
SPEED_FROM_POSITIONS = True
FORCE_DEFAULT_SPEED = False
DEFAULT_SPEED_KMH = 30
MIN_MOVING_SPEED_KMH = 2.0

# Maximum plausible vehicle speed used to filter GPS data.
# This is a configurable filter, not a safety threshold.
MAX_PLAUSIBLE_SPEED_KMH = 200

# ============================================================
# PHASE 8: TIME TO COLLISION
# ============================================================
CAMERA_TO_FRONT_M = 0.0
TTC_FAR_S = 60

OUTPUT_VIDEO_PHASE8 = os.path.join(
    BASE_DIR, "outputs", "annotated_phase8.mp4"
)
DAMAGE_CSV_PHASE8 = os.path.join(
    BASE_DIR, "outputs", "phase8_damage_events.csv"
)

# ============================================================
# PHASE 9: RISK ANALYSIS
# Prototype thresholds; not validated safety limits.
# ============================================================
RISK_TTC_CAUTION_S = 5.0
RISK_TTC_WARNING_S = 3.0
RISK_TTC_DANGER_S = 1.0

RISK_MIN_CONFIDENCE = 0.50
RISK_MAX_DISTANCE_M = 60

RISK_TYPE_ADJUSTMENT = {
    "pothole": 0,
    "alligator_crack": -1,
}

RISK_HIGH_SPEED_KMH = 60
RISK_HIGH_SPEED_FACTOR = 1.25

RISK_LEVELS = [
    "SAFE",
    "CAUTION",
    "WARNING",
    "IMMEDIATE_DANGER",
]

RISK_DISPLAY = {
    "SAFE": "SAFE / MONITOR",
    "CAUTION": "CAUTION",
    "WARNING": "WARNING",
    "IMMEDIATE_DANGER": "IMMEDIATE DANGER",
}

# OpenCV colors use BGR channel order.
RISK_COLORS = {
    "SAFE": (0, 200, 0),
    "CAUTION": (0, 255, 255),
    "WARNING": (0, 140, 255),
    "IMMEDIATE_DANGER": (0, 0, 255),
}

OUTPUT_VIDEO_PHASE9 = os.path.join(
    BASE_DIR, "outputs", "annotated_phase9.mp4"
)
DAMAGE_CSV_PHASE9 = os.path.join(
    BASE_DIR, "outputs", "phase9_damage_events.csv"
)
# ============================================================
# PHASE 9: DAMAGE LOCATION SETTINGS
# ============================================================
PROJECT_DAMAGE_LOCATION = True
