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


# Phase 4 settings
# ==== PHASE 4: vehicle-path analysis =======================================
# The "projected path" is a trapezoid in the lower-centre of the picture:
#
#            ____________            <- top edge at PATH_TOP_Y_RATIO (far away, narrow)
#           /            \
#          /              \
#         /________________\         <- bottom edge of the picture (close, wide)
#
PATH_WIDTH_RATIO = 0.35        # width of the path at the BOTTOM of the picture (35% of the width)
PATH_TOP_WIDTH_RATIO = 0.12    # width of the path at its far (top) edge
PATH_TOP_Y_RATIO = 0.55        # height of the far edge (0 = top of picture, 1 = bottom)
PATH_CENTER_X_RATIO = 0.50     # 0.5 = the camera points straight along the road
PATH_MIN_OVERLAP_RATIO = 0.30  # damage counts as "in path" if >= 30% of its box is inside the path
                               # (or if the bottom-centre of its box is inside)

PHASE4_USE_VEHICLES = True     # False = faster, damage + path only
OUTPUT_VIDEO_PHASE4 = os.path.join(BASE_DIR, "outputs", "annotated_phase4.mp4")
DAMAGE_CSV_PHASE4 = os.path.join(BASE_DIR, "outputs", "phase4_damage_events.csv")


# ============================================================
# PHASE 5: DISTANCE ESTIMATION
# ============================================================
# Camera height reported by the user: 1.2 metres.
CAMERA_HEIGHT_M = 1.2

# TEMPORARY PLACEHOLDERS ONLY — calibrate before trusting distances.
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
# ==== PHASE 6: GPS ==========================================================
# CSV columns:  timestamp,latitude,longitude,speed_kmh
# Example row:  2026-10-08T10:30:00,9.123456,39.123456,35.0
GPS_PATH = os.path.join(BASE_DIR, "data", "gps_track.csv")

# When did the VIDEO start, on the same clock and time zone as the GPS file?
#   None  = assume the video started at the first GPS record (fine for tests)
#   or a text such as "2026-10-08T10:30:00"
VIDEO_START_TIME = None
GPS_TIME_OFFSET_S = 0.0     # fine adjustment: if damage appears at the wrong place, shift by a few seconds
GPS_MAX_GAP_S = 5.0         # no position if the GPS file has a hole longer than this

# The GPS gives the position of the CAR. The damage is some metres ahead of it.
# True = also save an estimated position of the damage itself (needs distance + movement)
PROJECT_DAMAGE_LOCATION = True
HEADING_LOOKBACK_S = 2.0    # direction of travel is measured over the last 2 seconds
HEADING_MIN_MOVE_M = 3.0    # if the car moved less than this, the direction is not trusted

OUTPUT_VIDEO_PHASE6 = os.path.join(BASE_DIR, "outputs", "annotated_phase6.mp4")
DAMAGE_CSV_PHASE6 = os.path.join(BASE_DIR, "outputs", "phase6_damage_events.csv")

# ==== PHASE 7: Vehicle speed ================================================
DEFAULT_SPEED_KMH = 30
FORCE_DEFAULT_SPEED = False
SPEED_FROM_POSITIONS = True
SPEED_WINDOW_S = 2.0
MAX_PLAUSIBLE_SPEED_KMH = 200
MIN_MOVING_SPEED_KMH = 2.0

OUTPUT_VIDEO_PHASE7 = os.path.join(BASE_DIR, "outputs", "annotated_phase7.mp4")
DAMAGE_CSV_PHASE7 = os.path.join(BASE_DIR, "outputs", "phase7_damage_events.csv")
