"""
Settings for the Road Monitoring System.
Change the paths below to match your PC. Everything else can stay as it is for now.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Files ----------------------------------------------------------------
# Your trained two-class model (already copied to models\ for you).
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_best.pt")

# Put your test video here, or change this to the full path of your video.
VIDEO_PATH = os.path.join(BASE_DIR, "data", "test_video.mp4")

# Where the annotated video is saved.
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "test_video.mp4")

# ---- Model ----------------------------------------------------------------
# The model must have EXACTLY these two classes. The program stops if it does not.
EXPECTED_CLASSES = {0: "alligator_crack", 1: "pothole"}

CONFIDENCE_THRESHOLD = 0.40   # ignore detections below this confidence
IMAGE_SIZE = 640              # the size the model was trained with

# ---- Display --------------------------------------------------------------
SHOW_PREVIEW = False          # True = show a live window while processing (press q to stop)
BOX_THICKNESS = 3
# Colours are (Blue, Green, Red) because OpenCV uses BGR order.
CLASS_COLORS = {
    "alligator_crack": (0, 165, 255),   # orange
    "pothole": (0, 0, 255),             # red
}

# ---- Phase 2: Tracking and confirmation -------------------------------
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_MISSED_FRAMES = 5
MIN_CONFIRMATION_FRAMES = 3

OUTPUT_VIDEO_PHASE2 = os.path.join(BASE_DIR, "outputs", "phase2_tracked.mp4")
EVENTS_CSV_PATH = os.path.join(BASE_DIR, "outputs", "phase2_events.csv")
