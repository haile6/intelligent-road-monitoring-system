
import os
import cv2
import csv
import easyocr
from ultralytics import YOLO

VIDEO_PATH = r"data\new_test_video.mp4"
PLATE_MODEL_PATH = r"models\license_plate_detector.pt"
OUTPUT_DIR = r"outputs\plate_ocr_test"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "plate_ocr_annotated.mp4")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "plate_readings.csv")

CONFIDENCE = 0.25
IMG_SIZE = 960
FRAME_SKIP = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
if not os.path.exists(PLATE_MODEL_PATH):
    raise FileNotFoundError(f"Plate model not found: {PLATE_MODEL_PATH}")

print("Loading plate detector...")
detector = YOLO(PLATE_MODEL_PATH)

print("Loading EasyOCR (first run may download its recognition model)...")
reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    max(1, fps / FRAME_SKIP),
    (width, height),
)

rows = []
frame_no = 0
sampled_frames = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        if (frame_no - 1) % FRAME_SKIP != 0:
            continue

        sampled_frames += 1
        result = detector.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Upscale small plate crops for OCR.
            crop = cv2.resize(
                crop, None, fx=2, fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            ocr_results = reader.readtext(
                crop,
                detail=1,
                paragraph=False,
            )

            # Keep OCR text and confidence; don't assume the
            # characters are correct or enforce a guessed plate format.
            readings = [
                (text.strip(), float(score))
                for _, text, score in ocr_results
                if text.strip()
            ]
            plate_text = " ".join(t for t, _ in readings)
            ocr_conf = (
                sum(score for _, score in readings) / len(readings)
                if readings else None
            )

            rows.append({
                "frame": frame_no,
                "video_time_s": (frame_no - 1) / fps,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "plate_text": plate_text,
                "ocr_confidence": (
                    f"{ocr_conf:.3f}" if ocr_conf is not None else ""
                ),
            })

            label = plate_text if plate_text else "PLATE: unreadable"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label[:60],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if frame_no % 20 == 0:
            print(f"Processed {frame_no}/{total_frames} frames")

finally:
    cap.release()
    writer.release()

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "frame", "video_time_s", "x1", "y1", "x2", "y2",
        "plate_text", "ocr_confidence",
    ]
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(rows)

readable = sum(1 for row in rows if row["plate_text"])

print("\nOCR TEST COMPLETE")
print(f"Frames sampled: {sampled_frames}")
print(f"Plate detections: {len(rows)}")
print(f"Detections with OCR text: {readable}")
print(f"Annotated video: {OUTPUT_VIDEO}")
print(f"CSV results: {OUTPUT_CSV}")
print("Review the text manually; OCR confidence is not a guarantee of correctness.")
