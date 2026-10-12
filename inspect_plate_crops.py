import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = r"data\test_video.mp4"
MODEL = r"models\license_plate_detector.pt"
OUT = Path(r"outputs\plate_crops")

OUT.mkdir(parents=True, exist_ok=True)

detector = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

frame_no = 0
saved = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_no += 1

    # Inspect every second frame
    if (frame_no - 1) % 2 != 0:
        continue

    height, width = frame.shape[:2]

    result = detector.predict(
        frame, imgsz=960, conf=0.25, verbose=False
    )[0]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Enlarge the plate crop for easier inspection
        enlarged = cv2.resize(
            crop, None, fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        filename = OUT / f"frame_{frame_no:05d}_plate_{saved:04d}.jpg"
        cv2.imwrite(str(filename), enlarged)
        saved += 1

cap.release()

print(f"Finished. Saved {saved} plate crops.")
print(f"Folder: {OUT.resolve()}")