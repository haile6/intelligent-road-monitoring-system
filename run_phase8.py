
import os
import csv

import cv2
import easyocr
from ultralytics import YOLO

from detection.vehicle_detector import VehicleDetector
from detection.tracker import VehicleTracker


# Phase 6 video already contains GPS and road-damage overlays.
INPUT_VIDEO = r"outputs\annotated_phase7.mp4"

# License plate detection model.
PLATE_MODEL_PATH = r"models\license_plate_detector.pt"

OUTPUT_DIR = r"outputs"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR, "annotated_phase8.mp4"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, "phase8_plate_readings.csv"
)

PLATE_CONFIDENCE = 0.25
PLATE_IMAGE_SIZE = 960
OCR_EVERY_N_FRAMES = 3


def associate_vehicle(plate_box, vehicles):
    """Associate a plate with a confirmed tracked vehicle."""
    px1, py1, px2, py2 = plate_box
    cx = (px1 + px2) / 2
    cy = (py1 + py2) / 2

    candidates = []

    for vehicle in vehicles:
        if not vehicle.confirmed:
            continue

        x1, y1, x2, y2 = vehicle.bbox

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            candidates.append((area, vehicle.id))

    if not candidates:
        return ""

    candidates.sort()
    return candidates[0][1]


def main():
    if not os.path.isfile(INPUT_VIDEO):
        raise SystemExit(
            f"Phase 6 video not found: {INPUT_VIDEO}"
        )

    if not os.path.isfile(PLATE_MODEL_PATH):
        raise SystemExit(
            f"Plate model not found: {PLATE_MODEL_PATH}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading vehicle detector...")
    vehicle_detector = VehicleDetector()
    vehicle_tracker = VehicleTracker()

    print("Loading license plate detector...")
    plate_detector = YOLO(PLATE_MODEL_PATH)

    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open video: {INPUT_VIDEO}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit(
            f"Cannot create output video: {OUTPUT_VIDEO}"
        )

    best_reading = {}
    rows = []
    frame_no = 0
    plate_count = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1
            timestamp = (frame_no - 1) / fps

            # Detect and track vehicles.
            detections = vehicle_detector.detect(frame)

            vehicles, _ = vehicle_tracker.update(
                detections,
                frame_no,
                timestamp,
            )

            # Detect license plates.
            result = plate_detector.predict(
                frame,
                imgsz=PLATE_IMAGE_SIZE,
                conf=PLATE_CONFIDENCE,
                verbose=False,
            )[0]

            # Draw vehicle IDs.
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle.bbox

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 180, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Vehicle {vehicle.id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 180, 0),
                    2,
                )

            # Recognize each detected plate.
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_count += 1
                plate_box = (x1, y1, x2, y2)

                vehicle_id = associate_vehicle(
                    plate_box, vehicles
                )

                plate_text = ""
                ocr_confidence = None

                # OCR every third frame.
                if (frame_no - 1) % OCR_EVERY_N_FRAMES == 0:
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        crop = cv2.copyMakeBorder(
                            crop,
                            8, 8, 8, 8,
                            cv2.BORDER_REPLICATE,
                        )

                        crop = cv2.resize(
                            crop,
                            None,
                            fx=3,
                            fy=3,
                            interpolation=cv2.INTER_CUBIC,
                        )

                        readings = reader.readtext(
                            crop,
                            detail=1,
                            paragraph=False,
                            allowlist=(
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "0123456789"
                            ),
                        )

                        readings = [
                            (text.strip(), float(score))
                            for _, text, score in readings
                            if text.strip()
                        ]

                        if readings:
                            plate_text = " ".join(
                                text for text, _ in readings
                            )

                            ocr_confidence = sum(
                                score for _, score in readings
                            ) / len(readings)

                            if vehicle_id != "":
                                previous = best_reading.get(
                                    vehicle_id
                                )

                                if (
                                    previous is None
                                    or ocr_confidence > previous[1]
                                ):
                                    best_reading[vehicle_id] = (
                                        plate_text,
                                        ocr_confidence,
                                    )

                # Reuse a previous reading for this vehicle.
                if (
                    vehicle_id != ""
                    and vehicle_id in best_reading
                ):
                    old_text, old_confidence = best_reading[
                        vehicle_id
                    ]

                    if not plate_text:
                        plate_text = old_text
                        ocr_confidence = old_confidence

                if plate_text:
                    label = plate_text
                else:
                    label = "Plate unreadable"

                if vehicle_id != "":
                    label = f"ID {vehicle_id}: {label}"

                # Green boxes mark license plates.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label[:70],
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                rows.append({
                    "frame": frame_no,
                    "video_time_s": f"{timestamp:.3f}",
                    "vehicle_id": vehicle_id,
                    "plate_text": plate_text,
                    "ocr_confidence": (
                        f"{ocr_confidence:.3f}"
                        if ocr_confidence is not None
                        else ""
                    ),
                    "plate_x1": x1,
                    "plate_y1": y1,
                    "plate_x2": x2,
                    "plate_y2": y2,
                })

            # Keep the existing Phase 6 overlay visible.
            writer.write(frame)

            if frame_no % 30 == 0:
                print(
                    f"Processed {frame_no}/{total_frames}"
                )

    finally:
        cap.release()
        writer.release()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    columns = [
        "frame",
        "video_time_s",
        "vehicle_id",
        "plate_text",
        "ocr_confidence",
        "plate_x1",
        "plate_y1",
        "plate_x2",
        "plate_y2",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        csv_writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print("\nPHASE 6 + LICENSE PLATE RECOGNITION COMPLETE")
    print(f"Frames processed: {frame_no}")
    print(f"Plate detections: {plate_count}")
    print(f"CSV rows: {len(rows)}")
    print(f"Final video: {OUTPUT_VIDEO}")
    print(f"Plate report: {OUTPUT_CSV}")
    print(
        "OCR readings are estimates; verify them against "
        "the actual plates."
    )


if __name__ == "__main__":
    main()
