
import argparse
import csv
import os

import cv2
try:
    import easyocr
except ImportError:
    easyocr = None
from ultralytics import YOLO

from detection.vehicle_detector import VehicleDetector
from detection.tracker import VehicleTracker


PLATE_MODEL_PATH = r"models\license_plate_detector.pt"
DEFAULT_VIDEO = r"data\new_test_video.mp4"

OUTPUT_DIR = r"outputs\phase7"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR, "annotated_phase7.mp4"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, "phase7_plate_readings.csv"
)

PLATE_CONFIDENCE = 0.25
PLATE_IMAGE_SIZE = 960
OCR_EVERY_N_FRAMES = 3


def associate_vehicle(plate_box, vehicle_tracks):
    """
    Associate a detected plate with a confirmed vehicle
    when the plate center falls inside the vehicle box.
    """
    px1, py1, px2, py2 = plate_box

    center_x = (px1 + px2) / 2
    center_y = (py1 + py2) / 2

    candidates = []

    for vehicle in vehicle_tracks:
        if not vehicle.confirmed:
            continue

        x1, y1, x2, y2 = vehicle.bbox

        if (
            x1 <= center_x <= x2
            and y1 <= center_y <= y2
        ):
            area = max(
                1,
                (x2 - x1) * (y2 - y1)
            )
            candidates.append((area, vehicle.id))

    if not candidates:
        return ""

    # Choose the smallest vehicle box that contains the plate.
    candidates.sort()
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: License plate detection and OCR"
    )

    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="Path to the input video",
    )

    args = parser.parse_args()
    video_path = args.video

    # Check required input files.
    if not os.path.isfile(video_path):
        raise SystemExit(
            f"Video not found: {video_path}"
        )

    if not os.path.isfile(PLATE_MODEL_PATH):
        raise SystemExit(
            f"License plate model not found: "
            f"{PLATE_MODEL_PATH}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading vehicle detector...")
    vehicle_detector = VehicleDetector()
    vehicle_tracker = VehicleTracker()

    print("Loading license plate detector...")
    plate_detector = YOLO(PLATE_MODEL_PATH)

    reader = None
    if easyocr is not None:
        print(
            "Loading EasyOCR. The first run may take longer "
            "while recognition files are prepared..."
        )
        try:
            reader = easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            print(f"[!] Warning: Could not initialize EasyOCR reader: {e}")
            reader = None
    else:
        print("[!] Note: EasyOCR is not installed. Running plate detection without character recognition.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise SystemExit(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit(
            f"Could not create output video: "
            f"{OUTPUT_VIDEO}"
        )

    # Store the best OCR result for each tracked vehicle.
    best_reading = {}

    rows = []
    frame_no = 0
    plate_detections = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1
            timestamp = (frame_no - 1) / fps

            # Update vehicle detections and tracking.
            vehicle_detections = vehicle_detector.detect(
                frame
            )

            vehicle_visible, _ = vehicle_tracker.update(
                vehicle_detections,
                frame_no,
                timestamp,
            )

            # Detect license plates in the current frame.
            plate_result = plate_detector.predict(
                frame,
                imgsz=PLATE_IMAGE_SIZE,
                conf=PLATE_CONFIDENCE,
                verbose=False,
            )[0]

            # Draw tracked vehicle boxes.
            for vehicle in vehicle_visible:
                x1, y1, x2, y2 = vehicle.bbox

                color = (255, 180, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                vehicle_label = (
                    f"Vehicle ID {vehicle.id}: "
                    f"{vehicle.class_name}"
                )

                cv2.putText(
                    frame,
                    vehicle_label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Process every detected license plate.
            for box in plate_result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                # Keep the coordinates inside the image.
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_detections += 1

                plate_box = (
                    x1, y1, x2, y2
                )

                vehicle_id = associate_vehicle(
                    plate_box,
                    vehicle_visible,
                )

                plate_text = ""
                ocr_confidence = None

                # Run OCR every third frame.
                if (
                    (frame_no - 1) % OCR_EVERY_N_FRAMES
                    == 0
                ):
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # Add a small border around the plate.
                        crop = cv2.copyMakeBorder(
                            crop,
                            8, 8, 8, 8,
                            cv2.BORDER_REPLICATE,
                        )

                        # Enlarge the crop for OCR.
                        crop = cv2.resize(
                            crop,
                            None,
                            fx=3,
                            fy=3,
                            interpolation=cv2.INTER_CUBIC,
                        )

                        readings = []
                        if reader is not None:
                            try:
                                ocr_results = reader.readtext(
                                    crop,
                                    detail=1,
                                    paragraph=False,
                                )
                                readings = [
                                    (
                                        text.strip(),
                                        float(score),
                                    )
                                    for _, text, score
                                    in ocr_results
                                    if text.strip()
                                ]
                            except Exception:
                                readings = []

                        if readings:
                            plate_text = " ".join(
                                text
                                for text, _ in readings
                            )

                            ocr_confidence = (
                                sum(
                                    score
                                    for _, score in readings
                                )
                                / len(readings)
                            )

                            # Keep the highest-confidence
                            # reading for this vehicle.
                            if vehicle_id != "":
                                old = best_reading.get(
                                    vehicle_id
                                )

                                if (
                                    old is None
                                    or ocr_confidence > old[1]
                                ):
                                    best_reading[vehicle_id] = (
                                        plate_text,
                                        ocr_confidence,
                                    )

                # Reuse a previous reading for this vehicle
                # when the current OCR result is empty.
                if (
                    vehicle_id != ""
                    and vehicle_id in best_reading
                ):
                    known_text, known_conf = (
                        best_reading[vehicle_id]
                    )

                    if not plate_text:
                        plate_text = known_text
                        ocr_confidence = known_conf

                if vehicle_id != "" and plate_text:
                    label = (
                        f"ID {vehicle_id}: {plate_text}"
                    )
                elif vehicle_id != "":
                    label = (
                        f"ID {vehicle_id}: OCR pending"
                    )
                elif plate_text:
                    label = plate_text
                else:
                    label = "Plate detected - unreadable"

                # Draw the license plate box and label.
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
                    "plate_x1": x1,
                    "plate_y1": y1,
                    "plate_x2": x2,
                    "plate_y2": y2,
                    "plate_text": plate_text,
                    "ocr_confidence": (
                        f"{ocr_confidence:.3f}"
                        if ocr_confidence is not None
                        else ""
                    ),
                })

            # Add frame progress to the output video.
            cv2.putText(
                frame,
                f"Phase 7 | Frame {frame_no}/{total_frames}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

            if frame_no % 30 == 0:
                print(
                    f"Processed {frame_no}/"
                    f"{total_frames} frames"
                )

    finally:
        cap.release()
        writer.release()

        # Some OpenCV Windows installations do not have
        # GUI support. Handle that without stopping cleanup.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # Save the CSV after video processing has finished.
    columns = [
        "frame",
        "video_time_s",
        "vehicle_id",
        "plate_x1",
        "plate_y1",
        "plate_x2",
        "plate_y2",
        "plate_text",
        "ocr_confidence",
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

    unique_vehicles = {
        row["vehicle_id"]
        for row in rows
        if row["vehicle_id"] != ""
    }

    readable = sum(
        1
        for row in rows
        if row["plate_text"].strip()
    )

    print("\nPHASE 7 COMPLETE")
    print(f"Frames processed: {frame_no}")
    print(
        f"Plate detections across frames: "
        f"{plate_detections}"
    )
    print(f"Rows with OCR text: {readable}")
    print(
        f"Vehicle IDs associated with plates: "
        f"{len(unique_vehicles)}"
    )
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Plate report: {OUTPUT_CSV}")
    print(
        "Please verify OCR text against the actual "
        "license plates."
    )


if __name__ == "__main__":
    main()
