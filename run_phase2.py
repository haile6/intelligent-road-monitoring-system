r"""
PHASE 2: detect + TRACK road damage, confirm each damage once.

Run:   python run_phase2.py

Outputs:
  outputs\annotated_phase2.mp4   video with track IDs
  outputs\phase2_events.csv      one row per confirmed road-damage event
"""
import csv
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker
from visualization.overlay import draw_tracks, draw_banner


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def main(detector=None):
    if not os.path.exists(config.VIDEO_PATH):
        raise SystemExit(f"Video not found: {config.VIDEO_PATH}\nFix VIDEO_PATH in config.py")
    if detector is None:
        if not os.path.exists(config.MODEL_PATH):
            raise SystemExit(f"Model not found: {config.MODEL_PATH}")
        detector = RoadDamageDetector()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {width}x{height}, {fps:.1f} fps, {total} frames")
    print(f"Confirmation: {config.MIN_CONFIRMATION_FRAMES} frames | "
          f"IoU {config.TRACK_IOU_THRESHOLD} | max missed {config.TRACK_MAX_MISSED_FRAMES}\n")

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PHASE2), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PHASE2,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = DamageTracker()
    raw_detections = 0
    frame_no = 0
    confirmed_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        timestamp = frame_no / fps                     # seconds from the video start

        detections = detector.detect(frame)
        raw_detections += len(detections)
        visible, newly_confirmed = tracker.update(detections, frame_no, timestamp)

        for t in newly_confirmed:
            confirmed_counts[t.class_name] += 1
            print(f"[{fmt_time(timestamp)}] CONFIRMED  {t.class_name:16s} "
                  f"ID {t.id:<3d} confidence {t.confidence:.2f}  frame {frame_no}")

        draw_tracks(frame, visible)
        draw_banner(frame, [
            f"Confirmed damage events: {sum(confirmed_counts.values())}",
            f"Potholes: {confirmed_counts['pothole']}   "
            f"Alligator cracks: {confirmed_counts['alligator_crack']}",
        ])
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 2 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- save the confirmed events -------------------------------------
    events = tracker.confirmed_tracks()
    with open(config.EVENTS_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "best_confidence", "first_frame",
                    "confirmed_frame", "last_frame", "frames_detected", "video_time_s"])
        for t in events:
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.first_frame,
                        t.confirmed_frame, t.last_frame, t.hits, f"{t.confirmed_time:.2f}"])

    print("\nDONE")
    print(f"  Frames processed:              {frame_no}")
    print(f"  Raw detections (all frames):   {raw_detections}")
    print(f"  Confirmed road-damage events:  {len(events)}")
    for name, n in confirmed_counts.items():
        print(f"     {name}: {n}")
    print(f"  Saved video: {config.OUTPUT_VIDEO_PHASE2}")
    print(f"  Saved events: {config.EVENTS_CSV_PATH}")


if __name__ == "__main__":
    main()
