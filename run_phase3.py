r"""
PHASE 3: road damage (Phase 2) + OTHER vehicles (car, truck, bus, motorcycle).

Run:   python run_phase3.py

Outputs (folder outputs):
  annotated_phase3.mp4        video with damage boxes and vehicle boxes
  phase3_damage_events.csv    confirmed road-damage events
  phase3_vehicles.csv         confirmed vehicles (one row per tracked vehicle)
"""
import csv
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.vehicle_detector import VehicleDetector
from detection.tracker import DamageTracker, VehicleTracker
from visualization.overlay import draw_tracks, draw_vehicles, draw_banner


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def main(detector=None, vehicle_detector=None):
    if not os.path.exists(config.VIDEO_PATH):
        raise SystemExit(f"Video not found: {config.VIDEO_PATH}\nFix VIDEO_PATH in config.py")
    if detector is None:
        if not os.path.exists(config.MODEL_PATH):
            raise SystemExit(f"Model not found: {config.MODEL_PATH}")
        detector = RoadDamageDetector()
    if vehicle_detector is None:
        vehicle_detector = VehicleDetector()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {width}x{height}, {fps:.1f} fps, {total} frames\n")

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PHASE3), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PHASE3,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    damage_tracker = DamageTracker()
    vehicle_tracker = VehicleTracker()
    damage_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    vehicle_counts = {name: 0 for name in config.VEHICLE_CLASSES.values()}
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        timestamp = frame_no / fps

        # ---- road damage ----
        damage_visible, damage_new = damage_tracker.update(
            detector.detect(frame), frame_no, timestamp)
        for t in damage_new:
            damage_counts[t.class_name] += 1
            print(f"[{fmt_time(timestamp)}] DAMAGE   {t.class_name:16s} ID {t.id:<3d} "
                  f"confidence {t.confidence:.2f}")

        # ---- other vehicles ----
        vehicle_visible, vehicle_new = vehicle_tracker.update(
            vehicle_detector.detect(frame), frame_no, timestamp)
        for t in vehicle_new:
            vehicle_counts[t.class_name] += 1
            print(f"[{fmt_time(timestamp)}] VEHICLE  {t.class_name:16s} ID {t.id:<3d} "
                  f"confidence {t.confidence:.2f}")

        # ---- draw ----
        draw_vehicles(frame, vehicle_visible)
        draw_tracks(frame, damage_visible)
        in_view = sum(1 for t in vehicle_visible if t.confirmed)
        veh_text = "  ".join(f"{n}: {c}" for n, c in vehicle_counts.items() if c) or "none yet"
        draw_banner(frame, [
            f"Damage events: {sum(damage_counts.values())}  "
            f"(potholes {damage_counts['pothole']}, cracks {damage_counts['alligator_crack']})",
            f"Vehicles tracked: {sum(vehicle_counts.values())}  | in view now: {in_view}",
            f"By type -> {veh_text}",
        ])
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 3 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- save reports ----
    with open(config.DAMAGE_CSV_PHASE3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "best_confidence", "first_frame",
                    "confirmed_frame", "last_frame", "frames_detected", "video_time_s"])
        for t in damage_tracker.confirmed_tracks():
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.first_frame,
                        t.confirmed_frame, t.last_frame, t.hits, f"{t.confirmed_time:.2f}"])

    with open(config.VEHICLES_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "vehicle_type", "best_confidence", "first_frame",
                    "last_frame", "frames_detected", "first_seen_s", "last_seen_s"])
        for t in vehicle_tracker.confirmed_tracks():
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.first_frame,
                        t.last_frame, t.hits, f"{t.first_time:.2f}", f"{t.last_time:.2f}"])

    print("\nDONE")
    print(f"  Frames processed:            {frame_no}")
    print(f"  Confirmed damage events:     {sum(damage_counts.values())}  {damage_counts}")
    print(f"  Other vehicles tracked:      {sum(vehicle_counts.values())}  {vehicle_counts}")
    print(f"  Saved video:    {config.OUTPUT_VIDEO_PHASE3}")
    print(f"  Saved damage:   {config.DAMAGE_CSV_PHASE3}")
    print(f"  Saved vehicles: {config.VEHICLES_CSV_PATH}")


if __name__ == "__main__":
    main()
