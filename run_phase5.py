r"""
PHASE 5: estimated distance to road damage.

Run:   python run_phase5.py

Builds on Phase 4 (damage tracking, vehicles, path) and adds "Estimated distance".
The distance is an approximation from camera height, tilt and field of view.
Check it first with:  python check_distance.py

Outputs (folder outputs):
  annotated_phase5.mp4        video with path, damage (with ~distance) and vehicles
  phase5_damage_events.csv    confirmed damage events with path and distance
"""
import csv
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker, VehicleTracker
from risk.path_analyzer import PathAnalyzer
from risk.distance_estimator import make_estimator
from visualization.overlay import (draw_tracks, draw_vehicles, draw_path,
                                   draw_horizon, draw_banner)


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def fmt_m(value):
    return f"{value:.1f}" if value is not None else ""


def main(detector=None, vehicle_detector=None):
    if not os.path.exists(config.VIDEO_PATH):
        raise SystemExit(f"Video not found: {config.VIDEO_PATH}\nFix VIDEO_PATH in config.py")
    if detector is None:
        if not os.path.exists(config.MODEL_PATH):
            raise SystemExit(f"Model not found: {config.MODEL_PATH}")
        detector = RoadDamageDetector()
    if vehicle_detector is None and config.PHASE4_USE_VEHICLES:
        from detection.vehicle_detector import VehicleDetector
        vehicle_detector = VehicleDetector()
    use_vehicles = config.PHASE4_USE_VEHICLES and vehicle_detector is not None

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    path = PathAnalyzer(width, height)
    distance = make_estimator(width, height)
    print(f"\nVideo: {width}x{height}, {fps:.1f} fps, {total} frames")
    print(f"Distance method: {distance.method}")
    print(f"  camera height {config.CAMERA_HEIGHT_M} m, tilt {config.CAMERA_TILT_DEG} deg, "
          f"focal length {distance.focal:.0f} px, horizon at row {distance.horizon_row:.0f}")
    print("  (distances are ESTIMATES)\n")

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PHASE5), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PHASE5,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    damage_tracker = DamageTracker()
    vehicle_tracker = VehicleTracker()
    damage_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    vehicle_total = 0
    reported_in_path = set()
    frames_with_path_risk = 0
    frame_no = 0
    a = config.DISTANCE_SMOOTHING

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        timestamp = frame_no / fps

        # ---- road damage: detect, track, confirm ----
        visible, newly_confirmed = damage_tracker.update(
            detector.detect(frame), frame_no, timestamp)
        for t in newly_confirmed:
            damage_counts[t.class_name] += 1

        # ---- path + estimated distance for every visible damage ----
        danger_now = []
        for t in visible:
            result = path.analyze(t.bbox)
            t.in_path = result["in_path"]

            d = distance.estimate(t.bbox, frame)["distance_m"]
            if d is not None:                                   # smooth the jitter
                t.distance_m = d if t.distance_m is None else a * d + (1 - a) * t.distance_m
                t.min_distance_m = (t.distance_m if t.min_distance_m is None
                                    else min(t.min_distance_m, t.distance_m))
            if t.confirmed and t.first_distance_m is None:
                t.first_distance_m = t.distance_m

            if t.in_path:
                t.path_frames += 1
                t.max_path_overlap = max(t.max_path_overlap, result["overlap_ratio"])
                if t.first_path_frame is None:
                    t.first_path_frame = frame_no
                    t.path_entry_distance_m = t.distance_m
                if t.confirmed:
                    danger_now.append(t)
                    if t.id not in reported_in_path:
                        reported_in_path.add(t.id)
                        dist_text = f"~{t.distance_m:.0f} m" if t.distance_m is not None else "unknown"
                        print(f"[{fmt_time(timestamp)}] IN PATH  {t.class_name:16s} ID {t.id:<3d} "
                              f"estimated distance {dist_text}")
        if danger_now:
            frames_with_path_risk += 1

        # ---- other vehicles ----
        vehicle_visible = []
        if use_vehicles:
            vehicle_visible, vehicle_new = vehicle_tracker.update(
                vehicle_detector.detect(frame), frame_no, timestamp)
            vehicle_total += len(vehicle_new)

        # ---- draw ----
        draw_path(frame, path.polygon, danger=bool(danger_now))
        if config.SHOW_HORIZON_LINE:
            draw_horizon(frame, distance.horizon_row)
        if use_vehicles:
            draw_vehicles(frame, vehicle_visible)
        draw_tracks(frame, visible, path_mode=True, show_distance=True)

        distances = [t.distance_m for t in danger_now if t.distance_m is not None]
        if danger_now:
            names = ", ".join(f"{t.class_name} ID {t.id}" for t in danger_now)
            risk_line = f"PATH RISK: TRUE  ({names})"
        else:
            risk_line = "PATH RISK: FALSE"
        near_line = (f"Nearest damage in path: ~{min(distances):.0f} m (estimated)"
                     if distances else "Nearest damage in path: none")
        lines = [risk_line, near_line,
                 f"Damage events: {sum(damage_counts.values())}  "
                 f"(in path so far: {len(reported_in_path)})"]
        if use_vehicles:
            lines.append(f"Other vehicles tracked: {vehicle_total}")
        draw_banner(frame, lines)
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 5 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    events = damage_tracker.confirmed_tracks()
    with open(config.DAMAGE_CSV_PHASE5, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "best_confidence", "first_frame",
                    "confirmed_frame", "last_frame", "frames_detected", "video_time_s",
                    "ever_in_path", "frames_in_path", "first_in_path_frame", "max_path_overlap",
                    "est_distance_at_confirmation_m", "est_distance_at_path_entry_m",
                    "est_min_distance_m"])
        for t in events:
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.first_frame,
                        t.confirmed_frame, t.last_frame, t.hits, f"{t.confirmed_time:.2f}",
                        t.path_frames > 0, t.path_frames,
                        t.first_path_frame if t.first_path_frame else "",
                        f"{t.max_path_overlap:.2f}",
                        fmt_m(t.first_distance_m), fmt_m(t.path_entry_distance_m),
                        fmt_m(t.min_distance_m)])

    in_path_events = sum(1 for t in events if t.path_frames > 0)
    print("\nDONE")
    print(f"  Frames processed:              {frame_no}")
    print(f"  Confirmed damage events:       {len(events)}  {damage_counts}")
    print(f"  ...in our path / off our path: {in_path_events} / {len(events) - in_path_events}")
    print(f"  Frames with PATH RISK = TRUE:  {frames_with_path_risk}")
    print(f"  Saved video:  {config.OUTPUT_VIDEO_PHASE5}")
    print(f"  Saved events: {config.DAMAGE_CSV_PHASE5}")
    print("  Reminder: distances are estimates.")


if __name__ == "__main__":
    main()
