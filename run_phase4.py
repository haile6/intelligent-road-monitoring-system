r"""
PHASE 4: is the detected road damage in the path of our vehicle?

Run:   python run_phase4.py

The path is an APPROXIMATION (a fixed trapezoid in the lower-centre of the picture).
Tune it in config.py until the green shape covers the lane your car drives in.

Outputs (folder outputs):
  annotated_phase4.mp4        path shape + damage (thick = in path) + vehicles
  phase4_damage_events.csv    confirmed damage events with path information
"""
import csv
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker, VehicleTracker
from risk.path_analyzer import PathAnalyzer
from visualization.overlay import draw_tracks, draw_vehicles, draw_path, draw_banner


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


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
    print(f"\nVideo: {width}x{height}, {fps:.1f} fps, {total} frames")

    path = PathAnalyzer(width, height)
    print(f"Path polygon (x, y): {path.polygon.tolist()}\n")

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PHASE4), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PHASE4,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    damage_tracker = DamageTracker()
    vehicle_tracker = VehicleTracker()
    damage_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    vehicle_total = 0
    reported_in_path = set()
    frames_with_path_risk = 0
    frame_no = 0

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

        # ---- is each damage in our path? ----
        danger_now = []
        for t in visible:
            result = path.analyze(t.bbox)
            t.in_path = result["in_path"]
            if t.in_path:
                t.path_frames += 1
                t.max_path_overlap = max(t.max_path_overlap, result["overlap_ratio"])
                if t.first_path_frame is None:
                    t.first_path_frame = frame_no
                if t.confirmed:
                    danger_now.append(t)
                    if t.id not in reported_in_path:
                        reported_in_path.add(t.id)
                        print(f"[{fmt_time(timestamp)}] IN PATH  {t.class_name:16s} ID {t.id:<3d} "
                              f"overlap {result['overlap_ratio']:.0%}")
        if danger_now:
            frames_with_path_risk += 1

        # ---- other vehicles (from Phase 3) ----
        vehicle_visible = []
        if use_vehicles:
            vehicle_visible, vehicle_new = vehicle_tracker.update(
                vehicle_detector.detect(frame), frame_no, timestamp)
            vehicle_total += len(vehicle_new)

        # ---- draw ----
        draw_path(frame, path.polygon, danger=bool(danger_now))
        if use_vehicles:
            draw_vehicles(frame, vehicle_visible)
        draw_tracks(frame, visible, path_mode=True)

        if danger_now:
            names = ", ".join(f"{t.class_name} ID {t.id}" for t in danger_now)
            risk_line = f"PATH RISK: TRUE  ({names})"
        else:
            risk_line = "PATH RISK: FALSE"
        lines = [risk_line,
                 f"Damage events: {sum(damage_counts.values())}  "
                 f"(in path so far: {len(reported_in_path)})"]
        if use_vehicles:
            lines.append(f"Other vehicles tracked: {vehicle_total}")
        draw_banner(frame, lines)
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 4 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- save the confirmed events with path information ----
    events = damage_tracker.confirmed_tracks()
    with open(config.DAMAGE_CSV_PHASE4, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "best_confidence", "first_frame",
                    "confirmed_frame", "last_frame", "frames_detected", "video_time_s",
                    "ever_in_path", "frames_in_path", "first_in_path_frame", "max_path_overlap"])
        for t in events:
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.first_frame,
                        t.confirmed_frame, t.last_frame, t.hits, f"{t.confirmed_time:.2f}",
                        t.path_frames > 0, t.path_frames,
                        t.first_path_frame if t.first_path_frame else "",
                        f"{t.max_path_overlap:.2f}"])

    in_path_events = sum(1 for t in events if t.path_frames > 0)
    print("\nDONE")
    print(f"  Frames processed:                {frame_no}")
    print(f"  Confirmed damage events:         {len(events)}  {damage_counts}")
    print(f"  ...of which in our path:         {in_path_events}")
    print(f"  ...of which off our path:        {len(events) - in_path_events}")
    print(f"  Frames with PATH RISK = TRUE:    {frames_with_path_risk}")
    print(f"  Saved video:  {config.OUTPUT_VIDEO_PHASE4}")
    print(f"  Saved events: {config.DAMAGE_CSV_PHASE4}")


if __name__ == "__main__":
    main()
