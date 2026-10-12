r"""
PHASE 6: GPS synchronised with the video.

Run:   python run_phase6.py

Builds on Phase 5 and adds, for every confirmed road-damage event:
  timestamp, latitude, longitude, speed_kmh (from the GPS file), distance, frame number.
ttc_seconds and severity are left empty on purpose: they are calculated in later phases.

Outputs (folder outputs):
  annotated_phase6.mp4        video with GPS information
  phase6_damage_events.csv    confirmed damage events with GPS
"""
import csv
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker, VehicleTracker
from gps.gps_reader import load_gps_csv, describe
from gps.gps_sync import GpsSync
from risk.path_analyzer import PathAnalyzer
from risk.distance_estimator import make_estimator
from visualization.overlay import (draw_tracks, draw_vehicles, draw_path,
                                   draw_horizon, draw_banner)


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def num(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def main(detector=None, vehicle_detector=None):
    if not os.path.exists(config.VIDEO_PATH):
        raise SystemExit(f"Video not found: {config.VIDEO_PATH}\nFix VIDEO_PATH in config.py")
    if not os.path.exists(config.GPS_PATH):
        raise SystemExit(f"GPS file not found: {config.GPS_PATH}\n"
                         "Make a test file with:  python make_sample_gps.py --video <your video>")
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
    duration = total / fps

    # ---- GPS ----
    records = load_gps_csv(config.GPS_PATH)
    gps = GpsSync(records)
    print(f"\nGPS: {describe(records)}")
    print(f"Video: {width}x{height}, {fps:.1f} fps, {duration:.0f} s, "
          f"starts at {gps.video_start.isoformat(' ', 'seconds')} (GPS offset {gps.offset:+.1f} s)")
    coverage = gps.coverage(duration)
    print(f"GPS position available for {coverage:.0%} of the video")
    if coverage == 0:
        raise SystemExit(
            "\nERROR: the GPS file does not cover the video at all.\n"
            "  - Check VIDEO_START_TIME in config.py (same clock and time zone as the GPS file).\n"
            "  - GPX files from phones use UTC; Ethiopia is UTC+3 (see gpx_to_csv.py --tz-offset-hours).")
    if all(r["speed_kmh"] is None for r in records):
        print("NOTE: the GPS file has no speed column, so speed_kmh will be empty.")

    path = PathAnalyzer(width, height)
    distance = make_estimator(width, height)

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PHASE6), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PHASE6,
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
        timestamp = frame_no / fps                       # seconds from the video start
        fix = gps.at(timestamp)                          # car position now (or None)

        visible, newly_confirmed = damage_tracker.update(
            detector.detect(frame), frame_no, timestamp)
        for t in newly_confirmed:
            damage_counts[t.class_name] += 1

        danger_now = []
        for t in visible:
            result = path.analyze(t.bbox)
            t.in_path = result["in_path"]

            d = distance.estimate(t.bbox, frame)["distance_m"]
            if d is not None:
                t.distance_m = d if t.distance_m is None else a * d + (1 - a) * t.distance_m
                t.min_distance_m = (t.distance_m if t.min_distance_m is None
                                    else min(t.min_distance_m, t.distance_m))
            if t.confirmed and t.first_distance_m is None:
                t.first_distance_m = t.distance_m

            # ---- GPS for the event: looked up ONCE, at the moment of confirmation ----
            if t.confirmed and t.gps_status is None:
                t.gps_time = gps.absolute_time(timestamp)
                if fix is None:
                    t.gps_status = "no_fix"
                else:
                    t.gps_status = "ok"
                    t.latitude, t.longitude = fix["latitude"], fix["longitude"]
                    t.speed_kmh = fix["speed_kmh"]
                    if config.PROJECT_DAMAGE_LOCATION:
                        t.damage_latitude, t.damage_longitude = gps.damage_location(
                            timestamp, t.distance_m)
                dist_text = f"~{t.distance_m:.0f} m" if t.distance_m is not None else "unknown"
                gps_text = (f"GPS {t.latitude:.6f}, {t.longitude:.6f}"
                            f"{'' if t.speed_kmh is None else f', {t.speed_kmh:.0f} km/h'}"
                            if t.gps_status == "ok" else "GPS no fix")
                print(f"[{fmt_time(timestamp)}] CONFIRMED  {t.class_name:16s} ID {t.id:<3d} "
                      f"conf {t.confidence:.2f} | {dist_text} | {gps_text}")

            if t.in_path:
                t.path_frames += 1
                t.max_path_overlap = max(t.max_path_overlap, result["overlap_ratio"])
                if t.first_path_frame is None:
                    t.first_path_frame = frame_no
                    t.path_entry_distance_m = t.distance_m
                if t.confirmed:
                    danger_now.append(t)
                    reported_in_path.add(t.id)
        if danger_now:
            frames_with_path_risk += 1

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
        if fix is None:
            gps_line = "GPS: no fix"
        else:
            speed = "speed unknown" if fix["speed_kmh"] is None else f"{fix['speed_kmh']:.0f} km/h"
            gps_line = f"GPS {fix['latitude']:.5f}, {fix['longitude']:.5f}  {speed}"
        lines = ["PATH RISK: TRUE" if danger_now else "PATH RISK: FALSE",
                 (f"Nearest damage in path: ~{min(distances):.0f} m (estimated)"
                  if distances else "Nearest damage in path: none"),
                 gps_line,
                 f"Damage events: {sum(damage_counts.values())}"]
        draw_banner(frame, lines)
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 6 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- save the events ----
    events = damage_tracker.confirmed_tracks()
    with open(config.DAMAGE_CSV_PHASE6, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "confidence", "frame_number", "video_time_s",
                    "timestamp", "latitude", "longitude", "speed_kmh", "distance_m",
                    "ttc_seconds", "severity",
                    "damage_latitude", "damage_longitude",
                    "ever_in_path", "est_min_distance_m", "gps_status"])
        for t in events:
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.confirmed_frame,
                        f"{t.confirmed_time:.2f}",
                        t.gps_time.isoformat(" ", "seconds") if t.gps_time else "",
                        num(t.latitude, 6), num(t.longitude, 6), num(t.speed_kmh, 1),
                        num(t.first_distance_m, 1), "", "",
                        num(t.damage_latitude, 6), num(t.damage_longitude, 6),
                        t.path_frames > 0, num(t.min_distance_m, 1), t.gps_status])

    with_gps = sum(1 for t in events if t.gps_status == "ok")
    print("\nDONE")
    print(f"  Frames processed:           {frame_no}")
    print(f"  Confirmed damage events:    {len(events)}  {damage_counts}")
    print(f"  ...with a GPS position:     {with_gps}")
    print(f"  ...without GPS (no fix):    {len(events) - with_gps}")
    print(f"  Saved video:  {config.OUTPUT_VIDEO_PHASE6}")
    print(f"  Saved events: {config.DAMAGE_CSV_PHASE6}")
    print("  Reminder: the event position is the CAR position; damage_latitude/longitude is an estimate.")


if __name__ == "__main__":
    main()
