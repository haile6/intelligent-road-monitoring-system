r"""
PHASE 9: risk analysis.

Run:   python run_phase9.py

Builds on Phase 8. For every confirmed damage that is IN OUR PATH it decides a risk level:
    SAFE / MONITOR  |  CAUTION  |  WARNING  |  IMMEDIATE DANGER
from the time-to-impact, damage type, detection confidence, path, distance and speed.
The thresholds are PROTOTYPE values in config.py (RISK_*) - they are not validated safety limits.
Every level comes with its reasons (shown in the console and saved in the CSV).

Outputs (folder outputs):
  annotated_phase9.mp4        video: the path and the banner are coloured by risk level
  phase9_damage_events.csv    confirmed damage events with the highest risk each one reached
"""
import csv
import os
import cv2
import argparse

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker, VehicleTracker
from gps.gps_reader import load_gps_csv, describe
from gps.gps_sync import GpsSync
from gps.speed_provider import SpeedProvider
from risk.path_analyzer import PathAnalyzer
from risk.distance_estimator import make_estimator
from risk.ttc_calculator import calculate_ttc, format_ttc
from risk.risk_analyzer import assess_risk, scene_risk
from visualization.overlay import (draw_tracks, draw_vehicles, draw_path,
                                   draw_horizon, draw_banner)

WHITE, YELLOW, ORANGE = (255, 255, 255), (0, 255, 255), (0, 140, 255)


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def num(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def speed_text(info):
    """Text and colour for the speed line: a default value is clearly marked."""
    if info["is_default"]:
        return f"Speed: {info['speed_kmh']:.0f} km/h  (DEFAULT - test value)", ORANGE
    if info["source"] == "computed_from_positions":
        return f"Speed: {info['speed_kmh']:.0f} km/h  (computed from GPS positions)", YELLOW
    return f"Speed: {info['speed_kmh']:.0f} km/h  (GPS)", WHITE


def main(video_path=None, gps_path=None, show=False, output_video=None, output_csv=None,
         detector=None, vehicle_detector=None):
    video_path = video_path or config.VIDEO_PATH
    gps_path = gps_path or getattr(config, "GPS_PATH", os.path.join(config.BASE_DIR, "data", "gps_track.csv"))
    output_video = output_video or config.OUTPUT_VIDEO_PHASE9
    output_csv = output_csv or config.DAMAGE_CSV_PHASE9

    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}\nFix VIDEO_PATH in config.py or pass --video")
    if detector is None:
        if not os.path.exists(config.MODEL_PATH):
            raise SystemExit(f"Model not found: {config.MODEL_PATH}")
        detector = RoadDamageDetector()
    if vehicle_detector is None and config.PHASE4_USE_VEHICLES:
        from detection.vehicle_detector import VehicleDetector
        vehicle_detector = VehicleDetector()
    use_vehicles = config.PHASE4_USE_VEHICLES and vehicle_detector is not None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    # ---- GPS (optional in this phase) ----
    gps = None
    if gps_path and os.path.exists(gps_path):
        records = load_gps_csv(gps_path)
        gps = GpsSync(records)
        print(f"\nGPS: {describe(records)}")
        print(f"GPS position available for {gps.coverage(duration):.0%} of the video")
    else:
        print(f"\nNOTE: no GPS file at {gps_path}")
        print(f"      Every speed will be the DEFAULT test value ({config.DEFAULT_SPEED_KMH} km/h).")
    if config.FORCE_DEFAULT_SPEED:
        print(f"NOTE: FORCE_DEFAULT_SPEED is on: GPS speed is ignored.")
    speed_provider = SpeedProvider(gps)
    print(f"Video: {width}x{height}, {fps:.1f} fps, {duration:.0f} s\n")

    path = PathAnalyzer(width, height)
    distance = make_estimator(width, height)

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    writer = cv2.VideoWriter(output_video,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    damage_tracker = DamageTracker()
    vehicle_tracker = VehicleTracker()
    damage_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    speed_frames = {"gps": 0, "computed_from_positions": 0, "default": 0}
    speeds = []
    vehicle_total = 0
    frames_with_path_risk = 0
    risk_frames = {name: 0 for name in config.RISK_LEVELS}
    frame_no = 0
    a = config.DISTANCE_SMOOTHING

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        timestamp = frame_no / fps
        fix = gps.at(timestamp) if gps else None
        speed = speed_provider.get(timestamp)             # speed now, with its source
        speed_frames[speed["source"]] += 1
        speeds.append(speed["speed_kmh"])

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

            # ---- time-to-impact: estimated distance / CURRENT speed ----
            ttc = calculate_ttc(t.distance_m, speed["speed_mps"])
            t.ttc_s, t.ttc_status = ttc["ttc_s"], ttc["status"]
            if t.confirmed and t.in_path and t.ttc_s is not None:
                t.min_ttc_s = t.ttc_s if t.min_ttc_s is None else min(t.min_ttc_s, t.ttc_s)

            # ---- risk level of this damage ----
            if t.confirmed:
                risk = assess_risk(t, speed)
                t.risk_level, t.risk_rank = risk["level"], risk["rank"]
                if t.risk_rank > t.max_risk_rank:                # a new highest level for this damage
                    t.max_risk_level, t.max_risk_rank = risk["level"], risk["rank"]
                    t.max_risk_reasons = risk["reasons"]
                    print(f"[{fmt_time(timestamp)}] RISK {risk['label']:16s} {t.class_name} "
                          f"ID {t.id} | {'; '.join(risk['reasons'])}")

            # ---- GPS + speed for the event: looked up ONCE, when it is confirmed ----
            if t.confirmed and t.gps_status is None:
                if gps is not None:
                    t.gps_time = gps.absolute_time(timestamp)
                t.speed_kmh, t.speed_mps, t.speed_source = (
                    speed["speed_kmh"], speed["speed_mps"], speed["source"])
                t.first_ttc_s = t.ttc_s
                if gps is None:
                    t.gps_status = "no_gps_file"
                elif fix is None:
                    t.gps_status = "no_fix"
                else:
                    t.gps_status = "ok"
                    t.latitude, t.longitude = fix["latitude"], fix["longitude"]
                    if config.PROJECT_DAMAGE_LOCATION:
                        t.damage_latitude, t.damage_longitude = gps.damage_location(
                            timestamp, t.distance_m)
                dist_text = f"~{t.distance_m:.0f} m" if t.distance_m is not None else "unknown"
                tag = "DEFAULT test value" if speed["is_default"] else speed["source"]
                print(f"[{fmt_time(timestamp)}] CONFIRMED  {t.class_name:16s} ID {t.id:<3d} "
                      f"conf {t.confidence:.2f} | {dist_text} | "
                      f"{speed['speed_kmh']:.0f} km/h ({tag}, {speed['speed_mps']:.1f} m/s) | "
                      f"TTC {format_ttc(t.ttc_s, t.ttc_status)}"
                      f"{' [test value: default speed]' if speed['is_default'] and t.ttc_s is not None else ''}")

            if t.in_path:
                t.path_frames += 1
                t.max_path_overlap = max(t.max_path_overlap, result["overlap_ratio"])
                if t.first_path_frame is None:
                    t.first_path_frame = frame_no
                    t.path_entry_distance_m = t.distance_m
                if t.confirmed:
                    danger_now.append(t)
        if danger_now:
            frames_with_path_risk += 1

        vehicle_visible = []
        if use_vehicles:
            vehicle_visible, vehicle_new = vehicle_tracker.update(
                vehicle_detector.detect(frame), frame_no, timestamp)
            vehicle_total += len(vehicle_new)

        scene_level, scene_rank, scene_track = scene_risk(danger_now)
        risk_frames[scene_level] += 1
        scene_color = config.RISK_COLORS[scene_level]

        # ---- draw ----
        draw_path(frame, path.polygon, danger=bool(danger_now), color=scene_color)
        if config.SHOW_HORIZON_LINE:
            draw_horizon(frame, distance.horizon_row)
        if use_vehicles:
            draw_vehicles(frame, vehicle_visible)
        draw_tracks(frame, visible, path_mode=True, show_distance=True, show_ttc=True, show_risk=True)

        nearest = min((t for t in danger_now if t.distance_m is not None),
                      key=lambda t: t.distance_m, default=None)
        if nearest is None:
            near_line = "Nearest damage in path: none"
            ttc_line = ("Time to impact: -", WHITE)
        else:
            near_line = f"Nearest damage in path: ~{nearest.distance_m:.0f} m (estimated)"
            ttc_text = f"Time to impact: {format_ttc(nearest.ttc_s, nearest.ttc_status)}"
            if speed["is_default"] and nearest.ttc_s is not None:
                ttc_line = (ttc_text + "  (default speed - test)", ORANGE)
            else:
                ttc_line = (ttc_text, WHITE)
        gps_line = ("GPS: no GPS file" if gps is None else "GPS: no fix" if fix is None
                    else f"GPS {fix['latitude']:.5f}, {fix['longitude']:.5f}")
        risk_line = (f"RISK: {config.RISK_DISPLAY[scene_level]}"
                     + (f"  ({scene_track.class_name} ID {scene_track.id})" if scene_track else ""),
                     scene_color)
        lines = [risk_line,
                 near_line,
                 ttc_line,
                 gps_line,
                 speed_text(speed),
                 f"Damage events: {sum(damage_counts.values())}"]
        draw_banner(frame, lines)
        writer.write(frame)

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 9 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    if config.SHOW_PREVIEW:
    	try:
         cv2.destroyAllWindows()
    	except cv2.error:
         pass

    # ---- save the events ----
    events = damage_tracker.confirmed_tracks()
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "damage_type", "confidence", "frame_number", "video_time_s",
                    "timestamp", "latitude", "longitude",
                    "speed_kmh", "speed_mps", "speed_source", "speed_is_default",
                    "distance_m", "ttc_seconds", "min_ttc_in_path_s", "severity",
                    "max_risk_level", "risk_reasons",
                    "damage_latitude", "damage_longitude",
                    "ever_in_path", "est_min_distance_m", "gps_status"])
        for t in events:
            w.writerow([t.id, t.class_name, f"{t.best_confidence:.3f}", t.confirmed_frame,
                        f"{t.confirmed_time:.2f}",
                        t.gps_time.isoformat(" ", "seconds") if t.gps_time else "",
                        num(t.latitude, 6), num(t.longitude, 6),
                        num(t.speed_kmh, 1), num(t.speed_mps, 2), t.speed_source,
                        t.speed_source == "default",
                        num(t.first_distance_m, 1), num(t.first_ttc_s, 2), num(t.min_ttc_s, 2), "",
                        config.RISK_DISPLAY[t.max_risk_level], "; ".join(t.max_risk_reasons),
                        num(t.damage_latitude, 6), num(t.damage_longitude, 6),
                        t.path_frames > 0, num(t.min_distance_m, 1), t.gps_status])

    default_events = sum(1 for t in events if t.speed_source == "default")
    print("\nDONE")
    print(f"  Frames processed:          {frame_no}")
    print(f"  Confirmed damage events:   {len(events)}  {damage_counts}")
    if speeds:
        print(f"  Speed during the video:    min {min(speeds):.0f} / average {sum(speeds)/len(speeds):.0f}"
              f" / max {max(speeds):.0f} km/h")
    print(f"  Speed source (frames):     GPS {speed_frames['gps']}, "
          f"computed {speed_frames['computed_from_positions']}, DEFAULT {speed_frames['default']}")
    by_level = {name: sum(1 for t in events if t.max_risk_level == name) for name in config.RISK_LEVELS}
    print("  Events by highest risk reached:  " +
          ", ".join(f"{config.RISK_DISPLAY[n]} {c}" for n, c in by_level.items()))
    print("  Frames at each risk level:       " +
          ", ".join(f"{config.RISK_DISPLAY[n]} {c}" for n, c in risk_frames.items()))
    ttcs = [t.min_ttc_s for t in events if t.min_ttc_s is not None]
    if ttcs:
        print(f"  Shortest time-to-impact for damage in our path: {min(ttcs):.1f} s")
    if default_events:
        print(f"  WARNING: {default_events} of {len(events)} events use the DEFAULT speed "
              f"({config.DEFAULT_SPEED_KMH} km/h). Time-to-impact for them is only a test value.")
    print(f"  Saved video:  {output_video}")
    print(f"  Saved events: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 9: Unified Multi-Tier Risk Analysis & Collision Prediction"
    )
    parser.add_argument(
        "--video", "--source", dest="video",
        default=config.VIDEO_PATH,
        help="Path to input video file or camera index (default: config.VIDEO_PATH)"
    )
    parser.add_argument(
        "--gps",
        default=getattr(config, "GPS_PATH", os.path.join(config.BASE_DIR, "data", "gps_track.csv")),
        help="Path to synchronized GPS CSV telemetry file"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live preview window while processing"
    )
    parser.add_argument(
        "--output-video",
        default=config.OUTPUT_VIDEO_PHASE9,
        help="Output path for annotated video"
    )
    parser.add_argument(
        "--output-csv",
        default=config.DAMAGE_CSV_PHASE9,
        help="Output path for damage events CSV"
    )
    args = parser.parse_args()

    main(
        video_path=args.video,
        gps_path=args.gps,
        show=args.show,
        output_video=args.output_video,
        output_csv=args.output_csv
    )

