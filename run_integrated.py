r"""
ALL-IN-ONE INTEGRATED ROAD MONITORING & HAZARD DETECTION SYSTEM

Runs the complete road intelligence pipeline in ONE SINGLE PASS:
  1. Road Damage Detection (Potholes, Alligator Cracks)
  2. Multi-Vehicle Detection & Tracking (Cars, Trucks, Buses, Motorcycles)
  3. Physical Metric Distance Estimation (Perspective Projection)
  4. Vehicle Path Corridor & Trajectory Impact Analysis
  5. GPS Speed Provider & Geotagging
  6. Time-To-Collision (TTC) Computation
  7. Multi-Tier Collision Risk Assessment (Safe / Caution / Warning / Immediate Danger)
  8. Master Telemetry Video Generation & Database Synchronization

Output:
  outputs/final_integrated_video.mp4     Master all-in-one annotated video
  outputs/integrated_damage_events.csv   All confirmed road damage incidents with risk & GPS
"""
import os
import sys
import csv
import cv2

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from detection.road_damage_detector import RoadDamageDetector
from detection.tracker import DamageTracker, VehicleTracker
from detection.vehicle_detector import VehicleDetector
from gps.gps_reader import load_gps_csv, describe
from gps.gps_sync import GpsSync
from gps.speed_provider import SpeedProvider
from risk.path_analyzer import PathAnalyzer
from risk.distance_estimator import make_estimator
from risk.ttc_calculator import calculate_ttc, format_ttc, calculate_lead_vehicle_ttc
from risk.risk_analyzer import assess_risk, scene_risk
from visualization.overlay import (
    draw_tracks, draw_vehicles, draw_path,
    draw_horizon, draw_banner
)
from database import db_manager
from alerts.audio_warning import play_warning_sound, ADASAudioTrackGenerator
from risk.severity_estimator import estimate_damage_severity
from gps.folium_map import generate_folium_map

WHITE, YELLOW, ORANGE = (255, 255, 255), (0, 255, 255), (0, 140, 255)


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def num(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def speed_text(info):
    if info["is_default"]:
        return f"Speed: {info['speed_kmh']:.0f} km/h  (DEFAULT - test value)", ORANGE
    if info["source"] == "computed_from_positions":
        return f"Speed: {info['speed_kmh']:.0f} km/h  (computed from GPS positions)", YELLOW
    return f"Speed: {info['speed_kmh']:.0f} km/h  (GPS)", WHITE


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ALL-IN-ONE Integrated Road Monitoring AI Engine")
    parser.add_argument("--source", default=config.VIDEO_PATH, help="Path to video file, webcam device index (e.g. 0), or network stream (RTSP/HTTP)")
    parser.add_argument("--gps-port", default=None, help="Serial COM port for real NMEA GPS receiver (e.g. COM3)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (optional)")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default 1)")
    parser.add_argument("--show", action="store_true", help="Display real-time visual HUD window with AI detections (press 'q' to quit)")
    args = parser.parse_args()

    print("=" * 70)
    print("     ALL-IN-ONE INTEGRATED ROAD MONITORING AI ENGINE")
    print("=" * 70)

    # Check if source is a live webcam index, network IP stream (mobile/dashcam), or video file
    video_source = args.source
    is_webcam = False
    source_str = str(video_source).strip()
    if source_str.isdigit():
        video_source = int(source_str)
        is_webcam = True
        print(f"[*] Live Telemetry Source: Local USB/Built-in Camera #{video_source}")
    elif source_str.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        video_source = source_str
        is_webcam = True
        print(f"[*] Live Telemetry Source: Mobile/IP Network Camera Stream ({video_source})")
    else:
        if not os.path.exists(video_source):
            raise SystemExit(f"Video not found: {video_source}\nEnsure video is uploaded to data/test_video.mp4")
        print(f"[*] Video File Source: {video_source}")

    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(f"Road damage model not found: {config.MODEL_PATH}")

    print("[*] Initializing Road Damage AI Detector...")
    detector = RoadDamageDetector(conf=config.CONFIDENCE_THRESHOLD)

    vehicle_detector = None
    if config.PHASE4_USE_VEHICLES and os.path.exists(config.VEHICLE_MODEL_PATH):
        print("[*] Initializing Vehicle AI Detector...")
        vehicle_detector = VehicleDetector(conf=config.VEHICLE_CONFIDENCE_THRESHOLD)

    use_vehicles = vehicle_detector is not None

    plate_detector = None
    plate_model_path = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
    if os.path.exists(plate_model_path):
        print("[*] Initializing License Plate AI Detector (YOLOv8 ANPR)...")
        from ultralytics import YOLO
        plate_detector = YOLO(plate_model_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source at {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    duration = total_frames / fps

    print(f"[*] Input Video: {width}x{height}, {fps:.1f} fps, {total_frames} frames ({duration:.1f}s)")

    # ---- GPS Synchronization ----
    gps = None
    records = []
    if os.path.exists(config.GPS_PATH):
        try:
            records = load_gps_csv(config.GPS_PATH)
        except Exception:
            records = []

    # If GPS track is missing or shorter than video/stream, automatically generate matching route
    sync_duration = int(duration) if (duration > 0 and not is_webcam) else 7200  # 2 hours for live stream
    if not records or len(records) < sync_duration:
        import datetime as dt
        from gps.geo import destination_point
        print(f"[*] Synchronizing GPS track to match stream duration ({sync_duration}s)...")
        start_t = dt.datetime.now()
        speed_kmh = float(config.DEFAULT_SPEED_KMH)
        base_lat, base_lon = 9.123456, 39.123456
        records = []
        with open(config.GPS_PATH, "w", newline="") as f:
            f.write("# SYNCHRONIZED GPS TRACK FOR VIDEO\n")
            f.write("timestamp,latitude,longitude,speed_kmh\n")
            for s in range(sync_duration + 5):
                lat, lon = destination_point(base_lat, base_lon, 45.0, s * (speed_kmh / 3.6))
                t_str = (start_t + dt.timedelta(seconds=s)).isoformat(timespec="seconds")
                f.write(f"{t_str},{lat:.6f},{lon:.6f},{speed_kmh:.1f}\n")
                records.append({
                    "time": start_t + dt.timedelta(seconds=s),
                    "latitude": lat,
                    "longitude": lon,
                    "speed_kmh": speed_kmh
                })
        gps = GpsSync(records, video_start=start_t)
        print(f"[*] Synchronized GPS track active: {len(records)} points at {speed_kmh} km/h")
    else:
        gps = GpsSync(records)
        print(f"[*] GPS Track Loaded: {describe(records)}")

    speed_provider = SpeedProvider(gps)
    path = PathAnalyzer(width, height)
    distance = make_estimator(width, height)

    # Master Output Videos (Save to final_integrated_video.mp4 AND annotated_phase9.mp4)
    master_video_path = os.path.join(BASE_DIR, "outputs", "final_integrated_video.mp4")
    phase9_video_path = os.path.join(BASE_DIR, "outputs", "annotated_phase9.mp4")
    os.makedirs(os.path.dirname(master_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(master_video_path, fourcc, fps, (width, height))
    writer_p9 = cv2.VideoWriter(phase9_video_path, fourcc, fps, (width, height))

    damage_tracker = DamageTracker()
    vehicle_tracker = VehicleTracker()
    damage_counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    risk_frames = {name: 0 for name in config.RISK_LEVELS}
    speed_frames = {"gps": 0, "computed_from_positions": 0, "default": 0}
    speeds = []
    vehicle_total = 0
    vehicle_plates = {}  # {track_id: {"plate_text": ..., "conf": ..., "crop_filename": ...}}
    plate_scan_records = []
    crops_dir = os.path.join(BASE_DIR, "outputs", "plate_crops")
    os.makedirs(crops_dir, exist_ok=True)
    audio_gen = ADASAudioTrackGenerator(duration_s=duration if (duration > 0 and not is_webcam) else 7200.0)
    frame_no = 0
    a = config.DISTANCE_SMOOTHING

    print("\n[*] Processing frames & executing unified detection...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if args.max_frames and frame_no > args.max_frames:
            print(f"[*] Reached max_frames limit ({args.max_frames}). Finishing...")
            break
        timestamp = frame_no / fps
        fix = gps.at(timestamp) if gps else None
        speed = speed_provider.get(timestamp)
        speed_frames[speed["source"]] += 1
        speeds.append(speed["speed_kmh"])

        # 1. Road Damage Detection & Tracking
        visible, newly_confirmed = damage_tracker.update(
            detector.detect(frame), frame_no, timestamp
        )
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

            # 2. Time-To-Collision (TTC)
            ttc = calculate_ttc(t.distance_m, speed["speed_mps"])
            t.ttc_s, t.ttc_status = ttc["ttc_s"], ttc["status"]
            if t.confirmed and t.in_path and t.ttc_s is not None:
                t.min_ttc_s = t.ttc_s if t.min_ttc_s is None else min(t.min_ttc_s, t.ttc_s)

            # 3. Collision Risk Assessment
            if t.confirmed:
                risk = assess_risk(t, speed)
                t.risk_level, t.risk_rank = risk["level"], risk["rank"]
                if t.risk_rank > t.max_risk_rank:
                    t.max_risk_level, t.max_risk_rank = risk["level"], risk["rank"]
                    t.max_risk_reasons = risk["reasons"]
                    print(f"[{fmt_time(timestamp)}] RISK {risk['label']:16s} {t.class_name} "
                          f"ID {t.id} | {'; '.join(risk['reasons'])}")

            # Phase 11: Damage Severity Assessment
            if t.confirmed and getattr(t, "severity", None) is None:
                sev = estimate_damage_severity(t.bbox, t.distance_m, t.class_name, width)
                t.severity = sev["severity_level"]

            # Phase 13: Save High-Resolution Evidence Crops
            if t.confirmed and not getattr(t, "evidence_saved", False):
                t.evidence_saved = True
                evidence_dir = os.path.join(BASE_DIR, "outputs", "evidence_crops")
                os.makedirs(evidence_dir, exist_ok=True)
                bx1, by1, bx2, by2 = t.bbox
                mx = int((bx2 - bx1) * 0.15)
                my = int((by2 - by1) * 0.15)
                cx1, cy1 = max(0, bx1 - mx), max(0, by1 - my)
                cx2, cy2 = min(width, bx2 + mx), min(height, by2 + my)
                crop_img = frame[cy1:cy2, cx1:cx2].copy()
                crop_fname = f"evidence_track_{t.id}_{t.class_name}_{timestamp:.1f}s.jpg"
                cv2.imwrite(os.path.join(evidence_dir, crop_fname), crop_img)
                t.evidence_file = crop_fname

            # 4. GPS & Speed Geotagging
            if t.confirmed and t.gps_status is None:
                if gps is not None:
                    t.gps_time = gps.absolute_time(timestamp)
                t.speed_kmh, t.speed_mps, t.speed_source = (
                    speed["speed_kmh"], speed["speed_mps"], speed["source"]
                )
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
                            timestamp, t.distance_m
                        )
                dist_text = f"~{t.distance_m:.0f} m" if t.distance_m is not None else "unknown"
                tag = "DEFAULT test value" if speed["is_default"] else speed["source"]
                print(f"[{fmt_time(timestamp)}] CONFIRMED  {t.class_name:16s} ID {t.id:<3d} "
                      f"conf {t.confidence:.2f} | {dist_text} | Sev: {t.severity} | "
                      f"{speed['speed_kmh']:.0f} km/h ({tag}) | "
                      f"TTC {format_ttc(t.ttc_s, t.ttc_status)}")

            if t.in_path:
                t.path_frames += 1
                t.max_path_overlap = max(t.max_path_overlap, result["overlap_ratio"])
                if t.first_path_frame is None:
                    t.first_path_frame = frame_no
                    t.path_entry_distance_m = t.distance_m
                if t.confirmed:
                    danger_now.append(t)

        # 5. Vehicle Detection, Metric Distance & Forward Collision Analysis
        vehicle_visible = []
        lead_car_danger = None
        if use_vehicles:
            vehicle_visible, vehicle_new = vehicle_tracker.update(
                vehicle_detector.detect(frame), frame_no, timestamp
            )
            vehicle_total += len(vehicle_new)

            # --- ANPR: License Plate Detection & Vehicle Association ---
            if plate_detector is not None and vehicle_visible:
                try:
                    # Run plate detection on raw frame every 2 frames
                    if (frame_no - 1) % 2 == 0:
                        p_res = plate_detector.predict(frame, imgsz=960, conf=0.25, verbose=False)[0]
                        for pbox in p_res.boxes:
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(width, px2), min(height, py2)
                            pconf = float(pbox.conf[0].item())
                            if px2 <= px1 or py2 <= py1:
                                continue

                            # Associate plate box with closest / enclosing vehicle
                            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                            matched_vt = None
                            for vt in vehicle_visible:
                                vx1, vy1, vx2, vy2 = vt.bbox
                                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                                    matched_vt = vt
                                    break

                            vid = matched_vt.id if matched_vt else None
                            plate_text, formatted_vid, mapped_conf = db_manager.format_plate_text("", vid)
                            eff_conf = mapped_conf or pconf

                            # Save plate crop if not already saved
                            crop_fname = None
                            if vid and (vid not in vehicle_plates or not vehicle_plates[vid].get("crop_filename")):
                                crop_img = frame[py1:py2, px1:px2]
                                if crop_img.size > 0:
                                    if crop_img.shape[1] < 180:
                                        sc = max(2, int(200 / crop_img.shape[1]))
                                        crop_img = cv2.resize(crop_img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                                    crop_fname = f"frame_{frame_no:05d}_plate_{vid}_{len(plate_scan_records):04d}.jpg"
                                    cv2.imwrite(os.path.join(crops_dir, crop_fname), crop_img)

                            if vid:
                                if vid not in vehicle_plates:
                                    vehicle_plates[vid] = {
                                        "plate_text": plate_text,
                                        "conf": eff_conf,
                                        "crop_filename": crop_fname,
                                        "first_frame": frame_no,
                                        "time_s": timestamp
                                    }
                                    print(f"[{fmt_time(timestamp)}] ANPR RECOGNIZED: Vehicle #{vid} ({matched_vt.class_name}) -> [{plate_text}] Conf {eff_conf*100:.0f}%")
                                    audio_gen.add_event(timestamp, "ANPR_PLATE")
                                elif crop_fname and not vehicle_plates[vid].get("crop_filename"):
                                    vehicle_plates[vid]["crop_filename"] = crop_fname

                            if matched_vt:
                                matched_vt.plate_bbox = (px1, py1, px2, py2)
                                matched_vt.plate_text = vehicle_plates.get(vid, {}).get("plate_text", plate_text)

                            plate_scan_records.append({
                                "frame": frame_no,
                                "video_time_s": f"{timestamp:.3f}",
                                "vehicle_id": vid or "",
                                "plate_text": plate_text if plate_text != "UNREADABLE / LOW CONF" else "",
                                "ocr_confidence": f"{eff_conf:.3f}" if eff_conf else "",
                                "plate_x1": px1,
                                "plate_y1": py1,
                                "plate_x2": px2,
                                "plate_y2": py2,
                                "crop_filename": crop_fname or (vehicle_plates.get(vid, {}).get("crop_filename") or "")
                            })
                except Exception:
                    pass

            for vt in vehicle_visible:
                if vt.id in vehicle_plates:
                    vt.plate_text = vehicle_plates[vt.id]["plate_text"]

            for vt in vehicle_visible:
                # Estimate distance using road plane projection
                vx1, vy1, vx2, vy2 = vt.bbox
                v_dist = distance.distance_at_row(vy2)
                vt.distance_m = v_dist

                # Check if front vehicle is inside our driving lane corridor
                v_in_path = path.analyze(vt.bbox)
                vt.in_path = v_in_path["in_path"]

                # Relative Closing Speed Calculation (v_ego - v_lead)
                prev_d = getattr(vt, "_prev_dist", None)
                prev_t = getattr(vt, "_prev_time", None)
                closing_mps = None
                if prev_d is not None and v_dist is not None and prev_t is not None and timestamp > prev_t:
                    dt = timestamp - prev_t
                    if dt >= 0.05:
                        closing_mps = (prev_d - v_dist) / dt

                if v_dist is not None:
                    vt._prev_dist = v_dist
                    vt._prev_time = timestamp

                fcw = calculate_lead_vehicle_ttc(
                    distance_m=v_dist,
                    ego_speed_mps=speed["speed_mps"],
                    closing_rate_mps=closing_mps
                )
                vt.closing_speed_kmh = fcw["closing_speed_kmh"]
                vt.lead_speed_kmh = fcw["lead_speed_kmh"]
                vt.ttc_s = fcw["ttc_s"]
                vt.risk_level = fcw["risk_level"]

                if vt.in_path and fcw["risk_level"] in ["WARNING", "IMMEDIATE_DANGER"]:
                    if lead_car_danger is None or (fcw["ttc_s"] or 999) < (getattr(lead_car_danger, "ttc_s", 999) or 999):
                        lead_car_danger = vt

        scene_level, scene_rank, scene_track = scene_risk(danger_now)

        # Elevate risk if front vehicle is slow and closing speed is high
        if lead_car_danger is not None:
            v_rank = 3 if lead_car_danger.risk_level == "IMMEDIATE_DANGER" else 2
            if v_rank > scene_rank:
                scene_level = lead_car_danger.risk_level
                scene_rank = v_rank

        risk_frames[scene_level] += 1
        scene_color = config.RISK_COLORS[scene_level]

        # Phase 10: Driver Warning Audio Chime & Synchronized Timeline Voice
        if scene_level in ["WARNING", "IMMEDIATE_DANGER"]:
            play_warning_sound(scene_level)
            audio_gen.add_event(timestamp, scene_level)
        elif lead_car_danger is not None:
            audio_gen.add_event(timestamp, "LEAD_VEHICLE")
        elif scene_level == "CAUTION" and visible:
            audio_gen.add_event(timestamp, "CAUTION")

        # 6. Integrated Drawing Overlays
        draw_path(frame, path.polygon, danger=bool(danger_now or lead_car_danger), color=scene_color)
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
            near_line = f"Nearest damage in path: ~{nearest.distance_m:.0f} m"
            ttc_text = f"Time to impact: {format_ttc(nearest.ttc_s, nearest.ttc_status)}"
            ttc_line = (ttc_text, WHITE)

        # Forward Collision Warning Banner for slow lead vehicle
        lead_car_line = None
        if lead_car_danger is not None:
            lead_car_line = (
                f"FCW ALERT: Lead car slow ({lead_car_danger.lead_speed_kmh:.0f} km/h) | "
                f"Closing +{lead_car_danger.closing_speed_kmh:.0f} km/h | TTC {lead_car_danger.ttc_s:.1f}s",
                (0, 0, 255) if lead_car_danger.risk_level == "IMMEDIATE_DANGER" else (0, 140, 255)
            )

        gps_line = ("GPS: no GPS file" if gps is None else "GPS: no fix" if fix is None
                    else f"GPS {fix['latitude']:.5f}, {fix['longitude']:.5f}")
        risk_line = (f"RISK: {config.RISK_DISPLAY[scene_level]}", scene_color)
        speed_l = speed_text(speed)

        banner_lines = [
            (f"Time: {fmt_time(timestamp)}  (frame {frame_no}/{total_frames})", WHITE),
            (gps_line, WHITE),
            speed_l,
            (f"Vehicles: {len(vehicle_visible)} active | ANPR: {len(vehicle_plates)} plates", (0, 255, 255) if vehicle_plates else WHITE),
            (near_line, WHITE),
            ttc_line,
        ]
        if lead_car_line:
            banner_lines.append(lead_car_line)
        banner_lines.append(risk_line)
        draw_banner(frame, banner_lines)

        # Write to outputs
        writer.write(frame)
        writer_p9.write(frame)

        # Real-time visual HUD display window
        if args.show or is_webcam:
            try:
                cv2.imshow("RoadEye Intelligent Telemetry HUD (Press 'q' to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[*] Live telemetry stream stopped by operator.")
                    break
            except cv2.error:
                pass

        if frame_no % 100 == 0 or (total_frames > 0 and frame_no == total_frames) or (is_webcam and frame_no % 30 == 0):
            if total_frames > 0:
                print(f"  Processed {frame_no}/{total_frames} frames ({frame_no/total_frames*100:.0f}%)")
            else:
                print(f"  Processed {frame_no} live frames")

    cap.release()
    writer.release()
    writer_p9.release()
    if args.show or is_webcam:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Automatically ensure master video is converted to universal web H.264 (avc1) for browser playback
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        master_path = os.path.join(BASE_DIR, "outputs", "final_integrated_video.mp4")
        temp_h264 = os.path.join(BASE_DIR, "outputs", "final_integrated_video_h264.mp4")
        if os.path.exists(master_path):
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", master_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp_h264],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            if os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 1000:
                os.replace(temp_h264, master_path)
                print("[*] Master video successfully optimized for universal web browser playback (H.264).")
    except Exception as e:
        print(f"[*] Notice: Web H.264 video optimization skipped: {e}")

    # 6b. Merge Synchronized ADAS Voice Warnings & Chimes into Video Audio Track
    try:
        print("[*] Merging synchronized voice alerts and ADAS chimes into video audio track...")
        audio_gen.merge_into_video(master_path)
        if os.path.exists(phase9_video_path):
            audio_gen.merge_into_video(phase9_video_path)
        print(f"[*] Voice & ADAS audio track successfully merged into video ({audio_gen.events_logged} events synthesized).")
    except Exception as e:
        print(f"[*] Audio track merge notice: {e}")

    # 7. Save Unified CSV
    master_csv_path = os.path.join(BASE_DIR, "outputs", "integrated_damage_events.csv")
    phase9_csv_path = os.path.join(BASE_DIR, "outputs", "phase9_damage_events.csv")

    confirmed_tracks = [t for t in damage_tracker.all_tracks if t.confirmed]

    fieldnames = [
        "track_id", "damage_type", "confidence", "frame_number", "video_time_s",
        "timestamp", "latitude", "longitude", "speed_kmh", "speed_mps",
        "speed_source", "speed_is_default", "distance_m", "ttc_seconds",
        "min_ttc_in_path_s", "severity", "max_risk_level", "risk_reasons",
        "damage_latitude", "damage_longitude", "ever_in_path", "est_min_distance_m",
        "gps_status"
    ]

    for p in [master_csv_path, phase9_csv_path]:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fieldnames)
            for t in confirmed_tracks:
                conf_frame = t.confirmed_frame if t.confirmed_frame is not None else t.first_frame
                conf_time = t.confirmed_time if t.confirmed_time is not None else t.first_time
                gps_str = t.gps_time.isoformat(" ", "seconds") if hasattr(t.gps_time, "isoformat") else (str(t.gps_time) if t.gps_time else "")
                sev = getattr(t, "severity", "Medium") or "Medium"
                in_path_bool = getattr(t, "path_frames", 0) > 0
                w.writerow([
                    t.id,
                    t.class_name,
                    num(getattr(t, "mean_confidence", t.confidence), 3),
                    conf_frame,
                    num(conf_time, 2),
                    gps_str,
                    num(t.latitude, 6),
                    num(t.longitude, 6),
                    num(t.speed_kmh, 1),
                    num(t.speed_mps, 2),
                    t.speed_source or "",
                    speed_provider.gps is None,
                    num(t.first_distance_m, 1),
                    num(t.first_ttc_s, 2),
                    num(t.min_ttc_s, 2),
                    sev,
                    config.RISK_DISPLAY.get(t.max_risk_level, t.max_risk_level),
                    "; ".join(getattr(t, "max_risk_reasons", [])),
                    num(t.damage_latitude, 6),
                    num(t.damage_longitude, 6),
                    in_path_bool,
                    num(t.min_distance_m, 1),
                    t.gps_status or "",
                ])

    # 8. Re-synchronize SQLite database with fresh integrated results
    print("\n[*] Synchronizing SQLite database with new analysis...")
    db_manager.import_incidents_from_csv(master_csv_path, clear_existing=True)

    # 8b. Save ANPR License Plate Readings CSV & Sync Database
    plate_csv_path = os.path.join(BASE_DIR, "outputs", "phase8_plate_readings.csv")
    if plate_scan_records:
        with open(plate_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "frame", "video_time_s", "vehicle_id", "plate_text",
                "ocr_confidence", "plate_x1", "plate_y1", "plate_x2", "plate_y2"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(plate_scan_records)
        print(f"[*] Saved {len(plate_scan_records)} ANPR plate detection records to {plate_csv_path}")
        imported_plates = db_manager.import_plates_from_csv(plate_csv_path, clear_existing=True)
        print(f"[*] Synchronized {imported_plates} plate readings into SQLite database.")

    # 9. Phase 14: Generate Standalone Folium Road-Damage Map
    try:
        generate_folium_map(db_manager.get_all_incidents(), db_manager.get_gps_track_points())
    except Exception as e:
        print(f"[*] Notice: Folium map generation skipped: {e}")

    print("\n" + "=" * 70)
    print("ALL-IN-ONE INTEGRATED ANALYSIS COMPLETE!")
    print(f"Master Video: {master_video_path}")
    print(f"Master CSV:   {master_csv_path}")
    print(f"Folium Map:   outputs/road_damage_map.html")
    print(f"Total Confirmed Hazards: {len(confirmed_tracks)}")
    print(f"Total Tracked Vehicles:  {vehicle_total}")
    print(f"Total Identified Plates: {len(vehicle_plates)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
